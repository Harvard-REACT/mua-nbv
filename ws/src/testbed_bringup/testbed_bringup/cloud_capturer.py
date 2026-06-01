#!/usr/bin/env python3
import json
import os
import threading
import time
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time as RclTime
from rcl_interfaces.msg import SetParametersResult

from std_srvs.srv import Trigger
from std_msgs.msg import String as StringMsg
from std_msgs.msg import Header
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2

import tf2_ros

from mua_nbv_py_utils.transforms import quat_to_rot_np
from mua_nbv_py_utils.pcd_io import write_pcd_xyz_ascii
from mua_nbv_common.ros_helpers import stamp_str


def _is_epoch_like_stamp(stamp: TimeMsg) -> bool:
    """
    Heuristic: on real robots with system time, stamps are ~1e9 seconds since epoch.
    In simulation, stamps can be small. But for TF lookup on testbed we must avoid
    accidentally using a logical step-id token (sec=k) as a TF time.
    """
    try:
        return int(stamp.sec) > 1_000_000
    except Exception:
        return False


def _depth_image_to_xyz(
    img: Image,
    cam: CameraInfo,
    *,
    pixel_stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """
    Convert depth image to Nx3 xyz points in the *camera optical frame*.
    Supports encodings:
      - 16UC1 (uint16 millimeters)
      - 32FC1 (float32 meters)
    """
    enc = (img.encoding or "").lower()

    if enc in ("16uc1", "mono16"):
        depth = np.frombuffer(img.data, dtype=np.uint16).reshape(img.height, img.width)
        z = depth.astype(np.float32) * 1e-3  # mm -> m
    elif enc == "32fc1":
        z = np.frombuffer(img.data, dtype=np.float32).reshape(img.height, img.width)
    else:
        raise RuntimeError(f"Unsupported depth encoding '{img.encoding}' (expected 16UC1 or 32FC1)")

    if pixel_stride < 1:
        pixel_stride = 1
    z = z[::pixel_stride, ::pixel_stride]

    # Intrinsics
    fx = float(cam.k[0])
    fy = float(cam.k[4])
    cx = float(cam.k[2])
    cy = float(cam.k[5])

    if fx <= 0.0 or fy <= 0.0:
        raise RuntimeError(f"Bad intrinsics fx={fx}, fy={fy}")

    h, w = z.shape
    # Pixel coordinates in the *original* image coordinate system
    us = (np.arange(w, dtype=np.float32) * pixel_stride)
    vs = (np.arange(h, dtype=np.float32) * pixel_stride)
    uu, vv = np.meshgrid(us, vs)

    # Valid mask
    m = np.isfinite(z)
    if min_depth_m > 0.0:
        m &= (z >= float(min_depth_m))
    if max_depth_m > 0.0 and max_depth_m > min_depth_m:
        m &= (z <= float(max_depth_m))

    if not np.any(m):
        return np.empty((0, 3), dtype=np.float32)

    z1 = z[m]
    x1 = (uu[m] - cx) * z1 / fx
    y1 = (vv[m] - cy) * z1 / fy

    pts = np.stack([x1, y1, z1], axis=1).astype(np.float32, copy=False)
    return pts


def _write_ppm(path: str, *, w: int, h: int, rgb_bytes: bytes):
    # Binary PPM (P6)
    with open(path, "wb") as f:
        f.write(f"P6\n{int(w)} {int(h)}\n255\n".encode("ascii"))
        f.write(rgb_bytes)


def _write_pgm(path: str, *, w: int, h: int, gray_bytes: bytes):
    # Binary PGM (P5)
    with open(path, "wb") as f:
        f.write(f"P5\n{int(w)} {int(h)}\n255\n".encode("ascii"))
        f.write(gray_bytes)

def _largest_component_voxel_filter(
    pts: np.ndarray,
    *,
    voxel_size_m: float,
    min_component_points: int,
) -> np.ndarray:
    """
    Keep only the largest connected component under a voxel-grid adjacency.

    - Voxel size is the "gap" threshold: two clusters separated by >~voxel_size_m will
      generally fall into different components.
    - Connectivity: 26-neighborhood (including diagonals).

    This is a cheap, dependency-free alternative to DBSCAN / KDTree clustering.
    """
    if pts.size == 0:
        return pts
    vs = float(max(1e-6, voxel_size_m))
    # Compute voxel indices
    vid = np.floor(pts.astype(np.float64, copy=False) / vs).astype(np.int64, copy=False)
    voxel_map: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(int(vid.shape[0])):
        k = (int(vid[i, 0]), int(vid[i, 1]), int(vid[i, 2]))
        voxel_map.setdefault(k, []).append(i)

    keys = list(voxel_map.keys())
    if not keys:
        return pts

    # Precompute neighbor offsets (26-neighborhood)
    nbrs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == dy == dz == 0)]

    visited: set[Tuple[int, int, int]] = set()
    best_component_voxels: List[Tuple[int, int, int]] = []
    best_n = 0

    for k0 in keys:
        if k0 in visited:
            continue
        # BFS
        stack = [k0]
        visited.add(k0)
        comp: List[Tuple[int, int, int]] = []
        n_pts = 0
        while stack:
            k = stack.pop()
            comp.append(k)
            n_pts += len(voxel_map.get(k, []))
            x, y, z = k
            for (dx, dy, dz) in nbrs:
                kn = (x + dx, y + dy, z + dz)
                if kn in voxel_map and kn not in visited:
                    visited.add(kn)
                    stack.append(kn)
        if n_pts > best_n:
            best_n = n_pts
            best_component_voxels = comp

    if best_n < int(min_component_points):
        # Not enough points in any component: keep original.
        return pts

    keep_idx: List[int] = []
    for k in best_component_voxels:
        keep_idx.extend(voxel_map.get(k, []))
    keep = np.array(keep_idx, dtype=np.int64)
    return pts[keep]


class CloudCapturer(Node):
    """
    Testbed equivalent of simulation_bringup/cloud_capturer.py:
      depth Image + CameraInfo -> PointCloud2 in camera frame -> TF into target/base_link -> publish /target/points

    Step tokens:
      - In dynamic mode, /experiment/step_stamp should be produced by the stepper/coordinator (not by capturer).
      - For static convenience (no coordinator), capturer can optionally publish /experiment/step_stamp when capture is called.
    """

    def __init__(self):
        super().__init__("cloud_capturer")

        # ---- Params ----
        self.declare_parameter("input_depth_topic", "/pursuer/camera/depth/image_rect_raw")
        self.declare_parameter("input_camera_info_topic", "/pursuer/camera/depth/camera_info")
        # Optional RGB capture (slow rate is OK; we will wait during capture if enabled)
        self.declare_parameter("input_rgb_topic", "/pursuer/camera/color/image_raw")
        self.declare_parameter("save_rgb_enable", False)
        self.declare_parameter("rgb_wait_for_new_sec", 5.0)
        # If true, NEVER fall back to a cached RGB frame. If no *new* RGB arrives within
        # rgb_wait_for_new_sec during a capture, we simply skip saving RGB for that capture.
        self.declare_parameter("rgb_require_new", False)
        self.declare_parameter("output_cloud_topic", "/target/points")
        self.declare_parameter("service_name", "/experiment/capture/cloud")
        # Step token topic (normally input from stepper/coordinator; optionally published here for static convenience)
        self.declare_parameter("output_step_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("output_captured_stamp_topic", "/experiment/captured_cloud_stamp")
        self.declare_parameter("output_cam_pose_topic", "/experiment/captured_cam_pose")
        self.declare_parameter("publish_step_stamp", True)

        # Frames / TF
        self.declare_parameter("target_frame", "target/base_link")
        # Optional: also save a second cloud in a "GT" target frame (e.g., Optitrack VRPN base_link).
        # This does NOT affect published /target/points; it only controls extra files on disk.
        self.declare_parameter("save_gt_enable", False)
        self.declare_parameter("gt_target_frame", "target/base_link")
        self.declare_parameter("use_latest_tf", True)
        self.declare_parameter("tf_lookup_timeout_sec", 0.05)
        # Timestamp correction for multi-machine setups:
        # If depth stamps come from another machine with a different system clock, then
        # transforming depth using "latest TF" can mismatch depth time by seconds.
        # Modes:
        #  - "none": no correction (current behavior)
        #  - "auto": estimate (now - depth_stamp) online and use it to pick TF time
        #  - "fixed": use a fixed offset (sec) you provide
        self.declare_parameter("stamp_correction_mode", "none")  # "none" | "auto" | "fixed"
        self.declare_parameter("stamp_correction_fixed_offset_sec", 0.0)
        self.declare_parameter("stamp_correction_alpha", 0.05)  # EMA smoothing for auto mode
        # If the depth Image frame_id is empty, fall back to this
        self.declare_parameter("camera_frame_fallback", "pursuer/camera_depth_optical_frame")
        # Camera pose frame for voxel_map_node (target_frame <- cam_pose_frame)
        self.declare_parameter("cam_pose_frame", "pursuer/camera_depth_optical_frame")

        # Sampling / filtering
        self.declare_parameter("pixel_stride", 1)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 3.0)

        # Optional post-TF clipping in target frame (ground cut)
        self.declare_parameter("target_clip_enable", True)
        self.declare_parameter("target_min_z", 0.01)
        self.declare_parameter("target_max_z", 1.0)

        # Optional spatial filtering to drop far artifacts / keep main body
        self.declare_parameter("spatial_filter_enable", False)
        # "largest_component" keeps the largest connected component (voxel adjacency)
        self.declare_parameter("spatial_filter_mode", "largest_component")
        # This is the key knob for your request ("omit clusters >10cm away"):
        self.declare_parameter("spatial_filter_voxel_size_m", 0.10)
        self.declare_parameter("spatial_filter_min_component_points", 200)

        # Saving (match simulation_bringup feel)
        self.declare_parameter("out_dir", "debug/experiment")
        self.declare_parameter("group_by_run_id", False)
        self.declare_parameter("run_id_topic", "/experiment/run_id")
        self.declare_parameter("wait_for_inputs_sec", 2.0)
        self.declare_parameter("save_enable", True)
        # If true, keep publishing /target/points continuously from depth stream
        self.declare_parameter("publish_continuous", True)

        # ---- Read params ----
        self.depth_topic = str(self.get_parameter("input_depth_topic").value)
        self.info_topic = str(self.get_parameter("input_camera_info_topic").value)
        self.rgb_topic = str(self.get_parameter("input_rgb_topic").value)
        self.save_rgb_enable = bool(self.get_parameter("save_rgb_enable").value)
        self.rgb_wait_for_new_sec = float(self.get_parameter("rgb_wait_for_new_sec").value)
        self.rgb_require_new = bool(self.get_parameter("rgb_require_new").value)
        self.out_topic = str(self.get_parameter("output_cloud_topic").value)
        self.srv_name = str(self.get_parameter("service_name").value)
        self.step_stamp_topic = str(self.get_parameter("output_step_stamp_topic").value)
        self.captured_stamp_topic = str(self.get_parameter("output_captured_stamp_topic").value)
        self.cam_pose_topic = str(self.get_parameter("output_cam_pose_topic").value)
        self.publish_step_stamp = bool(self.get_parameter("publish_step_stamp").value)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.save_gt_enable = bool(self.get_parameter("save_gt_enable").value)
        self.gt_target_frame = str(self.get_parameter("gt_target_frame").value)
        self.use_latest_tf = bool(self.get_parameter("use_latest_tf").value)
        self.tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)
        self.camera_frame_fallback = str(self.get_parameter("camera_frame_fallback").value)
        self.cam_pose_frame = str(self.get_parameter("cam_pose_frame").value)
        self.stamp_correction_mode = str(self.get_parameter("stamp_correction_mode").value).strip()
        self.stamp_correction_fixed_offset_sec = float(self.get_parameter("stamp_correction_fixed_offset_sec").value)
        self.stamp_correction_alpha = float(self.get_parameter("stamp_correction_alpha").value)

        self.out_dir = str(self.get_parameter("out_dir").value)
        self.group_by_run_id = bool(self.get_parameter("group_by_run_id").value)
        self.run_id_topic = str(self.get_parameter("run_id_topic").value)
        self.wait_for_inputs_sec = float(self.get_parameter("wait_for_inputs_sec").value)
        self.save_enable = bool(self.get_parameter("save_enable").value)
        self.publish_continuous = bool(self.get_parameter("publish_continuous").value)

        # Allow runtime parameter updates (e.g., scripts switching frames without restart).
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---- State ----
        self._mtx = threading.Lock()
        self._cv = threading.Condition(self._mtx)
        self._last_info: Optional[CameraInfo] = None
        self._info_seq = 0
        self._last_depth: Optional[Image] = None
        self._depth_seq = 0
        self._last_rgb: Optional[Image] = None
        self._rgb_seq = 0
        self._run_id: Optional[str] = None
        self._last_step_stamp: Optional[TimeMsg] = None
        self._depth_to_local_offset_ns: Optional[int] = None  # estimated now_ns - depth_stamp_ns

        # ---- Pub/Sub ----
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        if self.save_rgb_enable and self.rgb_topic:
            self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self._on_rgb, qos_profile_sensor_data)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, 10)
        qos_token = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        )
        self.sub_step = self.create_subscription(TimeMsg, self.step_stamp_topic, self._on_step_stamp, qos_token)
        self.pub_step_stamp = self.create_publisher(TimeMsg, self.step_stamp_topic, qos_token)
        self.pub_captured_stamp = self.create_publisher(TimeMsg, self.captured_stamp_topic, qos_token)
        self.pub_cam_pose = self.create_publisher(PoseStamped, self.cam_pose_topic, qos_token)
        self.srv = self.create_service(Trigger, self.srv_name, self._on_capture)

        # Optional run_id topic (latched)
        if self.group_by_run_id:
            # Publisher is TRANSIENT_LOCAL; match QoS so we receive the last run_id
            # even if it was published before we subscribed.
            qos_run = rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            )
            self.sub_run = self.create_subscription(StringMsg, self.run_id_topic, self._on_run_id, qos_run)

        self._last_warn_time = self.get_clock().now()

        # Resolve out_dir early so it's obvious where files go (launch CWD can vary).
        self.out_dir = os.path.abspath(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📷 Depth: {self.depth_topic}")
        self.get_logger().info(f"📷 CameraInfo: {self.info_topic}")
        if self.save_rgb_enable:
            self.get_logger().info(
                f"🟥 RGB: {self.rgb_topic} (wait_for_new_sec={self.rgb_wait_for_new_sec:.1f} "
                f"require_new={bool(self.rgb_require_new)})"
            )
        self.get_logger().info(f"📢 Publishing: {self.out_topic} frame={self.target_frame}")
        self.get_logger().info(f"🔧 Service: {self.srv_name} (save_enable={self.save_enable} out_dir={self.out_dir})")
        if self.save_gt_enable:
            self.get_logger().info(f"💾 Also saving GT cloud frame={self.gt_target_frame}")
        self.get_logger().info(
            f"🟣 Step token: {self.step_stamp_topic} (publish_step_stamp={bool(self.publish_step_stamp)}) ; "
            f"captured={self.captured_stamp_topic} cam_pose={self.cam_pose_topic}"
        )
        self.get_logger().info(f"🧾 group_by_run_id={self.group_by_run_id} run_id_topic={self.run_id_topic}")
        self.get_logger().info(f"🔁 TF: use_latest_tf={self.use_latest_tf} tf_lookup_timeout_sec={self.tf_lookup_timeout_sec}")
        self.get_logger().info("----------------------------------------------------------------")

    def _on_set_parameters(self, params):
        """
        rclpy callback for dynamic parameter updates.
        Keep cached fields in sync with parameter values.
        """
        try:
            for p in params:
                if p.name == "target_frame":
                    self.target_frame = str(p.value)
                elif p.name == "gt_target_frame":
                    self.gt_target_frame = str(p.value)
                elif p.name == "save_gt_enable":
                    self.save_gt_enable = bool(p.value)
                elif p.name == "use_latest_tf":
                    self.use_latest_tf = bool(p.value)
                elif p.name == "tf_lookup_timeout_sec":
                    self.tf_lookup_timeout_sec = float(p.value)
                elif p.name == "camera_frame_fallback":
                    self.camera_frame_fallback = str(p.value)
                elif p.name == "cam_pose_frame":
                    self.cam_pose_frame = str(p.value)
                elif p.name == "stamp_correction_mode":
                    self.stamp_correction_mode = str(p.value).strip()
                elif p.name == "stamp_correction_fixed_offset_sec":
                    self.stamp_correction_fixed_offset_sec = float(p.value)
                elif p.name == "stamp_correction_alpha":
                    self.stamp_correction_alpha = float(p.value)
                elif p.name == "rgb_require_new":
                    self.rgb_require_new = bool(p.value)
            return SetParametersResult(successful=True)
        except Exception as e:
            return SetParametersResult(successful=False, reason=str(e))

    def _on_step_stamp(self, msg: TimeMsg):
        with self._mtx:
            self._last_step_stamp = msg

    def _warn_throttle(self, msg: str, period_s: float = 2.0):
        now = self.get_clock().now()
        if (now - self._last_warn_time) > Duration(seconds=period_s):
            self.get_logger().warn(msg)
            self._last_warn_time = now

    def _on_info(self, msg: CameraInfo):
        with self._cv:
            self._last_info = msg
            self._info_seq += 1
            self._cv.notify_all()

    def _on_rgb(self, msg: Image):
        with self._cv:
            self._last_rgb = msg
            self._rgb_seq += 1
            self._cv.notify_all()

    def _on_run_id(self, msg: StringMsg):
        if msg is None:
            return
        with self._mtx:
            self._run_id = str(msg.data)

    def _clip_target_z(self, cloud: PointCloud2) -> PointCloud2:
        if not bool(self.get_parameter("target_clip_enable").value):
            return cloud
        zmin = float(self.get_parameter("target_min_z").value)
        zmax = float(self.get_parameter("target_max_z").value)
        if zmax <= zmin:
            return cloud
        pts = point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
        kept = [(x, y, z) for (x, y, z) in pts if (zmin <= float(z) <= zmax)]
        return point_cloud2.create_cloud_xyz32(cloud.header, kept)

    def _clip_target_z_np(self, pts_target: np.ndarray) -> np.ndarray:
        if not bool(self.get_parameter("target_clip_enable").value):
            return pts_target
        zmin = float(self.get_parameter("target_min_z").value)
        zmax = float(self.get_parameter("target_max_z").value)
        if zmax <= zmin:
            return pts_target
        if pts_target.size == 0:
            return pts_target
        z = pts_target[:, 2]
        m = (z >= zmin) & (z <= zmax)
        return pts_target[m]

    def _spatial_filter_np(self, pts_target: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not bool(self.get_parameter("spatial_filter_enable").value):
            return pts_target, {"enable": False}
        mode = str(self.get_parameter("spatial_filter_mode").value).strip()
        voxel_size = float(self.get_parameter("spatial_filter_voxel_size_m").value)
        min_comp = int(self.get_parameter("spatial_filter_min_component_points").value)
        n0 = int(pts_target.shape[0])
        if mode == "largest_component":
            out = _largest_component_voxel_filter(pts_target, voxel_size_m=voxel_size, min_component_points=min_comp)
        else:
            out = pts_target
        n1 = int(out.shape[0])
        return out, {
            "enable": True,
            "mode": mode,
            "voxel_size_m": float(voxel_size),
            "min_component_points": int(min_comp),
            "num_points_before": n0,
            "num_points_after": n1,
        }

    def _remember_depth(self, img: Image):
        with self._cv:
            self._last_depth = img
            self._depth_seq += 1
            self._cv.notify_all()
        # Update depth->local clock offset estimate (for multi-machine timestamp skew).
        if self.stamp_correction_mode == "auto":
            try:
                st = img.header.stamp
                if int(st.sec) == 0 and int(st.nanosec) == 0:
                    return
                msg_ns = RclTime.from_msg(st).nanoseconds
                now_ns = self.get_clock().now().nanoseconds
                off = int(now_ns - msg_ns)
                a = float(min(max(self.stamp_correction_alpha, 0.0), 1.0))
                with self._mtx:
                    prev = self._depth_to_local_offset_ns
                    self._depth_to_local_offset_ns = off if prev is None else int((1.0 - a) * prev + a * off)
            except Exception:
                return

    def _corrected_time_for_stamp(self, st: TimeMsg) -> Optional[RclTime]:
        """
        Return a TF lookup time in the local clock domain corresponding to a depth stamp.
        None means "use latest TF".
        """
        if self.stamp_correction_mode == "none":
            return None
        if int(st.sec) == 0 and int(st.nanosec) == 0:
            return None
        msg_ns = RclTime.from_msg(st).nanoseconds
        if self.stamp_correction_mode == "fixed":
            off_ns = int(self.stamp_correction_fixed_offset_sec * 1e9)
            return RclTime(nanoseconds=int(msg_ns + off_ns))
        if self.stamp_correction_mode == "auto":
            with self._mtx:
                off_ns = self._depth_to_local_offset_ns
            if off_ns is None:
                return None
            return RclTime(nanoseconds=int(msg_ns + off_ns))
        return None

    def _wait_for_info(self, prev_seq: int, timeout_sec: float) -> Optional[CameraInfo]:
        deadline = time.time() + float(timeout_sec)
        with self._cv:
            while time.time() < deadline:
                if self._info_seq > prev_seq and self._last_info is not None:
                    return self._last_info
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._cv.wait(timeout=min(0.05, remaining))
            return self._last_info

    def _wait_for_depth(self, prev_seq: int, timeout_sec: float) -> Optional[Image]:
        deadline = time.time() + float(timeout_sec)
        with self._cv:
            while time.time() < deadline:
                if self._depth_seq > prev_seq and self._last_depth is not None:
                    return self._last_depth
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._cv.wait(timeout=min(0.05, remaining))
            return self._last_depth

    def _wait_for_rgb(self, prev_seq: int, timeout_sec: float, *, require_new: bool) -> Optional[Image]:
        deadline = time.time() + float(timeout_sec)
        with self._cv:
            while time.time() < deadline:
                if self._rgb_seq > prev_seq and self._last_rgb is not None:
                    return self._last_rgb
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._cv.wait(timeout=min(0.05, remaining))
            return None if bool(require_new) else self._last_rgb

    def _compute_target_cloud(self, img: Image, cam: CameraInfo, *, want_points: bool = False):
        # Read live in case params change at runtime (and to avoid cached/stale values)
        use_latest_tf = bool(self.get_parameter("use_latest_tf").value)
        tf_timeout = float(self.get_parameter("tf_lookup_timeout_sec").value)

        # Normalize frame_id (strip leading '/')
        src_frame = (img.header.frame_id or "").strip()
        if src_frame.startswith("/"):
            src_frame = src_frame[1:]
        if not src_frame:
            src_frame = self.camera_frame_fallback
        if not src_frame:
            raise RuntimeError("Depth image has empty frame_id and camera_frame_fallback is empty")

        # Build points in camera (optical) frame
        pts_src = _depth_image_to_xyz(
            img,
            cam,
            pixel_stride=int(self.get_parameter("pixel_stride").value),
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_depth_m=float(self.get_parameter("max_depth_m").value),
        )

        # Choose lookup time.
        # Even if configured for stamped TF, never use "tiny" stamps (e.g., step-id tokens),
        # because that will produce extrapolation errors on testbed where TF is epoch-stamped.
        corr_t = self._corrected_time_for_stamp(img.header.stamp)
        if use_latest_tf:
            lookup_time = corr_t if (corr_t is not None and _is_epoch_like_stamp(img.header.stamp)) else rclpy.time.Time()
        else:
            if not _is_epoch_like_stamp(img.header.stamp):
                lookup_time = rclpy.time.Time()
            else:
                lookup_time = corr_t if corr_t is not None else rclpy.time.Time.from_msg(img.header.stamp)

        def _lookup(target: str, src: str, t: rclpy.time.Time):
            return self.tf_buffer.lookup_transform(
                target,
                src,
                t,
                timeout=Duration(seconds=float(tf_timeout)),
            )

        # Transform to target frame (robust to frame-id mismatch + occasional timestamp skew)
        fb = (self.camera_frame_fallback or "").strip()
        if fb.startswith("/"):
            fb = fb[1:]

        # Try the requested target frame first, then fall back to GT/base_link if needed.
        # This prevents hard-failure when a config sets target_frame=target/est_base_link but that TF isn't being published.
        target_candidates: List[str] = []
        for t in [self.target_frame, self.gt_target_frame, "target/base_link"]:
            tt = (t or "").strip()
            if tt.startswith("/"):
                tt = tt[1:]
            if tt and tt not in target_candidates:
                target_candidates.append(tt)

        # Try the image frame, then fallback frame; for each, try target candidates.
        tf = None
        last_err = None
        used_src = src_frame
        used_target = self.target_frame
        for candidate_src in [src_frame, fb]:
            if not candidate_src:
                continue
            used_src = candidate_src
            for candidate_target in target_candidates:
                try:
                    tf = _lookup(candidate_target, candidate_src, lookup_time)
                    used_target = candidate_target
                    break
                except Exception as e:
                    last_err = e
                    # Retry once with latest TF as a fallback.
                    # This is needed for both:
                    # - use_latest_tf=false (stamped lookup)
                    # - use_latest_tf=true with stamp correction enabled (corrected stamped lookup)
                    # where TF can be slightly behind the requested time (future extrapolation).
                    if int(getattr(lookup_time, "nanoseconds", 0)) != 0:
                        try:
                            tf = _lookup(candidate_target, candidate_src, rclpy.time.Time())
                            used_target = candidate_target
                            break
                        except Exception as e2:
                            last_err = e2
                            continue
                    continue
            if tf is not None:
                break

        if tf is None:
            img_st = img.header.stamp
            epoch_like = _is_epoch_like_stamp(img_st)
            lt = lookup_time
            # For debugging: include the exact time policy used.
            lt_str = "LATEST(Time=0)" if int(lt.nanoseconds) == 0 else f"{lt.nanoseconds * 1e-9:.6f}s"
            raise RuntimeError(
                f"TF lookup failed targets={target_candidates} <- {used_src}: {last_err} ; "
                f"use_latest_tf={use_latest_tf} img_stamp={stamp_str(img_st)} epoch_like={epoch_like} lookup_time={lt_str}"
            )

        # Transform points in NumPy (much faster than do_transform_cloud + re-filtering)
        tr = tf.transform.translation
        qr = tf.transform.rotation
        R = quat_to_rot_np(float(qr.x), float(qr.y), float(qr.z), float(qr.w))  # target <- src
        tvec = np.array([float(tr.x), float(tr.y), float(tr.z)], dtype=np.float64)

        if pts_src.size == 0:
            pts_target = np.empty((0, 3), dtype=np.float32)
        else:
            P = pts_src.astype(np.float64, copy=False)
            # row vectors: p_target = p_src * R^T + t
            P_t = (P @ R.T) + tvec
            pts_target = P_t.astype(np.float32, copy=False)

        pts_target = self._clip_target_z_np(pts_target)
        pts_target, filt_info = self._spatial_filter_np(pts_target)

        # IMPORTANT: do NOT mutate img.header in-place.
        # We keep incoming message stamps intact (they may be reused if no new depth arrives),
        # and we avoid accidentally feeding step-tokens back into TF lookup paths.
        header = Header()
        header.frame_id = used_target
        header.stamp = self.get_clock().now().to_msg() if use_latest_tf else img.header.stamp
        cloud_target = point_cloud2.create_cloud_xyz32(header, pts_target)

        if not want_points:
            return cloud_target, tf, used_src, None, None, filt_info, used_target
        return cloud_target, tf, used_src, pts_target, pts_src, filt_info, used_target

    def _publish_tokens_and_cam_pose(self, *, stamp: TimeMsg, lookup_time: rclpy.time.Time, target_frame: Optional[str] = None):
        # Publish captured token for voxel_map gating. Step token is normally provided by stepper/coordinator;
        # for static convenience, capturer can publish it here if enabled.
        if self.publish_step_stamp and str(self.step_stamp_topic):
            self.pub_step_stamp.publish(stamp)
        self.pub_captured_stamp.publish(stamp)

        target = str(target_frame) if target_frame else str(self.target_frame)

        # Publish camera pose in target frame (matches voxel_map_node lookupCamPoseInTarget)
        try:
            tf_cam = self.tf_buffer.lookup_transform(
                target,
                self.cam_pose_frame,
                lookup_time,
                timeout=Duration(seconds=float(self.tf_lookup_timeout_sec)),
            )
        except Exception as e:
            # If we requested a stamped time (e.g. stamp correction), TF may lag slightly.
            # Fall back to latest TF for cam pose rather than failing the whole capture.
            if int(getattr(lookup_time, "nanoseconds", 0)) != 0:
                try:
                    tf_cam = self.tf_buffer.lookup_transform(
                        target,
                        self.cam_pose_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=float(self.tf_lookup_timeout_sec)),
                    )
                except Exception as e2:
                    self._warn_throttle(f"Cam pose TF lookup failed {target} <- {self.cam_pose_frame}: {e2}")
                    return
            else:
                self._warn_throttle(f"Cam pose TF lookup failed {target} <- {self.cam_pose_frame}: {e}")
                return

        ps = PoseStamped()
        ps.header.frame_id = target
        ps.header.stamp = stamp
        tr = tf_cam.transform.translation
        qr = tf_cam.transform.rotation
        ps.pose.position.x = float(tr.x)
        ps.pose.position.y = float(tr.y)
        ps.pose.position.z = float(tr.z)
        ps.pose.orientation.x = float(qr.x)
        ps.pose.orientation.y = float(qr.y)
        ps.pose.orientation.z = float(qr.z)
        ps.pose.orientation.w = float(qr.w)
        self.pub_cam_pose.publish(ps)

    def _save(
        self,
        stamp,
        *,
        img: Image,
        cam: CameraInfo,
        src_frame: str,
        tf_used,
        pts_target: np.ndarray,
        name_prefix: str = "cloud",
        tf_lookup_time_used: Optional[rclpy.time.Time] = None,
        rgb_img: Optional[Image] = None,
        filter_info: Optional[Dict[str, Any]] = None,
    ):
        s = stamp_str(stamp)
        out_base = self.out_dir
        rid = None
        if self.group_by_run_id:
            with self._mtx:
                rid = self._run_id
            if rid:
                out_base = os.path.join(self.out_dir, rid)
                os.makedirs(out_base, exist_ok=True)

        pcd_path = os.path.join(out_base, f"{name_prefix}_{s}.pcd")
        meta_path = os.path.join(out_base, f"meta_{name_prefix}_{s}.json")

        write_pcd_xyz_ascii(pcd_path, pts_target)

        rgb_path = None
        if rgb_img is not None:
            try:
                rgb_path = os.path.join(out_base, f"rgb_{s}.ppm")
                enc = (rgb_img.encoding or "").lower()
                w = int(rgb_img.width)
                h = int(rgb_img.height)
                if enc in ("rgb8",):
                    _write_ppm(rgb_path, w=w, h=h, rgb_bytes=bytes(rgb_img.data))
                elif enc in ("bgr8",):
                    arr = np.frombuffer(rgb_img.data, dtype=np.uint8).reshape(h, w, 3)
                    arr = arr[:, :, ::-1]  # BGR -> RGB
                    _write_ppm(rgb_path, w=w, h=h, rgb_bytes=arr.tobytes())
                elif enc in ("rgba8",):
                    arr = np.frombuffer(rgb_img.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
                    _write_ppm(rgb_path, w=w, h=h, rgb_bytes=arr.tobytes())
                elif enc in ("mono8",):
                    rgb_path = os.path.join(out_base, f"rgb_{s}.pgm")
                    _write_pgm(rgb_path, w=w, h=h, gray_bytes=bytes(rgb_img.data))
                else:
                    # Unknown encoding: don't write an image file
                    rgb_path = None
            except Exception:
                rgb_path = None

        tr = tf_used.transform.translation
        qr = tf_used.transform.rotation
        tf_st = tf_used.header.stamp

        meta = {
            "stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
            "depth_stamp": {"sec": int(img.header.stamp.sec), "nanosec": int(img.header.stamp.nanosec)},
            "target_frame": self.target_frame,
            "source_frame": src_frame,
            "depth_encoding": str(img.encoding),
            "depth_size": {"width": int(img.width), "height": int(img.height)},
            "camera_info": {
                "width": int(cam.width),
                "height": int(cam.height),
                "k": [float(x) for x in cam.k],
            },
            "sampling": {
                "pixel_stride": int(self.get_parameter("pixel_stride").value),
                "min_depth_m": float(self.get_parameter("min_depth_m").value),
                "max_depth_m": float(self.get_parameter("max_depth_m").value),
            },
            "target_clip": {
                "enable": bool(self.get_parameter("target_clip_enable").value),
                "min_z": float(self.get_parameter("target_min_z").value),
                "max_z": float(self.get_parameter("target_max_z").value),
            },
            "num_points": int(pts_target.shape[0]),
            "filter": filter_info,
            "tf_used": {
                "stamp": {"sec": int(tf_st.sec), "nanosec": int(tf_st.nanosec)},
                "translation": {"x": float(tr.x), "y": float(tr.y), "z": float(tr.z)},
                "rotation_xyzw": {"x": float(qr.x), "y": float(qr.y), "z": float(qr.z), "w": float(qr.w)},
            },
            "tf_lookup_time_used": (
                {"nanoseconds": int(tf_lookup_time_used.nanoseconds)} if tf_lookup_time_used is not None else None
            ),
            "group_by_run_id": bool(self.group_by_run_id),
            "run_id": str(rid) if rid else None,
            "rgb": (
                None
                if rgb_img is None
                else {
                    "path": str(rgb_path) if rgb_path else None,
                    "stamp": {"sec": int(rgb_img.header.stamp.sec), "nanosec": int(rgb_img.header.stamp.nanosec)},
                    "frame_id": str(rgb_img.header.frame_id),
                    "encoding": str(rgb_img.encoding),
                    "size": {"width": int(rgb_img.width), "height": int(rgb_img.height)},
                }
            ),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return pcd_path, meta_path

    def _on_depth(self, img: Image):
        # Always store the most recent frame for service capture.
        self._remember_depth(img)

        if not self.publish_continuous:
            return

        cam = self._last_info
        if cam is None:
            self._warn_throttle("No CameraInfo received yet; dropping depth frames.")
            return

        try:
            cloud_target, _tf, _src_frame, _pts, _pts_src, _filt, _tgt = self._compute_target_cloud(img, cam, want_points=False)
        except Exception as e:
            self._warn_throttle(str(e))
            return

        self.pub.publish(cloud_target)

    def _on_capture(self, req, resp):
        with self._cv:
            info_seq0 = int(self._info_seq)
            depth_seq0 = int(self._depth_seq)
            cam = self._last_info
            rgb_seq0 = int(self._rgb_seq)

        if cam is None:
            cam = self._wait_for_info(info_seq0, self.wait_for_inputs_sec)
            if cam is None:
                resp.success = False
                resp.message = f"❌ No CameraInfo received on {self.info_topic}"
                return resp

        img = self._wait_for_depth(depth_seq0, self.wait_for_inputs_sec)
        if img is None:
            resp.success = False
            resp.message = f"❌ No depth image received on {self.depth_topic}"
            return resp

        rgb_img = None
        if self.save_rgb_enable and self.rgb_topic:
            rgb_img = self._wait_for_rgb(
                rgb_seq0,
                float(max(0.0, self.rgb_wait_for_new_sec)),
                require_new=bool(self.rgb_require_new),
            )
            if rgb_img is None and bool(self.rgb_require_new):
                self._warn_throttle(
                    f"No NEW RGB received within {float(self.rgb_wait_for_new_sec):.1f}s; skipping RGB save for this capture."
                )

        # Choose TF lookup time policy.
        # Even if use_latest_tf=false, never use a non-epoch-like stamp (e.g., step-id tokens),
        # otherwise TF queries will request time ~k seconds and fail ("extrapolation into the past").
        corr_t = self._corrected_time_for_stamp(img.header.stamp)
        if self.use_latest_tf:
            lookup_time = corr_t if (corr_t is not None and _is_epoch_like_stamp(img.header.stamp)) else rclpy.time.Time()
        else:
            if not _is_epoch_like_stamp(img.header.stamp):
                lookup_time = rclpy.time.Time()
            else:
                lookup_time = corr_t if corr_t is not None else rclpy.time.Time.from_msg(img.header.stamp)

        # Choose the stamp used for downstream gating:
        # - Prefer the latest external step token if available (dynamic mode).
        # - Otherwise, fall back to now() / image stamp (static convenience).
        with self._mtx:
            step_stamp = self._last_step_stamp
        if step_stamp is not None:
            stamp = step_stamp
        else:
            stamp = self.get_clock().now().to_msg() if self.use_latest_tf else img.header.stamp

        try:
            cloud_target, tf_used, src_frame, pts_target, pts_src, filt_info, target_frame_used = self._compute_target_cloud(
                img, cam, want_points=self.save_enable
            )
        except Exception as e:
            resp.success = False
            resp.message = f"❌ Capture failed: {e}"
            return resp

        # Force exact stamp match for mua_nbv_planner gating
        cloud_target.header.stamp = stamp

        # Publish one-shot cloud (service semantics)
        self.pub.publish(cloud_target)
        self._publish_tokens_and_cam_pose(stamp=stamp, lookup_time=lookup_time, target_frame=target_frame_used)

        # Save PCD + meta
        if not self.save_enable:
            resp.success = True
            resp.message = "✅ Captured (save_enable=false)"
            return resp

        old_tf = self.target_frame
        try:
            # Ensure meta + downstream diagnostics reflect the actual TF frame used for this capture.
            self.target_frame = str(target_frame_used)
            if pts_target is None:
                raise RuntimeError("Internal error: pts_target is None while save_enable=true")
            pcd_path, meta_path = self._save(
                stamp,
                img=img,
                cam=cam,
                src_frame=src_frame,
                tf_used=tf_used,
                pts_target=pts_target,
                name_prefix="cloud",
                tf_lookup_time_used=lookup_time,
                rgb_img=rgb_img,
                filter_info=filt_info,
            )

            # Optional: also save GT cloud in target/base_link (Optitrack VRPN TF).
            gt_pcd_path = None
            gt_meta_path = None
            if self.save_gt_enable:
                if pts_src is None:
                    raise RuntimeError("Internal error: pts_src is None while save_gt_enable=true")
                # Lookup GT frame <- camera optical
                tf_gt = self.tf_buffer.lookup_transform(
                    self.gt_target_frame,
                    src_frame,
                    lookup_time,
                    timeout=Duration(seconds=float(self.tf_lookup_timeout_sec)),
                )
                tr = tf_gt.transform.translation
                qr = tf_gt.transform.rotation
                R = quat_to_rot_np(float(qr.x), float(qr.y), float(qr.z), float(qr.w))  # gt <- src
                tvec = np.array([float(tr.x), float(tr.y), float(tr.z)], dtype=np.float64)
                if pts_src.size == 0:
                    pts_gt = np.empty((0, 3), dtype=np.float32)
                else:
                    P = pts_src.astype(np.float64, copy=False)
                    pts_gt = ((P @ R.T) + tvec).astype(np.float32, copy=False)
                pts_gt = self._clip_target_z_np(pts_gt)
                pts_gt, filt_info_gt = self._spatial_filter_np(pts_gt)
                # Temporarily set target_frame in meta for GT file
                old_tf = self.target_frame
                try:
                    self.target_frame = str(self.gt_target_frame)
                    gt_pcd_path, gt_meta_path = self._save(
                        stamp,
                        img=img,
                        cam=cam,
                        src_frame=src_frame,
                        tf_used=tf_gt,
                        pts_target=pts_gt,
                        name_prefix="cloud_gt",
                        tf_lookup_time_used=lookup_time,
                        rgb_img=rgb_img,
                        filter_info=filt_info_gt,
                    )
                finally:
                    self.target_frame = old_tf
        except Exception as e:
            resp.success = False
            resp.message = f"❌ Save failed: {e}"
            return resp
        finally:
            self.target_frame = old_tf

        resp.success = True
        if self.save_gt_enable and gt_pcd_path and gt_meta_path:
            resp.message = (
                f"✅ Captured {int(pts_target.shape[0])} pts -> {pcd_path} (+ {meta_path}) ; "
                f"GT -> {gt_pcd_path} (+ {gt_meta_path})"
            )
        else:
            resp.message = f"✅ Captured {int(pts_target.shape[0])} pts -> {pcd_path} (+ {meta_path})"
        self.get_logger().info(resp.message)
        return resp


def main():
    rclpy.init()
    node = CloudCapturer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        # Under ros2 launch, shutdown can already be in progress.
        pass


if __name__ == "__main__":
    main()
