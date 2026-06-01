#!/usr/bin/env python3
import os
import json
import time
import threading
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time as rclpy_Time
from rclpy.utilities import ok as rclpy_ok

from std_srvs.srv import Trigger
from std_msgs.msg import Header
from std_msgs.msg import String as StringMsg
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import tf2_ros

from simulation_bringup.cloud_aligner import CloudAligner, CloudAlignerConfig
from mua_nbv_py_utils.transforms import quat_to_rot_np
from mua_nbv_py_utils.pcd_io import write_pcd_xyz_ascii
from mua_nbv_common.ros_helpers import time_tuple, stamp_str


class CloudCapturer(Node): 
    def __init__(self):
        super().__init__("cloud_capturer")
        self.cb = ReentrantCallbackGroup()

        # ---- Params ----
        # Frames
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("target_est_frame", "target/est_base_link")
        self.declare_parameter("debug_target_frame", "target/base_link")
        self.declare_parameter("cam_pose_frame", "pursuer/camera_depth_optical_frame")
        # Ground-truth / reference frames for logging (no effect on cloud output)
        self.declare_parameter("gt_target_frame", "")  # default: debug_target_frame
        self.declare_parameter("pursuer_frame", "pursuer/base_link")
        self.declare_parameter(
            "save_gt_cloud",
            True,
        )  # also save a cloud expressed in gt_target_frame (e.g. target/base_link)
        # Topics
        self.declare_parameter("input_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("input_cloud_topic", "/pursuer/camera/depth/color/points") 
        self.declare_parameter("output_cloud_topic", "/target/points")
        self.declare_parameter("output_captured_stamp_topic", "/experiment/captured_cloud_stamp")
        self.declare_parameter("output_cam_pose_topic", "/experiment/captured_cam_pose")
        # Filter params  
        self.declare_parameter("z_percentile", 2.0)     
        self.declare_parameter("z_margin", 0.05)    
        # Service
        self.declare_parameter("service_name", "/experiment/capture/cloud")
        # Path
        self.declare_parameter("out_dir", "debug/experiment") 
        self.declare_parameter("group_by_run_id", True)
        self.declare_parameter("run_id_topic", "/experiment/run_id")
        # Timeouts 
        self.declare_parameter("wait_for_inputs_sec", 2.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.05)
        self.declare_parameter("tf_time_source", "cloud")      # "cloud" or "step"
        self.declare_parameter("output_stamp_source", "cloud")# "cloud" or "step" 
        # Cloud selection policy (Invariant A support)
        self.declare_parameter("cloud_buf_max", 50)
        self.declare_parameter("require_cloud_stamp_eq_step", False)
        # Safety cap: subsample clouds larger than this to avoid downstream hangs
        self.declare_parameter("max_capture_points", 25000)
        # Override target frame
        self.declare_parameter("debug_use_fixed_target_frame", False)
        # Temporary simulation-only registration refinement
        self.declare_parameter("registration_mode", "none")  # none|uniform|overlap
        self.declare_parameter("registration_debug_save", False)
        self.declare_parameter("registration_submap_voxel_m", 0.02)
        self.declare_parameter("registration_submap_max_points", 30000)
        self.declare_parameter("registration_overlap_max_dist_m", 0.08)
        self.declare_parameter("registration_target_crop_dist_m", 0.10)
        self.declare_parameter("registration_min_overlap_points", 150)
        self.declare_parameter("registration_icp_voxel_m", 0.02)
        self.declare_parameter("registration_normal_radius_m", 0.06)
        self.declare_parameter("registration_normal_max_nn", 30)
        self.declare_parameter("registration_icp_max_corr_dist_m", 0.08)
        self.declare_parameter("registration_icp_max_iter", 40)
        self.declare_parameter("registration_icp_rel_fitness", 1e-6)
        self.declare_parameter("registration_icp_rel_rmse", 1e-6)
        self.declare_parameter("registration_accept_min_fitness", 0.30)
        self.declare_parameter("registration_accept_max_rmse", 0.06)
        self.declare_parameter("registration_accept_max_translation_m", 0.12)
        self.declare_parameter("registration_accept_max_rotation_deg", 12.0)
        
        # ---- Read params ----
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.target_est_frame = str(self.get_parameter("target_est_frame").value)
        self.debug_target_frame = str(self.get_parameter("debug_target_frame").value)
        self.cam_pose_frame = str(self.get_parameter("cam_pose_frame").value)
        self.gt_target_frame = str(self.get_parameter("gt_target_frame").value) or self.debug_target_frame
        self.pursuer_frame = str(self.get_parameter("pursuer_frame").value)
        self.save_gt_cloud = bool(self.get_parameter("save_gt_cloud").value)
        self.input_stamp_topic = str(self.get_parameter("input_stamp_topic").value)
        self.input_cloud_topic = str(self.get_parameter("input_cloud_topic").value)
        self.output_cloud_topic = str(self.get_parameter("output_cloud_topic").value)
        self.output_captured_stamp_topic = str(self.get_parameter("output_captured_stamp_topic").value)
        self.output_cam_pose_topic = str(self.get_parameter("output_cam_pose_topic").value)
        self.srv_name = str(self.get_parameter("service_name").value)
        self.out_dir = str(self.get_parameter("out_dir").value)
        self.group_by_run_id = bool(self.get_parameter("group_by_run_id").value)
        self.run_id_topic = str(self.get_parameter("run_id_topic").value)
        self.wait_for_inputs_sec = float(self.get_parameter("wait_for_inputs_sec").value)
        self.tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)
        self.tf_time_source = str(self.get_parameter("tf_time_source").value)
        self.output_stamp_source = str(self.get_parameter("output_stamp_source").value) 
        self.cloud_buf_max = int(self.get_parameter("cloud_buf_max").value)
        self.require_cloud_stamp_eq_step = bool(self.get_parameter("require_cloud_stamp_eq_step").value)
        self.z_percentile = float(self.get_parameter("z_percentile").value)
        self.z_margin = float(self.get_parameter("z_margin").value)
        self.debug_use_fixed_target_frame = bool(self.get_parameter("debug_use_fixed_target_frame").value)
        self.max_capture_points = int(self.get_parameter("max_capture_points").value)
        self.target_frame_effective = self.debug_target_frame if self.debug_use_fixed_target_frame else self.target_est_frame
        self.registration_mode = str(self.get_parameter("registration_mode").value).strip().lower()
        self.registration_debug_save = bool(self.get_parameter("registration_debug_save").value)

        # ---- Create output directory ----
        os.makedirs(self.out_dir, exist_ok=True)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---- State ---- 
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._cloud_buf = deque(maxlen=max(1, int(self.cloud_buf_max)))
        self._cloud_seq = 0  # monotonic counter for wakeups
        self._step_mtx = threading.Lock()
        self._last_step_stamp = None
        self._run_mtx = threading.Lock()
        self._run_id = None
        self._aligner = None
        if self.registration_mode in ("uniform", "overlap"):
            align_cfg = CloudAlignerConfig(
                mode=self.registration_mode,
                debug_save=self.registration_debug_save,
                submap_voxel_m=float(self.get_parameter("registration_submap_voxel_m").value),
                submap_max_points=int(self.get_parameter("registration_submap_max_points").value),
                overlap_max_dist_m=float(self.get_parameter("registration_overlap_max_dist_m").value),
                target_crop_dist_m=float(self.get_parameter("registration_target_crop_dist_m").value),
                min_overlap_points=int(self.get_parameter("registration_min_overlap_points").value),
                icp_voxel_m=float(self.get_parameter("registration_icp_voxel_m").value),
                normal_radius_m=float(self.get_parameter("registration_normal_radius_m").value),
                normal_max_nn=int(self.get_parameter("registration_normal_max_nn").value),
                icp_max_corr_dist_m=float(self.get_parameter("registration_icp_max_corr_dist_m").value),
                icp_max_iter=int(self.get_parameter("registration_icp_max_iter").value),
                icp_rel_fitness=float(self.get_parameter("registration_icp_rel_fitness").value),
                icp_rel_rmse=float(self.get_parameter("registration_icp_rel_rmse").value),
                accept_min_fitness=float(self.get_parameter("registration_accept_min_fitness").value),
                accept_max_rmse=float(self.get_parameter("registration_accept_max_rmse").value),
                accept_max_translation_m=float(self.get_parameter("registration_accept_max_translation_m").value),
                accept_max_rotation_deg=float(self.get_parameter("registration_accept_max_rotation_deg").value),
            )
            self._aligner = CloudAligner(align_cfg, self.get_logger())

        # ---- Pub/Sub/Service ----
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=3, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,)
        self.sub = self.create_subscription(PointCloud2, self.input_cloud_topic, self._on_cloud, qos, callback_group=self.cb) 
        qos_run = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        qos_token = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_step = self.create_subscription(TimeMsg, self.input_stamp_topic, self._on_step_stamp, qos_token)
        self.sub_run = self.create_subscription(StringMsg, self.run_id_topic, self._on_run_id, qos_run)
        self.pub = self.create_publisher(PointCloud2, self.output_cloud_topic, qos)
        self.pub_captured_stamp = self.create_publisher(TimeMsg, self.output_captured_stamp_topic, qos_token)
        self.pub_cam_pose = self.create_publisher(PoseStamped, self.output_cam_pose_topic, qos_token)
        self.srv = self.create_service(Trigger, self.srv_name, self._on_capture, callback_group=self.cb)

        self.get_logger().info("----------------------------------------------------------------") 
        self.get_logger().info(f"📡 Listening to: {self.input_stamp_topic}, {self.input_cloud_topic}")
        self.get_logger().info(f"📢 Publishing to: {self.output_cloud_topic}, {self.output_captured_stamp_topic}, {self.output_cam_pose_topic}")
        self.get_logger().info(f"🔧 Service: {self.srv_name}")
        self.get_logger().info(f"📊 Transform cloud: {self.input_cloud_topic} frame -> {self.target_frame_effective}")
        self.get_logger().info(f"📊 (meta) GT frames: target={self.gt_target_frame} pursuer={self.pursuer_frame}")
        self.get_logger().info(f"📊 Filtering: keep z > percentile(z,{self.z_percentile}) + {self.z_margin}")
        self.get_logger().info(f"📁 Saving to: {self.out_dir} wait_for_inputs_sec={self.wait_for_inputs_sec:.2f}")
        self.get_logger().info(f"📁 group_by_run_id: {self.group_by_run_id} (topic={self.run_id_topic})")
        self.get_logger().info(f"📊 registration_mode={self.registration_mode} debug_save={self.registration_debug_save}")
        self.get_logger().info("----------------------------------------------------------------")

    def _on_step_stamp(self, msg: TimeMsg):
        with self._step_mtx:
            self._last_step_stamp = msg

    def _on_run_id(self, msg: StringMsg):
        if msg is None:
            return
        with self._run_mtx:
            self._run_id = str(msg.data)

    def _on_cloud(self, msg: PointCloud2):
        with self._cv:
            self._cloud_buf.append(msg)
            self._cloud_seq += 1
            self._cv.notify_all()

    def _find_cloud_for_step(self, step_stamp: TimeMsg): 
        step_t = time_tuple(step_stamp)
        if not self._cloud_buf:
            return None

        # Work on a snapshot to avoid holding the lock too long
        clouds = list(self._cloud_buf)

        if self.require_cloud_stamp_eq_step:
            for c in clouds:
                if time_tuple(c.header.stamp) == step_t:
                    return c
            return None

        # earliest >= token
        best = None
        best_t = None
        for c in clouds:
            ct = time_tuple(c.header.stamp)
            if ct < step_t:
                continue
            if best is None or ct < best_t:
                best = c
                best_t = ct
        return best

    def _lookup_tf(self, target_frame: str, src_frame: str, stamp: TimeMsg):
        timeout = Duration(seconds=float(self.tf_lookup_timeout_sec))
        t_req = rclpy_Time.from_msg(stamp)
        return self.tf_buffer.lookup_transform(target_frame, src_frame, t_req, timeout=timeout)

    @staticmethod
    def _tf_to_dict(tf_msg) -> dict:
        tr = tf_msg.transform.translation
        qr = tf_msg.transform.rotation
        return {
            "parent_frame": str(tf_msg.header.frame_id),
            "child_frame": str(tf_msg.child_frame_id),
            "translation": {"x": float(tr.x), "y": float(tr.y), "z": float(tr.z)},
            "rotation_xyzw": {"x": float(qr.x), "y": float(qr.y), "z": float(qr.z), "w": float(qr.w)},
            "stamp": {"sec": int(tf_msg.header.stamp.sec), "nanosec": int(tf_msg.header.stamp.nanosec)},
        }

    def _read_and_filter_points_in_src(self, cloud: PointCloud2):
        """
        Read XYZ points in the incoming cloud's frame and apply the mandatory z-filter.
        Returns (pts_src_filtered, extra_meta).
        """
        pts = point_cloud2.read_points_numpy(cloud, field_names=["x", "y", "z"], skip_nans=True)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32), {"dropped_reason": "empty"}

        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32), {"dropped_reason": "nonfinite"}

        # ---- Mandatory z-filter (exactly your old logic) ----
        z = pts[:, 2]
        z0 = float(np.percentile(z, self.z_percentile))
        pts = pts[z > (z0 + self.z_margin)]
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32), {"dropped_reason": "zfilter_all_removed", "z0": z0}

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise RuntimeError(f"Unexpected pts shape: {pts.shape}")

        meta = {"z0": z0}
        cap = self.max_capture_points
        if cap > 0 and pts.shape[0] > cap:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(pts.shape[0], size=cap, replace=False)
            meta["subsampled_from"] = int(pts.shape[0])
            pts = pts[idx]

        return pts.astype(np.float32, copy=False), meta

    @staticmethod
    def _apply_tf_to_points(pts_src: np.ndarray, tf_msg) -> np.ndarray:
        if pts_src.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        t = tf_msg.transform.translation
        r = tf_msg.transform.rotation
        R = quat_to_rot_np(r.x, r.y, r.z, r.w)  # p_target = R * p_src + t
        tvec = np.array([t.x, t.y, t.z], dtype=np.float64)
        P = np.asarray(pts_src, dtype=np.float64, order="C")
        P_t = (P @ R.T) + tvec
        return P_t.astype(np.float32, copy=False)

    def _publish_cloud(self, stamp: TimeMsg, pts_target: np.ndarray):
        header = Header()
        header.frame_id = self.target_frame_effective
        header.stamp = stamp
        out = point_cloud2.create_cloud_xyz32(header, pts_target)
        self.pub.publish(out)

    def _publish_capture_tokens(self, stamp: TimeMsg, tf_cloud): 
        try:
            self.pub_captured_stamp.publish(stamp)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish captured_cloud_stamp: {e}")

        try:
            ps = PoseStamped()
            ps.header.frame_id = self.target_frame_effective
            ps.header.stamp = stamp
            tr = tf_cloud.transform.translation
            qr = tf_cloud.transform.rotation
            ps.pose.position.x = float(tr.x)
            ps.pose.position.y = float(tr.y)
            ps.pose.position.z = float(tr.z)
            ps.pose.orientation.x = float(qr.x)
            ps.pose.orientation.y = float(qr.y)
            ps.pose.orientation.z = float(qr.z)
            ps.pose.orientation.w = float(qr.w)
            self.pub_cam_pose.publish(ps)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish captured_cam_pose: {e}")

    def _out_base_dir(self) -> str:
        out_base = self.out_dir
        if self.group_by_run_id:
            with self._run_mtx:
                rid = self._run_id
            if rid:
                out_base = os.path.join(self.out_dir, rid)
        os.makedirs(out_base, exist_ok=True)
        return out_base

    def _save(
        self,
        *,
        step_stamp: TimeMsg,
        stamp: TimeMsg,
        src_frame: str,
        target_frame: str,
        tf_used,
        pts_target: np.ndarray,
        extra_meta: dict,
        file_tag: str = "",
    ):
        s = stamp_str(stamp)
        out_base = self.out_dir
        if self.group_by_run_id:
            with self._run_mtx:
                rid = self._run_id
            if rid:
                out_base = os.path.join(self.out_dir, rid)
                os.makedirs(out_base, exist_ok=True)

        pcd_path = os.path.join(out_base, f"cloud_{s}{file_tag}.pcd")
        meta_path = os.path.join(out_base, f"meta_{s}{file_tag}.json")

        write_pcd_xyz_ascii(pcd_path, pts_target)

        tf_used_dict = self._tf_to_dict(tf_used)

        meta = {
            "stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
            "step_stamp": {"sec": int(step_stamp.sec), "nanosec": int(step_stamp.nanosec)}, 
            "world_frame": self.world_frame,
            "target_frame": str(target_frame),
            "source_frame": src_frame,
            "num_points": int(pts_target.shape[0]),
            "z_filter": {
                "percentile": float(self.z_percentile),
                "margin": float(self.z_margin),
            },
            "tf_used": tf_used_dict,
        }
        meta.update(extra_meta or {})

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return pcd_path, meta_path

    def _on_capture(self, req, resp):
        deadline = time.time() + self.wait_for_inputs_sec

        with self._step_mtx:
            step_stamp = self._last_step_stamp

        if step_stamp is None:
            resp.success = False
            resp.message = f"❌ No step token received yet on {self.input_stamp_topic}"
            return resp

        self.get_logger().info(
            f"🟣 TOKEN capture request: step_stamp={stamp_str(step_stamp)} "
            f"(target_frame_effective={self.target_frame_effective}) "
            f"tf_time_source={self.tf_time_source} output_stamp_source={self.output_stamp_source}"
        )

        cloud = None
        while time.time() < deadline and rclpy_ok():
            with self._cv:
                cloud = self._find_cloud_for_step(step_stamp)
                if cloud is not None:
                    break
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._cv.wait(timeout=min(0.05, remaining))

        if cloud is None:
            resp.success = False
            mode = "==" if self.require_cloud_stamp_eq_step else ">="
            resp.message = (
                f"Timed out waiting for a cloud with stamp {mode} step_stamp on {self.input_cloud_topic} "
                f"(step_stamp={stamp_str(step_stamp)})"
            )
            return resp

        src_frame = cloud.header.frame_id
        cloud_stamp = cloud.header.stamp

        # Choose TF stamp and output stamp (critical for dynamic)
        tf_stamp = step_stamp if self.tf_time_source == "step" else cloud_stamp
        out_stamp = step_stamp if self.output_stamp_source == "step" else cloud_stamp 

        self.get_logger().info(
            f"🟣 TOKEN capture check: step={stamp_str(step_stamp)} cloud={stamp_str(cloud_stamp)} "
            f"src_frame={src_frame} cam_pose_frame={self.cam_pose_frame} "
            f"tf_stamp={stamp_str(tf_stamp)} out_stamp={stamp_str(out_stamp)}"
        )

        # TF lookup at tf_stamp (NOT necessarily the raw cloud stamp)
        try:
            tf_cloud_out = self._lookup_tf(self.target_frame_effective, src_frame, tf_stamp)
        except Exception as e:
            resp.success = False
            resp.message = (
                f"❌ TF lookup failed: {self.target_frame_effective} <- {src_frame} "
                f"at tf_stamp={stamp_str(tf_stamp)}: {e}"
            )
            return resp

        try:
            tf_cam = self._lookup_tf(self.target_frame_effective, self.cam_pose_frame, tf_stamp)
        except Exception as e:
            resp.success = False
            resp.message = (
                f"❌ TF lookup failed: {self.target_frame_effective} <- {self.cam_pose_frame} "
                f"at tf_stamp={stamp_str(tf_stamp)}: {e}"
            )
            return resp

        # If we are saving both est+gt clouds, we need both transforms.
        tf_cloud_est = None
        tf_cloud_gt = None
        try:
            tf_cloud_est = self._lookup_tf(self.target_est_frame, src_frame, tf_stamp)
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {self.target_est_frame} <- {src_frame} at {stamp_str(tf_stamp)}: {e}"
            )
        if self.save_gt_cloud:
            try:
                tf_cloud_gt = self._lookup_tf(self.gt_target_frame, src_frame, tf_stamp)
            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed: {self.gt_target_frame} <- {src_frame} at {stamp_str(tf_stamp)}: {e}"
                )

        # Optional "real" GT transforms for downstream evaluation / visualization.
        # These are NOT used for the captured cloud; they are only saved into meta_*.json.
        tf_gt_target_from_pursuer = None
        tf_world_from_gt_target = None
        tf_world_from_pursuer = None
        tf_world_from_est_target = None
        try:
            tf_gt_target_from_pursuer = self._lookup_tf(self.gt_target_frame, self.pursuer_frame, tf_stamp)
        except Exception as e:
            self.get_logger().warn(
                f"GT TF lookup failed: {self.gt_target_frame} <- {self.pursuer_frame} at {stamp_str(tf_stamp)}: {e}"
            )
        try:
            tf_world_from_gt_target = self._lookup_tf(self.world_frame, self.gt_target_frame, tf_stamp)
        except Exception as e:
            self.get_logger().warn(
                f"GT TF lookup failed: {self.world_frame} <- {self.gt_target_frame} at {stamp_str(tf_stamp)}: {e}"
            )
        try:
            tf_world_from_pursuer = self._lookup_tf(self.world_frame, self.pursuer_frame, tf_stamp)
        except Exception as e:
            self.get_logger().warn(
                f"GT TF lookup failed: {self.world_frame} <- {self.pursuer_frame} at {stamp_str(tf_stamp)}: {e}"
            )
        try:
            tf_world_from_est_target = self._lookup_tf(self.world_frame, self.target_est_frame, tf_stamp)
        except Exception as e:
            self.get_logger().warn(
                f"GT TF lookup failed: {self.world_frame} <- {self.target_est_frame} at {stamp_str(tf_stamp)}: {e}"
            )

        # Read/filter points once in source frame, then apply transforms.
        try:
            pts_src, extra = self._read_and_filter_points_in_src(cloud)
        except Exception as e:
            resp.success = False
            resp.message = f"❌ Transform/filter failed: {e}"
            return resp

        # Build the output cloud (as before) in target_frame_effective
        pts_target_out = self._apply_tf_to_points(pts_src, tf_cloud_out)
        reg_meta = None
        if self._aligner is not None:
            align_result = self._aligner.align(
                provisional_points=pts_target_out,
                out_base=self._out_base_dir(),
                stamp_label=stamp_str(out_stamp),
            )
            pts_target_out = np.asarray(
                align_result["points"], dtype=np.float32
            ).reshape(-1, 3)
            reg_meta = dict(align_result.get("metrics", {}))
            fit_str = (
                "None"
                if reg_meta.get("fitness") is None
                else f"{float(reg_meta['fitness']):.3f}"
            )
            rmse_str = (
                "None"
                if reg_meta.get("inlier_rmse") is None
                else f"{float(reg_meta['inlier_rmse']):.4f}"
            )
            self.get_logger().info(
                "🔧 registration: "
                f"accepted={str(reg_meta.get('icp_accepted', False)).lower()} "
                f"attempted={str(reg_meta.get('icp_attempted', False)).lower()} "
                f"fitness={fit_str} rmse={rmse_str} "
                f"msg='{str(reg_meta.get('message', ''))}'"
            )

        # Publish with out_stamp so downstream nodes gate consistently
        try:
            self._publish_cloud(out_stamp, pts_target_out)
            self._publish_capture_tokens(out_stamp, tf_cam)

            # OPTIONAL: include raw stamps in meta (recommended)
            extra = dict(extra or {})
            extra["raw_cloud_stamp"] = {"sec": int(cloud_stamp.sec), "nanosec": int(cloud_stamp.nanosec)}
            extra["tf_stamp_used"] = {"sec": int(tf_stamp.sec), "nanosec": int(tf_stamp.nanosec)}
            extra["out_stamp"] = {"sec": int(out_stamp.sec), "nanosec": int(out_stamp.nanosec)}
            extra["registration"] = reg_meta

            # Add GT/Reference transforms (if available)
            extra["gt_frames"] = {
                "gt_target_frame": self.gt_target_frame,
                "pursuer_frame": self.pursuer_frame,
                "target_est_frame": self.target_est_frame,
            }
            extra["tf_gt_target_from_pursuer"] = (
                self._tf_to_dict(tf_gt_target_from_pursuer) if tf_gt_target_from_pursuer is not None else None
            )
            extra["tf_world_from_gt_target"] = (
                self._tf_to_dict(tf_world_from_gt_target) if tf_world_from_gt_target is not None else None
            )
            extra["tf_world_from_pursuer"] = (
                self._tf_to_dict(tf_world_from_pursuer) if tf_world_from_pursuer is not None else None
            )
            extra["tf_world_from_est_target"] = (
                self._tf_to_dict(tf_world_from_est_target) if tf_world_from_est_target is not None else None
            )

            pcd_path = None
            meta_path = None

            if tf_cloud_est is not None:
                pts_est = self._apply_tf_to_points(pts_src, tf_cloud_est)
                extra_e = dict(extra)
                extra_e["cloud_variant"] = "est"
                pcd_path, meta_path = self._save(
                    step_stamp=step_stamp,
                    stamp=out_stamp,
                    src_frame=src_frame,
                    target_frame=self.target_est_frame,
                    tf_used=tf_cloud_est,
                    pts_target=pts_est,
                    extra_meta=extra_e,
                    file_tag="_est",
                )

            if self.save_gt_cloud and (tf_cloud_gt is not None):
                pts_gt = self._apply_tf_to_points(pts_src, tf_cloud_gt)
                extra_g = dict(extra)
                extra_g["cloud_variant"] = "gt"
                pcd_path, meta_path = self._save(
                    step_stamp=step_stamp,
                    stamp=out_stamp,
                    src_frame=src_frame,
                    target_frame=self.gt_target_frame,
                    tf_used=tf_cloud_gt,
                    pts_target=pts_gt,
                    extra_meta=extra_g,
                    file_tag="_gt",
                )
        except Exception as e:
            resp.success = False
            resp.message = f"❌ Publish/save failed: {e}"
            return resp

        resp.success = True
        if pts_target_out.shape[0] == 0:
            resp.message = f"⚠️ Captured EMPTY cloud ({extra.get('dropped_reason','unknown')}) -> {meta_path}"
        else:
            resp.message = f"✅ Captured {pts_target_out.shape[0]} pts -> {pcd_path} (+ {meta_path})"
        self.get_logger().info(resp.message)
        return resp


def main():
    rclpy.init()
    node = CloudCapturer()
    exec = MultiThreadedExecutor(num_threads=2)
    exec.add_node(node)
    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
