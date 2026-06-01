#!/usr/bin/env python3 
from collections import deque
import math  
import numpy as np
import os
import json
import threading
from typing import Any
from builtin_interfaces.msg import Time as TimeMsg

import jax.numpy as jnp

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time as RclTime

from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String as StringMsg
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from mua_nbv_prediction.tgpr import TGPR_CV
from mua_nbv_py_utils.transforms import quat_from_yaw, quat_to_rot_np, yaw_from_rot, yaw_var_from_vel
from mua_nbv_common.ros_helpers import stamp_str, pack_state_msg


class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__("trajectory_predictor")
        # ---- Params ----
        # Frames
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("target_est_frame", "target/est_base_link")
        # Output logging (pairs with cloud_capturer meta in debug/experiment/run_<run_id>/)
        self.declare_parameter("save_enable", True)
        self.declare_parameter("out_dir", "debug/experiment")
        self.declare_parameter("group_by_run_id", True)
        self.declare_parameter("run_id_topic", "/experiment/run_id")
        # Topics
        self.declare_parameter("input_target_gt_topic", "/sim/target/pose")
        self.declare_parameter("input_step_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("output_estimation_topic", "/nbv/target_estimation")
        self.declare_parameter("output_prediction_topic", "/nbv/target_prediction")
        self.declare_parameter("output_prediction_state_topic", "/nbv/target_prediction_state")
        self.declare_parameter("output_estimation_state_topic", "/nbv/target_estimation_state")
        # Smoothed estimate polyline for RViz (Path display)
        self.declare_parameter("publish_estimation_path", True)
        self.declare_parameter("output_estimation_path_topic", "/nbv/estimation_path")
        # 0 = unlimited history; else drop oldest poses (bounded memory)
        self.declare_parameter("estimation_path_max_poses", 0)
        # Noisy (px, py) measurements for RViz: polyline + axis-aligned σ ellipses
        self.declare_parameter("publish_measurement_markers", True)
        self.declare_parameter("output_measurement_markers_topic", "/nbv/measurement_markers")
        # Ellipse semi-axes = k * meas_std_{x,y} (k=2 ≈ ~95% for Gaussian axes)
        self.declare_parameter("measurement_markers_sigma_scale", 2.0)
        self.declare_parameter("measurement_markers_line_width", 0.025)
        self.declare_parameter("measurement_markers_z_thickness", 0.02)
        # Service
        self.declare_parameter("service_name", "/experiment/update/measurements")
        # Broadcasting rate
        self.declare_parameter("tf_publish_rate_hz", 15.0) 
        # GPR params
        self.declare_parameter("window_L", 10) 
        self.declare_parameter("dt_pred", 1.0)
        self.declare_parameter("q_c", 1.0)
        self.declare_parameter("meas_std_x", 0.05)
        self.declare_parameter("meas_std_y", 0.05)
        self.declare_parameter("init_pos_var", 0.01)
        self.declare_parameter("init_vel_var", 0.01)
        self.declare_parameter("v_min", 0.05)
        self.declare_parameter("pose_buffer_len", 200)
        # Simulation-only eval helpers (used for predmeta logging)
        self.declare_parameter("log_gt_in_predmeta", True)
        self.declare_parameter("rng_seed", 0)

        # ---- Read params ----
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.est_frame = str(self.get_parameter("target_est_frame").value)
        self.save_enable = bool(self.get_parameter("save_enable").value)
        self.out_dir = str(self.get_parameter("out_dir").value)
        self.group_by_run_id = bool(self.get_parameter("group_by_run_id").value)
        self.run_id_topic = str(self.get_parameter("run_id_topic").value)
        self.gt_topic = str(self.get_parameter("input_target_gt_topic").value)
        self.step_stamp_topic = str(self.get_parameter("input_step_stamp_topic").value)
        self.est_topic = str(self.get_parameter("output_estimation_topic").value)
        self.pred_topic = str(self.get_parameter("output_prediction_topic").value)
        self.state_topic = str(self.get_parameter("output_prediction_state_topic").value)
        self.est_state_topic = str(self.get_parameter("output_estimation_state_topic").value)
        self.publish_estimation_path = bool(self.get_parameter("publish_estimation_path").value)
        self.estimation_path_topic = str(self.get_parameter("output_estimation_path_topic").value)
        _max_path = int(self.get_parameter("estimation_path_max_poses").value)
        self._estimation_path_max = None if _max_path <= 0 else _max_path
        self.publish_measurement_markers = bool(self.get_parameter("publish_measurement_markers").value)
        self.measurement_markers_topic = str(self.get_parameter("output_measurement_markers_topic").value)
        self.measurement_markers_sigma_scale = float(self.get_parameter("measurement_markers_sigma_scale").value)
        self.measurement_markers_line_width = float(self.get_parameter("measurement_markers_line_width").value)
        self.measurement_markers_z_thickness = float(self.get_parameter("measurement_markers_z_thickness").value)
        self.srv_name = str(self.get_parameter("service_name").value)
        self.tf_publish_rate_hz = float(self.get_parameter("tf_publish_rate_hz").value)
        self.L = int(self.get_parameter("window_L").value)
        self.dt_pred = float(self.get_parameter("dt_pred").value)
        self.q_c = float(self.get_parameter("q_c").value)
        self.meas_std_x = float(self.get_parameter("meas_std_x").value)
        self.meas_std_y = float(self.get_parameter("meas_std_y").value)
        self.init_pos_var = float(self.get_parameter("init_pos_var").value)
        self.init_vel_var = float(self.get_parameter("init_vel_var").value)
        self.v_min = float(self.get_parameter("v_min").value)
        self.pose_buffer_len = int(self.get_parameter("pose_buffer_len").value)
        self.log_gt_in_predmeta = bool(self.get_parameter("log_gt_in_predmeta").value)
        self.rng_seed = int(self.get_parameter("rng_seed").value)

        # ---- State ----
        self._step_mtx = threading.Lock()
        self._last_step_stamp = None  # TimeMsg or None
        self._last_step_stamp_used = None  # tuple(sec, nsec) or None  

        # ---- Run grouping state ----
        self._run_mtx = threading.Lock()
        self._run_id = None  # str | None

        # ---- Previous one-step prediction (for ΔT_k) ----
        # Stores the prediction made at step k-1 for step k.
        self._prev_pred = None  # dict | None: {"stamp": (sec,nsec), "t": np.ndarray(3,), "q": (x,y,z,w)}

        # Smoothed trajectory + measurement viz for RViz (cleared when run_id changes)
        self._path_mtx = threading.Lock()
        self._viz_run_id: str | None = None
        maxlen = self._estimation_path_max
        self._estimation_path_poses = deque(maxlen=maxlen) if maxlen is not None else deque()
        self._measurement_viz_buf = deque(maxlen=maxlen) if maxlen is not None else deque()

        # ---- Model config ----
        K0 = jnp.diag(jnp.array([self.init_pos_var, self.init_pos_var, self.init_vel_var, self.init_vel_var], dtype=jnp.float32))
        rx = max(self.meas_std_x**2, 1e-8)
        ry = max(self.meas_std_y**2, 1e-8)
        R_meas = jnp.diag(jnp.array([float(rx), float(ry)], dtype=jnp.float32))

        self.gp = TGPR_CV(
            dataset_history=self.L,
            q_c=self.q_c,
            K0=K0,
            R=R_meas,
            dt=self.dt_pred,
            observe_position_only=True,
        )
        self._pose_buf = deque(maxlen=self.pose_buffer_len)  
        self._pos_buf = deque(maxlen=max(2, int(self.L) + 1))
        self._obs_buf = deque(maxlen=self.L)
        self._rng = np.random.default_rng(int(self.rng_seed))
        self._last_heading = 0.0 
        self.bootstrap_pos_var = float(min(self.init_pos_var, 1.0))
        self.bootstrap_vel_var = float(min(self.init_vel_var, 1.0))

        # ---- Subscribers / Publishers / Broadcasters / Services ----
        self.cb = ReentrantCallbackGroup()
        self.sub_gt = self.create_subscription(PoseStamped, self.gt_topic, self.cb_gt, 50, callback_group=self.cb)
        self.sub_step = self.create_subscription(TimeMsg, self.step_stamp_topic, self._on_step_stamp, 10, callback_group=self.cb)
        self.pub_est = self.create_publisher(PoseWithCovarianceStamped, self.est_topic, 10)
        self.pub_pred = self.create_publisher(PoseWithCovarianceStamped, self.pred_topic, 10)
        self.pub_pred_state = self.create_publisher(Float32MultiArray, self.state_topic, 10)
        self.pub_est_state = self.create_publisher(Float32MultiArray, self.est_state_topic, 10)
        qos_path = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.VOLATILE)
        self.pub_est_path = (
            self.create_publisher(Path, self.estimation_path_topic, qos_path)
            if self.publish_estimation_path
            else None
        )
        self.pub_meas_markers = (
            self.create_publisher(MarkerArray, self.measurement_markers_topic, qos_path)
            if self.publish_measurement_markers
            else None
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.srv = self.create_service(Trigger, self.srv_name, self.cb_trigger, callback_group=self.cb)

        # Run id subscriber (latched)
        qos_run = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )
        self.sub_run = self.create_subscription(StringMsg, self.run_id_topic, self._on_run_id, qos_run)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📍 TF: {self.world_frame} -> {self.est_frame} at {self.tf_publish_rate_hz:.1f} Hz")
        self.get_logger().info(f"📡 Listening to: {self.gt_topic}, {self.step_stamp_topic}")
        pub_list = f"{self.est_topic}, {self.pred_topic}, {self.state_topic}"
        if self.publish_estimation_path:
            pub_list += f", {self.estimation_path_topic}"
        if self.publish_measurement_markers:
            pub_list += f", {self.measurement_markers_topic}"
        self.get_logger().info(f"📢 Publishing to: {pub_list}")
        self.get_logger().info(f"🔧 Service: {self.srv_name}")
        self.get_logger().info(f"📁 save_enable={bool(self.save_enable)} out_dir='{self.out_dir}' group_by_run_id={bool(self.group_by_run_id)} run_id_topic='{self.run_id_topic}'")
        mm = "position_only"
        self.get_logger().info(
            f"📊 L={self.L}, dt_pred={self.dt_pred:.3f} measurement_model={mm}"
        )
        self.get_logger().info(f"📊 q_c={self.q_c:.3f}, init_pos_var={self.init_pos_var:.3f}, init_vel_var={self.init_vel_var:.3f}")
        self.get_logger().info(f"🧪 predmeta gt logging: log_gt_in_predmeta={bool(self.log_gt_in_predmeta)} rng_seed={int(self.rng_seed)}")
        self.get_logger().info("----------------------------------------------------------------")

    def _on_step_stamp(self, msg: TimeMsg):
        with self._step_mtx:
            self._last_step_stamp = msg

    def _on_run_id(self, msg: StringMsg):
        rid = str(msg.data).strip()
        with self._run_mtx:
            old = self._run_id
            self._run_id = rid if rid else None
            new = self._run_id
        if old != new:
            with self._path_mtx:
                self._estimation_path_poses.clear()
                self._measurement_viz_buf.clear()
                self._viz_run_id = new

    def _sync_viz_run(self, rid):
        if rid == self._viz_run_id:
            return
        self._estimation_path_poses.clear()
        self._measurement_viz_buf.clear()
        self._viz_run_id = rid

    def _append_and_publish_estimation_path(
        self,
        stamp,
        x: float,
        y: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> None:
        if not self.publish_estimation_path or self.pub_est_path is None:
            return
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.world_frame
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        with self._path_mtx:
            with self._run_mtx:
                rid = self._run_id
            self._sync_viz_run(rid)
            self._estimation_path_poses.append(ps)
            poses = list(self._estimation_path_poses)
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.world_frame
        path_msg.poses = poses
        self.pub_est_path.publish(path_msg)

    def _append_measurement_point(self, x: float, y: float) -> None:
        if not self.publish_measurement_markers:
            return
        with self._path_mtx:
            with self._run_mtx:
                rid = self._run_id
            self._sync_viz_run(rid)
            self._measurement_viz_buf.append((float(x), float(y)))

    def _publish_measurement_markers(self, stamp) -> None:
        if not self.publish_measurement_markers or self.pub_meas_markers is None:
            return
        k = max(float(self.measurement_markers_sigma_scale), 1e-6)
        sx = max(float(self.meas_std_x), 1e-6)
        sy = max(float(self.meas_std_y), 1e-6)
        zw = max(float(self.measurement_markers_z_thickness), 1e-6)
        lw = max(float(self.measurement_markers_line_width), 1e-6)

        with self._path_mtx:
            pts = list(self._measurement_viz_buf)

        ma = MarkerArray()
        clr = Marker()
        clr.header.frame_id = self.world_frame
        clr.header.stamp = stamp
        clr.ns = "target_meas"
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)

        if not pts:
            self.pub_meas_markers.publish(ma)
            return

        line = Marker()
        line.header.frame_id = self.world_frame
        line.header.stamp = stamp
        line.ns = "target_meas"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = lw
        line.color.r = 1.0
        line.color.g = 0.65
        line.color.b = 0.1
        line.color.a = 0.95
        for x, y in pts:
            line.points.append(Point(x=float(x), y=float(y), z=0.0))
        ma.markers.append(line)

        # Axis-aligned uncertainty: SPHERE scale (dx, dy, dz) renders as an ellipsoid in RViz2.
        for i, (x, y) in enumerate(pts):
            ell = Marker()
            ell.header.frame_id = self.world_frame
            ell.header.stamp = stamp
            ell.ns = "target_meas"
            ell.id = i + 1
            ell.type = Marker.SPHERE
            ell.action = Marker.ADD
            ell.pose.position.x = float(x)
            ell.pose.position.y = float(y)
            ell.pose.position.z = 0.0
            ell.pose.orientation.w = 1.0
            ell.scale.x = float(2.0 * k * sx)
            ell.scale.y = float(2.0 * k * sy)
            ell.scale.z = zw
            ell.color.r = 1.0
            ell.color.g = 0.85
            ell.color.b = 0.2
            ell.color.a = 0.22
            ma.markers.append(ell)

        self.pub_meas_markers.publish(ma)

    def _out_base_dir(self) -> str:
        out_base = self.out_dir
        if self.group_by_run_id:
            with self._run_mtx:
                rid = self._run_id
            if rid:
                out_base = os.path.join(out_base, rid)
        os.makedirs(out_base, exist_ok=True)
        return out_base

    def _save_pred_meta(self, *, stamp: TimeMsg, x_last, K_last, x_next, K_next, q_est, q_pred, gt_pose=None):
        if not self.save_enable:
            return

        s = stamp_str(stamp)
        out_base = self._out_base_dir()
        meta_path = os.path.join(out_base, f"predmeta_{s}.json")

        t_est = np.array([float(x_last[0]), float(x_last[1]), 0.0], dtype=np.float64)
        t_pred = np.array([float(x_next[0]), float(x_next[1]), 0.0], dtype=np.float64)
        qx_e, qy_e, qz_e, qw_e = (float(q_est[0]), float(q_est[1]), float(q_est[2]), float(q_est[3]))
        qx_p, qy_p, qz_p, qw_p = (float(q_pred[0]), float(q_pred[1]), float(q_pred[2]), float(q_pred[3]))

        R_est = quat_to_rot_np(qx_e, qy_e, qz_e, qw_e)

        # ΔT_k compares current estimate O_k with previous one-step prediction for this step.
        delta = None
        prev_pred_stamp = None
        with self._step_mtx:
            prev = self._prev_pred

        if prev is not None:
            prev_pred_stamp = {"sec": int(prev["stamp"][0]), "nanosec": int(prev["stamp"][1])}
            t_prev = np.asarray(prev["t"], dtype=np.float64).reshape(3,)
            qx_pp, qy_pp, qz_pp, qw_pp = prev["q"]
            R_prev = quat_to_rot_np(float(qx_pp), float(qy_pp), float(qz_pp), float(qw_pp))

            # ΔT = inv(T_est) * T_prev_pred
            dR = R_est.T @ R_prev
            dt = R_est.T @ (t_prev - t_est)

            # metrics
            dt_norm = float(np.linalg.norm(dt))
            yaw_err = yaw_from_rot(dR)
            tr = float(np.trace(dR))
            c = (tr - 1.0) * 0.5
            c = float(max(-1.0, min(1.0, c)))
            angle_err = float(math.acos(c))

            delta = {
                "translation": {"x": float(dt[0]), "y": float(dt[1]), "z": float(dt[2])},
                "t_norm": dt_norm,
                "yaw_err_rad": yaw_err,
                "angle_err_rad": angle_err,
            }

        meta = {
            "stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
            "world_frame": self.world_frame,
            "target_est_frame": self.est_frame,
            "dt_pred": float(self.dt_pred),
            "window_L": int(self.L),
            "meas_std_xy": {"x": float(self.meas_std_x), "y": float(self.meas_std_y)},
            "rng_seed": int(self.rng_seed),
            "run_id": None,
            "estimate": {
                "translation": {"x": float(t_est[0]), "y": float(t_est[1]), "z": float(t_est[2])},
                "rotation_xyzw": {"x": qx_e, "y": qy_e, "z": qz_e, "w": qw_e},
                "cov_xy": {
                    "xx": float(K_last[0, 0]),
                    "xy": float(K_last[0, 1]),
                    "yx": float(K_last[1, 0]),
                    "yy": float(K_last[1, 1]),
                },
            },
            "prediction_next": {
                "translation": {"x": float(t_pred[0]), "y": float(t_pred[1]), "z": float(t_pred[2])},
                "rotation_xyzw": {"x": qx_p, "y": qy_p, "z": qz_p, "w": qw_p},
                "cov_xy": {
                    "xx": float(K_next[0, 0]),
                    "xy": float(K_next[0, 1]),
                    "yx": float(K_next[1, 0]),
                    "yy": float(K_next[1, 1]),
                },
            },
            "prev_prediction_for_this_step": (prev_pred_stamp if prev_pred_stamp is not None else None),
            "delta_T": delta,  # None for first step (no previous prediction stored)
            "state_4d": {
                "est_mu": [float(x_last[i]) for i in range(4)],
                "est_cov": [float(K_last[i, j]) for i in range(4) for j in range(4)],
                "pred_mu": [float(x_next[i]) for i in range(4)],
                "pred_cov": [float(K_next[i, j]) for i in range(4) for j in range(4)],
            },
        }

        # Simulation-only: log GT pose at this step (in world_frame) if provided.
        if bool(self.log_gt_in_predmeta) and isinstance(gt_pose, dict):
            gt_t = gt_pose.get("t", None)
            gt_q = gt_pose.get("q", None)
            if gt_t is not None and gt_q is not None:
                gx, gy, gz = float(gt_t[0]), float(gt_t[1]), float(gt_t[2])
                qx_g, qy_g, qz_g, qw_g = (float(gt_q[0]), float(gt_q[1]), float(gt_q[2]), float(gt_q[3]))
                gt_block: dict[str, Any] = {
                    "translation": {"x": gx, "y": gy, "z": gz},
                    "rotation_xyzw": {"x": qx_g, "y": qy_g, "z": qz_g, "w": qw_g},
                }
                meta["gt"] = gt_block

        with self._run_mtx:
            rid = self._run_id
        meta["run_id"] = str(rid) if rid else None

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Update stored prediction: current x_next predicts the next step (k+1).
        with self._step_mtx:
            self._prev_pred = {
                "stamp": (int(stamp.sec), int(stamp.nanosec)),
                "t": t_pred,
                "q": (qx_p, qy_p, qz_p, qw_p),
            }
 
    def _broadcast_tf(self, stamp, x, y, qx, qy, qz, qw):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.est_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(t)

    def _publish_bootstrap(self, stamp, x, y):
        # Bootstrap state (x,y,0,0) 
        vx = 0.0
        vy = 0.0
        mu4 = np.array([x, y, vx, vy], dtype=float)
        Sigma4 = np.diag([self.bootstrap_pos_var, self.bootstrap_pos_var,
                          self.bootstrap_vel_var, self.bootstrap_vel_var]).astype(float)

        st = pack_state_msg(stamp, mu4, Sigma4)
        self.pub_pred_state.publish(st)
        self.pub_est_state.publish(st)

        # est
        yaw = float(self._last_heading)
        qx, qy, qz, qw = quat_from_yaw(yaw)
        est = PoseWithCovarianceStamped()
        est.header.stamp = stamp
        est.header.frame_id = self.world_frame
        est.pose.pose.position.x = float(x)
        est.pose.pose.position.y = float(y)
        est.pose.pose.position.z = 0.0
        est.pose.pose.orientation.x = float(qx)
        est.pose.pose.orientation.y = float(qy)
        est.pose.pose.orientation.z = float(qz)
        est.pose.pose.orientation.w = float(qw)
        cov = [0.0] * 36
        cov[0] = float(self.bootstrap_pos_var)
        cov[7] = float(self.bootstrap_pos_var)
        cov[35] = 1.0
        est.pose.covariance = cov
        self.pub_est.publish(est)

        # pred mirrors est during bootstrap
        pred = PoseWithCovarianceStamped()
        pred.header = est.header
        pred.pose.pose = est.pose.pose
        pred.pose.covariance = list(est.pose.covariance)
        self.pub_pred.publish(pred)

        # TF from estimate
        self._broadcast_tf(stamp, x, y, qx, qy, qz, qw)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"✅ Bootstrap Complete: {x:.3f}, {y:.3f}")
        self.get_logger().info(f"📏 State k (x,y,vx,vy): {mu4[0]:.3f}, {mu4[1]:.3f}, {mu4[2]:.3f}, {mu4[3]:.3f}")
        self.get_logger().info(f"📏 Cov k (x,y,vx,vy): {Sigma4[0, 0]:.3f}, {Sigma4[1, 1]:.3f}, {Sigma4[2, 2]:.3f}, {Sigma4[3, 3]:.3f}")
        self.get_logger().info("----------------------------------------------------------------")

    def _publish_est_pred_state(self, stamp, x_last, K_last, x_next, K_next, *, gt_pose=None):
        # ---- Pose estimation at k ----
        vx_k, vy_k = float(x_last[2]), float(x_last[3])
        speed_k = math.hypot(vx_k, vy_k)
        if speed_k >= self.v_min:
            yaw_k = math.atan2(vy_k, vx_k)
            self._last_heading = yaw_k
        else:
            yaw_k = self._last_heading
        qx_k, qy_k, qz_k, qw_k = quat_from_yaw(yaw_k)

        est = PoseWithCovarianceStamped()
        est.header.stamp = stamp
        est.header.frame_id = self.world_frame
        est.pose.pose.position.x = float(x_last[0])
        est.pose.pose.position.y = float(x_last[1])
        est.pose.pose.position.z = 0.0
        est.pose.pose.orientation.x = float(qx_k)
        est.pose.pose.orientation.y = float(qy_k)
        est.pose.pose.orientation.z = float(qz_k)
        est.pose.pose.orientation.w = float(qw_k)

        cov_est = [0.0] * 36
        Pxy_last = 0.5 * (K_last[0:2, 0:2] + K_last[0:2, 0:2].T)
        cov_est[0] = float(Pxy_last[0, 0]); cov_est[1] = float(Pxy_last[0, 1])
        cov_est[6] = float(Pxy_last[1, 0]); cov_est[7] = float(Pxy_last[1, 1])
        V_last = 0.5 * (K_last[2:4, 2:4] + K_last[2:4, 2:4].T)
        cov_est[35] = yaw_var_from_vel(vx_k, vy_k, V_last)
        est.pose.covariance = cov_est
        self.pub_est.publish(est)

        self._append_and_publish_estimation_path(
            stamp, float(x_last[0]), float(x_last[1]), qx_k, qy_k, qz_k, qw_k
        )

        # ---- Estimation state at k (for "no predictor" planning baseline) ----
        st_est = pack_state_msg(stamp, x_last, K_last)
        self.pub_est_state.publish(st_est)

        # TF from estimated pose
        self._broadcast_tf(stamp, x_last[0], x_last[1], qx_k, qy_k, qz_k, qw_k)  

        # ---- Pose prediction at k+1 ----
        vx_k1, vy_k1 = float(x_next[2]), float(x_next[3])
        speed_k1 = math.hypot(vx_k1, vy_k1)
        if speed_k1 >= self.v_min:
            yaw_k1 = math.atan2(vy_k1, vx_k1)
            self._last_heading = yaw_k1
        else:
            yaw_k1 = self._last_heading
        qx_k1, qy_k1, qz_k1, qw_k1 = quat_from_yaw(yaw_k1)

        pred = PoseWithCovarianceStamped()
        pred.header.stamp = stamp
        pred.header.frame_id = self.world_frame
        pred.pose.pose.position.x = float(x_next[0])
        pred.pose.pose.position.y = float(x_next[1])
        pred.pose.pose.position.z = 0.0
        pred.pose.pose.orientation.x = float(qx_k1)
        pred.pose.pose.orientation.y = float(qy_k1)
        pred.pose.pose.orientation.z = float(qz_k1)
        pred.pose.pose.orientation.w = float(qw_k1)

        cov_pred = [0.0] * 36
        Pxy_next = 0.5 * (K_next[0:2, 0:2] + K_next[0:2, 0:2].T)
        cov_pred[0] = float(Pxy_next[0, 0]); cov_pred[1] = float(Pxy_next[0, 1])
        cov_pred[6] = float(Pxy_next[1, 0]); cov_pred[7] = float(Pxy_next[1, 1])
        V_next = 0.5 * (K_next[2:4, 2:4] + K_next[2:4, 2:4].T)
        cov_pred[35] = yaw_var_from_vel(vx_k1, vy_k1, V_next)
        pred.pose.covariance = cov_pred
        self.pub_pred.publish(pred) 

        # ---- Prediction state at k+1 ----
        st = pack_state_msg(stamp, x_next, K_next)
        self.pub_pred_state.publish(st)

        # ---- Save per-step predictor meta (pairs with cloud_capturer meta by stamp) ----
        self._save_pred_meta(
            stamp=stamp,
            x_last=np.asarray(x_last, dtype=float),
            K_last=np.asarray(K_last, dtype=float),
            x_next=np.asarray(x_next, dtype=float),
            K_next=np.asarray(K_next, dtype=float),
            q_est=(qx_k, qy_k, qz_k, qw_k),
            q_pred=(qx_k1, qy_k1, qz_k1, qw_k1),
            gt_pose=gt_pose,
        )

        # self.get_logger().info("----------------------------------------------------------------")
        # self.get_logger().info(f"📏 State k (x,y,vx,vy) (yaw): {x_last[0]:.3f}, {x_last[1]:.3f}, {x_last[2]:.3f}, {x_last[3]:.3f}, {yaw_k:.3f}")
        # self.get_logger().info(f"📏 Cov k (x,y,vx,vy) (yaw): {K_last[0, 0]:.3f}, {K_last[1, 1]:.3f}, {K_last[2, 2]:.3f}, {K_last[3, 3]:.3f}")
        # self.get_logger().info(f"📏 State k+1 (x,y,vx,vy) (yaw): {x_next[0]:.3f}, {x_next[1]:.3f}, {x_next[2]:.3f}, {x_next[3]:.3f}, {yaw_k1:.3f}")  
        # self.get_logger().info(f"📏 Cov k+1 (x,y,vx,vy) (yaw): {K_next[0, 0]:.3f}, {K_next[1, 1]:.3f}, {K_next[2, 2]:.3f}, {K_next[3, 3]:.3f}")
        # self.get_logger().info("----------------------------------------------------------------")

    def cb_gt(self, msg: PoseStamped):
        # Just buffer; do NOT update the filter here (service owns the step)
        self._pose_buf.append(msg)

    def cb_trigger(self, req, resp):
        # One step happens here.
        if not self._pose_buf:
            resp.success = False
            resp.message = "❌ No target pose buffered yet."
            return resp

        with self._step_mtx:
            step_stamp = self._last_step_stamp

        if step_stamp is None:
            resp.success = False
            resp.message = f"❌ No step token received yet on {self.step_stamp_topic}"
            return resp

        self.get_logger().info(
            f"🟣 TOKEN predictor trigger: step_stamp={stamp_str(step_stamp)} last_used={self._last_step_stamp_used}"
        )

        step_key = (int(step_stamp.sec), int(step_stamp.nanosec))
        with self._step_mtx:
            if self._last_step_stamp_used == step_key:
                resp.success = False
                resp.message = f"❌ Step token already processed: {step_key}"
                return resp

        step_t = RclTime.from_msg(step_stamp)

        best = None
        best_dt = 0

        # Find the pose with stamp >= step_stamp and minimal (pose_stamp - step_stamp)
        for ps in reversed(self._pose_buf):  # newest-first, so we can break early if we want
            t = RclTime.from_msg(ps.header.stamp)
            dt = (t - step_t).nanoseconds
            if dt < 0:
                continue  # too old
            if best is None or dt < best_dt:
                best = ps
                best_dt = dt
                if dt == 0:
                    break  # exact match

        if best is None:
            resp.success = False
            resp.message = "❌ No pose measurement with stamp >= step_stamp yet."
            return resp

        self.get_logger().info(
            f"🟣 TOKEN predictor choose: step_stamp={stamp_str(step_stamp)} "
            f"best_pose_stamp={stamp_str(best.header.stamp)} dt_ns={best_dt}"
        )

        ps = best
        stamp = step_stamp  # IMPORTANT: output stamp = token stamp

        # Simulation-only: capture GT pose (un-noised) for this step for predmeta logging.
        gt_dict = None
        if bool(self.log_gt_in_predmeta):
            try:
                if str(ps.header.frame_id).strip() and str(ps.header.frame_id).strip() != str(self.world_frame):
                    self.get_logger().warn(
                        f"⚠️ GT pose frame_id='{ps.header.frame_id}' != world_frame='{self.world_frame}'. "
                        "Assuming they match (no transform applied)."
                    )
                gt_t = np.array([float(ps.pose.position.x), float(ps.pose.position.y), float(ps.pose.position.z)], dtype=float)
                gt_q = (
                    float(ps.pose.orientation.x),
                    float(ps.pose.orientation.y),
                    float(ps.pose.orientation.z),
                    float(ps.pose.orientation.w),
                )
                gt_dict = {"t": gt_t, "q": gt_q, "prev": None}
            except Exception:
                gt_dict = None

        # Noisy position measurement
        x = float(ps.pose.position.x) + float(self._rng.normal(0.0, self.meas_std_x))
        y = float(ps.pose.position.y) + float(self._rng.normal(0.0, self.meas_std_y))
        self._pos_buf.append((x, y))
        self._append_measurement_point(x, y)

        self._obs_buf.append((float(x), float(y)))

        if len(self._obs_buf) < self.L:
            # Keep original bootstrap behavior (vx=vy=0) for consistency with prior results.
            self._publish_bootstrap(stamp, x, y)
            self._publish_measurement_markers(stamp)
            with self._step_mtx:
                self._last_step_stamp_used = step_key
            resp.success = False
            resp.message = f"❌ Bootstrapping {len(self._obs_buf)}/{self.L} (need {self.L} measurements before planning)"
            return resp

        # Build fixed-lag dataset from measurement-derived state observations.
        obs_list = list(self._obs_buf)
        meas_np = np.array(obs_list, dtype=np.float32).reshape(-1, 2)
        self.gp.measurements = jnp.array(meas_np)

        try:
            x_next, K_next, x_last, K_last = self.gp.predict_one_step(self.dt_pred)
        except Exception as e:
            self._publish_measurement_markers(stamp)
            resp.success = False
            resp.message = f"❌ GPR update failed: {e}"
            return resp

        self._publish_est_pred_state(stamp,
            np.array(x_last, dtype=float), np.array(K_last, dtype=float),
            np.array(x_next, dtype=float), np.array(K_next, dtype=float),
            gt_pose=gt_dict,
        )
        self._publish_measurement_markers(stamp)
        with self._step_mtx:
            self._last_step_stamp_used = step_key

        resp.success = True
        resp.message = "✅ Predicted + published (est, pred, state, tf)." 
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
