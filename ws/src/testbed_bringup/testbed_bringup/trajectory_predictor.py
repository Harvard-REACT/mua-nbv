#!/usr/bin/env python3
"""
Testbed trajectory predictor (TGPR-CV, JAX).

Purpose:
  Publish the prediction topics expected by mua_nbv_planner dynamic mode:
    - /nbv/target_prediction_state  (Float32MultiArray, 22 floats)
    - /nbv/target_estimation_state  (Float32MultiArray, 22 floats)
    - /nbv/target_prediction        (PoseWithCovarianceStamped)
    - /nbv/target_estimation        (PoseWithCovarianceStamped)

Triggering:
  For Protocol-A, prediction is computed on demand by a Trigger service,
  using the latest buffered target VRPN pose at/after the current step token.
"""

import math
from collections import deque
from typing import Deque, Optional, Tuple
import numpy as np

import jax.numpy as jnp

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from mua_nbv_py_utils.transforms import quat_from_yaw, yaw_var_from_vel
from mua_nbv_prediction.tgpr import TGPR_CV
from mua_nbv_common.ros_helpers import pack_state_msg


class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__("trajectory_predictor")
        self.cb = ReentrantCallbackGroup()

        # Frames
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("target_est_frame", "target/est_base_link")

        # Topics
        # Measurement sources:
        # - "waypoint": DISCRETE per-step waypoints from target_stepper (recommended for Protocol-A)
        # - "vrpn": continuous-time mocap poses
        self.declare_parameter("measurement_source", "waypoint")  # "waypoint" | "vrpn"
        self.declare_parameter("input_target_gt_topic", "/vrpn_mocap/target/pose")
        self.declare_parameter("input_waypoint_topic", "/experiment/target/waypoint")
        self.declare_parameter("input_step_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("output_estimation_topic", "/nbv/target_estimation")
        self.declare_parameter("output_prediction_topic", "/nbv/target_prediction")
        self.declare_parameter("output_prediction_state_topic", "/nbv/target_prediction_state")
        self.declare_parameter("output_estimation_state_topic", "/nbv/target_estimation_state")

        # Service
        self.declare_parameter("service_name", "/experiment/update/measurements")

        # Model / buffers (align with simulation "bootstrap" behavior)
        self.declare_parameter("pose_buffer_len", 200) 
        self.declare_parameter("token_is_step_id", False)
        # Predictor bootstrap window (simulation: window_L). Until we have L measurements,
        # we publish a bootstrap state and return success=False.
        self.declare_parameter("window_L", 10)
        self.declare_parameter("dt_pred", 1.0)  # seconds ahead for prediction
        self.declare_parameter("v_min", 0.05)   # below this, keep previous heading
        # Discrete step interval for waypoint measurements (used to compute per-step velocity).
        self.declare_parameter("waypoint_dt_sec", 1.0)
        # TGPR smoother process PSD (continuous-time) (simulation uses q_c)
        self.declare_parameter("q_c", 1.0)
        self.declare_parameter("rng_seed", 0)

        # Measurement noise (simulation-like; optional)
        self.declare_parameter("meas_std_x", 0.0)
        self.declare_parameter("meas_std_y", 0.0)
        # Bootstrap covariance (simulation-like init_* vars; used during bootstrapping)
        self.declare_parameter("init_pos_var", 0.05)
        self.declare_parameter("init_vel_var", 0.10)

        # Covariance (simple diagonals for state [x,y,vx,vy])
        self.declare_parameter("pos_var", 0.05)  # m^2
        self.declare_parameter("vel_var", 0.10)  # (m/s)^2

        # TF broadcast rate (estimated frame)
        self.declare_parameter("tf_publish_rate_hz", 15.0)

        # ---- Read params ----
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.est_frame = str(self.get_parameter("target_est_frame").value)

        self.measurement_source = str(self.get_parameter("measurement_source").value).strip().lower()
        self.gt_topic = str(self.get_parameter("input_target_gt_topic").value)
        self.waypoint_topic = str(self.get_parameter("input_waypoint_topic").value)
        self.step_stamp_topic = str(self.get_parameter("input_step_stamp_topic").value)
        self.est_topic = str(self.get_parameter("output_estimation_topic").value)
        self.pred_topic = str(self.get_parameter("output_prediction_topic").value)
        self.pred_state_topic = str(self.get_parameter("output_prediction_state_topic").value)
        self.est_state_topic = str(self.get_parameter("output_estimation_state_topic").value)
        self.srv_name = str(self.get_parameter("service_name").value)

        self.pose_buffer_len = int(self.get_parameter("pose_buffer_len").value)
        self.token_is_step_id = bool(self.get_parameter("token_is_step_id").value)
        self.L = int(self.get_parameter("window_L").value)
        self.dt_pred = float(self.get_parameter("dt_pred").value)
        self.v_min = float(self.get_parameter("v_min").value)
        self.waypoint_dt_sec = float(self.get_parameter("waypoint_dt_sec").value)
        self.q_c = float(self.get_parameter("q_c").value)
        self.meas_std_x = float(self.get_parameter("meas_std_x").value)
        self.meas_std_y = float(self.get_parameter("meas_std_y").value)
        self.init_pos_var = float(self.get_parameter("init_pos_var").value)
        self.init_vel_var = float(self.get_parameter("init_vel_var").value)
        self.pos_var = float(self.get_parameter("pos_var").value)
        self.vel_var = float(self.get_parameter("vel_var").value)
        self.tf_publish_rate_hz = float(self.get_parameter("tf_publish_rate_hz").value)
        self.rng_seed = int(self.get_parameter("rng_seed").value)

        # ---- State ----
        self._pose_buf: Deque[PoseStamped] = deque(maxlen=max(2, self.pose_buffer_len))
        # Per-step position measurements (noisy).
        self._pos_buf: Deque[Tuple[float, float]] = deque(maxlen=max(2, self.L + 1))
        # Position-only observations for TGPR (x,y).
        self._obs_buf: Deque[Tuple[float, float]] = deque(maxlen=max(1, self.L))
        self._last_step_stamp: Optional[TimeMsg] = None
        self._last_step_used: Optional[Tuple[int, int]] = None
        self._last_heading: float = 0.0
        self._last_est_tf: Optional[TransformStamped] = None
        self._rng = np.random.default_rng(int(self.rng_seed))

        # ---- TGPR smoother model ----
        K0 = jnp.diag(
            jnp.array([float(self.init_pos_var), float(self.init_pos_var), float(self.init_vel_var), float(self.init_vel_var)], dtype=jnp.float32)
        )
        rx = max(float(self.meas_std_x) ** 2, 1e-8)
        ry = max(float(self.meas_std_y) ** 2, 1e-8)
        R2 = jnp.diag(jnp.array([float(rx), float(ry)], dtype=jnp.float32))
        self.gp = TGPR_CV(
            dataset_history=int(self.L),
            q_c=float(self.q_c),
            K0=K0,
            R=R2,
            dt=float(self.dt_pred),
            observe_position_only=True,
        )
        self.bootstrap_pos_var = float(min(float(self.init_pos_var), 1.0))
        self.bootstrap_vel_var = float(min(float(self.init_vel_var), 1.0))

        # ---- IO ----
        self.sub_gt = None
        self.sub_wp = None
        if self.measurement_source == "waypoint":
            # Waypoint publisher is TRANSIENT_LOCAL; subscribe with matching QoS so we always get the last waypoint.
            qos_wp = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.sub_wp = self.create_subscription(PoseStamped, self.waypoint_topic, self._on_gt, qos_wp, callback_group=self.cb)
        elif self.measurement_source == "vrpn":
            # VRPN is typically BEST_EFFORT; use sensor-data QoS to avoid QoS incompatibility.
            self.sub_gt = self.create_subscription(
                PoseStamped, self.gt_topic, self._on_gt, qos_profile_sensor_data, callback_group=self.cb
            )
        else:
            self.get_logger().warn(
                f"Unsupported measurement_source='{self.measurement_source}', falling back to 'waypoint'"
            )
            self.measurement_source = "waypoint"
            qos_wp = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.sub_wp = self.create_subscription(PoseStamped, self.waypoint_topic, self._on_gt, qos_wp, callback_group=self.cb)
        self.sub_step = self.create_subscription(TimeMsg, self.step_stamp_topic, self._on_step_stamp, 10, callback_group=self.cb)

        self.pub_est = self.create_publisher(PoseWithCovarianceStamped, self.est_topic, 10)
        self.pub_pred = self.create_publisher(PoseWithCovarianceStamped, self.pred_topic, 10)
        self.pub_pred_state = self.create_publisher(Float32MultiArray, self.pred_state_topic, 10)
        self.pub_est_state = self.create_publisher(Float32MultiArray, self.est_state_topic, 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.srv = self.create_service(Trigger, self.srv_name, self._on_trigger, callback_group=self.cb)

        # TF publish timer
        period = 1.0 / max(self.tf_publish_rate_hz, 1e-3)
        self.create_timer(period, self._tf_tick)

        self.get_logger().info("----------------------------------------------------------------")
        if self.measurement_source == "waypoint":
            self.get_logger().info(f"📡 Listening to: {self.waypoint_topic}, {self.step_stamp_topic} (measurement_source=waypoint)")
        elif self.measurement_source == "vrpn":
            self.get_logger().info(f"📡 Listening to: {self.gt_topic}, {self.step_stamp_topic} (measurement_source=vrpn)")
        else:
            self.get_logger().info(
                f"📡 Listening to: {self.waypoint_topic}, {self.step_stamp_topic} (measurement_source=waypoint)"
            )
        self.get_logger().info(f"📢 Publishing: {self.est_topic}, {self.pred_topic}, {self.pred_state_topic}, {self.est_state_topic}")
        self.get_logger().info(f"🔧 Service: {self.srv_name}")
        self.get_logger().info(f"📍 TF: {self.world_frame} -> {self.est_frame} ({self.tf_publish_rate_hz:.1f} Hz)")
        self.get_logger().info(
            f"📊 L={self.L} dt_pred={self.dt_pred:.3f}s "
            f"pos_var={self.pos_var:.3f} vel_var={self.vel_var:.3f} "
            f"meas_std=({self.meas_std_x:.3f},{self.meas_std_y:.3f})"
        )
        self.get_logger().info(
            f"🧠 TGPR smoother: measurement_model=position_only "
            f"L={int(self.L)} q_c={float(self.q_c):.3f} rng_seed={int(self.rng_seed)}"
        )
        self.get_logger().info("----------------------------------------------------------------")

    def _on_gt(self, msg: PoseStamped):
        if msg is None:
            return
        self._pose_buf.append(msg)

    def _on_step_stamp(self, msg: TimeMsg):
        self._last_step_stamp = msg

    def _tf_tick(self):
        tf = self._last_est_tf
        if tf is not None:
            self.tf_broadcaster.sendTransform(tf)

    def _find_pose_at_or_after(self, step: TimeMsg) -> Optional[PoseStamped]:
        if bool(self.token_is_step_id):
            # Token is not comparable to pose header stamps; use newest available.
            if len(self._pose_buf) < 1:
                return None
            return self._pose_buf[-1]

        # Find pose with stamp >= step and minimal dt
        step_t = RclTime.from_msg(step)
        best = None
        best_dt_ns = None
        for ps in reversed(self._pose_buf):  # newest first
            t = RclTime.from_msg(ps.header.stamp)
            dt_ns = (t - step_t).nanoseconds
            if dt_ns < 0:
                continue
            if best is None or dt_ns < best_dt_ns:
                best = ps
                best_dt_ns = dt_ns
                if dt_ns == 0:
                    break
        return best

    def _estimate_vel(self, cur: PoseStamped) -> Tuple[float, float]:
        # VRPN mode: use previous buffered pose for finite-diff velocity (world frame, continuous time).
        cur_t = RclTime.from_msg(cur.header.stamp)
        for prev in reversed(list(self._pose_buf)[:-1]):
            prev_t = RclTime.from_msg(prev.header.stamp)
            dt = (cur_t - prev_t).nanoseconds * 1e-9
            if dt <= 1e-3:
                continue
            vx = (float(cur.pose.position.x) - float(prev.pose.position.x)) / dt
            vy = (float(cur.pose.position.y) - float(prev.pose.position.y)) / dt
            return float(vx), float(vy)
        return 0.0, 0.0

    def _estimate_vel_from_meas(self) -> Tuple[float, float]:
        # Waypoint mode: finite-diff on the last two DISCRETE position measurements.
        if len(self._pos_buf) < 2:
            return 0.0, 0.0
        (x0, y0) = self._pos_buf[-2]
        (x1, y1) = self._pos_buf[-1]
        dt = float(max(1e-6, self.waypoint_dt_sec))
        return float((x1 - x0) / dt), float((y1 - y0) / dt)

    def _publish_state(
        self, stamp: TimeMsg, *, x: float, y: float, vx: float, vy: float, Sigma4: Optional[np.ndarray] = None, is_pred: bool
    ):
        if Sigma4 is None:
            Sigma4 = np.diag([float(self.pos_var), float(self.pos_var), float(self.vel_var), float(self.vel_var)]).astype(np.float64)
        msg = pack_state_msg(stamp, np.array([x, y, vx, vy]), Sigma4)
        if is_pred:
            self.pub_pred_state.publish(msg)
        else:
            self.pub_est_state.publish(msg)

    def _publish_pose(self, stamp: TimeMsg, *, x: float, y: float, vx: float, vy: float, P4: Optional[np.ndarray] = None, is_pred: bool):
        speed = math.hypot(vx, vy)
        if speed >= self.v_min:
            self._last_heading = math.atan2(vy, vx)
        yaw = float(self._last_heading)
        qx, qy, qz, qw = quat_from_yaw(yaw)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        cov = [0.0] * 36
        if P4 is None:
            cov[0] = float(self.pos_var)
            cov[7] = float(self.pos_var)
            cov[35] = 1.0
        else:
            P4 = np.asarray(P4, dtype=np.float64).reshape(4, 4)
            Pxy = 0.5 * (P4[0:2, 0:2] + P4[0:2, 0:2].T)
            cov[0] = float(Pxy[0, 0])
            cov[1] = float(Pxy[0, 1])
            cov[6] = float(Pxy[1, 0])
            cov[7] = float(Pxy[1, 1])
            V = 0.5 * (P4[2:4, 2:4] + P4[2:4, 2:4].T)
            cov[35] = float(yaw_var_from_vel(float(vx), float(vy), V))
        msg.pose.covariance = cov

        if is_pred:
            self.pub_pred.publish(msg)
        else:
            self.pub_est.publish(msg)
            tf = TransformStamped()
            # IMPORTANT (testbed): TF must be stamped with real clock time so it composes with
            # other epoch-stamped TF streams (VRPN, camera). The planner gating uses the
            # step token embedded in prediction_state/estimation_state; TF is only used
            # for geometric transforms (e.g., cloud_capturer), so "latest TF" should work.
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = self.world_frame
            tf.child_frame_id = self.est_frame
            tf.transform.translation.x = float(x)
            tf.transform.translation.y = float(y)
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = float(qx)
            tf.transform.rotation.y = float(qy)
            tf.transform.rotation.z = float(qz)
            tf.transform.rotation.w = float(qw)
            self._last_est_tf = tf

    def _publish_bootstrap(self, stamp: TimeMsg, *, x: float, y: float, vx: float, vy: float):
        # Bootstrap stage: publish topics but return success=False to gate planning.
        # We publish measurement-derived velocity so heading doesn't "snap" after bootstrapping.
        P0 = np.diag(
            [float(self.bootstrap_pos_var), float(self.bootstrap_pos_var), float(self.bootstrap_vel_var), float(self.bootstrap_vel_var)]
        ).astype(np.float64)
        self._publish_state(stamp, x=x, y=y, vx=vx, vy=vy, Sigma4=P0, is_pred=False)
        self._publish_pose(stamp, x=x, y=y, vx=vx, vy=vy, P4=P0, is_pred=False)
        self._publish_state(stamp, x=x, y=y, vx=vx, vy=vy, Sigma4=P0, is_pred=True)
        self._publish_pose(stamp, x=x, y=y, vx=vx, vy=vy, P4=P0, is_pred=True)

    def _on_trigger(self, req, resp):
        step = self._last_step_stamp
        if step is None:
            resp.success = False
            resp.message = f"❌ No step token received yet on {self.step_stamp_topic}"
            return resp

        step_key = (int(step.sec), int(step.nanosec))
        if self._last_step_used == step_key:
            resp.success = False
            resp.message = f"❌ Step token already processed: {step_key}"
            return resp

        if len(self._pose_buf) < 1:
            resp.success = False
            resp.message = "❌ No target pose buffered yet."
            return resp

        best = self._find_pose_at_or_after(step)
        if best is None:
            resp.success = False
            resp.message = "❌ No pose measurement available yet."
            return resp

        # Noisy measurement (simulation-style, optional)
        x = float(best.pose.position.x) + float(self._rng.normal(0.0, self.meas_std_x))
        y = float(best.pose.position.y) + float(self._rng.normal(0.0, self.meas_std_y))
        self._pos_buf.append((x, y))

        self._obs_buf.append((float(x), float(y)))
 
        if len(self._obs_buf) < int(max(1, self.L)):
            # Bootstrap stage: publish stable topics, but return success=False to gate planning.
            self._publish_bootstrap(step, x=x, y=y, vx=0.0, vy=0.0)
            self._last_step_used = step_key
            resp.success = False
            resp.message = f"❌ Bootstrapping {len(self._obs_buf)}/{int(max(1, self.L))} (need {int(max(1, self.L))} measurements before planning)"
            return resp

        # TGPR smoother: use the last L position-only observations.
        meas_np = np.array(list(self._obs_buf), dtype=np.float32).reshape(-1, 2)
        self.gp.measurements = jnp.array(meas_np)
        try:
            x_next, K_next, x_last, K_last = self.gp.predict_one_step(float(self.dt_pred))
        except Exception as e:
            resp.success = False
            resp.message = f"❌ TGPR update failed: {e}"
            return resp

        # Estimation at k (stamp = step token)
        x_last_np = np.array(x_last, dtype=float).reshape(4,)
        K_last_np = np.array(K_last, dtype=float).reshape(4, 4)
        x_next_np = np.array(x_next, dtype=float).reshape(4,)
        K_next_np = np.array(K_next, dtype=float).reshape(4, 4)
        self._publish_state(step, x=float(x_last_np[0]), y=float(x_last_np[1]), vx=float(x_last_np[2]), vy=float(x_last_np[3]), Sigma4=K_last_np, is_pred=False)
        self._publish_pose(step, x=float(x_last_np[0]), y=float(x_last_np[1]), vx=float(x_last_np[2]), vy=float(x_last_np[3]), P4=K_last_np, is_pred=False)

        # Prediction at k+1 (still stamped with step token for planner gating)
        self._publish_state(step, x=float(x_next_np[0]), y=float(x_next_np[1]), vx=float(x_next_np[2]), vy=float(x_next_np[3]), Sigma4=K_next_np, is_pred=True)
        self._publish_pose(step, x=float(x_next_np[0]), y=float(x_next_np[1]), vx=float(x_next_np[2]), vy=float(x_next_np[3]), P4=K_next_np, is_pred=True)

        self._last_step_used = step_key
        resp.success = True
        resp.message = "✅ Published estimation + prediction (state + pose + tf)."
        return resp


def main():
    rclpy.init()
    node = TrajectoryPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
