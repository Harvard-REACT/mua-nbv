#!/usr/bin/env python3
import math
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf2_ros

from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, TwistStamped

from mua_nbv_py_utils.transforms import yaw_from_quat, angle_wrap_pi, quat_from_yaw, quat_mul, quat_conj, quat_rotate, planarize


class PursuerMover(Node):
    """
    Protocol-A "act" stage:
      - Exposes Trigger service: /experiment/pursuer/go_to_best
      - Reads latest /nbv/best_candidate, converts optical->base if needed,
        then drives using cmd_vel with VRPN feedback until reached/timeout.

    This replaces the always-on "teleporter" behavior with a sequential, confirmable action.
    """

    def __init__(self):
        super().__init__("pursuer_mover")
        self.cb = ReentrantCallbackGroup()

        # ---- Params ----
        self.declare_parameter("service_name", "/experiment/pursuer/go_to_best")
        self.declare_parameter("best_candidate_topic", "/nbv/best_candidate")
        self.declare_parameter("candidate_is_optical_pose", True)

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("base_frame", "pursuer/base_link")
        self.declare_parameter("optical_frame", "pursuer/camera_depth_optical_frame")
        self.declare_parameter("tf_lookup_timeout_sec", 0.2)

        self.declare_parameter("vrpn_pursuer_pose_topic", "/vrpn_mocap/pursuer/pose")
        self.declare_parameter("vrpn_target_pose_topic", "/vrpn_mocap/target/pose")
        self.declare_parameter("cmd_vel_topic", "/pursuer/cmd_vel")
        self.declare_parameter("cmd_frame_id", "pursuer/base_link")
        self.declare_parameter("cmd_stamp_zero", True)

        # Feedback source
        # Recommended: use TF world->base_frame (already planarized by vrpn_tf_bridge).
        self.declare_parameter("use_tf_feedback", True)
        # Optional yaw offset (rad) if mocap base frame yaw is not aligned with robot forward.
        self.declare_parameter("yaw_offset_rad", 0.0)
        # Control
        self.declare_parameter("control_rate_hz", 20.0)
        # Safety cap: prevents accidental busy-looping if control_rate_hz is mis-set very high.
        self.declare_parameter("max_control_rate_hz", 60.0)
        # Safety floor: always sleep at least this long per control iteration.
        self.declare_parameter("min_loop_sleep_sec", 0.005)
        self.declare_parameter("max_v", 0.2)
        self.declare_parameter("max_w", 1.5)
        self.declare_parameter("k_heading", 1.5)
        self.declare_parameter("k_dist", 0.8)
        # ---- Controller parameters (EXACTLY matches target_stepper.py) ----
        self.declare_parameter("min_v", 0.05)
        self.declare_parameter("stop_distance_m", 0.05)
        self.declare_parameter("turn_in_place_rad", 1.57)  # kept for config compatibility; not used (matches target_stepper)
        self.declare_parameter("v_turn_penalty", 1.0)      # v /= (1 + v_turn_penalty*|w|)
        self.declare_parameter("heading_slow_rad", 1.05)   # if |heading_err| large, v *= heading_slow_scale
        self.declare_parameter("heading_slow_scale", 0.3)
        # Legacy params kept for backwards compatibility (ignored when using the target_stepper controller):
        self.declare_parameter("heading_stop_rad", 1.2)
        self.declare_parameter("max_w_align", 0.8)
        self.declare_parameter("k_heading_align", 1.0)
        self.declare_parameter("v_slow_radius", 0.6)
        self.declare_parameter("v_near_max", 0.08)
        self.declare_parameter("w_deadband_rad", 0.05)
        self.declare_parameter("v_slew_rate_m_s2", 0.0)
        self.declare_parameter("w_slew_rate_rad_s2", 0.0)
        self.declare_parameter("pose_match_tol_m", 0.10)
        # Kept for compatibility; ignored (target_stepper does not do yaw alignment).
        self.declare_parameter("align_goal_yaw", True)
        self.declare_parameter("yaw_match_tol_rad", 0.25)
        self.declare_parameter("align_hold_factor", 1.5)
        # Timeout policy: actual timeout = max(timeout_min_sec, dist*timeout_per_meter_sec) + yaw_budget_sec
        self.declare_parameter("timeout_min_sec", 15.0)
        self.declare_parameter("timeout_per_meter_sec", 25.0)
        self.declare_parameter("timeout_yaw_budget_sec", 10.0)

        # Safety / collision
        self.declare_parameter("min_target_sep_m", 0.50)  # do not drive closer than this

        # Planarization
        self.declare_parameter("flatten_z", True)
        self.declare_parameter("flatten_z_value", 0.0)
        self.declare_parameter("flatten_roll_pitch", True)

        # Enable output
        self.declare_parameter("enable_cmd", False)
        self.declare_parameter("dry_run", False)

        # Debug logging (throttled)
        self.declare_parameter("log_status", True)
        self.declare_parameter("log_period_sec", 1.0)

        # ---- Read params ----
        self.srv_name = str(self.get_parameter("service_name").value)
        self.best_topic = str(self.get_parameter("best_candidate_topic").value)
        self.candidate_is_optical = bool(self.get_parameter("candidate_is_optical_pose").value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.optical_frame = str(self.get_parameter("optical_frame").value)
        self.tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)

        self.pursuer_pose_topic = str(self.get_parameter("vrpn_pursuer_pose_topic").value)
        self.target_pose_topic = str(self.get_parameter("vrpn_target_pose_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cmd_frame_id = str(self.get_parameter("cmd_frame_id").value)
        self.cmd_stamp_zero = bool(self.get_parameter("cmd_stamp_zero").value)

        self.use_tf_feedback = bool(self.get_parameter("use_tf_feedback").value)
        self.yaw_offset_rad = float(self.get_parameter("yaw_offset_rad").value)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_control_rate_hz = float(self.get_parameter("max_control_rate_hz").value)
        self.min_loop_sleep_sec = float(self.get_parameter("min_loop_sleep_sec").value)
        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.k_dist = float(self.get_parameter("k_dist").value)
        self.min_v = float(self.get_parameter("min_v").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.v_turn_penalty = float(self.get_parameter("v_turn_penalty").value)
        self.heading_slow_rad = float(self.get_parameter("heading_slow_rad").value)
        self.heading_slow_scale = float(self.get_parameter("heading_slow_scale").value)
        self.heading_stop_rad = float(self.get_parameter("heading_stop_rad").value)  # legacy (unused)
        self.pose_match_tol_m = float(self.get_parameter("pose_match_tol_m").value)
        self.align_goal_yaw = bool(self.get_parameter("align_goal_yaw").value)
        self.max_w_align = float(self.get_parameter("max_w_align").value)
        self.k_heading_align = float(self.get_parameter("k_heading_align").value)
        self.yaw_match_tol_rad = float(self.get_parameter("yaw_match_tol_rad").value)
        self.align_hold_factor = float(self.get_parameter("align_hold_factor").value)
        self.timeout_min_sec = float(self.get_parameter("timeout_min_sec").value)
        self.timeout_per_meter_sec = float(self.get_parameter("timeout_per_meter_sec").value)
        self.timeout_yaw_budget_sec = float(self.get_parameter("timeout_yaw_budget_sec").value)
        # Ensure we never time out too aggressively on hardware.
        self.timeout_min_sec = max(30.0, float(self.timeout_min_sec))
        self.min_target_sep_m = float(self.get_parameter("min_target_sep_m").value)

        self.flatten_z = bool(self.get_parameter("flatten_z").value)
        self.flatten_z_value = float(self.get_parameter("flatten_z_value").value)
        self.flatten_roll_pitch = bool(self.get_parameter("flatten_roll_pitch").value)

        self.enable_cmd = bool(self.get_parameter("enable_cmd").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.log_status = bool(self.get_parameter("log_status").value)
        self.log_period_sec = float(self.get_parameter("log_period_sec").value)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---- State ----
        self._mtx = threading.Lock()
        self._active = False
        self._cv = threading.Condition()
        self._best_msg: Optional[PoseStamped] = None
        self._best_seq = 0
        self._pursuer_pose: Optional[PoseStamped] = None
        self._target_pose: Optional[PoseStamped] = None
        self._t_last_log_mono: Optional[float] = None
        # Last commanded velocities (for optional slew limiting).
        self._cmd_v_prev: float = 0.0
        self._cmd_w_prev: float = 0.0

        # ---- Pub/Sub/Service ----
        self.sub_best = self.create_subscription(
            PoseStamped, self.best_topic, self._on_best, qos_profile_sensor_data, callback_group=self.cb
        )
        self.sub_pursuer = self.create_subscription(
            PoseStamped, self.pursuer_pose_topic, self._on_pursuer_pose, qos_profile_sensor_data, callback_group=self.cb
        )
        self.sub_target = self.create_subscription(
            PoseStamped, self.target_pose_topic, self._on_target_pose, qos_profile_sensor_data, callback_group=self.cb
        )
        self.pub_cmd = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.srv = self.create_service(Trigger, self.srv_name, self._on_go_to_best, callback_group=self.cb)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"🔧 Service: {self.srv_name}")
        self.get_logger().info(f"📡 best_candidate: {self.best_topic}")
        self.get_logger().info(f"📡 VRPN: pursuer={self.pursuer_pose_topic} target={self.target_pose_topic}")
        self.get_logger().info(
            f"🕹️  cmd_vel: {self.cmd_vel_topic} (TwistStamped, frame_id={self.cmd_frame_id}) "
            f"enable_cmd={self.enable_cmd} dry_run={self.dry_run}"
        )
        self.get_logger().info(f"🧭 frames: world={self.world_frame} base={self.base_frame} optical={self.optical_frame}")
        self.get_logger().info(f"🧭 feedback: use_tf_feedback={self.use_tf_feedback} yaw_offset_rad={self.yaw_offset_rad:.3f}")
        self.get_logger().info(
            "🕹️  controller: using EXACT target_stepper control law "
            f"(min_v={self.min_v:.3f} stop_distance_m={self.stop_distance_m:.3f} "
            f"v_turn_penalty={self.v_turn_penalty:.3f} heading_slow_rad={self.heading_slow_rad:.3f} "
            f"heading_slow_scale={self.heading_slow_scale:.3f})"
        )
        self.get_logger().info(
            f"🧭 align_goal_yaw={bool(self.align_goal_yaw)} "
            f"yaw_match_tol_rad={self.yaw_match_tol_rad:.3f} "
            f"k_heading_align={self.k_heading_align:.3f} max_w_align={self.max_w_align:.3f} "
            f"align_hold_factor={self.align_hold_factor:.2f}"
        )
        self.get_logger().info(
            f"🧯 min_target_sep_m={self.min_target_sep_m} "
            f"timeout_min_sec={self.timeout_min_sec:.1f} timeout_per_meter_sec={self.timeout_per_meter_sec:.1f} "
            f"timeout_yaw_budget_sec={self.timeout_yaw_budget_sec:.1f}"
        )
        self.get_logger().info(f"🧾 log_status={self.log_status} log_period_sec={self.log_period_sec:.2f}")
        self.get_logger().info("----------------------------------------------------------------")

    def _on_best(self, msg: PoseStamped):
        if msg is None:
            return
        with self._cv:
            self._best_msg = msg
            self._best_seq += 1
            self._cv.notify_all()

    def _on_pursuer_pose(self, msg: PoseStamped):
        if msg is None:
            return
        with self._cv:
            self._pursuer_pose = msg
            self._cv.notify_all()

    def _on_target_pose(self, msg: PoseStamped):
        if msg is None:
            return
        with self._cv:
            self._target_pose = msg
            self._cv.notify_all()

    def _stop(self):
        cmd = TwistStamped()
        cmd.header.stamp = TimeMsg() if bool(self.cmd_stamp_zero) else self.get_clock().now().to_msg()
        cmd.header.frame_id = str(self.cmd_frame_id)
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        if self.enable_cmd and (not self.dry_run):
            self.pub_cmd.publish(cmd)
        # Reset slew state so the next action doesn't "ramp from" stale commands.
        self._cmd_v_prev = 0.0
        self._cmd_w_prev = 0.0

    def _reset_controller_state(self):
        self._t_last_log_mono = None
        self._cmd_v_prev = 0.0
        self._cmd_w_prev = 0.0

    def _log_throttled(self, msg: str):
        if not bool(self.get_parameter("log_status").value):
            return
        period = float(self.get_parameter("log_period_sec").value)
        period = max(0.05, period)
        now = time.monotonic()
        last = self._t_last_log_mono
        if last is None or (now - last) >= period:
            self._t_last_log_mono = now
            self.get_logger().info(msg)

    def _compute_target_stepper_cmd(self, *, dist: float, heading_err: float) -> Tuple[float, float]:
        """
        EXACT copy of target_stepper's heading controller:
          w = sat(k_heading * heading_err, max_w)
          v = min(max_v, k_dist * dist)
          v /= (1 + v_turn_penalty*|w|)
          if |heading_err| > heading_slow_rad: v *= heading_slow_scale
          if dist < stop_distance_m: v=w=0 else v=max(min_v, v) if v>0
        """
        w = float(self.k_heading) * float(heading_err)
        w = max(-float(self.max_w), min(float(self.max_w), float(w)))

        v = min(float(self.max_v), float(self.k_dist) * float(dist))
        v = max(0.0, float(v))

        pen = max(0.0, float(self.v_turn_penalty))
        v = float(v) / (1.0 + pen * abs(float(w)))

        if abs(float(heading_err)) > max(0.0, float(self.heading_slow_rad)):
            v *= max(0.0, min(1.0, float(self.heading_slow_scale)))

        if float(dist) < float(self.stop_distance_m):
            return 0.0, 0.0

        v = max(float(self.min_v), float(v)) if float(v) > 0.0 else 0.0
        return float(v), float(w)

    def _lookup_world_pose_xy_yaw(self, child_frame: str) -> Tuple[float, float, float]:
        tf = self.tf_buffer.lookup_transform(
            self.world_frame,
            child_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=float(self.tf_lookup_timeout_sec)),
        )
        t = tf.transform.translation
        r = tf.transform.rotation
        yaw = yaw_from_quat((float(r.x), float(r.y), float(r.z), float(r.w)))
        yaw = math.atan2(math.sin(yaw + self.yaw_offset_rad), math.cos(yaw + self.yaw_offset_rad))
        return float(t.x), float(t.y), float(yaw)

    def _lookup_optical_from_base(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        # optical <- base at Time(0) (static)
        tf = self.tf_buffer.lookup_transform(
            self.optical_frame,
            self.base_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=float(self.tf_lookup_timeout_sec)),
        )
        t = tf.transform.translation
        r = tf.transform.rotation
        return (float(t.x), float(t.y), float(t.z)), (float(r.x), float(r.y), float(r.z), float(r.w))

    def _optical_goal_to_base_goal(self, msg: PoseStamped) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        # world -> optical goal (from best_candidate)
        p = msg.pose.position
        o = msg.pose.orientation
        t_wo = (float(p.x), float(p.y), float(p.z))
        q_wo = (float(o.x), float(o.y), float(o.z), float(o.w))

        t_ob, q_ob = self._lookup_optical_from_base()  # optical <- base

        # We want world -> base such that: T_WO = T_WB * T_BO  =>  T_WB = T_WO * T_OB
        t_wb_off = quat_rotate(q_wo, t_ob)
        t_wb = (t_wo[0] + t_wb_off[0], t_wo[1] + t_wb_off[1], t_wo[2] + t_wb_off[2])
        q_wb = quat_mul(q_wo, q_ob)

        t_wb, q_wb = planarize(t_wb, q_wb, flatten_z=self.flatten_z, z0=self.flatten_z_value, flatten_roll_pitch=self.flatten_roll_pitch)
        return t_wb, q_wb

    def _wait_for_best(self, start_seq: int, timeout_sec: float) -> Optional[PoseStamped]:
        deadline = time.time() + float(timeout_sec)
        with self._cv:
            while time.time() < deadline:
                if self._best_seq > start_seq and self._best_msg is not None:
                    return self._best_msg
                self._cv.wait(timeout=0.05)
            return self._best_msg

    def _on_go_to_best(self, req, resp):
        # Prevent concurrent long-running service calls from stacking up (can peg CPU and send conflicting cmd_vel).
        with self._mtx:
            if self._active:
                resp.success = False
                resp.message = "❌ Previous go_to_best still in progress"
                return resp
            self._active = True

        try:
            # Block until we have a goal + VRPN pose feedback.
            self._reset_controller_state()
            with self._cv:
                start_seq = int(self._best_seq)

            best = self._wait_for_best(start_seq, timeout_sec=2.0)
            if best is None:
                resp.success = False
                resp.message = "❌ No /nbv/best_candidate received"
                return resp

            if (not self.use_tf_feedback) and (self._pursuer_pose is None):
                resp.success = False
                resp.message = "❌ No pursuer VRPN pose received"
                return resp

            # Convert to base goal in world
            try:
                if self.candidate_is_optical:
                    (g_t, g_q) = self._optical_goal_to_base_goal(best)
                else:
                    p = best.pose.position
                    o = best.pose.orientation
                    g_t = (float(p.x), float(p.y), float(p.z))
                    g_q = (float(o.x), float(o.y), float(o.z), float(o.w))
                    g_t, g_q = planarize(g_t, g_q, flatten_z=self.flatten_z, z0=self.flatten_z_value, flatten_roll_pitch=self.flatten_roll_pitch)
            except Exception as e:
                resp.success = False
                resp.message = f"❌ Goal conversion failed: {e}"
                return resp

            goal_yaw = yaw_from_quat(g_q)
            goal_x, goal_y = float(g_t[0]), float(g_t[1])
            self.get_logger().info(
                f"🎯 best_candidate frame={best.header.frame_id} "
                f"pos=({best.pose.position.x:.3f},{best.pose.position.y:.3f},{best.pose.position.z:.3f})"
            )
            self.get_logger().info(f"🎯 goal_base(world) xy=({goal_x:.3f},{goal_y:.3f}) yaw={goal_yaw:.3f} rad")

            # Initial pose for adaptive timeout.
            try:
                if self.use_tf_feedback:
                    px0, py0, yaw0 = self._lookup_world_pose_xy_yaw(self.base_frame)
                else:
                    purs0 = self._pursuer_pose
                    if purs0 is None:
                        raise RuntimeError("Lost pursuer pose")
                    px0 = float(purs0.pose.position.x)
                    py0 = float(purs0.pose.position.y)
                    yaw0 = yaw_from_quat(
                        (
                            float(purs0.pose.orientation.x),
                            float(purs0.pose.orientation.y),
                            float(purs0.pose.orientation.z),
                            float(purs0.pose.orientation.w),
                        )
                    )
                    yaw0 = math.atan2(math.sin(yaw0 + self.yaw_offset_rad), math.cos(yaw0 + self.yaw_offset_rad))
            except Exception:
                px0, py0, yaw0 = goal_x, goal_y, goal_yaw

            dist0 = math.hypot(goal_x - px0, goal_y - py0)
            adaptive_timeout = max(self.timeout_min_sec, dist0 * max(0.0, self.timeout_per_meter_sec))
            if bool(self.align_goal_yaw):
                adaptive_timeout += max(0.0, float(self.timeout_yaw_budget_sec))
            deadline = time.time() + float(adaptive_timeout)

            # Guard against misconfiguration: clamp Hz and enforce a minimum sleep.
            hz = float(self.get_parameter("control_rate_hz").value)
            hz_cap = max(1.0, float(self.max_control_rate_hz))
            hz = min(max(0.1, hz), hz_cap)
            sleep_sec = max(1.0 / hz, max(0.0, float(self.min_loop_sleep_sec)))

            last_dist = float("inf")
            last_yaw_err = 0.0
            in_align = False

            while time.time() < deadline:
                with self._cv:
                    purs_msg = self._pursuer_pose
                    targ_msg = self._target_pose

                try:
                    if self.use_tf_feedback:
                        px, py, yaw = self._lookup_world_pose_xy_yaw(self.base_frame)
                    else:
                        if purs_msg is None:
                            raise RuntimeError("Lost pursuer pose")
                        px = float(purs_msg.pose.position.x)
                        py = float(purs_msg.pose.position.y)
                        yaw = yaw_from_quat(
                            (
                                float(purs_msg.pose.orientation.x),
                                float(purs_msg.pose.orientation.y),
                                float(purs_msg.pose.orientation.z),
                                float(purs_msg.pose.orientation.w),
                            )
                        )
                        yaw = math.atan2(math.sin(yaw + self.yaw_offset_rad), math.cos(yaw + self.yaw_offset_rad))
                except Exception as e:
                    self._stop()
                    resp.success = False
                    resp.message = f"❌ Pose feedback failed: {e}"
                    return resp

                dx = goal_x - px
                dy = goal_y - py
                dist = math.hypot(dx, dy)
                last_dist = float(dist)

                # Collision safety: stop if too close to target
                if self.min_target_sep_m > 0.0:
                    try:
                        if self.use_tf_feedback:
                            tx, ty, _ = self._lookup_world_pose_xy_yaw("target/base_link")
                        else:
                            if targ_msg is None:
                                tx = ty = None  # no check
                            else:
                                tx = float(targ_msg.pose.position.x)
                                ty = float(targ_msg.pose.position.y)
                        if tx is not None and ty is not None:
                            dpt = math.hypot(tx - px, ty - py)
                            if dpt < self.min_target_sep_m:
                                self._stop()
                                resp.success = False
                                resp.message = f"❌ Safety stop: pursuer-target dist={dpt:.3f} < {self.min_target_sep_m:.3f}"
                                return resp
                    except Exception:
                        # If target TF isn't available yet, skip this cycle rather than failing.
                        pass

                tol = float(self.pose_match_tol_m)
                hold = max(1.0, float(self.align_hold_factor))
                # If yaw alignment is disabled, arrival check is just distance (like target_stepper).
                if (not bool(self.align_goal_yaw)) and dist <= tol:
                    self._stop()
                    resp.success = True
                    resp.message = f"✅ Reached goal (dist={dist:.3f})"
                    return resp

                # If yaw alignment is enabled, enter/exit align mode with hysteresis.
                if bool(self.align_goal_yaw):
                    if (not in_align) and dist <= tol:
                        in_align = True
                    if in_align and dist > tol * hold:
                        in_align = False

                if in_align:
                    # If we drift outside the position tolerance while turning-in-place, we must
                    # return to move mode; otherwise v=0 alignment can never reduce dist again.
                    if dist > tol:
                        in_align = False
                    else:
                        yaw_err = angle_wrap_pi(float(goal_yaw) - float(yaw))
                        last_yaw_err = float(yaw_err)
                        if abs(float(yaw_err)) <= float(self.yaw_match_tol_rad) and dist <= tol:
                            self._stop()
                            resp.success = True
                            resp.message = f"✅ Reached goal (dist={dist:.3f}, yaw_err={yaw_err:.3f})"
                            return resp

                        w = float(self.k_heading_align) * float(yaw_err)
                        w = max(-float(self.max_w_align), min(float(self.max_w_align), float(w)))
                        v = 0.0
                        self._log_throttled(
                            f"[align] dist={dist:.3f} yaw_err={yaw_err:.3f} v={v:.3f} w={w:.3f} "
                            f"pos=({px:.2f},{py:.2f})"
                        )

                        cmd = TwistStamped()
                        cmd.header.stamp = TimeMsg() if bool(self.cmd_stamp_zero) else self.get_clock().now().to_msg()
                        cmd.header.frame_id = str(self.cmd_frame_id)
                        cmd.twist.linear.x = float(v)
                        cmd.twist.angular.z = float(w)
                        if self.enable_cmd and (not self.dry_run):
                            self.pub_cmd.publish(cmd)
                        time.sleep(sleep_sec)
                        continue

                target_heading = math.atan2(dy, dx)
                heading_error = math.atan2(math.sin(target_heading - yaw), math.cos(target_heading - yaw))
                last_yaw_err = float(heading_error)
                v, w = self._compute_target_stepper_cmd(dist=dist, heading_err=heading_error)
                self._log_throttled(
                    f"[move] dist={dist:.3f} heading_err={heading_error:.3f} v={v:.3f} w={w:.3f} "
                    f"pos=({px:.2f},{py:.2f}) goal=({goal_x:.2f},{goal_y:.2f})"
                )

                cmd = TwistStamped()
                cmd.header.stamp = TimeMsg() if bool(self.cmd_stamp_zero) else self.get_clock().now().to_msg()
                cmd.header.frame_id = str(self.cmd_frame_id)
                cmd.twist.linear.x = float(v)
                cmd.twist.angular.z = float(w)

                if self.enable_cmd and (not self.dry_run):
                    self.pub_cmd.publish(cmd)

                time.sleep(sleep_sec)

            self._stop()
            resp.success = False
            resp.message = f"❌ Timeout moving to best candidate (dist={last_dist:.3f}, yaw_err={last_yaw_err:.3f})"
            return resp
        finally:
            with self._mtx:
                self._active = False


def main():
    rclpy.init()
    node = PursuerMover()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    try:
        node._stop()
    except Exception:
        pass
    node.destroy_node()
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
