#!/usr/bin/env python3
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import Trigger
from std_msgs.msg import Int32
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, TwistStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from mua_nbv_py_utils.transforms import yaw_from_quat


def figure8_points(n: int, scale: float):
    """
    Figure-8 waypoint generator:
      x = scale * cos(t)
      y = scale * sin(2t) / 2
    """
    pts = []
    for i in range(int(n)):
        t = 2.0 * math.pi * float(i) / float(max(int(n), 1))
        x = scale * math.cos(t)
        y = scale * math.sin(2.0 * t) / 2.0
        pts.append((x, y))
    return pts


def s_curve_points(n: int, length_x: float, width_y: float):
    """
    S-curve waypoint generator inside a 2D rectangle aligned with +x:
      - x spans length_x meters
      - y spans width_y meters (peak-to-peak)

    We use one full sine period so:
      y=0 at both ends, and the curve makes an "S" as x increases.

      x(t) = (t - 0.5) * length_x
      y(t) = (width_y/2) * sin(2*pi*t)
      t in [0,1]
    """
    n = int(max(int(n), 2))
    L = float(length_x)
    W = float(width_y)
    pts = []
    for i in range(n):
        t = float(i) / float(n - 1)
        x = (t - 0.5) * L
        y = 0.5 * W * math.sin(2.0 * math.pi * t)
        pts.append((x, y))
    return pts


class TargetStepper(Node):
    """
    Testbed equivalent of simulation_bringup/target_stepper.py:
    - Exposes a Trigger service to "advance" to the next waypoint.
    - Drives a Turtlebot using /cmd_vel until it reaches the waypoint (VRPN feedback).
    - Publishes /experiment/step as a simple counter (no /experiment/step_stamp token on testbed).
    """

    def __init__(self):
        super().__init__("target_stepper")
        self.cb = ReentrantCallbackGroup()

        # ---- Params (modeled after simulation_bringup target_stepper + our simple heading controller) ----
        # IO
        self.declare_parameter("vrpn_pose_topic", "/vrpn_mocap/pursuer/pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("cmd_frame_id", "target/base_link")
        # TwistStamped header.stamp handling:
        # - If your ROS machines are not time-synced, using "now()" here can make commands
        #   appear too far in the future/past for the base driver and be ignored.
        # - Setting stamp to zero is the most robust cross-machine behavior.
        self.declare_parameter("cmd_stamp_zero", True)
        self.declare_parameter("output_step_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("output_step_index_topic", "/experiment/step")
        # Publish the active planned waypoint (world frame) for downstream prediction/debug.
        self.declare_parameter("output_waypoint_topic", "/experiment/target/waypoint")
        self.declare_parameter("service_name", "/experiment/target/advance")
        # Step token semantics:
        # - "pose_stamp": publish the incoming VRPN PoseStamped.header.stamp (simulation-like time token)
        # - "step_index": publish a small monotonic id (sec=step_k, nanosec=0)
        #
        # NOTE: mua_nbv_planner encodes prediction/estimation stamp inside Float32MultiArray.
        # Float32 cannot represent wall-clock epoch seconds (~1.7e9) precisely, which can make
        # the planner's "pred_stamp >= step_stamp" gate fail forever on testbed.
        # For dynamic testbed, prefer "step_index".
        self.declare_parameter("step_stamp_mode", "pose_stamp")

        # Trajectory
        self.declare_parameter("trajectory_type", "figure8")
        self.declare_parameter("trajectory_scale", 1.5)
        # S-curve parameters (lab frame): 2x5 rectangle means width_y=2, length_x=5.
        self.declare_parameter("s_length_x_m", 5.0)
        self.declare_parameter("s_width_y_m", 2.0)
        self.declare_parameter("num_points", 200)
        self.declare_parameter("center_on_start", True)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("start_index", 0)
        self.declare_parameter("yaw_offset", 0.0)

        # Debug visualization (RViz): publish whole trajectory markers
        self.declare_parameter("publish_waypoints_markers", True)
        self.declare_parameter("waypoints_markers_topic", "/experiment/target/trajectory_markers")
        self.declare_parameter("waypoints_frame_id", "world")
        self.declare_parameter("waypoints_line_width", 0.03)
        self.declare_parameter("waypoints_point_scale", 0.08)
        self.declare_parameter("waypoints_alpha", 0.9)
        self.declare_parameter("waypoints_close_loop", False)
        # 0 => publish once (and on init); >0 => republish periodically
        self.declare_parameter("waypoints_publish_period_sec", 0.0)

        # Control
        self.declare_parameter("control_rate_hz", 20.0)
        # Safety cap: prevents accidental busy-looping if control_rate_hz is mis-set very high.
        self.declare_parameter("max_control_rate_hz", 60.0)
        # Safety floor: always sleep at least this long per control iteration.
        self.declare_parameter("min_loop_sleep_sec", 0.005)
        self.declare_parameter("max_v", 0.2)
        self.declare_parameter("max_w", 1.5)
        self.declare_parameter("k_heading", 1.5)
        # Distance gain for forward motion (was previously implicit/constant; now tunable)
        self.declare_parameter("k_dist", 0.8)
        self.declare_parameter("min_v", 0.05)
        self.declare_parameter("stop_distance_m", 0.05)
        self.declare_parameter("pose_match_tol_m", 0.05)
        # Legacy fixed timeout (kept for backwards compatibility)
        self.declare_parameter("pose_match_timeout_sec", 8.0)
        # Adaptive timeout (like pursuer_mover): actual timeout = max(timeout_min_sec, dist0 * timeout_per_meter_sec)
        self.declare_parameter("timeout_min_sec", 8.0)
        self.declare_parameter("timeout_per_meter_sec", 20.0)

        self.declare_parameter("turn_in_place_rad", 1.57)     # ~pi/2
        self.declare_parameter("v_turn_penalty", 1.0)         # v /= (1 + v_turn_penalty*|w|)
        self.declare_parameter("heading_slow_rad", 1.05)      # ~pi/3
        self.declare_parameter("heading_slow_scale", 0.3)

        # Safety
        self.declare_parameter("enable_cmd", False)
        self.declare_parameter("dry_run", True)

        # Debug logging (throttled, like pursuer_mover)
        self.declare_parameter("log_status", True)
        self.declare_parameter("log_period_sec", 1.0)

        # ---- Read params ----
        self.vrpn_pose_topic = str(self.get_parameter("vrpn_pose_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.cmd_frame_id = str(self.get_parameter("cmd_frame_id").value)
        self.cmd_stamp_zero = bool(self.get_parameter("cmd_stamp_zero").value)
        self.step_stamp_topic = str(self.get_parameter("output_step_stamp_topic").value)
        self.step_index_topic = str(self.get_parameter("output_step_index_topic").value)
        self.waypoint_topic = str(self.get_parameter("output_waypoint_topic").value)
        self.srv_name = str(self.get_parameter("service_name").value)
        self.step_stamp_mode = str(self.get_parameter("step_stamp_mode").value).strip()

        self.trajectory_type = str(self.get_parameter("trajectory_type").value)
        self.scale = float(self.get_parameter("trajectory_scale").value)
        self.s_length_x_m = float(self.get_parameter("s_length_x_m").value)
        self.s_width_y_m = float(self.get_parameter("s_width_y_m").value)
        self.num_points = int(self.get_parameter("num_points").value)
        self.center_on_start = bool(self.get_parameter("center_on_start").value)
        self.center_x = float(self.get_parameter("center_x").value)
        self.center_y = float(self.get_parameter("center_y").value)
        self.idx = int(self.get_parameter("start_index").value)
        self.yaw_offset = float(self.get_parameter("yaw_offset").value)

        self.publish_waypoints_markers = bool(self.get_parameter("publish_waypoints_markers").value)
        self.waypoints_markers_topic = str(self.get_parameter("waypoints_markers_topic").value)
        self.waypoints_frame_id = str(self.get_parameter("waypoints_frame_id").value)
        self.waypoints_line_width = float(self.get_parameter("waypoints_line_width").value)
        self.waypoints_point_scale = float(self.get_parameter("waypoints_point_scale").value)
        self.waypoints_alpha = float(self.get_parameter("waypoints_alpha").value)
        self.waypoints_close_loop = bool(self.get_parameter("waypoints_close_loop").value)
        self.waypoints_publish_period_sec = float(self.get_parameter("waypoints_publish_period_sec").value)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_control_rate_hz = float(self.get_parameter("max_control_rate_hz").value)
        self.min_loop_sleep_sec = float(self.get_parameter("min_loop_sleep_sec").value)
        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.k_dist = float(self.get_parameter("k_dist").value)
        self.min_v = float(self.get_parameter("min_v").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.pose_match_tol_m = float(self.get_parameter("pose_match_tol_m").value)
        self.pose_match_timeout_sec = float(self.get_parameter("pose_match_timeout_sec").value)
        self.timeout_min_sec = float(self.get_parameter("timeout_min_sec").value)
        self.timeout_per_meter_sec = float(self.get_parameter("timeout_per_meter_sec").value)
        self.turn_in_place_rad = float(self.get_parameter("turn_in_place_rad").value)
        self.v_turn_penalty = float(self.get_parameter("v_turn_penalty").value)
        self.heading_slow_rad = float(self.get_parameter("heading_slow_rad").value)
        self.heading_slow_scale = float(self.get_parameter("heading_slow_scale").value)

        self.enable_cmd = bool(self.get_parameter("enable_cmd").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.log_status = bool(self.get_parameter("log_status").value)
        self.log_period_sec = float(self.get_parameter("log_period_sec").value)

        if self.trajectory_type not in ("figure8", "s_curve"):
            raise ValueError(
                f"Unsupported trajectory_type='{self.trajectory_type}' (supported: 'figure8', 's_curve')"
            )

        # ---- Path state ----
        self._center_initialized = False
        if self.trajectory_type == "figure8":
            self._raw_path = figure8_points(max(self.num_points, 2), self.scale)
        else:
            self._raw_path = s_curve_points(max(self.num_points, 2), self.s_length_x_m, self.s_width_y_m)
        self.path = [(px + self.center_x, py + self.center_y) for (px, py) in self._raw_path]

        # ---- Robot state ----
        self.current_pose: PoseStamped | None = None

        # ---- Stepper state ----
        self._mtx = threading.Lock()
        self._active = False
        self.step_k = 0
        self._t_log_prev = 0.0
        self._last_step_ok: bool | None = None
        self._last_step_msg: str = ""
        self._last_step_stamp: tuple[int, int] | None = None  # (sec, nsec)

        # ---- ROS I/O ----
        self.sub_pose = self.create_subscription(
            PoseStamped, self.vrpn_pose_topic, self._on_pose, qos_profile_sensor_data, callback_group=self.cb
        )
        self.pub_cmd = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        qos_token = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        )
        self.pub_step_stamp = self.create_publisher(TimeMsg, self.step_stamp_topic, qos_token)
        self.pub_step_index = self.create_publisher(Int32, self.step_index_topic, qos_token)
        self.pub_waypoint = self.create_publisher(PoseStamped, self.waypoint_topic, qos_token)
        self.srv = self.create_service(Trigger, self.srv_name, self._on_advance, callback_group=self.cb)
        # Waypoint markers: latch so RViz can join anytime and still see them (matches simulation behavior)
        qos_markers = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_waypoints = self.create_publisher(MarkerArray, self.waypoints_markers_topic, qos_markers)

        # Optional marker republish
        if self.publish_waypoints_markers and self.waypoints_publish_period_sec > 0.0:
            self.create_timer(float(self.waypoints_publish_period_sec), self._publish_waypoints_markers, callback_group=self.cb)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📡 Pose: {self.vrpn_pose_topic}")
        self.get_logger().info(f"🕹️  cmd_vel: {self.cmd_vel_topic} (TwistStamped)")
        self.get_logger().info(f"🔧 Service: {self.srv_name}")
        self.get_logger().info(f"🟣 Step token: {self.step_stamp_topic} (TimeMsg, latched)")
        self.get_logger().info(f"📢 Step index: {self.step_index_topic} (Int32, latched)")
        self.get_logger().info(f"📌 Waypoint: {self.waypoint_topic} (PoseStamped, latched)")
        self.get_logger().info(f"📈 Trajectory: {self.trajectory_type} scale={self.scale} num_points={self.num_points} start_index={self.idx}")
        self.get_logger().info(f"🧭 Waypoints markers: enable={self.publish_waypoints_markers} topic={self.waypoints_markers_topic}")
        self.get_logger().info(f"🔧 Velocity control: max_v={self.max_v} max_w={self.max_w} k_heading={self.k_heading} k_dist={self.k_dist} min_v={self.min_v} stop_distance_m={self.stop_distance_m} pose_match_tol_m={self.pose_match_tol_m} pose_match_timeout_sec={self.pose_match_timeout_sec} timeout_min_sec={self.timeout_min_sec} timeout_per_meter_sec={self.timeout_per_meter_sec}")
        self.get_logger().info(f"🔧 Turn control: turn_in_place_rad={self.turn_in_place_rad} v_turn_penalty={self.v_turn_penalty} heading_slow_rad={self.heading_slow_rad} heading_slow_scale={self.heading_slow_scale}")
        self.get_logger().info(f"🧯 Safety: enable_cmd={self.enable_cmd} dry_run={self.dry_run}")
        self.get_logger().info(f"🧾 log_status={self.log_status} log_period_sec={self.log_period_sec:.2f}")
        self.get_logger().info("----------------------------------------------------------------")

        # If center_on_start is false, we can publish markers immediately.
        if self.publish_waypoints_markers and (not self.center_on_start):
            self._publish_waypoints_markers()

    def _on_pose(self, msg: PoseStamped):
        self.current_pose = msg

        if self.center_on_start and (not self._center_initialized):
            self.center_x = float(msg.pose.position.x)
            self.center_y = float(msg.pose.position.y)
            self.path = [(px + self.center_x, py + self.center_y) for (px, py) in self._raw_path]

            # start at nearest waypoint to avoid an initial "jump"
            best_i = 0
            best_d2 = float("inf")
            for i, (px, py) in enumerate(self.path):
                dx = px - self.center_x
                dy = py - self.center_y
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            self.idx = best_i
            self._center_initialized = True
            self.get_logger().info(f"Initialized center at ({self.center_x:.3f}, {self.center_y:.3f}); idx={self.idx}")
            if self.publish_waypoints_markers:
                self._publish_waypoints_markers()

    def _publish_waypoints_markers(self):
        if not self.publish_waypoints_markers:
            return
        if self.center_on_start and (not self._center_initialized):
            return  # no stable path yet

        pts = list(self.path)
        if self.waypoints_close_loop and len(pts) >= 2:
            pts = pts + [pts[0]]

        stamp = self.get_clock().now().to_msg()

        mk_line = Marker()
        mk_line.header.stamp = stamp
        mk_line.header.frame_id = self.waypoints_frame_id
        mk_line.ns = "target_waypoints"
        mk_line.id = 0
        mk_line.type = Marker.LINE_STRIP
        mk_line.action = Marker.ADD
        mk_line.scale.x = float(self.waypoints_line_width)
        mk_line.color.r = 0.1
        mk_line.color.g = 0.9
        mk_line.color.b = 0.2
        mk_line.color.a = float(self.waypoints_alpha)
        mk_line.pose.orientation.w = 1.0

        mk_pts = Marker()
        mk_pts.header.stamp = stamp
        mk_pts.header.frame_id = self.waypoints_frame_id
        mk_pts.ns = "target_waypoints"
        mk_pts.id = 1
        mk_pts.type = Marker.POINTS
        mk_pts.action = Marker.ADD
        mk_pts.scale.x = float(self.waypoints_point_scale)
        mk_pts.scale.y = float(self.waypoints_point_scale)
        mk_pts.color.r = 0.1
        mk_pts.color.g = 0.6
        mk_pts.color.b = 1.0
        mk_pts.color.a = float(self.waypoints_alpha)
        mk_pts.pose.orientation.w = 1.0

        for (x, y) in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            mk_line.points.append(p)
            mk_pts.points.append(p)

        arr = MarkerArray()
        arr.markers = [mk_line, mk_pts]
        self.pub_waypoints.publish(arr)

    def _stop(self):
        cmd = TwistStamped()
        cmd.header.stamp = TimeMsg() if bool(self.cmd_stamp_zero) else self.get_clock().now().to_msg()
        cmd.header.frame_id = str(self.cmd_frame_id)
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        if self.enable_cmd and (not self.dry_run):
            self.pub_cmd.publish(cmd)

    def _publish_token(self, stamp):
        # Step token is a TimeMsg (used by predictor + planner gating).
        tok = TimeMsg()
        if self.step_stamp_mode == "step_index":
            tok.sec = int(self.step_k)
            tok.nanosec = 0
        else:
            tok.sec = int(stamp.sec)
            tok.nanosec = int(stamp.nanosec)
        self.pub_step_stamp.publish(tok)
        self._last_step_stamp = (int(tok.sec), int(tok.nanosec))

        kmsg = Int32()
        kmsg.data = int(self.step_k)
        self.pub_step_index.publish(kmsg)

    def _publish_waypoint(self, *, x: float, y: float):
        ps = PoseStamped()
        ps.header.frame_id = str(self.waypoints_frame_id)
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        self.pub_waypoint.publish(ps)

    def _log_throttled(self, msg: str):
        if not bool(self.get_parameter("log_status").value):
            return
        period = float(self.get_parameter("log_period_sec").value)
        period = max(0.05, period)
        now = time.monotonic()
        if (now - float(getattr(self, "_t_log_prev", 0.0))) >= period:
            self._t_log_prev = now
            self.get_logger().info(msg)

    def _on_advance(self, req, resp):
        # Called by orchestrator/CLI to move to the next waypoint.
        # Pursuer-style semantics: publish cmd_vel continuously INSIDE this service call
        # until reached/timeout, then stop and return.
        if self.current_pose is None:
            resp.success = False
            resp.message = f"No pose received yet (waiting for VRPN pose on {self.vrpn_pose_topic})"
            return resp

        if self.center_on_start and (not self._center_initialized):
            resp.success = False
            resp.message = "Center not initialized yet (waiting for first pose)"
            return resp

        with self._mtx:
            if self._active:
                resp.success = False
                resp.message = "Previous step still in progress"
                return resp
            self._active = True

        try:
            # Reset completion state for this step.
            self._last_step_ok = None
            self._last_step_msg = ""
            self._last_step_stamp = None

            k0 = int(self.step_k)
            idx0 = int(self.idx) % max(self.num_points, 1)
            tx, ty = self.path[idx0]
            tx, ty = float(tx), float(ty)

            # advance for next call immediately (like before)
            self.idx = (idx0 + 1) % max(self.num_points, 1)

            # Adaptive timeout based on current distance to target.
            p0 = self.current_pose.pose.position
            dist0 = math.hypot(float(tx) - float(p0.x), float(ty) - float(p0.y))
            t_adapt = max(float(self.timeout_min_sec), float(dist0) * float(self.timeout_per_meter_sec))
            # Respect legacy fixed timeout if user set it larger.
            t_fixed = float(self.pose_match_timeout_sec)
            timeout = max(t_adapt, t_fixed)
            deadline = time.time() + float(timeout)

            # Guard against misconfiguration: clamp Hz and enforce a minimum sleep.
            hz = float(self.get_parameter("control_rate_hz").value)
            hz_cap = max(1.0, float(self.max_control_rate_hz))
            hz = min(max(0.1, hz), hz_cap)
            sleep_sec = max(1.0 / hz, max(0.0, float(self.min_loop_sleep_sec)))
            last_dist = float("inf")

            while time.time() < deadline:
                if self.current_pose is None:
                    break

                p = self.current_pose.pose.position
                x, y = float(p.x), float(p.y)
                o = self.current_pose.pose.orientation
                yaw = yaw_from_quat((o.x, o.y, o.z, o.w)) + float(self.yaw_offset)

                dx = tx - x
                dy = ty - y
                dist = math.hypot(dx, dy)
                last_dist = float(dist)

                # arrival check
                if dist <= float(self.pose_match_tol_m):
                    self._stop()
                    self._publish_token(self.current_pose.header.stamp)
                    self._publish_waypoint(x=tx, y=ty)
                    self._last_step_ok = True
                    self._last_step_msg = f"✅ Reached waypoint dist={dist:.3f}"
                    self.get_logger().info(f"{self._last_step_msg} (k={self.step_k})")
                    self.step_k += 1
                    ok = True
                    msg = self._last_step_msg
                    stamp = self._last_step_stamp
                    stamp_str = f"{stamp[0]}.{stamp[1]:09d}" if stamp is not None else "None"
                    resp.success = ok
                    resp.message = f"{msg} (k={k0} status=ARRIVED step_stamp={stamp_str})"
                    return resp

                # heading controller
                target_heading = math.atan2(dy, dx)
                heading_error = math.atan2(math.sin(target_heading - yaw), math.cos(target_heading - yaw))

                w = float(self.k_heading) * heading_error
                w = max(-float(self.max_w), min(float(self.max_w), w))

                # forward speed: P on distance, capped
                v = min(float(self.max_v), float(self.k_dist) * float(dist))
                v = max(0.0, float(v))

                # reduce v while turning (helps traction / reduces sideways slip)
                pen = max(0.0, float(self.v_turn_penalty))
                v = float(v) / (1.0 + pen * abs(float(w)))

                # No rotate-in-place mode: always allow forward motion (subject to v_turn_penalty).
                if abs(heading_error) > max(0.0, float(self.heading_slow_rad)):
                    v *= max(0.0, min(1.0, float(self.heading_slow_scale)))

                # taper near goal, but avoid stalling too early
                if dist < float(self.stop_distance_m):
                    v = 0.0
                    w = 0.0
                else:
                    v = max(float(self.min_v), v) if v > 0.0 else 0.0

                cmd = TwistStamped()
                cmd.header.stamp = TimeMsg() if bool(self.cmd_stamp_zero) else self.get_clock().now().to_msg()
                cmd.header.frame_id = str(self.cmd_frame_id)
                cmd.twist.linear.x = float(v)
                cmd.twist.angular.z = float(w)

                self._log_throttled(
                    f"[move] dist={dist:.3f} heading_err={heading_error:.3f} v={v:.3f} w={w:.3f} "
                    f"pos=({x:.2f},{y:.2f}) goal=({tx:.2f},{ty:.2f}) "
                    f"enable_cmd={bool(self.enable_cmd)} dry_run={bool(self.dry_run)}"
                )

                if (not self.dry_run) and bool(self.enable_cmd):
                    self.pub_cmd.publish(cmd)

                time.sleep(sleep_sec)

            # timeout / failure
            self._stop()
            if self.current_pose is not None:
                self._publish_token(self.current_pose.header.stamp)
                self._publish_waypoint(x=tx, y=ty)
            self._last_step_ok = False
            self._last_step_msg = f"❌ Timeout reaching waypoint (dist={last_dist:.3f})"
            self.get_logger().warn(f"{self._last_step_msg}. Advancing anyway. (k={self.step_k})")
            self.step_k += 1
            stamp = self._last_step_stamp
            stamp_str = f"{stamp[0]}.{stamp[1]:09d}" if stamp is not None else "None"
            resp.success = False
            resp.message = f"{self._last_step_msg} (k={k0} status=TIMEOUT step_stamp={stamp_str})"
            return resp
        finally:
            with self._mtx:
                self._active = False

    # No timer-based control loop: publishing occurs inside the service call (pursuer-style).


def main():
    rclpy.init()
    node = TargetStepper()
    try:
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(node)
        ex.spin()
    except KeyboardInterrupt:
        pass
    try:
        node._stop()
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
