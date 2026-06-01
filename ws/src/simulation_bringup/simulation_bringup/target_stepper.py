#!/usr/bin/env python3
import math
import numpy as np
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Standard ROS messages
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Int32
from builtin_interfaces.msg import Time as TimeMsg
from visualization_msgs.msg import Marker, MarkerArray

# GZ Interfaces (Make sure ros_gz_interfaces is installed!)
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from geometry_msgs.msg import Point

from mua_nbv_py_utils.transforms import quat_from_yaw
from mua_nbv_common.ros_helpers import stamp_str


def figure8_points(n: int, scale: float):
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * float(i) / float(n)
        x = scale * math.sin(t)
        y = scale * math.sin(t) * math.cos(t)
        pts.append((x, y))
    return pts


def circle_points(n: int, radius: float):
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * float(i) / float(n)
        x = radius * math.cos(t)
        y = radius * math.sin(t)
        pts.append((x, y))
    return pts


def line_points(n: int, half_length: float):
    n = max(int(n), 2)
    xs = np.linspace(-float(half_length), float(half_length), n, dtype=float)
    forward = [(float(x), 0.0) for x in xs]
    if len(forward) <= 2:
        return forward
    back = list(reversed(forward[1:-1]))
    return forward + back


def wiggle_points(n: int, scale: float):
    """Sinusoidal wave with 2 full cycles, forward + backward (closed path).

    X-extent is 2x figure-8 (±2*scale).  Y amplitude matches figure-8 (±scale/2).
    Every segment has curvature, so the CV model consistently struggles.
    """
    n_fwd = max(n // 2, 2)
    half_x = 2.0 * scale
    amp_y = scale * 0.5
    fwd = []
    for i in range(n_fwd):
        frac = float(i) / float(n_fwd - 1)
        x = half_x * (2.0 * frac - 1.0)
        y = amp_y * math.sin(4.0 * math.pi * frac)
        fwd.append((x, y))
    bwd = list(reversed(fwd[1:-1]))
    return fwd + bwd

def cv_random_points(n: int, scale: float, seed: int = 42, dt: float = 1.0,
                     q_c: float = 0.1, v0_mag: float = 0.5):
    """Generate a trajectory from the Constant-Velocity stochastic process.

    The GP smoother with matching q_c is calibrated by construction.

    Parameters
    ----------
    n : int
        Number of waypoints.
    scale : float
        Approximate spatial extent — initial velocity magnitude is ``v0_mag``
        and the trajectory is clipped to ±2*scale from the origin.
    seed : int
        RNG seed for reproducibility across runs.
    dt : float
        Time step between waypoints (should match dt_pred).
    q_c : float
        Process-noise spectral density (same as the GP smoother).
    v0_mag : float
        Initial speed (m/s).
    """
    rng = np.random.default_rng(seed)
    # Initial heading ~ uniform
    theta0 = rng.uniform(-math.pi, math.pi)
    vx, vy = v0_mag * math.cos(theta0), v0_mag * math.sin(theta0)
    x, y = 0.0, 0.0

    sigma_a = math.sqrt(q_c * dt)
    pts = [(x, y)]
    for _ in range(n - 1):
        ax = rng.normal(0.0, sigma_a)
        ay = rng.normal(0.0, sigma_a)
        vx += ax
        vy += ay
        x += vx * dt
        y += vy * dt
        # Soft reflection to keep trajectory bounded
        bound = 2.0 * scale
        if abs(x) > bound:
            vx = -vx * 0.5
            x = max(-bound, min(bound, x))
        if abs(y) > bound:
            vy = -vy * 0.5
            y = max(-bound, min(bound, y))
        pts.append((x, y))
    return pts


def dist2_pose_xy(a: Pose, b: Pose) -> float:
    dx = float(a.position.x) - float(b.position.x)
    dy = float(a.position.y) - float(b.position.y)
    return dx * dx + dy * dy


class TargetStepper(Node):
    def __init__(self):
        super().__init__("target_stepper")

        # ---- Params ----
        # Frames
        self.declare_parameter("world_name", "world")
        self.declare_parameter("entity_name", "target")
        # Topics
        self.declare_parameter("input_target_pose_topic", "/sim/target/pose")
        self.declare_parameter("output_step_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("output_step_index_topic", "/experiment/step")
        # Service
        self.declare_parameter("service_name", "/experiment/target/advance") 
        # Path
        self.declare_parameter("trajectory_type", "cv_random")
        self.declare_parameter("trajectory_scale", 3.0)
        self.declare_parameter("num_points", 100)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("z", 0.01) # Slightly up to avoid clipping
        self.declare_parameter("yaw_offset", 0.0)
        self.declare_parameter("start_index", 20)
        # Tolerances
        self.declare_parameter("pose_match_tol_m", 0.005) # Loosened slightly for reliability
        self.declare_parameter("pose_match_timeout_sec", 2.0)
        self.declare_parameter("skip_pose_confirmation", False)
        # cv_random trajectory params
        self.declare_parameter("cv_random_seed", 42)
        self.declare_parameter("cv_random_q_c", 0.1)
        self.declare_parameter("cv_random_v0", 0.5)
        self.declare_parameter("cv_random_dt", 1.0)

        # Debug visualization (RViz): publish the full waypoint trajectory
        self.declare_parameter("publish_waypoints_markers", True)
        self.declare_parameter("waypoints_markers_topic", "/experiment/target/trajectory_markers")
        self.declare_parameter("waypoints_frame_id", "sim_world")
        self.declare_parameter("waypoints_line_width", 0.03)
        self.declare_parameter("waypoints_point_scale", 0.08)
        self.declare_parameter("waypoints_alpha", 0.9)
        self.declare_parameter("waypoints_close_loop", True)
        self.declare_parameter("waypoints_publish_period_sec", 0.0)  # 0 => publish once on startup

        # ---- Read params ----
        self.world_name = str(self.get_parameter("world_name").value)
        self.entity_name = str(self.get_parameter("entity_name").value)
        self.target_pose_topic = str(self.get_parameter("input_target_pose_topic").value)
        self.step_stamp_topic = str(self.get_parameter("output_step_stamp_topic").value)
        self.step_index_topic = str(self.get_parameter("output_step_index_topic").value)
        self.srv_name = str(self.get_parameter("service_name").value)
        self.trajectory_type = str(self.get_parameter("trajectory_type").value)
        self.scale = float(self.get_parameter("trajectory_scale").value)
        self.num_points = int(self.get_parameter("num_points").value)
        self.center_x = float(self.get_parameter("center_x").value)
        self.center_y = float(self.get_parameter("center_y").value)
        self.z = float(self.get_parameter("z").value)
        self.yaw_offset = float(self.get_parameter("yaw_offset").value)
        self.idx = int(self.get_parameter("start_index").value)
        self.pose_match_tol_m = float(self.get_parameter("pose_match_tol_m").value)
        self.pose_match_timeout_sec = float(self.get_parameter("pose_match_timeout_sec").value)
        self.skip_pose_confirmation = bool(self.get_parameter("skip_pose_confirmation").value)

        self.publish_waypoints_markers = bool(self.get_parameter("publish_waypoints_markers").value)
        self.waypoints_markers_topic = str(self.get_parameter("waypoints_markers_topic").value)
        self.waypoints_frame_id = str(self.get_parameter("waypoints_frame_id").value)
        self.waypoints_line_width = float(self.get_parameter("waypoints_line_width").value)
        self.waypoints_point_scale = float(self.get_parameter("waypoints_point_scale").value)
        self.waypoints_alpha = float(self.get_parameter("waypoints_alpha").value)
        self.waypoints_close_loop = bool(self.get_parameter("waypoints_close_loop").value)
        self.waypoints_publish_period_sec = float(self.get_parameter("waypoints_publish_period_sec").value)

        # ---- Generate path ----
        if self.trajectory_type == "figure8":
            base_pts = figure8_points(max(self.num_points, 2), self.scale)
        elif self.trajectory_type == "circle":
            base_pts = circle_points(max(self.num_points, 3), self.scale)
        elif self.trajectory_type == "line":
            base_pts = line_points(max(self.num_points, 2), self.scale)
        elif self.trajectory_type == "wiggle":
            base_pts = wiggle_points(max(self.num_points, 4), self.scale)
        elif self.trajectory_type == "cv_random":
            base_pts = cv_random_points(
                max(self.num_points, 2), self.scale,
                seed=int(self.get_parameter("cv_random_seed").value),
                q_c=float(self.get_parameter("cv_random_q_c").value),
                v0_mag=float(self.get_parameter("cv_random_v0").value),
                dt=float(self.get_parameter("cv_random_dt").value),
            )
        else:
            raise ValueError(
                f"Invalid trajectory type: {self.trajectory_type} "
                f"(expected 'figure8'|'circle'|'line'|'wiggle'|'cv_random')"
            )
        self.pts = [(px + self.center_x, py + self.center_y) for (px, py) in base_pts]
        if self.trajectory_type == "cv_random":
            self._init_cv_random_online_state()
            # Keep online generator state synchronized with pre-generated seed path.
            self.pts = [(self.center_x, self.center_y)]
            self._extend_cv_random_points(max(self.num_points, 2))

        # ---- State vars ----
        self._mtx = threading.Lock()
        self._pending_pose = None         
        self._pending_idx = None           
        self._pending_deadline_wall = None
        self._in_flight = False
        self.step_k = 0
        
        # ---- Clients / Subscribers / Publishers ----
        self.pub_waypoints = None
        self._gz_service_name = f"/world/{self.world_name}/set_pose"
        self._client = self.create_client(SetEntityPose, self._gz_service_name)
        qos_in = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, self.target_pose_topic, self._on_target_pose, qos_in)
        qos_token = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_step_stamp = self.create_publisher(TimeMsg, self.step_stamp_topic, qos_token)
        self.pub_step_index = self.create_publisher(Int32, self.step_index_topic, qos_token)
        self.srv = self.create_service(Trigger, self.srv_name, self._on_advance)
        self.create_timer(0.25, self._check_pending_timeout)

        # Waypoint markers: latch so RViz can join anytime and still see the trajectory
        if self.publish_waypoints_markers:
            qos_markers = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.pub_waypoints = self.create_publisher(MarkerArray, self.waypoints_markers_topic, qos_markers)

            # Publish once immediately, and optionally republish periodically
            self._publish_waypoints_markers()
            if self.waypoints_publish_period_sec > 0.0:
                self.create_timer(self.waypoints_publish_period_sec, self._publish_waypoints_markers)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"🌐 World: {self.world_name}, Entity: {self.entity_name}")
        self.get_logger().info(f"📡 Listening to: {self.target_pose_topic}")
        self.get_logger().info(f"📢 Publishing to: {self.step_stamp_topic}, {self.step_index_topic}")
        self.get_logger().info(f"🔧 Service: {self.srv_name}")
        self.get_logger().info(f"📊 Trajectory Type: {self.trajectory_type}, Scale: {self.scale}, Num Points: {self.num_points}")
        self.get_logger().info(f"📊 Center: ({self.center_x}, {self.center_y}), Z: {self.z}")
        self.get_logger().info(f"📊 Yaw Offset: {self.yaw_offset}, Start Index: {self.idx}")
        self.get_logger().info(f"📊 Pose Match Tol: {self.pose_match_tol_m}m, Timeout: {self.pose_match_timeout_sec}s")
        if self.publish_waypoints_markers:
            self.get_logger().info(
                f"🧭 Waypoints markers: topic={self.waypoints_markers_topic} frame={self.waypoints_frame_id} "
                f"close_loop={self.waypoints_close_loop} period={self.waypoints_publish_period_sec}"
            )
        self.get_logger().info("----------------------------------------------------------------")

    def _init_cv_random_online_state(self):
        """Initialize online CV-random generator state from current trajectory params."""
        self._cv_rng = np.random.default_rng(int(self.get_parameter("cv_random_seed").value))
        self._cv_dt = float(self.get_parameter("cv_random_dt").value)
        self._cv_qc = float(self.get_parameter("cv_random_q_c").value)
        self._cv_bound = 2.0 * float(self.scale)

        theta0 = self._cv_rng.uniform(-math.pi, math.pi)
        v0_mag = float(self.get_parameter("cv_random_v0").value)
        self._cv_vx = v0_mag * math.cos(theta0)
        self._cv_vy = v0_mag * math.sin(theta0)
        self._cv_x = 0.0
        self._cv_y = 0.0

    def _extend_cv_random_points(self, min_count: int):
        """Extend self.pts for cv_random trajectory so len(self.pts) >= min_count."""
        if self.trajectory_type != "cv_random":
            return
        if not hasattr(self, "_cv_rng"):
            self._init_cv_random_online_state()

        sigma_a = math.sqrt(max(1e-9, self._cv_qc * self._cv_dt))
        while len(self.pts) < int(min_count):
            ax = self._cv_rng.normal(0.0, sigma_a)
            ay = self._cv_rng.normal(0.0, sigma_a)
            self._cv_vx += ax
            self._cv_vy += ay
            self._cv_x += self._cv_vx * self._cv_dt
            self._cv_y += self._cv_vy * self._cv_dt

            if abs(self._cv_x) > self._cv_bound:
                self._cv_vx = -self._cv_vx * 0.5
                self._cv_x = max(-self._cv_bound, min(self._cv_bound, self._cv_x))
            if abs(self._cv_y) > self._cv_bound:
                self._cv_vy = -self._cv_vy * 0.5
                self._cv_y = max(-self._cv_bound, min(self._cv_bound, self._cv_y))

            self.pts.append((self._cv_x + self.center_x, self._cv_y + self.center_y))

    def _publish_waypoints_markers(self):
        if not self.publish_waypoints_markers:
            return

        now = self.get_clock().now().to_msg()

        # Build list of points (optionally close the loop)
        pts = list(self.pts)
        if self.waypoints_close_loop and len(pts) >= 2:
            pts = pts + [pts[0]]

        # LINE_STRIP for trajectory curve
        line = Marker()
        line.header.frame_id = self.waypoints_frame_id
        line.header.stamp = now
        line.ns = "target_trajectory"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = max(1e-4, float(self.waypoints_line_width))
        line.color.r = 0.1
        line.color.g = 0.1
        line.color.b = 1.0
        line.color.a = float(max(0.0, min(1.0, self.waypoints_alpha)))
        for (x, y) in pts:
            line.points.append(Point(x=float(x), y=float(y), z=float(self.z)))

        # SPHERE_LIST for waypoint vertices
        pts_m = Marker()
        pts_m.header.frame_id = self.waypoints_frame_id
        pts_m.header.stamp = now
        pts_m.ns = "target_trajectory"
        pts_m.id = 1
        pts_m.type = Marker.SPHERE_LIST
        pts_m.action = Marker.ADD
        s = max(1e-4, float(self.waypoints_point_scale))
        pts_m.scale.x = s
        pts_m.scale.y = s
        pts_m.scale.z = s
        pts_m.color.r = 1.0
        pts_m.color.g = 0.2
        pts_m.color.b = 0.2
        pts_m.color.a = float(max(0.0, min(1.0, self.waypoints_alpha)))
        for (x, y) in self.pts:
            pts_m.points.append(Point(x=float(x), y=float(y), z=float(self.z)))

        ma = MarkerArray()
        ma.markers.append(line)
        ma.markers.append(pts_m)

        self.pub_waypoints.publish(ma)

    def _on_advance(self, req, resp):
        """Called by the Orchestrator/CLI to move the target 1 step."""
        
        # 1. Check if can talk to Gazebo
        if not self._client.service_is_ready():
            if not self._client.wait_for_service(timeout_sec=1.0):
                resp.success = False
                resp.message = "Gazebo /set_pose service not ready"
                return resp

        # 2. Check if busy
        with self._mtx:
            if self._in_flight or self._pending_pose is not None:
                resp.success = False
                resp.message = "Previous step still in progress (waiting for pose confirmation)"
                return resp
            self._in_flight = True
            
        # 3. Calculate next pose
        if self.trajectory_type == "cv_random":
            cur_idx = max(0, int(self.idx))
            # Ensure a valid look-ahead segment exists; grow trajectory on demand.
            self._extend_cv_random_points(cur_idx + 2)
            x0, y0 = self.pts[cur_idx]
            x1, y1 = self.pts[cur_idx + 1]
        else:
            n = len(self.pts)
            cur_idx = self.idx % n
            x0, y0 = self.pts[cur_idx]
            x1, y1 = self.pts[(cur_idx + 1) % n]

        yaw = math.atan2(y1 - y0, x1 - x0) + self.yaw_offset
        qx, qy, qz, qw = quat_from_yaw(yaw)

        pose = Pose()
        pose.position.x = float(x0)
        pose.position.y = float(y0)
        pose.position.z = self.z
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw

        # 4. Construct Request
        req_gz = SetEntityPose.Request()
        req_gz.entity.name = self.entity_name
        req_gz.entity.type = Entity.MODEL
        req_gz.pose = pose

        if self.trajectory_type == "cv_random":
            self.idx = cur_idx + 1
            # Keep RViz trajectory markers representative of the generated path.
            if self.publish_waypoints_markers and self.waypoints_publish_period_sec <= 0.0:
                self._publish_waypoints_markers()
        else:
            self.idx = (cur_idx + 1) % n

        # 5. Send to Gazebo Async
        fut = self._client.call_async(req_gz)
        fut.add_done_callback(lambda f: self._on_set_pose_done(f, pose, cur_idx))

        resp.success = True
        resp.message = f"✅ Step initiated (idx={cur_idx}). Waiting for physical arrival..."
        return resp

    def _on_set_pose_done(self, fut, pose, idx):
        with self._mtx:
            self._in_flight = False
        
        try:
            res = fut.result()
            if not res.success:
                self.get_logger().warn(f"Gazebo rejected set_pose! Success=False")
                return 
        except Exception as e:
            self.get_logger().error(f"Gazebo service call failed: {e}")
            return

        if self.skip_pose_confirmation:
            now = self.get_clock().now().to_msg()
            with self._mtx:
                self._confirm_step(now, success=True)
            return

        # Arm the confirmation logic
        with self._mtx:
            self._pending_pose = pose
            self._pending_idx = int(idx)
            self._pending_deadline_wall = time.time() + self.pose_match_timeout_sec

    def _on_target_pose(self, msg: PoseStamped):
        """Checks if the robot has actually arrived at the requested pose."""
        with self._mtx:
            if self._pending_pose is None:
                return # We aren't waiting for anything

            # Check timeout
            if time.time() > self._pending_deadline_wall:
                self.get_logger().warn("Timeout waiting for target to physically reach pose. Proceeding anyway.")
                self._confirm_step(msg.header.stamp, success=False)
                return

            # Check distance
            if dist2_pose_xy(msg.pose, self._pending_pose) <= (self.pose_match_tol_m**2): 
                self._confirm_step(msg.header.stamp, success=True)

    def _check_pending_timeout(self):
        """Timer-based fallback: force-confirm if the subscription-based check missed the deadline."""
        with self._mtx:
            if self._pending_pose is None or self._pending_deadline_wall is None:
                return
            if time.time() <= self._pending_deadline_wall:
                return
            self.get_logger().warn("Timer-based timeout: force-confirming pending step.")
            now = self.get_clock().now().to_msg()
            self._confirm_step(now, success=False)

    def _confirm_step(self, stamp, success):
        """Publishes the synchronization token so other nodes know we are ready."""

        # Clear state BEFORE publishing token to avoid race with fast callers
        self._pending_pose = None
        self._pending_idx = None
        self.step_k += 1
        
        s = TimeMsg()
        s.sec = int(stamp.sec)
        s.nanosec = int(stamp.nanosec)
        self.pub_step_stamp.publish(s)
        self.get_logger().info(f"🟣 TOKEN publish: step_stamp={stamp_str(s)} (k={self.step_k-1}, success={success})")

        kmsg = Int32()
        kmsg.data = int(self.step_k - 1)
        self.pub_step_index.publish(kmsg)
        
        status = "ARRIVED" if success else "TIMEOUT"
        self.get_logger().info(f"✅ Target Step {self.step_k-1} Complete: {status}.")


def main():
    rclpy.init()
    node = TargetStepper()
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
