#!/usr/bin/env python3
"""
Coordinated trajectory follower for target and pursuer with trigger-based synchronization.

Protocol:
1. Target moves through waypoints 0-4 alone (pursuer stays still)
2. After waypoint 4, pursuer can start moving
3. If pursuer arrives at waypoint 4+n before target, pursuer waits
4. Once target arrives at waypoint 4+n, pursuer can proceed

Usage:
  ros2 run testbed_bringup coordinated_trajectory_follower \
    --ros-args \
    -p waypoint_csv:=/path/to/waypoints.csv
"""

from __future__ import annotations

import csv
import math
import signal
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from builtin_interfaces.msg import Time as TimeMsg
from visualization_msgs.msg import Marker, MarkerArray

from mua_nbv_py_utils.transforms import yaw_from_quat, quat_from_yaw


class CoordinatedTrajectoryFollower(Node):
    """
    Coordinated trajectory follower for target and pursuer.
    
    Implements trigger-based synchronization where pursuer waits for target
    at each waypoint after waypoint 4.
    """

    def __init__(self):
        super().__init__("coordinated_trajectory_follower")
        self.cb = ReentrantCallbackGroup()

        # ---- Parameters ----
        self.declare_parameter("waypoint_csv", "")
        self.declare_parameter("target_vrpn_pose_topic", "/vrpn_mocap/target/pose")
        self.declare_parameter("pursuer_vrpn_pose_topic", "/vrpn_mocap/pursuer/pose")
        self.declare_parameter("target_cmd_vel_topic", "/target/cmd_vel")
        self.declare_parameter("pursuer_cmd_vel_topic", "/pursuer/cmd_vel")
        self.declare_parameter("cmd_frame_id", "base_link")
        self.declare_parameter("cmd_stamp_zero", True)

        # Control parameters
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("max_control_rate_hz", 60.0)
        self.declare_parameter("target_max_v", 0.045)  # ~1/3 of pursuer
        self.declare_parameter("pursuer_max_v", 0.20)
        self.declare_parameter("max_w", 0.6)
        self.declare_parameter("k_heading", 1.0)
        self.declare_parameter("k_dist", 0.8)
        self.declare_parameter("min_v", 0.02)
        self.declare_parameter("stop_distance_m", 0.1)
        self.declare_parameter("v_turn_penalty", 0.5)
        self.declare_parameter("heading_slow_rad", 1.2)
        self.declare_parameter("heading_slow_scale", 0.4)

        # Waypoint following
        self.declare_parameter("waypoint_tolerance_m", 0.1)  
        self.declare_parameter("solo_phase_end_waypoint", 4)  # Waypoint 0-4 are solo

        # Waypoint visualization
        self.declare_parameter("publish_waypoints_markers", True)
        self.declare_parameter("waypoints_markers_topic", "/experiment/coordinated_trajectory/waypoints")
        self.declare_parameter("waypoints_frame_id", "world")
        self.declare_parameter("waypoints_line_width", 0.03)
        self.declare_parameter("waypoints_point_scale", 0.08)
        self.declare_parameter("waypoints_alpha", 0.9)
        self.declare_parameter("waypoints_publish_period_sec", 1.0)  # Republish periodically for RViz

        # Debug: publish pursuer pose at waypoints
        self.declare_parameter("pursuer_waypoint_pose_topic", "/experiment/pursuer/waypoint_pose")

        # Pursuer 2-stage control (position then orientation)
        self.declare_parameter("pursuer_align_goal_yaw", True)
        self.declare_parameter("pursuer_yaw_match_tol_rad", 0.25)
        self.declare_parameter("pursuer_k_heading_align", 1.0)
        self.declare_parameter("pursuer_max_w_align", 0.8)
        self.declare_parameter("pursuer_align_hold_factor", 1.5)

        # Safety
        self.declare_parameter("enable_cmd", True)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("log_status", True)
        self.declare_parameter("log_period_sec", 1.0)

        # ---- Read parameters ----
        waypoint_csv = str(self.get_parameter("waypoint_csv").value)
        if not waypoint_csv:
            raise ValueError("waypoint_csv parameter must be set")
        self.waypoint_csv = Path(waypoint_csv)
        if not self.waypoint_csv.is_file():
            raise FileNotFoundError(f"Waypoint CSV not found: {self.waypoint_csv}")

        self.target_vrpn_pose_topic = str(self.get_parameter("target_vrpn_pose_topic").value)
        self.pursuer_vrpn_pose_topic = str(self.get_parameter("pursuer_vrpn_pose_topic").value)
        self.target_cmd_vel_topic = str(self.get_parameter("target_cmd_vel_topic").value)
        self.pursuer_cmd_vel_topic = str(self.get_parameter("pursuer_cmd_vel_topic").value)
        self.cmd_frame_id = str(self.get_parameter("cmd_frame_id").value)
        self.cmd_stamp_zero = bool(self.get_parameter("cmd_stamp_zero").value)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_control_rate_hz = float(self.get_parameter("max_control_rate_hz").value)
        self.target_max_v = float(self.get_parameter("target_max_v").value)
        self.pursuer_max_v = float(self.get_parameter("pursuer_max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.k_dist = float(self.get_parameter("k_dist").value)
        self.min_v = float(self.get_parameter("min_v").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.v_turn_penalty = float(self.get_parameter("v_turn_penalty").value)
        self.heading_slow_rad = float(self.get_parameter("heading_slow_rad").value)
        self.heading_slow_scale = float(self.get_parameter("heading_slow_scale").value)

        self.waypoint_tolerance_m = float(self.get_parameter("waypoint_tolerance_m").value)
        self.solo_phase_end_waypoint = int(self.get_parameter("solo_phase_end_waypoint").value)

        self.publish_waypoints_markers = bool(self.get_parameter("publish_waypoints_markers").value)
        self.waypoints_markers_topic = str(self.get_parameter("waypoints_markers_topic").value)
        self.waypoints_frame_id = str(self.get_parameter("waypoints_frame_id").value)
        self.waypoints_line_width = float(self.get_parameter("waypoints_line_width").value)
        self.waypoints_point_scale = float(self.get_parameter("waypoints_point_scale").value)
        self.waypoints_alpha = float(self.get_parameter("waypoints_alpha").value)
        self.waypoints_publish_period_sec = float(self.get_parameter("waypoints_publish_period_sec").value)

        self.pursuer_waypoint_pose_topic = str(self.get_parameter("pursuer_waypoint_pose_topic").value)

        self.pursuer_align_goal_yaw = bool(self.get_parameter("pursuer_align_goal_yaw").value)
        self.pursuer_yaw_match_tol_rad = float(self.get_parameter("pursuer_yaw_match_tol_rad").value)
        self.pursuer_k_heading_align = float(self.get_parameter("pursuer_k_heading_align").value)
        self.pursuer_max_w_align = float(self.get_parameter("pursuer_max_w_align").value)
        self.pursuer_align_hold_factor = float(self.get_parameter("pursuer_align_hold_factor").value)

        self.enable_cmd = bool(self.get_parameter("enable_cmd").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.log_status = bool(self.get_parameter("log_status").value)
        self.log_period_sec = float(self.get_parameter("log_period_sec").value)

        # ---- Load waypoints ----
        self.waypoints = self._load_waypoints()
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints from {self.waypoint_csv}")

        # ---- State ----
        self._mtx = threading.Lock()
        self.target_pose: Optional[PoseStamped] = None
        self.pursuer_pose: Optional[PoseStamped] = None
        
        # Waypoint tracking
        self.target_waypoint_idx = 0
        self.pursuer_waypoint_idx = 0
        self.target_waypoint_reached = [False] * len(self.waypoints)
        self.pursuer_waypoint_reached = [False] * len(self.waypoints)
        
        # Pursuer 2-stage control state
        self.pursuer_in_align_mode = False
        self.pursuer_waypoint_pose_published = [False] * len(self.waypoints)  # Track published poses
        self.pursuer_alignment_entered = [False] * len(self.waypoints)  # Track if alignment was entered for each waypoint
        self.pursuer_prev_target_waypoint = -1  # Track previous waypoint to detect changes
        
        self._t_log_prev = 0.0

        # ---- ROS I/O ----
        qos_profile_sensor_data = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.sub_target_pose = self.create_subscription(
            PoseStamped, self.target_vrpn_pose_topic, self._on_target_pose, qos_profile_sensor_data, callback_group=self.cb
        )
        self.sub_pursuer_pose = self.create_subscription(
            PoseStamped, self.pursuer_vrpn_pose_topic, self._on_pursuer_pose, qos_profile_sensor_data, callback_group=self.cb
        )
        self.pub_target_cmd = self.create_publisher(TwistStamped, self.target_cmd_vel_topic, 10)
        self.pub_pursuer_cmd = self.create_publisher(TwistStamped, self.pursuer_cmd_vel_topic, 10)

        # Waypoint markers publisher
        if self.publish_waypoints_markers:
            qos_markers = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.pub_waypoints = self.create_publisher(MarkerArray, self.waypoints_markers_topic, qos_markers)
            # Republish periodically for RViz (like target_stepper)
            if self.waypoints_publish_period_sec > 0.0:
                self.create_timer(self.waypoints_publish_period_sec, self._publish_waypoints_markers, callback_group=self.cb)

        # Pursuer waypoint pose publisher (for debugging alignment)
        qos_waypoint_pose = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_pursuer_waypoint_pose = self.create_publisher(PoseStamped, self.pursuer_waypoint_pose_topic, qos_waypoint_pose)

        # Control timer
        dt_control = 1.0 / min(self.control_rate_hz, self.max_control_rate_hz)
        self.create_timer(dt_control, self._control_tick, callback_group=self.cb)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📁 Waypoints: {self.waypoint_csv} ({len(self.waypoints)} waypoints)")
        self.get_logger().info(f"📡 Target pose: {self.target_vrpn_pose_topic}")
        self.get_logger().info(f"📡 Pursuer pose: {self.pursuer_vrpn_pose_topic}")
        self.get_logger().info(f"🕹️  Target cmd_vel: {self.target_cmd_vel_topic} (max_v={self.target_max_v})")
        self.get_logger().info(f"🕹️  Pursuer cmd_vel: {self.pursuer_cmd_vel_topic} (max_v={self.pursuer_max_v})")
        self.get_logger().info(f"🔧 Solo phase: waypoints 0-{self.solo_phase_end_waypoint}")
        self.get_logger().info(f"🧯 Safety: enable_cmd={self.enable_cmd} dry_run={self.dry_run}")
        if self.publish_waypoints_markers:
            self.get_logger().info(f"📊 Waypoints markers: {self.waypoints_markers_topic}")
        self.get_logger().info(f"🐛 Debug: Pursuer waypoint pose: {self.pursuer_waypoint_pose_topic}")
        self.get_logger().info("----------------------------------------------------------------")

        # Publish waypoint markers on startup
        if self.publish_waypoints_markers:
            self._publish_waypoints_markers()

    def _load_waypoints(self) -> list[Tuple[float, float, float, float, float, float]]:
        """Load waypoints from CSV: (time_s, target_x, target_y, target_z, pursuer_x, pursuer_y, pursuer_z)."""
        waypoints = []
        with open(self.waypoint_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row['time_s'])
                tx = float(row['target_x'])
                ty = float(row['target_y'])
                tz = float(row['target_z'])
                px = float(row['pursuer_x'])
                py = float(row['pursuer_y'])
                pz = float(row['pursuer_z'])
                waypoints.append((t, tx, ty, tz, px, py, pz))
        return waypoints

    def _on_target_pose(self, msg: PoseStamped):
        with self._mtx:
            self.target_pose = msg

    def _on_pursuer_pose(self, msg: PoseStamped):
        with self._mtx:
            self.pursuer_pose = msg

    def _check_waypoint_arrival(self, x: float, y: float, waypoint_idx: int, is_target: bool) -> bool:
        if waypoint_idx >= len(self.waypoints):
            return False
        
        _, tx, ty, _, px, py, _ = self.waypoints[waypoint_idx]
        goal_x, goal_y = (tx, ty) if is_target else (px, py)
        
        dist = math.hypot(x - goal_x, y - goal_y)
        return dist <= self.waypoint_tolerance_m

    def _compute_control(self, x: float, y: float, yaw: float, goal_x: float, goal_y: float, max_v: float) -> Tuple[float, float]:
        dx = goal_x - x
        dy = goal_y - y
        dist = math.hypot(dx, dy)

        if dist < self.stop_distance_m:
            return 0.0, 0.0

        target_heading = math.atan2(dy, dx)
        heading_error = math.atan2(math.sin(target_heading - yaw), math.cos(target_heading - yaw))

        w = self.k_heading * heading_error
        w = max(-self.max_w, min(self.max_w, w))

        v = min(max_v, self.k_dist * dist)
        v = max(0.0, v)

        # Reduce speed while turning
        v = v / (1.0 + self.v_turn_penalty * abs(w))

        # Slow down for large heading errors
        if abs(heading_error) > self.heading_slow_rad:
            v *= self.heading_slow_scale

        v = max(self.min_v, v) if v > 0.0 else 0.0

        return v, w

    def _control_tick(self):
        if not self.enable_cmd:
            return

        with self._mtx:
            if self.target_pose is None or self.pursuer_pose is None:
                return

            # Get current poses
            tp = self.target_pose.pose.position
            target_x, target_y = float(tp.x), float(tp.y)
            to = self.target_pose.pose.orientation
            target_yaw = yaw_from_quat((to.x, to.y, to.z, to.w))

            pp = self.pursuer_pose.pose.position
            pursuer_x, pursuer_y = float(pp.x), float(pp.y)
            po = self.pursuer_pose.pose.orientation
            pursuer_yaw = yaw_from_quat((po.x, po.y, po.z, po.w))

            # Check target waypoint arrivals (check all waypoints up to current target)
            for wp_idx in range(min(self.target_waypoint_idx + 1, len(self.waypoints))):
                if not self.target_waypoint_reached[wp_idx]:
                    if self._check_waypoint_arrival(target_x, target_y, wp_idx, is_target=True):
                        self.target_waypoint_reached[wp_idx] = True
                        self.get_logger().info(f"✅ Target reached waypoint {wp_idx}")
                        # Update target waypoint index to next unreached waypoint
                        if wp_idx == self.target_waypoint_idx:
                            self.target_waypoint_idx = min(wp_idx + 1, len(self.waypoints))

            # Control target - move toward current target waypoint
            if self.target_waypoint_idx < len(self.waypoints):
                _, tx, ty, _, _, _, _ = self.waypoints[self.target_waypoint_idx]
                target_v, target_w = self._compute_control(target_x, target_y, target_yaw, tx, ty, self.target_max_v)
            else:
                target_v, target_w = 0.0, 0.0

            # Control pursuer
            pursuer_target_waypoint = None
            # Phase 1: Solo phase (waypoints 0-4) - pursuer stays still
            if self.target_waypoint_idx <= self.solo_phase_end_waypoint:
                pursuer_v, pursuer_w = 0.0, 0.0
                pursuer_target_waypoint = "wait"
                self.pursuer_in_align_mode = False
                self.pursuer_prev_target_waypoint = -1  # Reset when in solo phase
            else:
                # Phase 2: Coordinated phase - pursuer tries to reach same waypoint as target
                pursuer_target_waypoint = self.target_waypoint_idx
                
                # Reset alignment state when moving to a new waypoint
                if pursuer_target_waypoint != self.pursuer_prev_target_waypoint:
                    self.pursuer_in_align_mode = False
                    self.pursuer_prev_target_waypoint = pursuer_target_waypoint
                
                # Check if pursuer has reached the target waypoint (position)
                if pursuer_target_waypoint < len(self.waypoints):
                    _, _, _, _, px, py, _ = self.waypoints[pursuer_target_waypoint]
                    dist_to_waypoint = math.hypot(pursuer_x - px, pursuer_y - py)
                    
                    # Stage 1: Position control (move to waypoint)
                    # If already in alignment mode, skip position control and go straight to alignment logic
                    # This prevents exiting alignment mode due to small position drift during rotation
                    if dist_to_waypoint > self.waypoint_tolerance_m and not self.pursuer_in_align_mode:
                        # Not at waypoint yet and NOT in alignment mode - move toward it
                        pursuer_v, pursuer_w = self._compute_control(pursuer_x, pursuer_y, pursuer_yaw, px, py, self.pursuer_max_v)
                    elif dist_to_waypoint <= self.waypoint_tolerance_m or self.pursuer_in_align_mode:
                        # At waypoint position - mark as reached
                        if not self.pursuer_waypoint_reached[pursuer_target_waypoint]:
                            self.pursuer_waypoint_reached[pursuer_target_waypoint] = True
                            self.get_logger().info(f"✅ Pursuer reached waypoint {pursuer_target_waypoint} (position)")
                        
                        # Stage 2: Orientation control (align heading toward target's current waypoint)
                        # The pursuer aligns toward pursuer_target_waypoint (which equals target_waypoint_idx)
                        if self.pursuer_align_goal_yaw:
                            # Compute desired yaw: from pursuer's current position to target's waypoint
                            _, tx, ty, _, _, _, _ = self.waypoints[pursuer_target_waypoint]
                            desired_yaw = math.atan2(ty - pursuer_y, tx - pursuer_x)
                            yaw_error = math.atan2(math.sin(desired_yaw - pursuer_yaw), math.cos(desired_yaw - pursuer_yaw))
                            
                            # Enter align mode (only once per waypoint, with hysteresis)
                            hold_factor = max(1.0, self.pursuer_align_hold_factor)
                            # Use a much larger threshold for exiting alignment mode to prevent shivering
                            # Once in alignment, stay in it until alignment completes or waypoint changes
                            exit_align_threshold = self.waypoint_tolerance_m * max(5.0, hold_factor * 3.0)
                            
                            # Check if we should enter alignment mode
                            if not self.pursuer_in_align_mode and dist_to_waypoint <= self.waypoint_tolerance_m:
                                # Only enter and log once per waypoint
                                if not self.pursuer_alignment_entered[pursuer_target_waypoint]:
                                    self.pursuer_in_align_mode = True
                                    self.pursuer_alignment_entered[pursuer_target_waypoint] = True
                                    self.get_logger().info(f"🔄 Pursuer entering alignment mode for waypoint {pursuer_target_waypoint} (aligning toward target waypoint {pursuer_target_waypoint})")
                                else:
                                    # Re-enter silently if we're back within tolerance
                                    self.pursuer_in_align_mode = True
                            
                            # Once in alignment mode for a waypoint, stay in it until waypoint changes
                            # Only exit if we've moved EXTREMELY far away (10x tolerance = 1.0m for 0.1m tolerance)
                            # This prevents shivering from small position drift during rotation
                            if self.pursuer_in_align_mode:
                                # Use a very large threshold - only exit if we've moved extremely far away
                                # This should only happen if we're actually moving to a new waypoint
                                extreme_exit_threshold = self.waypoint_tolerance_m * 10.0
                                if dist_to_waypoint > extreme_exit_threshold:
                                    # Moved extremely far away - exit alignment mode (shouldn't happen normally)
                                    self.pursuer_in_align_mode = False
                                    self.get_logger().warn(f"⚠️  Pursuer exited alignment mode for waypoint {pursuer_target_waypoint} due to large position drift ({dist_to_waypoint:.3f}m > {extreme_exit_threshold:.3f}m)")
                            
                            if self.pursuer_in_align_mode:
                                # Align mode: rotate in place
                                if abs(yaw_error) <= self.pursuer_yaw_match_tol_rad:
                                    # Aligned! Publish pose for debugging (only once per waypoint)
                                    if not self.pursuer_waypoint_pose_published[pursuer_target_waypoint]:
                                        self._publish_pursuer_waypoint_pose(pursuer_target_waypoint, pursuer_x, pursuer_y, pursuer_yaw, desired_yaw)
                                        self.pursuer_waypoint_pose_published[pursuer_target_waypoint] = True
                                        self.get_logger().info(f"✅ Pursuer aligned at waypoint {pursuer_target_waypoint} (yaw error: {math.degrees(abs(yaw_error)):.1f}°)")
                                    # Wait if target hasn't reached this waypoint yet
                                    if not self.target_waypoint_reached[pursuer_target_waypoint]:
                                        pursuer_v, pursuer_w = 0.0, 0.0
                                    else:
                                        # Both aligned and reached - can proceed to next waypoint
                                        pursuer_v, pursuer_w = 0.0, 0.0
                                    # Keep alignment mode active until we move to next waypoint
                                    # Don't reset pursuer_in_align_mode here to prevent re-entry
                                else:
                                    # Rotate to align - stay in alignment mode regardless of small position drift
                                    # Once we're actively aligning, we should not exit due to position drift
                                    pursuer_w = self.pursuer_k_heading_align * yaw_error
                                    pursuer_w = max(-self.pursuer_max_w_align, min(self.pursuer_max_w_align, pursuer_w))
                                    pursuer_v = 0.0  # No forward motion during alignment
                            else:
                                # Still moving to position
                                pursuer_v, pursuer_w = self._compute_control(pursuer_x, pursuer_y, pursuer_yaw, px, py, self.pursuer_max_v)
                        else:
                            # No alignment needed - wait if target hasn't reached this waypoint yet
                            if not self.target_waypoint_reached[pursuer_target_waypoint]:
                                pursuer_v, pursuer_w = 0.0, 0.0
                            else:
                                pursuer_v, pursuer_w = 0.0, 0.0
                            self.pursuer_in_align_mode = False
                else:
                    pursuer_v, pursuer_w = 0.0, 0.0
                    self.pursuer_in_align_mode = False

            # Publish commands
            target_cmd = TwistStamped()
            target_cmd.header.stamp = TimeMsg() if self.cmd_stamp_zero else self.get_clock().now().to_msg()
            target_cmd.header.frame_id = self.cmd_frame_id
            target_cmd.twist.linear.x = float(target_v)
            target_cmd.twist.angular.z = float(target_w)

            pursuer_cmd = TwistStamped()
            pursuer_cmd.header.stamp = TimeMsg() if self.cmd_stamp_zero else self.get_clock().now().to_msg()
            pursuer_cmd.header.frame_id = self.cmd_frame_id
            pursuer_cmd.twist.linear.x = float(pursuer_v)
            pursuer_cmd.twist.angular.z = float(pursuer_w)

            if not self.dry_run:
                self.pub_target_cmd.publish(target_cmd)
                self.pub_pursuer_cmd.publish(pursuer_cmd)

            # Log status
            if self.log_status:
                t_now = time.time()
                if t_now - self._t_log_prev >= self.log_period_sec:
                    phase = "solo" if self.target_waypoint_idx <= self.solo_phase_end_waypoint else "coordinated"
                    pursuer_wp_str = str(pursuer_target_waypoint) if pursuer_target_waypoint != "wait" else "wait"
                    align_str = " [ALIGN]" if self.pursuer_in_align_mode else ""
                    self.get_logger().info(
                        f"Phase: {phase} | Target: wp={self.target_waypoint_idx}/{len(self.waypoints)} "
                        f"v={target_v:.3f} w={target_w:.3f} | "
                        f"Pursuer: wp={pursuer_wp_str}{align_str} v={pursuer_v:.3f} w={pursuer_w:.3f}"
                    )
                    self._t_log_prev = t_now

    def _publish_waypoints_markers(self):
        if not self.publish_waypoints_markers or len(self.waypoints) == 0:
            return

        stamp = self.get_clock().now().to_msg()

        # Target waypoints (blue)
        mk_target_line = Marker()
        mk_target_line.header.stamp = stamp
        mk_target_line.header.frame_id = self.waypoints_frame_id
        mk_target_line.ns = "target_waypoints"
        mk_target_line.id = 0
        mk_target_line.type = Marker.LINE_STRIP
        mk_target_line.action = Marker.ADD
        mk_target_line.scale.x = float(self.waypoints_line_width)
        mk_target_line.color.r = 0.1
        mk_target_line.color.g = 0.3
        mk_target_line.color.b = 1.0
        mk_target_line.color.a = float(self.waypoints_alpha)
        mk_target_line.pose.orientation.w = 1.0

        mk_target_pts = Marker()
        mk_target_pts.header.stamp = stamp
        mk_target_pts.header.frame_id = self.waypoints_frame_id
        mk_target_pts.ns = "target_waypoints"
        mk_target_pts.id = 1
        mk_target_pts.type = Marker.POINTS
        mk_target_pts.action = Marker.ADD
        mk_target_pts.scale.x = float(self.waypoints_point_scale)
        mk_target_pts.scale.y = float(self.waypoints_point_scale)
        mk_target_pts.color.r = 0.1
        mk_target_pts.color.g = 0.3
        mk_target_pts.color.b = 1.0
        mk_target_pts.color.a = float(self.waypoints_alpha)
        mk_target_pts.pose.orientation.w = 1.0

        # Pursuer waypoints (red)
        mk_pursuer_line = Marker()
        mk_pursuer_line.header.stamp = stamp
        mk_pursuer_line.header.frame_id = self.waypoints_frame_id
        mk_pursuer_line.ns = "pursuer_waypoints"
        mk_pursuer_line.id = 2
        mk_pursuer_line.type = Marker.LINE_STRIP
        mk_pursuer_line.action = Marker.ADD
        mk_pursuer_line.scale.x = float(self.waypoints_line_width)
        mk_pursuer_line.color.r = 1.0
        mk_pursuer_line.color.g = 0.1
        mk_pursuer_line.color.b = 0.1
        mk_pursuer_line.color.a = float(self.waypoints_alpha)
        mk_pursuer_line.pose.orientation.w = 1.0

        mk_pursuer_pts = Marker()
        mk_pursuer_pts.header.stamp = stamp
        mk_pursuer_pts.header.frame_id = self.waypoints_frame_id
        mk_pursuer_pts.ns = "pursuer_waypoints"
        mk_pursuer_pts.id = 3
        mk_pursuer_pts.type = Marker.POINTS
        mk_pursuer_pts.action = Marker.ADD
        mk_pursuer_pts.scale.x = float(self.waypoints_point_scale)
        mk_pursuer_pts.scale.y = float(self.waypoints_point_scale)
        mk_pursuer_pts.color.r = 1.0
        mk_pursuer_pts.color.g = 0.1
        mk_pursuer_pts.color.b = 0.1
        mk_pursuer_pts.color.a = float(self.waypoints_alpha)
        mk_pursuer_pts.pose.orientation.w = 1.0

        # Add waypoints
        for _, tx, ty, tz, px, py, pz in self.waypoints:
            # Target waypoint
            pt = Point()
            pt.x = float(tx)
            pt.y = float(ty)
            pt.z = float(tz)
            mk_target_line.points.append(pt)
            mk_target_pts.points.append(pt)

            # Pursuer waypoint
            pp = Point()
            pp.x = float(px)
            pp.y = float(py)
            pp.z = float(pz)
            mk_pursuer_line.points.append(pp)
            mk_pursuer_pts.points.append(pp)

        # Publish MarkerArray
        arr = MarkerArray()
        arr.markers = [mk_target_line, mk_target_pts, mk_pursuer_line, mk_pursuer_pts]
        self.pub_waypoints.publish(arr)
        # Only log once per second to reduce spam
        t_now = time.time()
        if not hasattr(self, '_last_marker_log_time') or t_now - self._last_marker_log_time >= 1.0:
            self.get_logger().info(f"Published {len(self.waypoints)} waypoints to {self.waypoints_markers_topic} (frame: {self.waypoints_frame_id})")
            self._last_marker_log_time = t_now

    def _publish_pursuer_waypoint_pose(self, waypoint_idx: int, x: float, y: float, yaw: float, desired_yaw: float):
        ps = PoseStamped()
        ps.header.frame_id = self.waypoints_frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        qx, qy, qz, qw = quat_from_yaw(yaw)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        
        self.pub_pursuer_waypoint_pose.publish(ps)
        
        yaw_error = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
        self.get_logger().info(
            f"📌 Published pursuer waypoint pose: wp={waypoint_idx} "
            f"pos=({x:.3f}, {y:.3f}) yaw={math.degrees(yaw):.1f}° "
            f"desired={math.degrees(desired_yaw):.1f}° error={math.degrees(yaw_error):.1f}°"
        )

    def _stop(self):
        try:
            target_cmd = TwistStamped()
            target_cmd.header.stamp = TimeMsg() if self.cmd_stamp_zero else self.get_clock().now().to_msg()
            target_cmd.header.frame_id = self.cmd_frame_id
            target_cmd.twist.linear.x = 0.0
            target_cmd.twist.angular.z = 0.0

            pursuer_cmd = TwistStamped()
            pursuer_cmd.header.stamp = TimeMsg() if self.cmd_stamp_zero else self.get_clock().now().to_msg()
            pursuer_cmd.header.frame_id = self.cmd_frame_id
            pursuer_cmd.twist.linear.x = 0.0
            pursuer_cmd.twist.angular.z = 0.0

            if self.enable_cmd and not self.dry_run:
                # Publish multiple times to ensure the stop command is received
                for _ in range(3):
                    try:
                        self.pub_target_cmd.publish(target_cmd)
                        self.pub_pursuer_cmd.publish(pursuer_cmd)
                        time.sleep(0.05)  # Small delay between publishes
                    except Exception:
                        pass  # Ignore errors during shutdown
                self.get_logger().info("🛑 Sent stop commands to both robots")
        except Exception:
            # Ignore errors during shutdown
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CoordinatedTrajectoryFollower()
    
    # Set up signal handlers to stop robots on SIGINT/SIGTERM
    def signal_handler(signum, frame):
        node.get_logger().info(f"🛑 Received signal {signum}, stopping robots...")
        node._stop()
        # Raise KeyboardInterrupt to exit spin loop naturally
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure stop commands are sent even if something else caused shutdown
        node._stop()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Ignore shutdown errors


if __name__ == "__main__":
    main()
