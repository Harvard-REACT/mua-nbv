#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, TwistStamped

import tf2_ros

from mua_nbv_py_utils.transforms import yaw_from_quat, quat_from_yaw, quat_mul, quat_conj, quat_rotate, planarize


class PursuerTeleporter(Node):  
    def __init__(self):
        super().__init__("pursuer_teleporter")

        # ---- Params ----
        self.declare_parameter("best_candidate_topic", "/nbv/best_candidate")

        # Frames (assume best_candidate is optical pose by default)
        self.declare_parameter("candidate_is_optical_pose", True)
        self.declare_parameter("base_frame", "pursuer/base_link")
        self.declare_parameter("optical_frame", "pursuer/camera_depth_optical_frame")
        self.declare_parameter("tf_lookup_timeout_sec", 0.2)

        # Planarization (recommended)
        self.declare_parameter("flatten_z", True)
        self.declare_parameter("flatten_z_value", 0.0)
        self.declare_parameter("flatten_roll_pitch", True)

        # cmd_vel mode
        self.declare_parameter("vrpn_pose_topic", "/vrpn_mocap/pursuer/pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_v", 0.2)
        self.declare_parameter("max_w", 1.5)
        self.declare_parameter("k_heading", 1.5)
        self.declare_parameter("pose_match_tol_m", 0.10)
        self.declare_parameter("enable_cmd", True)

        # ---- Read params ----
        self.best_topic = str(self.get_parameter("best_candidate_topic").value)

        self.candidate_is_optical = bool(self.get_parameter("candidate_is_optical_pose").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.optical_frame = str(self.get_parameter("optical_frame").value)
        self.tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)

        self.flatten_z = bool(self.get_parameter("flatten_z").value)
        self.flatten_z_value = float(self.get_parameter("flatten_z_value").value)
        self.flatten_roll_pitch = bool(self.get_parameter("flatten_roll_pitch").value)
        self.vrpn_pose_topic = str(self.get_parameter("vrpn_pose_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.pose_match_tol_m = float(self.get_parameter("pose_match_tol_m").value)
        self.enable_cmd = bool(self.get_parameter("enable_cmd").value)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---- State ----
        self._last_goal_base: Optional[PoseStamped] = None
        self._current_pose: Optional[PoseStamped] = None  # vrpn feedback in cmd_vel mode

        # ---- Pub/Sub ----
        self.sub_best = self.create_subscription(PoseStamped, self.best_topic, self._on_best, qos_profile_sensor_data)

        self.pub_cmd = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.sub_pose = self.create_subscription(PoseStamped, self.vrpn_pose_topic, self._on_pose, qos_profile_sensor_data)

        period = 1.0 / max(self.control_rate_hz, 1e-3)
        self.timer = self.create_timer(period, self._control_tick)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📡 best_candidate: {self.best_topic}")
        self.get_logger().info(f"🧭 candidate_is_optical_pose={self.candidate_is_optical} optical_frame={self.optical_frame} base_frame={self.base_frame}")
        self.get_logger().info(f"🕹️ cmd_vel_topic={self.cmd_vel_topic}")
        self.get_logger().info(f"🧯 enable_cmd={self.enable_cmd}")
        self.get_logger().info("----------------------------------------------------------------")

    def _on_pose(self, msg: PoseStamped):
        self._current_pose = msg

    def _lookup_optical_from_base(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Returns T_optical_base as (t_OB, q_OB) where p_O = R(q_OB) p_B + t_OB
        """
        tf = self.tf_buffer.lookup_transform(
            self.optical_frame,
            self.base_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=float(self.tf_lookup_timeout_sec)),
        )
        t = tf.transform.translation
        r = tf.transform.rotation
        t_ob = (float(t.x), float(t.y), float(t.z))
        q_ob = (float(r.x), float(r.y), float(r.z), float(r.w))
        return t_ob, q_ob

    def _optical_goal_to_base_goal(self, msg: PoseStamped) -> PoseStamped:
        # world -> optical goal from best_candidate
        p = msg.pose.position
        o = msg.pose.orientation
        t_wo = (float(p.x), float(p.y), float(p.z))
        q_wo = (float(o.x), float(o.y), float(o.z), float(o.w))

        t_ob, q_ob = self._lookup_optical_from_base()

        # T_WB = T_WO * T_OB
        t_wb = quat_rotate(q_wo, t_ob)
        t_wb = (t_wo[0] + t_wb[0], t_wo[1] + t_wb[1], t_wo[2] + t_wb[2])
        q_wb = quat_mul(q_wo, q_ob)

        (t_wb, q_wb) = planarize(t_wb, q_wb, flatten_z=self.flatten_z, z0=self.flatten_z_value, flatten_roll_pitch=self.flatten_roll_pitch)

        out = PoseStamped()
        out.header.frame_id = msg.header.frame_id
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose.position.x = float(t_wb[0])
        out.pose.position.y = float(t_wb[1])
        out.pose.position.z = float(t_wb[2])
        out.pose.orientation.x = float(q_wb[0])
        out.pose.orientation.y = float(q_wb[1])
        out.pose.orientation.z = float(q_wb[2])
        out.pose.orientation.w = float(q_wb[3])
        return out

    def _stop(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        if self.enable_cmd:
            self.pub_cmd.publish(cmd)

    def _control_tick(self):
        # Simple VRPN-feedback controller toward the last goal pose (base frame in world coords).
        if not self.enable_cmd:
            return
        if self._last_goal_base is None or self._current_pose is None:
            return

        # Use VRPN pose as current base pose in world
        p = self._current_pose.pose.position
        x, y = float(p.x), float(p.y)
        yaw = yaw_from_quat(
            (
                float(self._current_pose.pose.orientation.x),
                float(self._current_pose.pose.orientation.y),
                float(self._current_pose.pose.orientation.z),
                float(self._current_pose.pose.orientation.w),
            )
        )

        gx = float(self._last_goal_base.pose.position.x)
        gy = float(self._last_goal_base.pose.position.y)
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist <= self.pose_match_tol_m:
            self._stop()
            return

        target_heading = math.atan2(dy, dx)
        heading_error = math.atan2(math.sin(target_heading - yaw), math.cos(target_heading - yaw))

        w = self.k_heading * heading_error
        w = max(-self.max_w, min(self.max_w, w))

        v = self.max_v / (1.0 + 1.0 * abs(w))
        v = max(0.0, min(self.max_v, v))
        if abs(heading_error) > math.pi / 2:
            v = 0.0

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(w)
        self.pub_cmd.publish(cmd)

    def _on_best(self, msg: PoseStamped):
        if msg is None:
            return

        try:
            goal_base = msg if (not self.candidate_is_optical) else self._optical_goal_to_base_goal(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to compute base goal from best_candidate: {e}")
            return

        self._last_goal_base = goal_base
        # Controller will drive toward _last_goal_base in timer


def main():
    rclpy.init()
    node = PursuerTeleporter()
    try:
        rclpy.spin(node)
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
