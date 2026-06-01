#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf2_ros

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from mua_nbv_py_utils.transforms import yaw_from_quat, quat_from_yaw, rotate_z, planarize


class VrpnTFBridge(Node):
    def __init__(self):
        super().__init__("vrpn_tf_bridge")

        # frames / topics
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("pursuer_pose_topic", "/vrpn_mocap/pursuer/pose")
        self.declare_parameter("target_pose_topic", "/vrpn_mocap/target/pose")
        self.declare_parameter("pursuer_base_frame", "pursuer/base_link")
        self.declare_parameter("target_base_frame", "target/base_link")

        # planarization
        self.declare_parameter("flatten_z", True)
        self.declare_parameter("flatten_z_value", 0.0)
        self.declare_parameter("flatten_roll_pitch", True)

        # Static offsets from VRPN rigid-body origin to desired base_link origins
        # expressed in the VRPN body frame (meters). This compensates lever-arm
        # effects where turning-in-place appears as translation.
        self.declare_parameter("pursuer_offset_x", 0.0)
        self.declare_parameter("pursuer_offset_y", 0.0)
        self.declare_parameter("pursuer_offset_z", 0.0)
        self.declare_parameter("target_offset_x", 0.0)
        self.declare_parameter("target_offset_y", 0.0)
        self.declare_parameter("target_offset_z", 0.0)
        # Optional yaw offsets (rad) if the VRPN body frame yaw is rotated
        # relative to the intended base_link yaw.
        self.declare_parameter("pursuer_yaw_offset_rad", 0.0)
        self.declare_parameter("target_yaw_offset_rad", 0.0)
        # Timestamp handling:
        # - Some VRPN/Optitrack PoseStamped streams come with stamp=0 (or unsynced clocks).
        # - If we forward that stamp into TF, TF2 will consider it ancient and drop it, so
        #   the 'world' frame appears missing.
        # - Setting TF stamps to "now" is robust and matches how the rest of testbed uses latest TF.
        self.declare_parameter("tf_stamp_mode", "now")  # "now" | "msg" | "auto"

        self.world = self.get_parameter("world_frame").value
        self.pursuer_pose_topic = self.get_parameter("pursuer_pose_topic").value
        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.pursuer_base = self.get_parameter("pursuer_base_frame").value
        self.target_base = self.get_parameter("target_base_frame").value

        self.flatten_z = bool(self.get_parameter("flatten_z").value)
        self.flatten_z_value = float(self.get_parameter("flatten_z_value").value)
        self.flatten_roll_pitch = bool(self.get_parameter("flatten_roll_pitch").value)

        self.pursuer_offset = (
            float(self.get_parameter("pursuer_offset_x").value),
            float(self.get_parameter("pursuer_offset_y").value),
            float(self.get_parameter("pursuer_offset_z").value),
        )
        self.target_offset = (
            float(self.get_parameter("target_offset_x").value),
            float(self.get_parameter("target_offset_y").value),
            float(self.get_parameter("target_offset_z").value),
        )
        self.pursuer_yaw_offset = float(self.get_parameter("pursuer_yaw_offset_rad").value)
        self.target_yaw_offset = float(self.get_parameter("target_yaw_offset_rad").value)
        self.tf_stamp_mode = str(self.get_parameter("tf_stamp_mode").value).strip()

        # TF broadcaster
        self.br = tf2_ros.TransformBroadcaster(self)

        qos_vrpn = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, self.pursuer_pose_topic, self.cb_pursuer, qos_vrpn)
        self.create_subscription(PoseStamped, self.target_pose_topic, self.cb_target, qos_vrpn)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📍 TF: {self.world} -> {self.target_base}, {self.world} -> {self.pursuer_base}")
        self.get_logger().info(f"📡 VRPN: {self.target_pose_topic}, {self.pursuer_pose_topic}")
        self.get_logger().info(f"🧭 pursuer_offset(xyz)={self.pursuer_offset} yaw_offset={self.pursuer_yaw_offset:.3f}")
        self.get_logger().info(f"🧭 target_offset(xyz)={self.target_offset} yaw_offset={self.target_yaw_offset:.3f}")
        self.get_logger().info("----------------------------------------------------------------")

    def cb_target(self, msg: PoseStamped):
        t_raw = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        q_raw = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        t_use, q_use = planarize(
            t_raw, q_raw,
            flatten_z=self.flatten_z, z0=self.flatten_z_value,
            flatten_roll_pitch=self.flatten_roll_pitch, yaw_offset_rad=self.target_yaw_offset,
        )
        # Apply lever-arm offset in the planarized (yaw-only) body frame.
        yaw_use = yaw_from_quat(q_use)
        off_w = rotate_z(yaw_use, self.target_offset)
        t_use = (t_use[0] + off_w[0], t_use[1] + off_w[1], t_use[2] + off_w[2])

        t = TransformStamped()
        if self.tf_stamp_mode == "msg":
            t.header.stamp = msg.header.stamp
        elif self.tf_stamp_mode == "auto":
            st = msg.header.stamp
            if int(st.sec) == 0 and int(st.nanosec) == 0:
                t.header.stamp = self.get_clock().now().to_msg()
            else:
                t.header.stamp = st
        else:
            t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world
        t.child_frame_id = self.target_base
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = t_use
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q_use
        self.br.sendTransform(t)

    def cb_pursuer(self, msg: PoseStamped):
        t_raw = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        q_raw = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        t_use, q_use = planarize(
            t_raw, q_raw,
            flatten_z=self.flatten_z, z0=self.flatten_z_value,
            flatten_roll_pitch=self.flatten_roll_pitch, yaw_offset_rad=self.pursuer_yaw_offset,
        )
        # Apply lever-arm offset in the planarized (yaw-only) body frame.
        yaw_use = yaw_from_quat(q_use)
        off_w = rotate_z(yaw_use, self.pursuer_offset)
        t_use = (t_use[0] + off_w[0], t_use[1] + off_w[1], t_use[2] + off_w[2])
        
        out = TransformStamped()
        if self.tf_stamp_mode == "msg":
            out.header.stamp = msg.header.stamp
        elif self.tf_stamp_mode == "auto":
            st = msg.header.stamp
            if int(st.sec) == 0 and int(st.nanosec) == 0:
                out.header.stamp = self.get_clock().now().to_msg()
            else:
                out.header.stamp = st
        else:
            out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.world
        out.child_frame_id = self.pursuer_base
        out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = t_use
        out.transform.rotation.x, out.transform.rotation.y, out.transform.rotation.z, out.transform.rotation.w = q_use
        self.br.sendTransform(out)


def main():
    rclpy.init()
    node = VrpnTFBridge()
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
