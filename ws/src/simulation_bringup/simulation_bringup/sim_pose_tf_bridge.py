#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from typing import cast

from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros

from mua_nbv_py_utils.transforms import planarize


class SimPoseTFBridge(Node): 
    def __init__(self):
        super().__init__("sim_pose_tf_bridge")

        # ---- Params ----
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("pursuer_base_frame", "pursuer/base_link")
        self.declare_parameter("target_base_frame", "target/base_link")  
        self.declare_parameter("pursuer_pose_topic", "/sim/pursuer/pose")
        self.declare_parameter("target_pose_topic", "/sim/target/pose")
        # Planarization  
        self.declare_parameter("flatten_z", True)
        self.declare_parameter("flatten_z_value", 0.0)
        self.declare_parameter("flatten_roll_pitch", True)

        # ---- Read params ----
        self.world = str(self.get_parameter("world_frame").value)
        self.pursuer_base = str(self.get_parameter("pursuer_base_frame").value)
        self.target_base = str(self.get_parameter("target_base_frame").value)
        self.pursuer_pose_topic = str(self.get_parameter("pursuer_pose_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.flatten_z = bool(self.get_parameter("flatten_z").value)
        self.flatten_z_value = float(cast(float, self.get_parameter("flatten_z_value").value))
        self.flatten_roll_pitch = bool(self.get_parameter("flatten_roll_pitch").value)

        # ---- Broadcasters / Subscribers ----
        self.br = tf2_ros.TransformBroadcaster(self)
        qos_in = QoSProfile(history=HistoryPolicy.KEEP_LAST,
                            depth=10,
                            reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, self.pursuer_pose_topic, self.cb_pursuer, qos_in)
        self.create_subscription(PoseStamped, self.target_pose_topic, self.cb_target, qos_in)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📍 TF: {self.world} -> {self.target_base}, {self.world} -> {self.pursuer_base}")
        self.get_logger().info("----------------------------------------------------------------")

    def cb_target(self, msg: PoseStamped):
        t_raw = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        q_raw = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        t_use, q_use = planarize(t_raw, q_raw, flatten_z=self.flatten_z, z0=self.flatten_z_value, flatten_roll_pitch=self.flatten_roll_pitch)

        t = TransformStamped() 
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.world
        t.child_frame_id = self.target_base
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = t_use
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q_use
        self.br.sendTransform(t)

    def cb_pursuer(self, msg: PoseStamped):
        t_raw = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        q_raw = (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        t_use, q_use = planarize(t_raw, q_raw, flatten_z=self.flatten_z, z0=self.flatten_z_value, flatten_roll_pitch=self.flatten_roll_pitch)

        out = TransformStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.world
        out.child_frame_id = self.pursuer_base
        out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = t_use
        out.transform.rotation.x, out.transform.rotation.y, out.transform.rotation.z, out.transform.rotation.w = q_use
        self.br.sendTransform(out)


def main():
    rclpy.init()
    node = SimPoseTFBridge()
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
