#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from mua_nbv_py_utils.transforms import quat_from_rpy


class SimStaticTF(Node):  
    def __init__(self):
        super().__init__("sim_static_tf")

        # ---- Params ----
        self.declare_parameter("base_frame", "pursuer/base_link")
        self.declare_parameter("camera_link_frame", "pursuer/camera_link") 
        self.declare_parameter("camera_optical_frame", "pursuer/camera_depth_optical_frame")
        # Camera mount position
        self.declare_parameter("cam_x", 0.0)
        self.declare_parameter("cam_y", 0.0)
        self.declare_parameter("cam_z", 0.3)
        self.declare_parameter("optical_roll", -1.57079632679) 
        self.declare_parameter("optical_pitch", 0.0)
        self.declare_parameter("optical_yaw", -1.57079632679)

        # ---- Read params ----
        base = self.get_parameter("base_frame").value
        cam_link = self.get_parameter("camera_link_frame").value
        cam_opt = self.get_parameter("camera_optical_frame").value
        cam_x = float(self.get_parameter("cam_x").value)
        cam_y = float(self.get_parameter("cam_y").value)
        cam_z = float(self.get_parameter("cam_z").value)
        roll = float(self.get_parameter("optical_roll").value)
        pitch = float(self.get_parameter("optical_pitch").value)
        yaw = float(self.get_parameter("optical_yaw").value)
        qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)

        # ---- Broadcaster ----
        self.broadcaster = StaticTransformBroadcaster(self)

        # ---- Camera mount position ----
        t_cam = TransformStamped()
        t_cam.header.stamp = self.get_clock().now().to_msg()
        t_cam.header.frame_id = base
        t_cam.child_frame_id = cam_link
        t_cam.transform.translation.x = cam_x
        t_cam.transform.translation.y = cam_y
        t_cam.transform.translation.z = cam_z
        t_cam.transform.rotation.x = 0.0
        t_cam.transform.rotation.y = 0.0
        t_cam.transform.rotation.z = 0.0
        t_cam.transform.rotation.w = 1.0

        # ---- Camera optical rotation ----
        t_opt = TransformStamped()
        t_opt.header.stamp = self.get_clock().now().to_msg()
        t_opt.header.frame_id = cam_link
        t_opt.child_frame_id = cam_opt
        t_opt.transform.translation.x = 0.0
        t_opt.transform.translation.y = 0.0
        t_opt.transform.translation.z = 0.0
        t_opt.transform.rotation.x = qx
        t_opt.transform.rotation.y = qy
        t_opt.transform.rotation.z = qz
        t_opt.transform.rotation.w = qw

        self.broadcaster.sendTransform([t_cam, t_opt])

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"📍 Static TF: {base} -> {cam_link} xyz=({cam_x},{cam_y},{cam_z})")
        self.get_logger().info(f"📍 Static TF: {cam_link} -> {cam_opt} rpy=({roll:.3f},{pitch:.3f},{yaw:.3f})")
        self.get_logger().info("----------------------------------------------------------------")


def main():
    rclpy.init()
    node = SimStaticTF()
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
