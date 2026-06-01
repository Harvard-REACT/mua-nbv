#!/usr/bin/env python3
import os, time, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

class RGBSaver(Node):
    def __init__(self):
        super().__init__('rgb_saver')
        # Params (relative topics work nicely with namespace)
        self.declare_parameter('image_topic', 'camera/color/image_raw')
        self.declare_parameter('save_dir', os.path.expanduser('~/captures'))
        self.declare_parameter('fmt', 'jpg')                # 'jpg' or 'png'
        self.declare_parameter('jpeg_quality', 95)          # 0-100
        self.declare_parameter('burst_n', 1)                # how many frames per trigger

        self.image_topic = self.get_parameter('image_topic').value
        self.save_dir = self.get_parameter('save_dir').value
        self.fmt = self.get_parameter('fmt').value.lower()
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.burst_n_cfg = int(self.get_parameter('burst_n').value)

        os.makedirs(self.save_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.want_left = 0

        # Sub to RGB
        self.sub = self.create_subscription(
            Image, self.image_topic, self._on_img, qos_profile_sensor_data
        )
        # Service: /camera/save_rgb (namespaced by __ns)
        self.srv = self.create_service(Trigger, 'camera/save_rgb', self._on_save)

        self.get_logger().info(f"RGBSaver watching: {self.image_topic}")
        self.get_logger().info(f"Saves to: {self.save_dir} (fmt={self.fmt}, burst_n={self.burst_n_cfg})")

    def _on_save(self, req, resp):
        self.want_left = max(1, self.burst_n_cfg)
        resp.success = True
        resp.message = f"Capturing next {self.want_left} frame(s) from {self.image_topic}"
        return resp

    def _on_img(self, msg: Image):
        if self.want_left <= 0:
            return
        try:
            cvimg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ts = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            fname = f"rgb_{ts}.{self.fmt}"
            path = os.path.join(self.save_dir, fname)
            ok = True
            if self.fmt == 'jpg' or self.fmt == 'jpeg':
                ok = cvimg is not None and cv2.imwrite(
                    path, cvimg, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
            else:
                ok = cvimg is not None and cv2.imwrite(path, cvimg)
            if ok:
                self.get_logger().info(f"Saved: {path}")
                self.want_left -= 1
            else:
                self.get_logger().error(f"Failed to save: {path}")
        except Exception as e:
            self.get_logger().error(f"Capture failed: {e}")
            self.want_left = 0

def main():
    rclpy.init()
    rclpy.spin(RGBSaver())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
