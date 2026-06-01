#!/usr/bin/env python3
import math
import random
import threading
import time
from typing import cast

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.utilities import ok as rclpy_ok

import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time as rclpy_Time

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from builtin_interfaces.msg import Time as TimeMsg
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from mua_nbv_py_utils.transforms import quat_mul, quat_to_rot_np, rot_apply
from mua_nbv_common.ros_helpers import stamp_str, time_tuple


def dist2_xy(a: Pose, b: Pose) -> float:
    dx = float(a.position.x) - float(b.position.x)
    dy = float(a.position.y) - float(b.position.y)
    return dx * dx + dy * dy


class PursuerSpawner(Node): 
    def __init__(self):
        super().__init__("pursuer_spawner")
        # ---- Params ----
        # World
        self.declare_parameter("world_name", "sim_world")
        self.declare_parameter("entity_name", "pursuer")
        # Frames
        self.declare_parameter("base_frame", "pursuer/base_link")
        self.declare_parameter("optical_frame", "pursuer/camera_depth_optical_frame")
        # Topics
        self.declare_parameter("input_step_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("input_best_candidate_topic", "/nbv/best_candidate")
        self.declare_parameter("input_candidates_topic", "/nbv/candidates_world_base")
        self.declare_parameter("input_pursuer_pose_topic", "/sim/pursuer/pose")
        # Service
        self.declare_parameter("service_name", "/experiment/pursuer/spawn")
        self.declare_parameter("service_random_name", "/experiment/pursuer/spawn_random")
        # Timeouts / waits
        self.declare_parameter("wait_for_candidate_sec", 2.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.2)
        self.declare_parameter("pose_match_tol_m", 0.01)
        self.declare_parameter("pose_match_timeout_sec", 2.0)
        self.declare_parameter("random_seed", 0)
        # First spawn_random() only: "random", "nearest_origin", "fixed", "closest_pursuer"
        self.declare_parameter("seed_spawn_mode", "closest_pursuer")
        self.declare_parameter("seed_spawn_x", 0.0)
        self.declare_parameter("seed_spawn_y", 0.0)
        self.declare_parameter("seed_spawn_z", 0.3)
        self.declare_parameter("seed_spawn_yaw", 0.0)

        # ---- Read params ----
        self.world_name = str(self.get_parameter("world_name").value)
        self.entity_name = str(self.get_parameter("entity_name").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.optical_frame = str(self.get_parameter("optical_frame").value)
        self.step_stamp_topic = str(self.get_parameter("input_step_stamp_topic").value)
        self.best_topic = str(self.get_parameter("input_best_candidate_topic").value)
        self.candidates_topic = str(self.get_parameter("input_candidates_topic").value)
        self.pursuer_pose_topic = str(self.get_parameter("input_pursuer_pose_topic").value)
        self.srv_name = str(self.get_parameter("service_name").value)
        self.srv_random_name = str(self.get_parameter("service_random_name").value)
        self.wait_for_candidate_sec = float(cast(float, self.get_parameter("wait_for_candidate_sec").value))
        self.tf_lookup_timeout_sec = float(cast(float, self.get_parameter("tf_lookup_timeout_sec").value))
        self.pose_match_tol_m = float(cast(float, self.get_parameter("pose_match_tol_m").value))
        self.pose_match_timeout_sec = float(cast(float, self.get_parameter("pose_match_timeout_sec").value))
        self.random_seed = int(cast(int, self.get_parameter("random_seed").value))
        self._rng = random.Random(self.random_seed)
        self._seed_spawn_mode = str(self.get_parameter("seed_spawn_mode").value).strip().lower()
        self._seed_spawn_x = float(self.get_parameter("seed_spawn_x").value)
        self._seed_spawn_y = float(self.get_parameter("seed_spawn_y").value)
        self._seed_spawn_z = float(self.get_parameter("seed_spawn_z").value)
        self._seed_spawn_yaw = float(self.get_parameter("seed_spawn_yaw").value)
        self._first_random_done = False

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---- State ----
        self._cv = threading.Condition()
        self._last_step_stamp = None
        self._best_msg = None
        self._best_seq = 0
        self._candidates_msg = None
        self._candidates_seq = 0
        self._last_pursuer_pose = None
        self._pose_seq = 0

        # ---- Gazebo client ----
        self._gz_service_name = f"/world/{self.world_name}/set_pose"
        self._client = self.create_client(SetEntityPose, self._gz_service_name)

        # ---- Subscriptions ----
        qos_in = QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1, 
                            reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE,)
        self.cb = ReentrantCallbackGroup()
        self.sub_best = self.create_subscription(PoseStamped, self.best_topic, self._on_best, qos_in, callback_group=self.cb)
        self.sub_candidates = self.create_subscription(PoseArray, self.candidates_topic, self._on_candidates, qos_in, callback_group=self.cb)
        self.sub_step = self.create_subscription(TimeMsg, self.step_stamp_topic, self._on_step_stamp, 10, callback_group=self.cb)
        self.sub_pose = self.create_subscription(PoseStamped, self.pursuer_pose_topic, self._on_pursuer_pose, qos_in, callback_group=self.cb)

        # ---- Service ----
        self.srv = self.create_service(Trigger, self.srv_name, self._on_spawn, callback_group=self.cb)
        self.srv_random = self.create_service(Trigger, self.srv_random_name, self._on_spawn_random, callback_group=self.cb)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"🌐 World: {self.world_name}, Entity: {self.entity_name}")
        self.get_logger().info(f"📍 TF static: {self.optical_frame} <- {self.base_frame}")
        self.get_logger().info(f"📡 Listening to: {self.step_stamp_topic}, {self.best_topic}, {self.candidates_topic}, {self.pursuer_pose_topic}")
        self.get_logger().info(f"🔧 Service: {self.srv_name} (best), {self.srv_random_name} (random)")
        self.get_logger().info("----------------------------------------------------------------")

    def _on_best(self, msg: PoseStamped):
        if msg is None:
            return
        with self._cv:
            self._best_msg = msg
            self._best_seq += 1
            self._cv.notify_all()

    def _on_candidates(self, msg: PoseArray):
        if msg is None:
            return
        with self._cv:
            self._candidates_msg = msg
            self._candidates_seq += 1
            self._cv.notify_all()

    def _on_step_stamp(self, msg: TimeMsg):
        if msg is None:
            return
        with self._cv:
            self._last_step_stamp = msg
            self._cv.notify_all()

    def _on_pursuer_pose(self, msg: PoseStamped):
        if msg is None:
            return
        with self._cv:
            self._last_pursuer_pose = msg
            self._pose_seq += 1
            self._cv.notify_all()

    def _wait_for_best(self, start_seq: int, deadline: float):
        with self._cv:
            while rclpy_ok() and self._best_seq <= start_seq:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return None
                self._cv.wait(timeout=min(0.05, remaining))
            return self._best_msg

    def _wait_for_candidates(self, start_seq: int, deadline: float):
        with self._cv:
            while rclpy_ok() and self._candidates_seq <= start_seq:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return None
                self._cv.wait(timeout=min(0.05, remaining))
            return self._candidates_msg

    def _lookup_static_cam_from_base(self):
        timeout = Duration(seconds=float(self.tf_lookup_timeout_sec))
        return self.tf_buffer.lookup_transform(
            self.optical_frame, self.base_frame, rclpy_Time(), timeout=timeout
        )

    def _cam_pose_to_base_pose(self, cam_pose_msg: PoseStamped, tf_cam_base):
        pc = cam_pose_msg.pose.position
        qc = cam_pose_msg.pose.orientation
        q_WC = (float(qc.x), float(qc.y), float(qc.z), float(qc.w))

        t = tf_cam_base.transform.translation
        r = tf_cam_base.transform.rotation
        t_CB = (float(t.x), float(t.y), float(t.z))
        q_CB = (float(r.x), float(r.y), float(r.z), float(r.w))

        R_WC = quat_to_rot_np(*q_WC)
        dt = rot_apply(R_WC, t_CB)

        pose = Pose()
        pose.position.x = float(pc.x) + float(dt[0])
        pose.position.y = float(pc.y) + float(dt[1])
        pose.position.z = float(pc.z) + float(dt[2])

        q_WB = quat_mul(q_WC, q_CB)
        n = math.sqrt(q_WB[0] ** 2 + q_WB[1] ** 2 + q_WB[2] ** 2 + q_WB[3] ** 2)
        if n > 1e-12:
            q_WB = (q_WB[0] / n, q_WB[1] / n, q_WB[2] / n, q_WB[3] / n)
        pose.orientation.x = q_WB[0]
        pose.orientation.y = q_WB[1]
        pose.orientation.z = q_WB[2]
        pose.orientation.w = q_WB[3]
        return pose

    def _spawn_from_cam_pose(self, cam_pose: PoseStamped, resp):
        try:
            tf_cam_base = self._lookup_static_cam_from_base()
        except Exception as e:
            resp.success = False
            resp.message = f"❌ TF lookup failed: {self.optical_frame} <- {self.base_frame}: {e}"
            return resp

        try:
            base_pose = self._cam_pose_to_base_pose(cam_pose, tf_cam_base)
        except Exception as e:
            resp.success = False
            resp.message = f"❌ Pose conversion failed: {e}"
            return resp

        try:
            with self._cv:
                cand = self._candidates_msg
                step_stamp = self._last_step_stamp
            if cand is not None and step_stamp is not None and len(cand.poses) > 0:
                if time_tuple(cand.header.stamp) >= time_tuple(step_stamp):
                    d2s = [dist2_xy(p, base_pose) for p in cand.poses]
                    i_min = int(min(range(len(d2s)), key=lambda i: d2s[i]))
                    d_min = math.sqrt(float(d2s[i_min]))
                    if d_min > 0.05:
                        self.get_logger().warn(
                            f"⚠️ spawn(best): converted base pose is far from nearest candidate "
                            f"(d={d_min:.3f}m idx={i_min} N={len(d2s)}). "
                            "This may indicate a frame mismatch for /nbv/best_candidate."
                        )
                    else:
                        self.get_logger().info(
                            f"✅ spawn(best): converted base pose matches candidate idx={i_min} (d={d_min:.3f}m)"
                        )
        except Exception:
            pass

        req_gz = SetEntityPose.Request()
        req_gz.entity.name = self.entity_name
        req_gz.entity.type = Entity.MODEL
        req_gz.pose = base_pose

        fut = self._client.call_async(req_gz)
        gz_deadline = time.time() + 2.0
        while rclpy_ok() and (not fut.done()) and time.time() < gz_deadline:
            time.sleep(0.01)

        if not fut.done():
            resp.success = False
            resp.message = "❌ Gazebo set_pose timed out"
            return resp

        try:
            res = fut.result()
            if res is None:
                resp.success = False
                resp.message = "❌ Gazebo set_pose returned None"
                return resp
            if not res.success:
                resp.success = False
                resp.message = "❌ Gazebo rejected set_pose (success=False)"
                return resp
        except Exception as e:
            resp.success = False
            resp.message = f"❌ Gazebo set_pose call failed: {e}"
            return resp

        desired = base_pose
        pose_deadline = time.time() + float(self.pose_match_timeout_sec)
        tol2 = float(self.pose_match_tol_m) ** 2

        with self._cv:
            start_pose_seq = self._pose_seq

        ok = False
        last_msg = None
        while rclpy_ok() and time.time() < pose_deadline:
            with self._cv:
                if self._pose_seq <= start_pose_seq:
                    self._cv.wait(timeout=0.05)
                msg = self._last_pursuer_pose
                last_msg = msg
            if msg is None:
                continue
            if dist2_xy(msg.pose, desired) <= tol2:
                ok = True
                break

        if not ok:
            resp.success = False
            if last_msg is not None:
                dx = float(last_msg.pose.position.x) - float(desired.position.x)
                dy = float(last_msg.pose.position.y) - float(desired.position.y)
                resp.message = f"❌ Timed out waiting for pursuer to reach commanded pose (dx={dx:.3f} dy={dy:.3f})"
            else:
                resp.message = "❌ Timed out waiting for pursuer to reach commanded pose"
            return resp

        resp.success = True
        resp.message = "✅ Pursuer teleported and confirmed."
        return resp

    def _spawn_from_base_pose(self, base_pose: Pose, resp):
        req_gz = SetEntityPose.Request()
        req_gz.entity.name = self.entity_name
        req_gz.entity.type = Entity.MODEL
        req_gz.pose = base_pose

        fut = self._client.call_async(req_gz)
        gz_deadline = time.time() + 2.0
        while rclpy_ok() and (not fut.done()) and time.time() < gz_deadline:
            time.sleep(0.01)

        if not fut.done():
            resp.success = False
            resp.message = "❌ Gazebo set_pose timed out"
            return resp

        try:
            res = fut.result()
            if res is None:
                resp.success = False
                resp.message = "❌ Gazebo set_pose returned None"
                return resp
            if not res.success:
                resp.success = False
                resp.message = "❌ Gazebo rejected set_pose (success=False)"
                return resp
        except Exception as e:
            resp.success = False
            resp.message = f"❌ Gazebo set_pose call failed: {e}"
            return resp

        desired = base_pose
        pose_deadline = time.time() + float(self.pose_match_timeout_sec)
        tol2 = float(self.pose_match_tol_m) ** 2

        with self._cv:
            start_pose_seq = self._pose_seq

        ok = False
        while rclpy_ok() and time.time() < pose_deadline:
            with self._cv:
                if self._pose_seq <= start_pose_seq:
                    self._cv.wait(timeout=0.05)
                msg = self._last_pursuer_pose
            if msg is None:
                continue
            if dist2_xy(msg.pose, desired) <= tol2:
                ok = True
                break

        if not ok:
            resp.success = False
            resp.message = "❌ Timed out waiting for pursuer to reach commanded pose"
            return resp

        resp.success = True
        resp.message = "✅ Pursuer teleported and confirmed."
        return resp

    def _on_spawn(self, req, resp):
        if not self._client.service_is_ready():
            if not self._client.wait_for_service(timeout_sec=1.0):
                resp.success = False
                resp.message = f"Gazebo {self._gz_service_name} not ready"
                return resp

        with self._cv:
            step_stamp = self._last_step_stamp
        if step_stamp is None:
            resp.success = False
            resp.message = f"❌ No step token received yet on {self.step_stamp_topic}"
            return resp
        self.get_logger().info(f"🟣 TOKEN spawn(best) request: step_stamp={stamp_str(step_stamp)}")

        deadline = time.time() + float(self.wait_for_candidate_sec)
        with self._cv:
            best = self._best_msg
            start_seq = self._best_seq
        if best is None or time_tuple(best.header.stamp) < time_tuple(step_stamp):
            best = self._wait_for_best(start_seq, deadline)
        if best is None:
            resp.success = False
            resp.message = f"❌ Timed out waiting for best candidate on {self.best_topic}"
            return resp
        if time_tuple(best.header.stamp) < time_tuple(step_stamp):
            resp.success = False
            resp.message = "❌ Best candidate is older than step token"
            return resp
        self.get_logger().info(
            f"🟣 TOKEN spawn(best) check: step_stamp={stamp_str(step_stamp)} best_stamp={stamp_str(best.header.stamp)}"
        )

        out = self._spawn_from_cam_pose(best, resp)
        if out.success:
            out.message = "✅ Pursuer teleported to best candidate and confirmed."
        return out

    def _on_spawn_random(self, req, resp):
        if not self._client.service_is_ready():
            if not self._client.wait_for_service(timeout_sec=1.0):
                resp.success = False
                resp.message = f"Gazebo {self._gz_service_name} not ready"
                return resp

        deadline = time.time() + float(self.wait_for_candidate_sec)

        with self._cv:
            step_stamp = self._last_step_stamp
            cand = self._candidates_msg
            start_seq = self._candidates_seq

        if step_stamp is None:
            resp.success = False
            resp.message = f"❌ No step token received yet on {self.step_stamp_topic}"
            return resp
        self.get_logger().info(f"🟣 TOKEN spawn(random) request: step_stamp={stamp_str(step_stamp)}")
 
        if cand is None or len(cand.poses) == 0 or time_tuple(cand.header.stamp) < time_tuple(step_stamp):
            cand = self._wait_for_candidates(start_seq, deadline)
        if cand is None or len(cand.poses) == 0:
            resp.success = False
            resp.message = f"❌ Timed out waiting for candidates on {self.candidates_topic}"
            return resp

        if time_tuple(cand.header.stamp) < time_tuple(step_stamp):
            resp.success = False
            resp.message = "❌ Candidates are older than step token"
            return resp
        self.get_logger().info(
            f"🟣 TOKEN spawn(random) check: step_stamp={stamp_str(step_stamp)} cand_stamp={stamp_str(cand.header.stamp)} "
            f"N={len(cand.poses)}"
        )

        poses = list(cand.poses)

        if not self._first_random_done and self._seed_spawn_mode == "fixed":
            self._first_random_done = True
            yaw = self._seed_spawn_yaw
            cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
            base_pose = Pose()
            base_pose.position.x = self._seed_spawn_x
            base_pose.position.y = self._seed_spawn_y
            base_pose.position.z = self._seed_spawn_z
            base_pose.orientation.x = 0.0
            base_pose.orientation.y = 0.0
            base_pose.orientation.z = sy
            base_pose.orientation.w = cy
            self.get_logger().info(
                f"🎯 FIXED seed spawn: ({self._seed_spawn_x:.3f}, {self._seed_spawn_y:.3f}, "
                f"{self._seed_spawn_z:.3f}) yaw={math.degrees(yaw):.1f}°"
            )
            out = self._spawn_from_base_pose(base_pose, resp)
            if out.success:
                out.message = (
                    f"✅ Spawned pursuer to FIXED pose "
                    f"({self._seed_spawn_x:.3f}, {self._seed_spawn_y:.3f}) yaw={math.degrees(yaw):.1f}° and confirmed."
                )
            return out

        i: int
        label: str
        if not self._first_random_done and self._seed_spawn_mode == "closest_pursuer":
            pose_deadline = time.time() + max(2.0, float(self.wait_for_candidate_sec))
            pursuer_msg = None
            while rclpy_ok() and time.time() < pose_deadline:
                with self._cv:
                    if self._last_pursuer_pose is not None:
                        pursuer_msg = self._last_pursuer_pose
                if pursuer_msg is not None:
                    break
                time.sleep(0.05)
            if pursuer_msg is None:
                resp.success = False
                resp.message = f"❌ Timed out waiting for pursuer pose on {self.pursuer_pose_topic}"
                return resp
            d2s = [dist2_xy(p, pursuer_msg.pose) for p in poses]
            i = int(min(range(len(d2s)), key=lambda j: d2s[j]))
            self._first_random_done = True
            label = "CLOSEST_PURSUER"
            self.get_logger().info(
                f"🎯 CLOSEST_PURSUER seed spawn: idx={i} "
                f"d={math.sqrt(float(d2s[i])):.3f}m (XY base vs candidates)"
            )
        elif not self._first_random_done and self._seed_spawn_mode == "nearest_origin":
            dists = [float(p.position.x) ** 2 + float(p.position.y) ** 2 for p in poses]
            i = int(min(range(len(dists)), key=lambda j: dists[j]))
            self._first_random_done = True
            label = "NEAREST_ORIGIN"
        else:
            if not self._first_random_done:
                self._first_random_done = True
            i = self._rng.randrange(0, len(poses))
            label = "RANDOM"

        base_pose = poses[i]

        out = self._spawn_from_base_pose(base_pose, resp)
        if out.success:
            out.message = f"✅ Spawned pursuer to {label} candidate idx={i} and confirmed."
        return out


def main():
    rclpy.init()
    node = PursuerSpawner()
    exec = MultiThreadedExecutor(num_threads=2)
    exec.add_node(node)
    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
