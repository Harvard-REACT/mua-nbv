#!/usr/bin/env python3
"""
Testbed "closest-candidate" coordinator (dynamic mode).

Protocol:
  step token -> predictor -> generator -> pick closest candidate -> move -> capture (save)

Differences vs experiment_coordinator:
  - NO voxelize / score (but DOES capture+save)
  - Always selects the closest candidate (world-base distance) every iteration
  - Supports run_id_override (same as experiment_coordinator) so sweep outputs land under prog_cv_*_tracking_closest_*

This is useful for debugging candidate generation + motion without involving mapping.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.utilities import ok as rclpy_ok
from rclpy.duration import Duration

import tf2_ros

from std_srvs.srv import Trigger
from std_msgs.msg import String as StringMsg
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseArray, PoseStamped

from mua_nbv_common.ros_helpers import time_tuple, stamp_str, call_trigger


class ClosestCandidateCoordinator(Node):
    def __init__(self):
        super().__init__("closest_candidate_coordinator")

        # ---- Params ----
        self.declare_parameter("iteration", 10)
        self.declare_parameter("do_step", True)
        self.declare_parameter("service_timeout_sec", 60.0)
        self.declare_parameter("token_timeout_sec", 10.0)
        self.declare_parameter("retry_sleep_sec", 0.2)
        self.declare_parameter("sleep_after_predict_sec", 0.1)

        # Topics
        self.declare_parameter("input_step_stamp_topic", "/experiment/step_stamp")
        self.declare_parameter("run_id_topic", "/experiment/run_id")
        self.declare_parameter("run_id_override", "")
        self.declare_parameter("candidates_base_topic", "/nbv/candidates_world_base")
        self.declare_parameter("candidates_optical_topic", "/nbv/candidates_world_optical")
        self.declare_parameter("best_candidate_topic", "/nbv/best_candidate")

        # Services
        self.declare_parameter("step_service", "/experiment/target/advance")
        self.declare_parameter("predict_service", "/experiment/update/measurements")
        self.declare_parameter("generate_service", "/nbv/generate/candidates")
        self.declare_parameter("pursuer_service", "/experiment/pursuer/go_to_best")
        self.declare_parameter("capture_service", "/experiment/capture/cloud")

        # Capture timing (save a cloud once the pursuer arrives)
        self.declare_parameter("capture_enable", True)
        self.declare_parameter("sleep_before_capture_sec", 0.5)
        self.declare_parameter("sleep_after_capture_sec", 0.1)

        # TF for pursuer pose (closest-candidate selection)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("pursuer_base_frame", "pursuer/base_link")
        self.declare_parameter("tf_lookup_timeout_sec", 0.2)

        # ---- Read params ----
        self.iteration = int(self.get_parameter("iteration").value)
        self.do_step = bool(self.get_parameter("do_step").value)
        self.service_timeout = float(self.get_parameter("service_timeout_sec").value)
        self.token_timeout = float(self.get_parameter("token_timeout_sec").value)
        self.retry_sleep = float(self.get_parameter("retry_sleep_sec").value)
        self.sleep_after_predict = float(self.get_parameter("sleep_after_predict_sec").value)

        self.step_stamp_topic = str(self.get_parameter("input_step_stamp_topic").value)
        self.run_id_topic = str(self.get_parameter("run_id_topic").value)
        self.run_id_override = str(self.get_parameter("run_id_override").value).strip()
        self.cand_base_topic = str(self.get_parameter("candidates_base_topic").value)
        self.cand_opt_topic = str(self.get_parameter("candidates_optical_topic").value)
        self.best_candidate_topic = str(self.get_parameter("best_candidate_topic").value)

        self.step_srv = str(self.get_parameter("step_service").value)
        self.predict_srv = str(self.get_parameter("predict_service").value)
        self.generate_srv = str(self.get_parameter("generate_service").value)
        self.pursuer_srv = str(self.get_parameter("pursuer_service").value)
        self.capture_srv = str(self.get_parameter("capture_service").value)

        self.capture_enable = bool(self.get_parameter("capture_enable").value)
        self.sleep_before_capture = float(self.get_parameter("sleep_before_capture_sec").value)
        self.sleep_after_capture = float(self.get_parameter("sleep_after_capture_sec").value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.pursuer_base_frame = str(self.get_parameter("pursuer_base_frame").value)
        self.tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # ---- Token state ----
        self._token_mtx = threading.Lock()
        self._last_step_token: Optional[Tuple[int, int]] = None
        self._token_evt = threading.Event()

        # ---- Candidate state ----
        self._cand_mtx = threading.Lock()
        self._cand_base: Optional[PoseArray] = None
        self._cand_opt: Optional[PoseArray] = None
        self._cand_seq = 0
        self._cand_evt = threading.Event()

        # ---- Subscribers / Publishers ----
        self.create_subscription(TimeMsg, self.step_stamp_topic, self._on_step_stamp, 10)
        self.create_subscription(PoseArray, self.cand_base_topic, self._on_candidates_base, 10)
        self.create_subscription(PoseArray, self.cand_opt_topic, self._on_candidates_optical, 10)

        # Latching run_id publisher (so downstream can group outputs if they want)
        qos_run = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )
        self.pub_run_id = self.create_publisher(StringMsg, self.run_id_topic, qos_run)
        self.pub_best = self.create_publisher(PoseStamped, self.best_candidate_topic, 10)

        # ---- Clients ----
        self.cli_step = self.create_client(Trigger, self.step_srv)
        self.cli_predict = self.create_client(Trigger, self.predict_srv)
        self.cli_generate = self.create_client(Trigger, self.generate_srv)
        self.cli_pursuer = self.create_client(Trigger, self.pursuer_srv)
        self.cli_capture = self.create_client(Trigger, self.capture_srv)

        self.get_logger().info("----------------------------------------------------------------")
        self.get_logger().info(f"🧭 mode=closest_candidate(iter={self.iteration} do_step={self.do_step})")
        self.get_logger().info(f"🟣 token: {self.step_stamp_topic}")
        self.get_logger().info(f"🔧 step: {self.step_srv}  predict: {self.predict_srv}")
        self.get_logger().info(f"🔧 gen:  {self.generate_srv}  move: {self.pursuer_srv}  capture: {self.capture_srv}")
        self.get_logger().info(f"📡 candidates: base={self.cand_base_topic} optical={self.cand_opt_topic}")
        self.get_logger().info(f"💾 capture_enable={self.capture_enable} sleep_before_capture_sec={self.sleep_before_capture:.2f}")
        self.get_logger().info("----------------------------------------------------------------")

    def _on_step_stamp(self, msg: TimeMsg):
        tok = time_tuple(msg)
        with self._token_mtx:
            self._last_step_token = tok
        self._token_evt.set()

    def _on_candidates_base(self, msg: PoseArray):
        with self._cand_mtx:
            self._cand_base = msg
            self._cand_seq += 1
        self._cand_evt.set()

    def _on_candidates_optical(self, msg: PoseArray):
        with self._cand_mtx:
            self._cand_opt = msg
            self._cand_seq += 1
        self._cand_evt.set()

    def _wait_for_next_token(self, prev: Optional[Tuple[int, int]], timeout_sec: float) -> Optional[Tuple[int, int]]:
        deadline = time.time() + float(timeout_sec)
        self._token_evt.clear()
        while rclpy_ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._token_mtx:
                tok = self._last_step_token
            if tok is not None and (prev is None or tok > prev):
                return tok
        return None

    def _wait_for_candidates(self, prev_seq: int, timeout_sec: float) -> bool:
        deadline = time.time() + float(timeout_sec)
        self._cand_evt.clear()
        while rclpy_ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._cand_mtx:
                seq = int(self._cand_seq)
                have = (self._cand_base is not None) and (self._cand_opt is not None)
            if have and seq > prev_seq:
                return True
        return False

    def _pursuer_world_xy(self) -> Tuple[float, float]:
        tf = self.tf_buffer.lookup_transform(
            self.world_frame,
            self.pursuer_base_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=float(self.tf_lookup_timeout_sec)),
        )
        return float(tf.transform.translation.x), float(tf.transform.translation.y)

    def _publish_closest_candidate(self) -> tuple[bool, int, float]:
        with self._cand_mtx:
            base = self._cand_base
            opt = self._cand_opt
        if base is None or opt is None or len(base.poses) == 0 or len(opt.poses) == 0:
            return (False, -1, float("inf"))
        if len(base.poses) != len(opt.poses):
            return (False, -1, float("inf"))

        try:
            px, py = self._pursuer_world_xy()
        except Exception as e:
            # If we can't get pursuer pose, fall back to last candidate deterministically.
            self.get_logger().warn(f"⚠️ TF lookup failed for pursuer pose: {e} ; using last candidate")
            idx = len(opt.poses) - 1
            msg = PoseStamped()
            msg.header.frame_id = str(opt.header.frame_id) if opt.header.frame_id else self.world_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose = opt.poses[idx]
            self.pub_best.publish(msg)
            return (True, idx, float("inf"))

        best_i = 0
        best_d2 = float("inf")
        for i, pose in enumerate(base.poses):
            dx = float(pose.position.x) - px
            dy = float(pose.position.y) - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        msg = PoseStamped()
        msg.header.frame_id = str(opt.header.frame_id) if opt.header.frame_id else self.world_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = opt.poses[best_i]
        self.pub_best.publish(msg)
        return (True, best_i, math.sqrt(best_d2))

    def run(self):
        if self.iteration < 1:
            raise RuntimeError(f"Invalid iteration={self.iteration} (must be >= 1)")

        # publish run_id (for downstream grouping / logs)
        run_id = self.run_id_override or time.strftime("%Y%m%d_%H%M%S", time.localtime())
        rid = StringMsg()
        rid.data = run_id
        self.pub_run_id.publish(rid)
        self.get_logger().info(f"🧾 RUN_ID publish: {run_id} -> {self.run_id_topic}")

        # wait for services
        services = [
            (self.cli_predict, "predictor"),
            (self.cli_generate, "generator"),
            (self.cli_pursuer, "pursuer"),
        ]
        if self.do_step:
            services.insert(0, (self.cli_step, "stepper"))
        if self.capture_enable:
            services.append((self.cli_capture, "capturer"))
        for (cli, nm) in services:
            if not cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"Service not ready: {nm}")

        bootstrapping = True
        prev_token = None
        completed = 0

        while rclpy_ok():
            # 1) step target (publishes new token)
            if self.do_step:
                st = call_trigger(self, self.cli_step, "stepper", self.service_timeout, require_success=False)
                self.get_logger().info(f"🔧 stepper: success={bool(st.success)} msg='{st.message}'")

            # 2) wait for token
            tok = self._wait_for_next_token(prev_token, timeout_sec=self.token_timeout)
            if tok is None:
                raise RuntimeError(f"❌ No new token on {self.step_stamp_topic} within {self.token_timeout:.1f}s")
            prev_token = tok
            token_msg = TimeMsg()
            token_msg.sec = int(tok[0])
            token_msg.nanosec = int(tok[1])
            self.get_logger().info(f"🟣 TOKEN cycle start: {stamp_str(token_msg)}")

            # 3) predictor for this token (call once)
            pred = call_trigger(self, self.cli_predict, "predictor", self.service_timeout, require_success=False)
            self.get_logger().info(f"🔧 predictor: success={bool(pred.success)} msg='{pred.message}'")

            if not pred.success:
                msg = str(pred.message)
                # Bootstrap stage: step again to accumulate measurements.
                if "Bootstrapping" in msg:
                    bootstrapping = True
                    self.get_logger().info("🔄 BOOTSTRAP: predictor not ready; stepping again.")
                    continue
                # If we already processed this token, that's OK: proceed to generator.
                if "already processed" in msg:
                    pass
                else:
                    self.get_logger().warn(f"⚠️ predictor failed (non-bootstrap): {msg} ; stepping again.")
                    continue

            if bootstrapping:
                self.get_logger().info("✅ Predictor READY -> entering closest-candidate loop")
                bootstrapping = False

            time.sleep(max(0.0, self.sleep_after_predict))

            # 4) generator retry loop (do NOT re-call predictor for the same token)
            with self._cand_mtx:
                cand_seq0 = int(self._cand_seq)
            gen_deadline = time.time() + float(self.service_timeout)
            gen = None
            while rclpy_ok() and time.time() < gen_deadline:
                gen = call_trigger(self, self.cli_generate, "generator", self.service_timeout, require_success=False)
                self.get_logger().info(f"🔧 generator: success={bool(gen.success)} msg='{gen.message}'")
                if gen.success:
                    break
                if "Prediction stamp is older than step token" in str(gen.message):
                    time.sleep(max(0.0, self.retry_sleep))
                    continue
                self.get_logger().warn(f"⚠️ generator failed: {gen.message} ; stepping again.")
                gen = None
                break

            if gen is None or not getattr(gen, "success", False):
                continue

            # Wait for candidates topics.
            _ = self._wait_for_candidates(cand_seq0, timeout_sec=max(0.5, self.token_timeout))

            ok, idx, dist = self._publish_closest_candidate()
            if not ok:
                self.get_logger().warn("⚠️ No candidates available; stepping again.")
                continue
            if math.isfinite(dist):
                self.get_logger().info(f"🎯 closest candidate idx={idx} dist={dist:.3f}m -> publish {self.best_candidate_topic}")
            else:
                self.get_logger().info(f"🎯 fallback candidate idx={idx} -> publish {self.best_candidate_topic}")

            # 5) move pursuer to best
            mv = call_trigger(self, self.cli_pursuer, "pursuer(go_to_best)", self.service_timeout, require_success=False)
            self.get_logger().info(f"🔧 pursuer: success={bool(mv.success)} msg='{mv.message}'")

            # 6) capture (save point cloud for later comparison)
            if self.capture_enable:
                time.sleep(max(0.0, self.sleep_before_capture))
                cap = call_trigger(self, self.cli_capture, "capturer", self.service_timeout, require_success=False)
                self.get_logger().info(f"🔧 capturer: success={bool(cap.success)} msg='{cap.message}'")
                time.sleep(max(0.0, self.sleep_after_capture))

            completed += 1
            self.get_logger().info(f"✅ Iteration {completed}/{self.iteration} complete.")

            if completed >= self.iteration:
                self.get_logger().info("✅ Completed iterations. Stopping.")
                break


def main():
    rclpy.init()
    node = ClosestCandidateCoordinator()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
