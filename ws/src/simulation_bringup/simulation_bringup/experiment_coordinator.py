import threading
import time
from typing import cast

import rclpy
from rclpy.node import Node
from rclpy.utilities import ok as rclpy_ok
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String as StringMsg
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped

from mua_nbv_common.ros_helpers import time_tuple, stamp_str, stamp_tuple_str, call_trigger


class ExperimentCoordinator(Node):
    def __init__(self):
        super().__init__("experiment_coordinator")

        # ---- Params ----
        # Mode
        self.declare_parameter("target_mode", "dynamic")  # "dynamic" or "static" 
        self.declare_parameter("iteration", 1)
        # Pipeline mode:
        # - "full": stepper->predictor->planner/capture/score/spawn (NBV)
        # - "predict_only": stepper->predictor only (fast predictor evaluation)
        self.declare_parameter("pipeline_mode", "full")
        # Optional: deterministic run id for sweeps (otherwise uses wall-clock timestamp).
        self.declare_parameter("run_id_override", "")
        # Topics
        self.declare_parameter("input_step_stamp_topic", "/experiment/step_stamp")
        # One "run id" per experiment_coordinator execution; used by other nodes
        # to group debug outputs per run.
        self.declare_parameter("run_id_topic", "/experiment/run_id") 
        self.declare_parameter("static_token_source", "pose")  # "pose" | "cloud"
        self.declare_parameter("token_pose_topic", "/sim/pursuer/pose")
        self.declare_parameter("token_cloud_topic", "/pursuer/camera/depth/color/points")
        self.static_token_source = str(self.get_parameter("static_token_source").value)
        self.token_pose_topic = str(self.get_parameter("token_pose_topic").value)

        # Services
        self.declare_parameter("step_service", "/experiment/target/advance")
        self.declare_parameter("predict_service", "/experiment/update/measurements")
        self.declare_parameter("capture_service", "/experiment/capture/cloud")
        self.declare_parameter("voxelize_service", "/nbv/update/voxels")
        self.declare_parameter("generate_service", "/nbv/generate/candidates")
        self.declare_parameter("score_service", "/nbv/generate/scores")
        self.declare_parameter("pursuer_service", "/experiment/pursuer/spawn")
        self.declare_parameter("pursuer_random_service", "/experiment/pursuer/spawn_random")
        # Timeouts
        self.declare_parameter("service_timeout_sec", 10.0)
        self.declare_parameter("step_timeout_sec", 5.0)
        self.declare_parameter("wait_for_clock_sec", 5.0)
        # Seeding behavior
        # In static mode you already do a seed random spawn once. This adds the same idea for dynamic mode:
        # once predictor is ready AND generator can succeed, do one random pursuer spawn to avoid starting from
        # the default spawn pose.
        self.declare_parameter("seed_random_spawn_dynamic", True)
        self.declare_parameter("seed_wait_cloud_after_spawn_sec", 2.0)

        # ---- Read params ----
        self.target_mode = str(self.get_parameter("target_mode").value)
        self.iteration = int(cast(int, self.get_parameter("iteration").value))
        self.pipeline_mode = str(self.get_parameter("pipeline_mode").value).strip()
        self.step_stamp_topic = str(self.get_parameter("input_step_stamp_topic").value)
        self.run_id_topic = str(self.get_parameter("run_id_topic").value)
        self.run_id_override = str(self.get_parameter("run_id_override").value).strip()
        self.token_cloud_topic = str(self.get_parameter("token_cloud_topic").value)
        self.step_srv = str(self.get_parameter("step_service").value)
        self.predict_srv = str(self.get_parameter("predict_service").value)
        self.capture_srv = str(self.get_parameter("capture_service").value)
        self.voxelize_srv = str(self.get_parameter("voxelize_service").value)
        self.generate_srv = str(self.get_parameter("generate_service").value)
        self.score_srv = str(self.get_parameter("score_service").value)
        self.pursuer_srv = str(self.get_parameter("pursuer_service").value)
        self.pursuer_random_srv = str(self.get_parameter("pursuer_random_service").value)
        self.service_timeout = float(cast(float, self.get_parameter("service_timeout_sec").value))
        self.step_timeout = float(cast(float, self.get_parameter("step_timeout_sec").value))
        self.wait_for_clock_sec = float(cast(float, self.get_parameter("wait_for_clock_sec").value))
        self.seed_random_spawn_dynamic = bool(self.get_parameter("seed_random_spawn_dynamic").value)
        self.seed_wait_cloud_after_spawn_sec = float(cast(float, self.get_parameter("seed_wait_cloud_after_spawn_sec").value))

        # ---- Token state ----
        self._token_mtx = threading.Lock()
        self._last_step_token = None  # (sec, nsec)
        self._token_event = threading.Event()

        # ---- Cloud token source state ----
        self._cloud_mtx = threading.Lock()
        self._last_cloud_stamp = None  # (sec, nsec)
        self._cloud_event = threading.Event()

        self._pose_mtx = threading.Lock()
        self._last_pose_stamp = None  # (sec, nsec)
        self._pose_event = threading.Event()

        # ---- Subscribers / Publishers ----
        self.create_subscription(TimeMsg, self.step_stamp_topic, self._on_step_stamp, 10)
        qos_token = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_step_stamp = self.create_publisher(TimeMsg, self.step_stamp_topic, qos_token)
        self.create_subscription(PointCloud2, self.token_cloud_topic, self._on_token_cloud, 10)
        self.create_subscription(PoseStamped, self.token_pose_topic, self._on_token_pose, 10)
        self.create_subscription(Float64, "/nbv/score_time_ms", self._on_score_time, 10)
        self._last_score_time_ms = 0.0
        # Latching publisher so late joiners still learn the current run_id
        qos_run = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )
        self.pub_run_id = self.create_publisher(StringMsg, self.run_id_topic, qos_run)

        # ---- Clients ----
        self.cli_step = self.create_client(Trigger, self.step_srv)
        self.cli_predict = self.create_client(Trigger, self.predict_srv)
        self.cli_capture = self.create_client(Trigger, self.capture_srv)
        self.cli_voxelize = self.create_client(Trigger, self.voxelize_srv)
        self.cli_generate = self.create_client(Trigger, self.generate_srv)
        self.cli_score = self.create_client(Trigger, self.score_srv)
        self.cli_pursuer = self.create_client(Trigger, self.pursuer_srv)
        self.cli_pursuer_random = self.create_client(Trigger, self.pursuer_random_srv)

    def _on_step_stamp(self, msg: TimeMsg):
        tok = time_tuple(msg)
        with self._token_mtx:
            self._last_step_token = tok
        self._token_event.set()

    def _on_token_cloud(self, msg: PointCloud2):
        st = msg.header.stamp
        tup = (int(st.sec), int(st.nanosec))
        with self._cloud_mtx:
            self._last_cloud_stamp = tup
        self._cloud_event.set()

    def _on_token_pose(self, msg: PoseStamped):
        st = msg.header.stamp
        tup = (int(st.sec), int(st.nanosec))
        with self._pose_mtx:
            self._last_pose_stamp = tup
        self._pose_event.set()

    def _on_score_time(self, msg: Float64):
        self._last_score_time_ms = float(msg.data)

    def wait_for_next_token(self, prev_token, timeout_sec):
        deadline = time.time() + timeout_sec
        self._token_event.clear()

        while rclpy_ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._token_mtx:
                tok = self._last_step_token
            if tok is not None and (prev_token is None or tok > prev_token):
                return tok

        return None

    def wait_for_next_cloud_stamp(self, prev_token, timeout_sec):
        deadline = time.time() + float(timeout_sec)
        self._cloud_event.clear()
        while rclpy_ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._cloud_mtx:
                cs = self._last_cloud_stamp
            if cs is not None and (prev_token is None or cs > prev_token):
                return cs
        return None

    def get_last_cloud_stamp(self):
        with self._cloud_mtx:
            return self._last_cloud_stamp

    def wait_for_next_pose_stamp(self, prev_token, timeout_sec):
        deadline = time.time() + float(timeout_sec)
        self._pose_event.clear()
        while rclpy_ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self._pose_mtx:
                ps = self._last_pose_stamp
            if ps is not None and (prev_token is None or ps > prev_token):
                return ps
        return None

    def get_last_pose_stamp(self):
        with self._pose_mtx:
            return self._last_pose_stamp

    def log_trigger_result(self, name: str, res: Trigger.Response, token: TimeMsg | None = None):
        tok_str = stamp_str(token) if token is not None else "None"
        self.get_logger().info(f"🔧 {name}: success={bool(res.success)} token={tok_str} msg='{res.message}'")

    def wait_for_clock(self, timeout_sec: float) -> bool: 
        deadline = time.time() + float(timeout_sec)
        while rclpy_ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.get_clock().now().nanoseconds > 0:
                return True
        return False

    def run(self):
        if self.target_mode not in ("dynamic", "static"):
            raise RuntimeError(f"Invalid target_mode='{self.target_mode}' (expected 'dynamic' or 'static')")
        if self.iteration < 1:
            raise RuntimeError(f"Invalid iteration={self.iteration} (must be >= 1)")
        if self.pipeline_mode not in ("full", "predict_only"):
            raise RuntimeError(f"Invalid pipeline_mode='{self.pipeline_mode}' (expected 'full' or 'predict_only')")
 
        run_id = self.run_id_override or time.strftime("%Y%m%d_%H%M%S", time.localtime())
        msg = StringMsg()
        msg.data = run_id
        self.pub_run_id.publish(msg)
        self.get_logger().info(f"🧾 RUN_ID publish: {run_id} -> {self.run_id_topic}")

        if bool(self.get_parameter("use_sim_time").value):
            if not self.wait_for_clock(self.wait_for_clock_sec):
                raise RuntimeError(f"❌ Timed out waiting for /clock to become active (>{0}s). ")

        # wait for services (mode-dependent)
        do_planning = (self.pipeline_mode == "full")

        services = []
        if self.target_mode == "dynamic":
            services = [(self.cli_step, "stepper"), (self.cli_predict, "predictor")]
            if do_planning:
                services += [
                    (self.cli_generate, "generator"),
                    (self.cli_capture, "capturer"),
                    (self.cli_voxelize, "voxelizer"),
                    (self.cli_score, "scorer"),
                    (self.cli_pursuer, "spawner"),
                    (self.cli_pursuer_random, "spawner_random"),
                ]
        else:
            services = [
                (self.cli_generate, "generator"),
                (self.cli_capture, "capturer"),
                (self.cli_voxelize, "voxelizer"),
                (self.cli_score, "scorer"),
                (self.cli_pursuer, "spawner"),
                (self.cli_pursuer_random, "spawner_random"),
            ]

        for (cli, nm) in services:
            if not cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"Service not ready: {nm}")

        bootstrapping = (self.target_mode == "dynamic")
        did_seed_dynamic = False
        prev_token = None
        completed_iters = 0
 
        if self.target_mode == "static":
            tmp = self.get_clock().now().to_msg()
            self.pub_step_stamp.publish(tmp)
            tmp_tok = time_tuple(tmp)
            self.get_logger().info(f"🟣 TOKEN publish (static seed tmp): {stamp_str(tmp)}")

            gen0 = call_trigger(self, self.cli_generate, "generator(seed)", self.service_timeout, require_success=False)
            self.log_trigger_result("generator(seed)", gen0, token=tmp)
            if not gen0.success: raise RuntimeError(f"generator(seed) failed: {gen0.message}")

            sr0 = call_trigger(self, self.cli_pursuer_random, "spawner_random(seed)", self.service_timeout, require_success=True)  
            self.log_trigger_result("spawner_random(seed)", sr0, token=tmp)

            if self.static_token_source == "pose":
                ps0 = self.wait_for_next_pose_stamp(None, timeout_sec=self.step_timeout)
                if ps0 is None:
                    raise RuntimeError(f"❌ Timed out waiting for token pose on {self.token_pose_topic}")
                prev_token = ps0
                self.get_logger().info(f"🟣 TOKEN seed complete (pose): {stamp_tuple_str(prev_token)}")
            else:
                cs0 = self.wait_for_next_cloud_stamp(tmp_tok, timeout_sec=self.step_timeout)
                if cs0 is None:
                    raise RuntimeError(f"❌ Timed out waiting for token cloud on {self.token_cloud_topic}")
                prev_token = cs0
                self.get_logger().info(f"🟣 TOKEN seed complete (cloud): {stamp_tuple_str(prev_token)}")

        while rclpy_ok():
            if self.target_mode == "dynamic":
                # 1) Stepper (initiate motion), retry if previous step pending
                for _retry in range(20):
                    _resp = call_trigger(self, self.cli_step, "stepper", self.service_timeout, require_success=False)
                    if _resp.success:
                        break
                    time.sleep(0.15)
                else:
                    raise RuntimeError(f"stepper failed after retries: {_resp.message}")
                time.sleep(0.1)

                # 2) Wait for arrival token
                token = self.wait_for_next_token(prev_token, self.step_timeout)
                if token is None: raise RuntimeError(f"❌ No new token received on {self.step_stamp_topic} within {self.step_timeout}s")
                prev_token = token

                # 3) Predictor (must correspond to this token)
                pred = call_trigger(self, self.cli_predict, "predictor", self.service_timeout, require_success=False)

                if not pred.success:
                    bootstrapping = True
                    self.get_logger().info(f"🔄 BOOTSTRAP: {pred.message}")
                    if not do_planning:
                        time.sleep(0.1)
                    continue

                if bootstrapping:
                    self.get_logger().info("✅ Predictor READY -> Entering planning phase")
                    bootstrapping = False

                if not do_planning:
                    completed_iters += 1
                    self.get_logger().info(
                        f"✅ Predict-only step {completed_iters}/{self.iteration} completed @token={stamp_tuple_str(prev_token)}"
                    )
                    if completed_iters >= self.iteration:
                        self.get_logger().info(f"✅ Completed predict-only iterations: {completed_iters}/{self.iteration}. Stopping.")
                        break
                    time.sleep(0.1)
                    continue
            else:  
                if prev_token is None:
                    raise RuntimeError("Static mode internal error: prev_token is None after seeding.")
                stamp = TimeMsg()
                stamp.sec = int(prev_token[0])
                stamp.nanosec = int(prev_token[1])
                self.pub_step_stamp.publish(stamp)
                self.get_logger().info(f"🟣 TOKEN publish (static post-spawn cloud): {stamp_str(stamp)}")

            token_msg = TimeMsg()
            token_msg.sec = int(prev_token[0])
            token_msg.nanosec = int(prev_token[1])
            self.get_logger().info(f"🟣 TOKEN cycle start: {stamp_str(token_msg)} (mode={self.target_mode})")

            gen = call_trigger(self, self.cli_generate, "generator", self.service_timeout, require_success=False)
            self.log_trigger_result("generator", gen, token=token_msg)
 
            if (
                self.target_mode == "dynamic"
                and (not bootstrapping)
                and self.seed_random_spawn_dynamic
                and (not did_seed_dynamic)
            ):
                if not gen.success:
                    self.get_logger().info(f"🔄 Waiting for generator before dynamic seed @token={stamp_str(token_msg)}: {gen.message}")
                    continue

                marker = self.get_last_cloud_stamp()
                sr = call_trigger(
                    self,
                    self.cli_pursuer_random,
                    "spawner_random(seed_dynamic)",
                    self.service_timeout,
                    require_success=True,
                )
                self.log_trigger_result("spawner_random(seed_dynamic)", sr, token=token_msg)

                wait_s = max(0.0, float(self.seed_wait_cloud_after_spawn_sec))
                if wait_s > 0.0:
                    cs_after = self.wait_for_next_cloud_stamp(marker, timeout_sec=wait_s)
                    if cs_after is None:
                        self.get_logger().warn(f"⚠️ dynamic seed: no post-spawn cloud within {wait_s:.2f}s on {self.token_cloud_topic}")
                    else:
                        self.get_logger().info(f"🟣 dynamic seed: post-spawn cloud={stamp_tuple_str(cs_after)}")

                did_seed_dynamic = True
                self.get_logger().info("🧪 dynamic seed complete; restarting loop for next token.")
                continue

            if not gen.success:
                self.get_logger().info(f"🔄 Waiting for generator @token={stamp_str(token_msg)}: {gen.message}")
                continue

            # Capture
            cap = call_trigger(self, self.cli_capture, "capturer", self.service_timeout, require_success=True)
            time.sleep(0.1)
            self.log_trigger_result("capturer", cap, token=token_msg) 
            vox_deadline = time.time() + float(self.service_timeout)
            vox = None
            while rclpy_ok() and time.time() < vox_deadline: 
                req = Trigger.Request()
                fut = self.cli_voxelize.call_async(req)

                while rclpy_ok() and time.time() < vox_deadline and not fut.done():
                    rclpy.spin_once(self, timeout_sec=0.05)

                if not fut.done(): break

                vox = fut.result()
                if vox is None: raise RuntimeError("voxelizer returned None")

                if vox.success: break

                if "Inputs not ready for voxel update" in str(vox.message):
                    time.sleep(0.05)
                    continue

                break

            if vox is None or not vox.success:
                fail_msg = vox.message if vox is not None else "timeout"
                self.get_logger().error(
                    f"⏰ voxelizer failed/timed out: {fail_msg} -> random spawn fallback"
                )
                sr = call_trigger(
                    self, self.cli_pursuer_random,
                    "spawner_random(fallback_vox)", self.service_timeout,
                    require_success=True,
                )
                self.log_trigger_result("spawner_random(fallback_vox)", sr, token=token_msg)
                completed_iters += 1
                if completed_iters >= self.iteration:
                    self.get_logger().info(f"✅ Completed iterations: {completed_iters}/{self.iteration}. Stopping.")
                    break
                self.get_logger().info(f"✅ Iteration {completed_iters}/{self.iteration} completed (vox fallback). Advancing...")
                continue

            self.log_trigger_result("voxelizer", vox, token=token_msg)
            # Score — catch timeouts (e.g. CGAL hang on degenerate clusters)
            scorer_failed_recoverable = False
            scorer_fail_msg = ""
            try:
                sc = call_trigger(self, self.cli_score, "scorer", self.service_timeout, require_success=False)
                self.log_trigger_result("scorer", sc, token=token_msg)
                if not sc.success:
                    scorer_fail_msg = str(sc.message)
                    if ("no ellipsoids fit" in scorer_fail_msg.lower()) or \
                       ("scoring skipped" in scorer_fail_msg.lower()):
                        scorer_failed_recoverable = True
                    else:
                        raise RuntimeError(f"scorer failed: {scorer_fail_msg}")
            except RuntimeError as e:
                err_str = str(e)
                if "timed out" in err_str.lower():
                    self.get_logger().error(
                        f"⏰ scorer timed out ({self.service_timeout:.0f}s) -> random spawn fallback"
                    )
                    scorer_failed_recoverable = True
                    scorer_fail_msg = err_str
                else:
                    raise

            if scorer_failed_recoverable:
                self.get_logger().warn(
                    f"⚠️ scorer failed (recoverable): {scorer_fail_msg} -> random spawn fallback"
                )
                sr = call_trigger(
                    self,
                    self.cli_pursuer_random,
                    "spawner_random(fallback_scorer)",
                    self.service_timeout,
                    require_success=True,
                )
                self.log_trigger_result("spawner_random(fallback)", sr, token=token_msg)

                if self.target_mode == "static":
                    marker_for_next = (
                        self.get_last_pose_stamp() if self.static_token_source == "pose" else self.get_last_cloud_stamp()
                    )
                    if self.static_token_source == "pose":
                        ps_next = self.wait_for_next_pose_stamp(marker_for_next, timeout_sec=self.step_timeout)
                        if ps_next is None:
                            raise RuntimeError(
                                f"❌ Timed out waiting for post-spawn token pose on {self.token_pose_topic}"
                            )
                        prev_token = ps_next
                        self.get_logger().info(f"🟣 TOKEN next (post-spawn pose): {stamp_tuple_str(prev_token)}")
                    else:
                        cs_next = self.wait_for_next_cloud_stamp(marker_for_next, timeout_sec=self.step_timeout)
                        if cs_next is None:
                            raise RuntimeError(
                                f"❌ Timed out waiting for post-spawn token cloud on {self.token_cloud_topic}"
                            )
                        prev_token = cs_next
                        self.get_logger().info(f"🟣 TOKEN next (post-spawn cloud): {stamp_tuple_str(prev_token)}")

                completed_iters += 1
                if completed_iters >= self.iteration:
                    self.get_logger().info(f"✅ Completed iterations: {completed_iters}/{self.iteration}. Stopping.")
                    break
                self.get_logger().info(f"✅ Iteration {completed_iters}/{self.iteration} completed (fallback random spawn). Advancing...")
                continue

            # Teleport pursuer to best candidate 
            marker_for_next = None
            if self.target_mode == "static":
                marker_for_next = (self.get_last_pose_stamp() if self.static_token_source == "pose" else self.get_last_cloud_stamp())
            sp = call_trigger(self, self.cli_pursuer, "spawner", self.service_timeout, require_success=True)
            self.log_trigger_result("spawner(best)", sp, token=token_msg)

            if self.target_mode == "static":
                if self.static_token_source == "pose":
                    ps_next = self.wait_for_next_pose_stamp(marker_for_next, timeout_sec=self.step_timeout)
                    if ps_next is None:
                        raise RuntimeError(f"❌ Timed out waiting for post-spawn token pose on {self.token_pose_topic}")
                    prev_token = ps_next
                    self.get_logger().info(f"🟣 TOKEN next (post-spawn pose): {stamp_tuple_str(prev_token)}")
                else:
                    cs_next = self.wait_for_next_cloud_stamp(marker_for_next, timeout_sec=self.step_timeout)
                    if cs_next is None:
                        raise RuntimeError(f"❌ Timed out waiting for post-spawn token cloud on {self.token_cloud_topic}")
                    prev_token = cs_next
                    self.get_logger().info(f"🟣 TOKEN next (post-spawn cloud): {stamp_tuple_str(prev_token)}")
            completed_iters += 1

            self.get_logger().info(
                f"⏱️ score_time_ms={self._last_score_time_ms:.1f} "
                f"(iter {completed_iters}/{self.iteration})"
            )

            if completed_iters >= self.iteration:
                self.get_logger().info(f"✅ Completed iterations: {completed_iters}/{self.iteration}. Stopping.")
                break
        
            self.get_logger().info(f"✅ Iteration {completed_iters}/{self.iteration} completed. Advancing...")


def main():
    rclpy.init()
    node = ExperimentCoordinator()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
