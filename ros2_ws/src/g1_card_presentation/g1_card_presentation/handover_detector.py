"""Detect a recipient entering the presented-card transfer region."""

from collections import deque
import json
import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger


def depth_statistics(depth, roi):
    """Return median and near-25 depth in millimetres for valid pixels."""
    x, y, width, height = roi
    region = depth[y:y + height, x:x + width]
    valid = region[(region > 100.0) & (region < 4000.0)]
    if not valid.size:
        return None, None
    return float(np.median(valid)), float(np.percentile(valid, 25))


def rgb_change_ratio(frame, baseline, roi, pixel_delta):
    """Return the changed-pixel ratio inside an RGB region."""
    import cv2

    x, y, width, height = roi
    current = frame[y:y + height, x:x + width]
    reference = baseline[y:y + height, x:x + width]
    if current.size == 0 or current.shape != reference.shape:
        return 0.0
    difference = cv2.absdiff(current, reference)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray >= pixel_delta)) / gray.size


class CardHandoverDetector(Node):
    """Trigger handover from fresh RGB and depth observations."""

    def __init__(self):
        super().__init__("card_handover_detector")

        defaults = {
            "enabled": False,
            "state_topic": "/event/card/state",
            "trigger_service": "/event/card/handover",
            "rgb_topic": "/g1/camera/rgb/compressed",
            "depth_topic": "/g1/camera/depth/image_raw",
            "debug_topic": "/event/card/handover/debug/compressed",
            "poll_rate_hz": 10.0,
            "rgb_stale_after_s": 0.25,
            "depth_stale_after_s": 0.5,
            "baseline_delay_s": 0.5,
            "required_hits": 2,
            "window_size": 4,
            "gate_memory_s": 0.5,
            "waiting_hand_required_hits": 7,
            "waiting_hand_window_size": 10,
            "rgb_roi_x": 0,
            "rgb_roi_y": 0,
            "rgb_roi_width": 1,
            "rgb_roi_height": 1,
            "depth_roi_x": 0,
            "depth_roi_y": 0,
            "depth_roi_width": 1,
            "depth_roi_height": 1,
            "rgb_pixel_delta": 25.0,
            "rgb_changed_ratio": 0.08,
            "depth_approach_mm": 80.0,
            "depth_max_mm": 1000.0,
            "log_interval_s": 1.0,
            "publish_debug_image": True,
            "show_calibration_windows": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def value(name):
            return self.get_parameter(name).value

        self.enabled = bool(value("enabled"))
        self.rgb_stale_after_s = float(value("rgb_stale_after_s"))
        self.depth_stale_after_s = float(value("depth_stale_after_s"))
        self.baseline_delay_s = float(value("baseline_delay_s"))
        self.required_hits = int(value("required_hits"))
        self.window_size = int(value("window_size"))
        self.gate_memory_s = float(value("gate_memory_s"))
        self.waiting_hand_required_hits = int(
            value("waiting_hand_required_hits"))
        self.waiting_hand_window_size = int(
            value("waiting_hand_window_size"))
        self.rgb_pixel_delta = float(value("rgb_pixel_delta"))
        self.rgb_changed_ratio = float(value("rgb_changed_ratio"))
        self.depth_approach_mm = float(value("depth_approach_mm"))
        self.depth_max_mm = float(value("depth_max_mm"))
        self.log_interval_s = float(value("log_interval_s"))
        self.publish_debug_image = bool(value("publish_debug_image"))
        self.show_windows = bool(value("show_calibration_windows"))
        self.rgb_roi = tuple(int(value(f"rgb_roi_{part}")) for part in (
            "x", "y", "width", "height"))
        self.depth_roi = tuple(int(value(f"depth_roi_{part}")) for part in (
            "x", "y", "width", "height"))

        if not 1 <= self.required_hits <= self.window_size:
            raise ValueError("required_hits must be between 1 and window_size")
        if self.gate_memory_s <= 0.0:
            raise ValueError("gate_memory_s must be positive")
        if not (
            1 <= self.waiting_hand_required_hits
            <= self.waiting_hand_window_size
        ):
            raise ValueError(
                "waiting_hand_required_hits must be between 1 and "
                "waiting_hand_window_size")
        if min(self.rgb_roi[2:] + self.depth_roi[2:]) < 1:
            raise ValueError("RGB and depth ROI sizes must be positive")

        self.armed = False
        self.fired = False
        self.pending = False
        self.armed_at = None
        self.baseline_rgb = None
        self.baseline_depth_near25 = None
        self.hits = deque(maxlen=self.window_size)
        self.waiting_hand_hits = deque(
            maxlen=self.waiting_hand_window_size)
        self.rgb = None
        self.depth = None
        self.rgb_at = None
        self.depth_at = None
        self.last_log_at = 0.0
        self.last_rgb_hit_at = None
        self.last_depth_hit_at = None
        self.drag_start = {}

        self.client = self.create_client(
            Trigger, str(value("trigger_service")))
        self.debug_publisher = self.create_publisher(
            CompressedImage, str(value("debug_topic")), 1)
        self.create_subscription(
            String, str(value("state_topic")), self.state_callback, 10)
        self.create_subscription(
            CompressedImage, str(value("rgb_topic")), self.rgb_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, str(value("depth_topic")), self.depth_callback,
            qos_profile_sensor_data)
        self.create_timer(1.0 / float(value("poll_rate_hz")), self.poll)

        if self.show_windows:
            self._start_windows()
        self.get_logger().warning(
            "Card handover detector ready; "
            f"physical trigger enabled={self.enabled}")

    def state_callback(self, message):
        """Arm only while the card is gripped and the arm is presented."""
        try:
            state = json.loads(message.data)
            armed = (
                state["hand_state"] == "GRIPPED"
                and state["arm_state"] == "PRESENTED"
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"invalid card state: {exc}")
            return

        if armed and not self.armed:
            self.fired = False
            self.pending = False
            self.armed_at = time.monotonic()
            self.baseline_rgb = None
            self.baseline_depth_near25 = None
            self.hits.clear()
            self.waiting_hand_hits.clear()
            self.last_rgb_hit_at = None
            self.last_depth_hit_at = None
            self.get_logger().info("card handover detector armed")
        elif not armed and self.armed:
            self.hits.clear()
            self.waiting_hand_hits.clear()
            self.last_rgb_hit_at = None
            self.last_depth_hit_at = None
            self.get_logger().info("card handover detector disarmed")
        self.armed = armed

    def rgb_callback(self, message):
        """Decode and retain the newest bridge RGB frame."""
        import cv2

        self.rgb = cv2.imdecode(
            np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.rgb_at = time.monotonic() if self.rgb is not None else None

    def depth_callback(self, message):
        """Decode and retain the newest bridge depth frame."""
        if message.encoding != "16UC1":
            return
        values = np.frombuffer(message.data, dtype=np.uint16)
        if values.size != int(message.height) * int(message.width):
            return
        self.depth = values.reshape(message.height, message.width).astype(
            np.float32)
        self.depth_at = time.monotonic()

    def poll(self):
        """Evaluate the configured RGB and depth gates."""
        now = time.monotonic()
        if self.show_windows:
            self._show_calibration()
        if not self.armed or self.fired or self.pending:
            return
        if (
            self.rgb is None or self.depth is None
            or now - self.rgb_at > self.rgb_stale_after_s
            or now - self.depth_at > self.depth_stale_after_s
        ):
            self.hits.clear()
            self.waiting_hand_hits.clear()
            return

        median, near25 = depth_statistics(self.depth, self.depth_roi)
        if near25 is None:
            self.hits.clear()
            self.waiting_hand_hits.clear()
            return
        if now - self.armed_at < self.baseline_delay_s:
            return

        absolute_depth_hit = near25 <= self.depth_max_mm
        self.waiting_hand_hits.append(absolute_depth_hit)

        if self.baseline_rgb is None:
            if absolute_depth_hit:
                self._log_waiting_hand(median, near25, now)
                self._trigger_if_waiting_hand()
                return
            self.baseline_rgb = self.rgb.copy()
            self.baseline_depth_near25 = near25
            self.get_logger().info(
                "handover baseline: "
                f"median={median:.0f}mm near25={near25:.0f}mm")
            return

        # Recover when the first baseline was captured while a person or
        # transient object was already inside the transfer region.
        if near25 > self.baseline_depth_near25:
            self.baseline_rgb = self.rgb.copy()
            self.baseline_depth_near25 = near25
            self.hits.clear()

        changed = rgb_change_ratio(
            self.rgb, self.baseline_rgb, self.rgb_roi, self.rgb_pixel_delta)
        depth_delta = self.baseline_depth_near25 - near25
        rgb_hit = changed >= self.rgb_changed_ratio
        depth_hit = (
            absolute_depth_hit
            and depth_delta >= self.depth_approach_mm
        )
        if rgb_hit:
            self.last_rgb_hit_at = now
        if depth_hit:
            self.last_depth_hit_at = now
        recent_rgb = (
            self.last_rgb_hit_at is not None
            and now - self.last_rgb_hit_at <= self.gate_memory_s
        )
        recent_depth = (
            self.last_depth_hit_at is not None
            and now - self.last_depth_hit_at <= self.gate_memory_s
        )
        self.hits.append(recent_rgb and recent_depth)

        if now - self.last_log_at >= self.log_interval_s:
            self.last_log_at = now
            self.get_logger().info(
                "handover metrics: "
                f"rgb_change={changed:.3f} depth_median={median:.0f}mm "
                f"depth_near25={near25:.0f}mm depth_delta={depth_delta:.0f}mm "
                f"hits={sum(self.hits)}/{len(self.hits)} "
                "waiting_hits="
                f"{sum(self.waiting_hand_hits)}/{len(self.waiting_hand_hits)}")

        self._publish_debug(rgb_hit, depth_hit)
        if (
            len(self.hits) >= self.required_hits
            and sum(self.hits) >= self.required_hits
        ):
            self._trigger()
            return
        self._trigger_if_waiting_hand()

    def _log_waiting_hand(self, median, near25, now):
        if now - self.last_log_at < self.log_interval_s:
            return
        self.last_log_at = now
        self.get_logger().info(
            "handover metrics: rgb_change=unavailable "
            f"depth_median={median:.0f}mm depth_near25={near25:.0f}mm "
            "baseline=waiting-for-clear-region "
            f"waiting_hits={sum(self.waiting_hand_hits)}/"
            f"{len(self.waiting_hand_hits)}")

    def _trigger_if_waiting_hand(self):
        if (
            len(self.waiting_hand_hits) >= self.waiting_hand_required_hits
            and sum(self.waiting_hand_hits)
            >= self.waiting_hand_required_hits
        ):
            self._trigger()

    def _trigger(self):
        if not self.enabled:
            self.fired = True
            self.get_logger().warning(
                "stable card handover detected; physical trigger is DISABLED")
            return
        if not self.client.service_is_ready():
            self.get_logger().warning("card handover service is unavailable")
            self.hits.clear()
            return
        self.pending = True
        future = self.client.call_async(Trigger.Request())
        future.add_done_callback(self._trigger_response)

    def _trigger_response(self, future):
        self.pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"card handover call failed: {exc}")
            self.hits.clear()
            return
        self.fired = bool(response.success)
        if not response.success:
            self.hits.clear()
        self.get_logger().info(f"card handover response: {response.message}")

    def _publish_debug(self, rgb_hit, depth_hit):
        if (
            not self.publish_debug_image
            or self.debug_publisher.get_subscription_count() == 0
        ):
            return
        import cv2

        preview = self.rgb.copy()
        x, y, width, height = self.rgb_roi
        color = (0, 255, 0) if rgb_hit and depth_hit else (0, 200, 255)
        cv2.rectangle(preview, (x, y), (x + width, y + height), color, 3)
        success, encoded = cv2.imencode(".jpg", preview)
        if success:
            message = CompressedImage()
            message.format = "jpeg"
            message.data = encoded.tobytes()
            self.debug_publisher.publish(message)

    def _start_windows(self):
        import cv2

        cv2.namedWindow("card RGB")
        cv2.namedWindow("card depth")
        cv2.setMouseCallback("card RGB", self._mouse, "rgb")
        cv2.setMouseCallback("card depth", self._mouse, "depth")

    def _mouse(self, event, x, y, _flags, kind):
        import cv2

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start[kind] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and kind in self.drag_start:
            x0, y0 = self.drag_start.pop(kind)
            roi = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            if roi[2] > 0 and roi[3] > 0:
                if kind == "rgb":
                    self.rgb_roi = roi
                else:
                    self.depth_roi = roi
                self.baseline_rgb = None
                self.baseline_depth_near25 = None
                self.hits.clear()
                self.waiting_hand_hits.clear()
                self.last_rgb_hit_at = None
                self.last_depth_hit_at = None
                self.get_logger().warning(f"selected {kind}_roi={roi}")

    def _show_calibration(self):
        import cv2

        if self.rgb is not None:
            preview = self.rgb.copy()
            x, y, width, height = self.rgb_roi
            cv2.rectangle(
                preview,
                (x, y),
                (x + width, y + height),
                (0, 255, 255),
                3,
            )
            cv2.imshow("card RGB", preview)
        if self.depth is not None:
            depth_view = cv2.applyColorMap(
                cv2.convertScaleAbs(self.depth, alpha=255.0 / 2000.0),
                cv2.COLORMAP_TURBO)
            x, y, width, height = self.depth_roi
            cv2.rectangle(
                depth_view,
                (x, y),
                (x + width, y + height),
                (255, 255, 255),
                2,
            )
            cv2.imshow("card depth", depth_view)
        cv2.waitKey(1)

    def shutdown(self):
        if self.show_windows:
            import cv2
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CardHandoverDetector()
    stopped = threading.Event()

    def stop(_signum, _frame):
        stopped.set()

    old_int = signal.signal(signal.SIGINT, stop)
    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        while not stopped.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    main()
