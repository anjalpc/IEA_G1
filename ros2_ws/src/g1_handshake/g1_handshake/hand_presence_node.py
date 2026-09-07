"""ROS node that detects stable human-hand presence for the handshake.

RGB is received from a DDS-isolated helper through shared memory.

This process deliberately contains:
    - rclpy / ROS
    - MediaPipe
    - shared-memory RGB reader

This process deliberately DOES NOT contain:
    - Unitree SDK2
    - ChannelFactoryInitialize
    - Unitree VideoClient
    - RealSense depth fusion

Physical grip triggering is disabled by default and must be explicitly
enabled through the enable_grip_trigger ROS parameter.
"""

import json
import signal
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger

from g1_handshake.presence.hand_detector import (
    HandDetector,
)
from g1_handshake.presence.hand_presence import (
    HandPresenceDetector,
)
from g1_handshake.presence.target_region import TargetRegion
from g1_handshake.video.rgb_frame_client import (
    RgbFrameClient,
)
from g1_handshake.video.depth_camera import DepthCamera

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None


ARMED_STATE = "ARM_PRESENTED"

RGB_UNAVAILABLE = "UNAVAILABLE"
RGB_STALE = "STALE"
RGB_READY = "READY"


class HandPresenceNode(Node):
    """Detect stable human-hand presence while the arm is presented."""

    def __init__(self):
        super().__init__("hand_presence")

        self.declare_parameter(
            "min_detection_confidence",
            0.5,
        )
        self.declare_parameter(
            "detection_max_width",
            1280,
        )

        self.declare_parameter(
            "required_stable_frames",
            5,
        )

        self.declare_parameter(
            "stability_window_frames",
            7,
        )

        self.declare_parameter(
            "poll_rate_hz",
            10.0,
        )

        self.declare_parameter(
            "rgb_stale_after_s",
            0.25,
        )

        self.declare_parameter(
            "enable_grip_trigger",
            False,
        )
        for name, value in (
            ("target_x_min", 0.30),
            ("target_y_min", 0.35),
            ("target_x_max", 0.70),
            ("target_y_max", 0.85),
            ("min_hand_area_ratio", 0.01),
        ):
            self.declare_parameter(name, value)
        self.declare_parameter("publish_debug_image", True)
        for name, value in (
            ("depth_roi_x", 381),
            ("depth_roi_y", 170),
            ("depth_roi_width", 193),
            ("depth_roi_height", 181),
            ("depth_near25_below_mm", 800.0),
            ("rgb_recent_s", 1.0),
        ):
            self.declare_parameter(name, value)

        min_detection_confidence = float(
            self.get_parameter(
                "min_detection_confidence"
            ).value
        )
        detection_max_width = int(
            self.get_parameter("detection_max_width").value
        )

        required_stable_frames = int(
            self.get_parameter(
                "required_stable_frames"
            ).value
        )

        stability_window_frames = int(
            self.get_parameter(
                "stability_window_frames"
            ).value
        )

        poll_rate_hz = float(
            self.get_parameter(
                "poll_rate_hz"
            ).value
        )

        self.rgb_stale_after_s = float(
            self.get_parameter(
                "rgb_stale_after_s"
            ).value
        )

        self.enable_grip_trigger = bool(
            self.get_parameter(
                "enable_grip_trigger"
            ).value
        )
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        self.target_region = TargetRegion(
            self.get_parameter("target_x_min").value,
            self.get_parameter("target_y_min").value,
            self.get_parameter("target_x_max").value,
            self.get_parameter("target_y_max").value,
            self.get_parameter("min_hand_area_ratio").value,
        )
        self.depth_roi = tuple(
            int(self.get_parameter(name).value)
            for name in (
                "depth_roi_x", "depth_roi_y",
                "depth_roi_width", "depth_roi_height",
            )
        )
        self.depth_near25_below_mm = float(
            self.get_parameter("depth_near25_below_mm").value
        )
        self.rgb_recent_s = float(
            self.get_parameter("rgb_recent_s").value
        )

        if poll_rate_hz <= 0.0:
            raise ValueError(
                "poll_rate_hz must be positive"
            )

        if self.rgb_stale_after_s <= 0.0:
            raise ValueError(
                "rgb_stale_after_s must be positive"
            )

        self.detector = HandPresenceDetector(
            required_stable_frames=required_stable_frames,
            window_size=stability_window_frames,
        )

        self.hand_detector = HandDetector(
            min_detection_confidence=(
                min_detection_confidence
            ),
            max_width=detection_max_width,
        )

        self.hand_detector.start()

        self.rgb_client = RgbFrameClient()
        self.depth_camera = DepthCamera()
        self.depth_camera.start()

        self._armed = False
        self._fired_this_presentation = False
        self._rgb_state = None
        self._grip_pending = False
        self._last_rgb_hit_at = None
        self._depth_median_mm = None
        self._depth_near25_mm = None
        self._rgb_hit_state = None
        self._depth_hit_state = None

        self._debug_publisher = self.create_publisher(
            CompressedImage,
            "/event/handshake/hand_presence/debug/compressed",
            1,
        )

        self._grip_client = self.create_client(
            Trigger,
            "/event/handshake/grip",
        )

        self.create_subscription(
            String,
            "/event/handshake/state",
            self._state_callback,
            10,
        )

        self.create_timer(
            1.0 / poll_rate_hz,
            self._poll,
        )

        self.get_logger().warning(
            "Hand-presence node ready; "
            f"physical grip trigger enabled="
            f"{self.enable_grip_trigger}"
        )

    def _set_rgb_state(
        self,
        new_state,
        age=None,
    ):
        """Log only RGB-state transitions, not every poll."""
        if new_state == self._rgb_state:
            return

        self._rgb_state = new_state

        if new_state == RGB_READY:
            self.get_logger().info(
                "RGB helper is READY"
            )

        elif new_state == RGB_STALE:
            if age is None:
                self.get_logger().warning(
                    "RGB helper is STALE"
                )
            else:
                self.get_logger().warning(
                    "RGB helper is STALE "
                    f"(frame age={age:.3f}s)"
                )

        elif new_state == RGB_UNAVAILABLE:
            self.get_logger().warning(
                "RGB helper is UNAVAILABLE"
            )

    def _state_callback(
        self,
        message,
    ):
        try:
            state = str(
                json.loads(
                    message.data
                )["state"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self.get_logger().error(
                "invalid handshake state: "
                f"{exc}"
            )
            return

        armed = (
            state == ARMED_STATE
        )

        if armed and not self._armed:
            self._fired_this_presentation = False
            self._grip_pending = False
            self._last_rgb_hit_at = None
            self._rgb_hit_state = None
            self._depth_hit_state = None
            self.detector.reset()

            self.get_logger().info(
                "hand-presence trigger armed"
            )

        elif (
            not armed
            and self._armed
        ):
            self.detector.reset()

            self.get_logger().info(
                "hand-presence trigger disarmed"
            )

        self._armed = armed

    def _log_gate_transitions(self, rgb_hit, depth_hit):
        """Log RGB and depth gate changes once while armed."""
        if not self._armed:
            return

        rgb_hit = bool(rgb_hit)
        depth_hit = bool(depth_hit)

        if rgb_hit != self._rgb_hit_state:
            self._rgb_hit_state = rgb_hit
            self.get_logger().info(
                f"RGB gate {'HIT' if rgb_hit else 'CLEAR'}"
            )

        if depth_hit != self._depth_hit_state:
            self._depth_hit_state = depth_hit
            detail = ""
            if self._depth_near25_mm is not None:
                detail = f" (near25={self._depth_near25_mm:.0f}mm)"
            self.get_logger().info(
                f"depth gate {'HIT' if depth_hit else 'CLEAR'}{detail}"
            )

    def _poll(self):
        (
            color_bgr,
            age,
            _sequence,
        ) = self.rgb_client.read_latest()

        if color_bgr is None:
            self._set_rgb_state(
                RGB_UNAVAILABLE
            )

            self.detector.reset()
            return

        if (
            age is None
            or age > self.rgb_stale_after_s
        ):
            self._set_rgb_state(
                RGB_STALE,
                age=age,
            )

            self.detector.reset()
            return

        self._set_rgb_state(
            RGB_READY
        )

        try:
            hand = self.hand_detector.detect(
                color_bgr
            )

        except Exception as exc:
            self.get_logger().error(
                "hand detection failed: "
                f"{exc}"
            )

            self.detector.reset()
            return

        height, width = color_bgr.shape[:2]
        rgb_hit = self.target_region.accepts(hand, width, height)

        depth_hit = False
        try:
            depth = self.depth_camera.read_frame_mm(timeout_ms=100)
            x, y, roi_width, roi_height = self.depth_roi
            region = depth[y:y + roi_height, x:x + roi_width]
            valid = region[(region > 100.0) & (region < 4000.0)]
            if valid.size:
                import numpy as np
                self._depth_median_mm = float(np.median(valid))
                self._depth_near25_mm = float(np.percentile(valid, 25))
                depth_hit = (
                    self._depth_near25_mm < self.depth_near25_below_mm
                )
        except Exception as exc:
            self._depth_median_mm = None
            self._depth_near25_mm = None
            self.get_logger().warning(
                f"depth frame unavailable: {exc}"
            )

        if (
            self.publish_debug_image
            and self._debug_publisher.get_subscription_count() > 0
        ):
            self._publish_debug_image(
                color_bgr, hand, rgb_hit, depth_hit
            )

        self._log_gate_transitions(rgb_hit, depth_hit)

        if (
            not self._armed
            or self._fired_this_presentation
            or self._grip_pending
        ):
            return

        now = time.monotonic()
        if rgb_hit:
            self._last_rgb_hit_at = now

        recent_rgb = (
            self._last_rgb_hit_at is not None
            and now - self._last_rgb_hit_at <= self.rgb_recent_s
        )
        hit = recent_rgb and depth_hit

        if not self.detector.observe(hit):
            return

        if not self.enable_grip_trigger:
            self._fired_this_presentation = True
            self.get_logger().warning(
                "stable hand presence detected; "
                "physical grip trigger is DISABLED"
            )
            return

        self._grip_pending = self._call_grip()

    def _publish_debug_image(self, frame, hand, rgb_hit, depth_hit):
        """Publish an annotated JPEG without requiring cv_bridge."""
        import cv2

        preview = frame.copy()
        height, width = preview.shape[:2]
        left, top, right, bottom = (
            self.target_region.pixel_bounds(
                width,
                height,
            )
        )
        accepted = rgb_hit and depth_hit
        region_color = (0, 255, 0) if accepted else (0, 200, 255)
        cv2.rectangle(preview, (left, top), (right, bottom), region_color, 4)
        if hand is not None:
            for point in hand.landmarks:
                cv2.circle(preview, point, 3, (255, 255, 0), -1)
            cv2.circle(
                preview,
                (hand.center_x, hand.center_y),
                12,
                (255, 0, 255),
                -1,
            )
            if hand.bbox is not None:
                x1, y1, x2, y2 = hand.bbox
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 3)
        status = (
            f"armed={self._armed} "
            f"rgb={'HIT' if rgb_hit else 'CLEAR'} "
            f"depth={'HIT' if depth_hit else 'CLEAR'} "
            f"trigger={self.enable_grip_trigger}"
        )
        cv2.putText(preview, status, (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        depth_label = (
            "depth unavailable"
            if self._depth_median_mm is None
            else (
                f"depth median={self._depth_median_mm:.0f}mm "
                f"near25={self._depth_near25_mm:.0f}mm"
            )
        )
        cv2.putText(preview, depth_label, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        success, encoded = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, 75],
        )
        if not success:
            return
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self._debug_publisher.publish(message)

    def _call_grip(self):
        if not self._grip_client.service_is_ready():
            self.get_logger().warning(
                "stable hand detected, but "
                "/event/handshake/grip "
                "is not available"
            )
            return False

        future = self._grip_client.call_async(
            Trigger.Request()
        )

        future.add_done_callback(
            self._grip_response
        )
        return True

    def _grip_response(
        self,
        future,
    ):
        self._grip_pending = False
        try:
            response = future.result()

        except Exception as exc:
            self.get_logger().error(
                "grip call failed: "
                f"{exc}"
            )
            self.detector.reset()
            return

        if response.success:
            self._fired_this_presentation = True
            self.get_logger().info(
                "hand detected, "
                "grip triggered: "
                f"{response.message}"
            )

        else:
            self.detector.reset()
            self.get_logger().warning(
                "hand detected, "
                "but grip was rejected: "
                f"{response.message}"
            )

    def shutdown(self):
        try:
            self.depth_camera.stop()
        except Exception as exc:
            self.get_logger().error(
                f"depth camera shutdown failed: {exc}"
            )

        try:
            self.rgb_client.close()

        except Exception as exc:
            self.get_logger().error(
                "RGB client shutdown failed: "
                f"{exc}"
            )

        try:
            self.hand_detector.stop()

        except Exception as exc:
            self.get_logger().error(
                "hand detector shutdown failed: "
                f"{exc}"
            )


def main(args=None):
    if SignalHandlerOptions is None:
        rclpy.init(
            args=args
        )
    else:
        rclpy.init(
            args=args,
            signal_handler_options=(
                SignalHandlerOptions.NO
            ),
        )

    node = HandPresenceNode()

    stop_requested = threading.Event()

    def request_stop(
        _signum,
        _frame,
    ):
        stop_requested.set()

    previous_sigint = signal.signal(
        signal.SIGINT,
        request_stop,
    )

    previous_sigterm = signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    try:
        while (
            not stop_requested.is_set()
            and rclpy.ok()
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.2,
            )

    except KeyboardInterrupt:
        pass

    finally:
        node.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        signal.signal(
            signal.SIGINT,
            previous_sigint,
        )

        signal.signal(
            signal.SIGTERM,
            previous_sigterm,
        )


if __name__ == "__main__":
    main()
