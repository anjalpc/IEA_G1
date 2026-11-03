"""ROS-facing owner for the G1 RGB and depth compatibility feeds."""

import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage, Image

from g1_hardware_bridge.camera.ros_color import (
    ColorFrameCache,
    color_image_to_bgr,
)
from g1_hardware_bridge.camera.ros_depth import DepthFrameCache

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None


DEFAULT_DEPTH_INPUT_TOPIC = "/d435/depth/image_rect_raw"
DEFAULT_RGB_INPUT_TOPIC = "/d435/color/image_raw"
DEPTH_OUTPUT_TOPIC = "/g1/camera/depth/image_raw"
RGB_OUTPUT_TOPIC = "/g1/camera/rgb/compressed"

CAMERA_INPUT_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def depth_image_message(depth_mm, stamp):
    """Encode a 2-D millimetre depth array as a ROS 16UC1 image."""
    depth_u16 = np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = "g1_depth_camera"
    message.height, message.width = depth_u16.shape
    message.encoding = "16UC1"
    message.is_bigendian = False
    message.step = message.width * 2
    message.data = depth_u16.tobytes(order="C")
    return message


class CameraBridgeNode(Node):
    """Bridge driver-owned D435i color and depth into stable topics."""

    def __init__(
        self,
        monotonic=time.monotonic,
    ):
        """Create read-only subscriptions to driver-owned camera streams."""
        super().__init__("g1_camera_bridge")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("rgb_stale_after_s", 0.25)
        self.declare_parameter(
            "rgb_input_topic",
            DEFAULT_RGB_INPUT_TOPIC,
        )
        self.declare_parameter("depth_stale_after_s", 0.5)
        self.declare_parameter(
            "depth_input_topic",
            DEFAULT_DEPTH_INPUT_TOPIC,
        )
        self.declare_parameter("jpeg_quality", 75)

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.rgb_stale_after_s = float(
            self.get_parameter("rgb_stale_after_s").value
        )
        rgb_input_topic = str(
            self.get_parameter("rgb_input_topic").value
        )
        depth_stale_after_s = float(
            self.get_parameter("depth_stale_after_s").value
        )
        depth_input_topic = str(
            self.get_parameter("depth_input_topic").value
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")

        self._publish_period_s = 1.0 / rate
        self._next_rgb_publish_at = float("-inf")
        self._monotonic = monotonic
        self._color_cache = ColorFrameCache(self.rgb_stale_after_s)
        self._depth_cache = DepthFrameCache(depth_stale_after_s)
        self._rgb_available = None
        self._depth_available = None
        self._color_callback_group = MutuallyExclusiveCallbackGroup()
        self._depth_callback_group = MutuallyExclusiveCallbackGroup()
        self._publish_callback_group = MutuallyExclusiveCallbackGroup()

        self.rgb_publisher = self.create_publisher(
            CompressedImage,
            RGB_OUTPUT_TOPIC,
            qos_profile_sensor_data,
        )
        self.depth_publisher = self.create_publisher(
            Image,
            DEPTH_OUTPUT_TOPIC,
            qos_profile_sensor_data,
        )
        self._depth_subscription = self.create_subscription(
            Image,
            depth_input_topic,
            self._depth_callback,
            CAMERA_INPUT_QOS,
            callback_group=self._depth_callback_group,
        )
        self._color_subscription = self.create_subscription(
            Image,
            rgb_input_topic,
            self._color_callback,
            CAMERA_INPUT_QOS,
            callback_group=self._color_callback_group,
        )
        self._publish_timer = self.create_timer(
            1.0 / rate,
            self._publish_frames,
            callback_group=self._publish_callback_group,
        )
        self.get_logger().info("G1 camera bridge ready")

    def _depth_callback(self, message):
        """Accept only structurally valid driver-owned depth frames."""
        if self._depth_cache.offer(message, self._monotonic()):
            return
        if self._depth_available is not False:
            self.get_logger().warning("invalid depth frame rejected")
        self._depth_available = False

    def _color_callback(self, message):
        """Validate and immediately bridge rate-limited color frames."""
        now = self._monotonic()
        if not self._color_cache.offer(message, now):
            if self._rgb_available is not False:
                self.get_logger().warning("invalid RGB frame rejected")
            self._rgb_available = False
            return

        if now < self._next_rgb_publish_at:
            return

        import cv2

        frame = color_image_to_bgr(message)
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            self._rgb_available = False
            return

        output = CompressedImage()
        output.header = message.header
        output.format = "jpeg"
        output.data = encoded.tobytes()
        self.rgb_publisher.publish(output)
        self._next_rgb_publish_at = now + self._publish_period_s

        if self._rgb_available is not True:
            self.get_logger().info("RGB stream is READY")
        self._rgb_available = True

    def _publish_frames(self):
        depth_message = self._depth_cache.latest(self._monotonic())
        if depth_message is None:
            if self._depth_available is not False:
                self.get_logger().warning("depth stream unavailable or stale")
            self._depth_available = False
            return

        if self._depth_available is not True:
            self.get_logger().info("depth stream is READY")
        self._depth_available = True
        self.depth_publisher.publish(depth_message)

    def shutdown(self):
        """Release bridge-owned resources."""


def create_executor(node):
    """Create one executor thread for each independent callback group."""
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    return executor


def main(args=None):
    """Run the camera bridge until interrupted."""
    if SignalHandlerOptions is None:
        rclpy.init(args=args)
    else:
        rclpy.init(
            args=args,
            signal_handler_options=SignalHandlerOptions.NO,
        )

    node = CameraBridgeNode()
    executor = create_executor(node)
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
