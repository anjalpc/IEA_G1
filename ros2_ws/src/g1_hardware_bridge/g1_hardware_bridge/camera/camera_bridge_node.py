"""Single ROS-facing owner for the G1 RGB and depth camera feeds."""

import signal
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from g1_hardware_bridge.camera.depth_camera import DepthCamera
from g1_hardware_bridge.camera.rgb_frame_client import RgbFrameClient

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:
    SignalHandlerOptions = None


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
    """Publish camera frames while exclusively owning the live depth device."""

    def __init__(self):
        super().__init__("g1_camera_bridge")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("rgb_stale_after_s", 0.25)
        self.declare_parameter("jpeg_quality", 75)

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.rgb_stale_after_s = float(
            self.get_parameter("rgb_stale_after_s").value
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")

        self.rgb_client = RgbFrameClient()
        self.depth_camera = DepthCamera()
        self.depth_camera.start()
        self._depth_available = None

        self.rgb_publisher = self.create_publisher(
            CompressedImage,
            "/g1/camera/rgb/compressed",
            qos_profile_sensor_data,
        )
        self.depth_publisher = self.create_publisher(
            Image,
            "/g1/camera/depth/image_raw",
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / rate, self._publish_frames)
        self.get_logger().info("G1 camera bridge ready")

    def _publish_frames(self):
        import cv2

        frame, age, _sequence = self.rgb_client.read_latest()
        if (
            frame is not None
            and age is not None
            and age <= self.rgb_stale_after_s
        ):
            success, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if success:
                message = CompressedImage()
                message.header.stamp = self.get_clock().now().to_msg()
                message.header.frame_id = "g1_rgb_camera"
                message.format = "jpeg"
                message.data = encoded.tobytes()
                self.rgb_publisher.publish(message)

        try:
            depth_mm = self.depth_camera.read_frame_mm(timeout_ms=100)
        except Exception as exc:
            if self._depth_available is not False:
                self.get_logger().warning(f"depth camera unavailable: {exc}")
            self._depth_available = False
            return

        if self._depth_available is not True:
            self.get_logger().info("depth camera is READY")
        self._depth_available = True
        self.depth_publisher.publish(
            depth_image_message(depth_mm, self.get_clock().now().to_msg())
        )

    def shutdown(self):
        """Close both physical camera transports."""
        self.depth_camera.stop()
        self.rgb_client.close()


def main(args=None):
    """Run the camera bridge until interrupted."""
    if SignalHandlerOptions is None:
        rclpy.init(args=args)
    else:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    node = CameraBridgeNode()
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
