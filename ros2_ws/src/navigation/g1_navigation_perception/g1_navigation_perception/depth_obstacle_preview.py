"""Read-only experimental depth-obstacle preview; never a Nav2 input."""

from collections import deque
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, read_points_numpy
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener

from .depth_context import cloud_context_reason
from .depth_geometry import heights_above_plane, select_floor_plane, transform_points


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


class DepthObstaclePreview(Node):
    """Publish diagnostic candidates and explicit rejection status."""

    def __init__(self):
        super().__init__("g1_depth_obstacle_preview")
        self.bounds = (
            self.declare_parameter("minimum_floor_distance_m", 0.65).value,
            self.declare_parameter("maximum_floor_distance_m", 0.95).value)
        self.imus = deque(maxlen=100)
        self.pending = None
        self.last_processed_at = None
        cloud_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.previous = self.received = self.active_stamp = self.reason = None
        self.buffer = Buffer(cache_time=Duration(seconds=3))
        self.listener = TransformListener(self.buffer, self)
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/g1/diagnostics/depth/obstacles", qos_profile_sensor_data)
        self.status_pub = self.create_publisher(
            String, "/g1/diagnostics/depth/status", 1)
        self.subscriptions_owned = [
            self.create_subscription(Imu, "/g1/state/imu_raw",
                                     lambda m: self.imus.append(m), qos_profile_sensor_data),
            self.create_subscription(PointCloud2, "/g1/diagnostics/depth/points",
                                     self.cloud_callback, cloud_qos)]
        self.timer = self.create_timer(0.1, self.watchdog)
        self.processing_timer = self.create_timer(0.02, self.process_latest)

    def emit(self, points, stamp, reason):
        header = Header(stamp=stamp, frame_id="pelvis")
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        self.cloud_pub.publish(create_cloud_xyz32(header, points))
        self.status_pub.publish(String(data=reason))
        if reason != self.reason:
            self.get_logger().info(reason)
            self.reason = reason

    def clear(self, reason):
        self.active_stamp = None
        self.emit([], self.get_clock().now().to_msg(), reason)

    def watchdog(self):
        if self.received is None or time.monotonic() - self.received > 0.35:
            self.clear("CLOUD_TIMEOUT")
        elif self.active_stamp is not None:
            age = self.get_clock().now().nanoseconds - self.active_stamp
            if not -50_000_000 <= age <= 350_000_000:
                self.clear("CLOUD_STALE")

    def cloud_callback(self, message):
        self.received = time.monotonic()
        self.pending = message

    def can_defer(self):
        if self.active_stamp is None or self.received is None:
            return False
        age = self.get_clock().now().nanoseconds - self.active_stamp
        return (-50_000_000 <= age <= 350_000_000
                and time.monotonic() - self.received < 0.06)

    def process_latest(self):
        if (self.last_processed_at is not None
                and time.monotonic() - self.last_processed_at < 0.1):
            return
        message = self.pending
        if message is None:
            return
        if not self.imus:
            self.clear("IMU_MISSING")
            return
        source_stamp = stamp_ns(message.header)
        imu = min(self.imus, key=lambda m: abs(stamp_ns(m.header) - source_stamp))
        reason = cloud_context_reason(
            source_stamp, stamp_ns(imu.header), self.get_clock().now().nanoseconds,
            cloud_frame=message.header.frame_id, imu_frame=imu.header.frame_id,
            previous_stamp_ns=self.previous)
        if reason != "READY":
            if reason == "IMU_UNSYNCHRONIZED" and self.can_defer():
                return
            self.clear(reason)
            return
        try:
            t = self.buffer.lookup_transform(
                "pelvis", message.header.frame_id, Time.from_msg(message.header.stamp)).transform
            self.pending = None
            self.previous = source_stamp
            self.last_processed_at = time.monotonic()
            v, r, q = t.translation, t.rotation, imu.orientation
            points = read_points_numpy(message, field_names=("x", "y", "z"),
                                       skip_nans=True)[::4]
            points = points[np.isfinite(points).all(axis=1)]
            points = transform_points(points, [v.x, v.y, v.z], [r.x, r.y, r.z, r.w])
            up = transform_points([[0., 0., 1.]], [0., 0., 0.],
                                  [-q.x, -q.y, -q.z, q.w])[0]
            forward = np.array([1., 0., 0.])
            forward -= np.dot(forward, up) * up
            length = np.linalg.norm(forward)
            if length < 1e-6:
                raise ValueError("forward axis is vertical")
            forward /= length
            side = np.cross(up, forward)
            floor_region = ((points @ forward > 0.2) & (points @ forward < 1.2)
                            & (np.abs(points @ side) < 0.4))
            plane = select_floor_plane(points[floor_region], up, distance_bounds=self.bounds)
            if plane is None:
                self.clear("FLOOR_UNKNOWN")
                return
            height = heights_above_plane(points, plane)
            keep = ((height > 0.04) & (height < 1.50) & (points[:, 0] > 0)
                    & (points[:, 0] < 2.0) & (np.abs(points[:, 1]) < 0.75))
            age = self.get_clock().now().nanoseconds - source_stamp
            if not -50_000_000 <= age <= 350_000_000:
                self.clear("PROCESSING_STALE")
                return
            self.emit(points[keep], message.header.stamp, "DIAGNOSTIC_CANDIDATE")
            self.active_stamp = source_stamp
        except TransformException as error:
            if self.can_defer():
                return
            if self.reason != "TF_UNAVAILABLE":
                self.get_logger().warning(str(error))
            self.clear("TF_UNAVAILABLE")
        except (ValueError, TypeError, AssertionError):
            self.clear("GEOMETRY_INVALID")


def main():
    rclpy.init()
    node = DepthObstaclePreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
