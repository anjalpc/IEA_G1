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
        self.gpu_selector = None
        backend = self.declare_parameter("compute_backend", "cpu").value
        if backend not in ("cpu", "cuda"):
            raise ValueError("compute_backend must be cpu or cuda")
        if backend == "cuda":
            try:
                from .depth_geometry_gpu import select_floor_plane_gpu
                x, y = np.meshgrid(np.linspace(0.3, 0.8, 10),
                                   np.linspace(-0.3, 0.3, 10))
                warmup = np.column_stack((x.ravel(), y.ravel(),
                                          np.full(x.size, -0.8)))
                wall_y, wall_z = np.meshgrid(
                    np.linspace(-0.3, 0.3, 50), np.linspace(-0.3, 0.3, 80))
                wall = np.column_stack((np.full(wall_y.size, 0.9),
                                        wall_y.ravel(), wall_z.ravel()))
                scenes = (
                    ("clear_floor", warmup),
                    ("wall_fallback", np.vstack((warmup, wall))),
                )
                for label, scene in scenes:
                    started = time.monotonic()
                    plane = select_floor_plane_gpu(
                        scene, [0., 0., 1.], distance_bounds=self.bounds)
                    if plane is None:
                        raise RuntimeError(f"CUDA warm-up failed: {label}")
                    self.get_logger().info(
                        f"CUDA_WARMUP path={label} "
                        f"elapsed_ms={(time.monotonic() - started) * 1000:.1f}")
                self.gpu_selector = select_floor_plane_gpu
            except Exception as error:
                self.get_logger().warning(
                    f"CUDA unavailable; using CPU floor fitting: {error}")
        selected = "cuda" if self.gpu_selector is not None else "cpu"
        self.get_logger().info(f"Floor fitting backend: {selected}")
        self.imus = deque(maxlen=100)
        self.pending = None
        self.last_processed_at = None
        self.first_geometry_pending = True
        self.last_timing_log = time.monotonic()
        self.max_callback_gap_ms = 0.0
        self.max_input_age_ms = 0.0
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

    def select_floor(self, points, up):
        if self.gpu_selector is not None:
            try:
                started = time.monotonic()
                plane = self.gpu_selector(points, up, distance_bounds=self.bounds)
                elapsed_ms = (time.monotonic() - started) * 1000
                if elapsed_ms > 100:
                    self.get_logger().warning(
                        f"CUDA_FIT_SLOW elapsed_ms={elapsed_ms:.1f} "
                        f"region_points={len(points)}")
                return plane
            except Exception as error:
                self.gpu_selector = None
                self.get_logger().warning(
                    f"CUDA floor fitting failed; switching to CPU: {error}")
        return select_floor_plane(points, up, distance_bounds=self.bounds)

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
        received = time.monotonic()
        if self.received is not None:
            self.max_callback_gap_ms = max(
                self.max_callback_gap_ms, (received - self.received) * 1000)
        self.received = received
        age_ms = (self.get_clock().now().nanoseconds
                  - stamp_ns(message.header)) / 1e6
        self.max_input_age_ms = max(self.max_input_age_ms, age_ms)
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
            plane = self.select_floor(points[floor_region], up)
            if plane is None:
                self.clear("FLOOR_UNKNOWN")
                return
            height = heights_above_plane(points, plane)
            keep = ((height > 0.04) & (height < 1.50) & (points[:, 0] > 0)
                    & (points[:, 0] < 2.0) & (np.abs(points[:, 1]) < 0.75))
            if self.first_geometry_pending:
                elapsed_ms = (time.monotonic() - self.last_processed_at) * 1000
                self.get_logger().info(
                    f"FIRST_GEOMETRY elapsed_ms={elapsed_ms:.1f}")
                self.first_geometry_pending = False
            age = self.get_clock().now().nanoseconds - source_stamp
            if not -50_000_000 <= age <= 350_000_000:
                self.clear("PROCESSING_STALE")
                return
            self.emit(points[keep], message.header.stamp, "DIAGNOSTIC_CANDIDATE")
            self.active_stamp = source_stamp
            finished = time.monotonic()
            if finished - self.last_timing_log >= 5.0:
                elapsed_ms = (finished - self.last_processed_at) * 1000
                output_age_ms = (
                    self.get_clock().now().nanoseconds - source_stamp) / 1e6
                backend = "cuda" if self.gpu_selector is not None else "cpu"
                self.get_logger().info(
                    f"TIMING backend={backend} callback_ms={elapsed_ms:.1f} "
                    f"output_age_ms={output_age_ms:.1f} "
                    f"max_input_age_ms={self.max_input_age_ms:.1f} "
                    f"max_input_gap_ms={self.max_callback_gap_ms:.1f}")
                self.last_timing_log = finished
                self.max_callback_gap_ms = 0.0
                self.max_input_age_ms = 0.0
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
