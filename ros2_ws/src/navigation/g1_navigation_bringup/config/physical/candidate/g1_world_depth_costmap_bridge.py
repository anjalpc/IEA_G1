"""World-referenced diagnostic bridge only; never publishes robot commands."""
from collections import OrderedDict, deque
from pathlib import Path
import json
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, Imu
from sensor_msgs_py.point_cloud2 import read_points_numpy, create_cloud_xyz32
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster, TransformException
from g1_navigation_perception.depth_geometry import transform_points, select_floor_plane

FRAME = 'depth_world_preview_ground'
def ns(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec

class Bridge(Node):
    def __init__(self):
        super().__init__('g1_world_depth_costmap_bridge')
        self.cache = {key: OrderedDict() for key in ('raw', 'marks', 'health')}
        self.imus = deque(maxlen=100)
        self.reference = None
        self.last = 0
        self.last_log = 0.
        self.buffer = Buffer(cache_time=Duration(seconds=5))
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.marks_pub = self.create_publisher(PointCloud2, '/g1/diagnostics/depth/costmap_input', 1)
        self.rays_pub = self.create_publisher(PointCloud2, '/g1/diagnostics/depth/costmap_clear', 1)
        self.subs = [self.create_subscription(cls, topic, lambda m, k=key: self.store(k, m), qos)
                     for key, cls, topic, qos in (
                         ('raw', PointCloud2, '/g1/diagnostics/depth/points', qos_profile_sensor_data),
                         ('marks', PointCloud2, '/g1/diagnostics/depth/obstacles', qos_profile_sensor_data),
                         ('health', DiagnosticArray, '/g1/diagnostics/depth/health', 1))]
        self.subs.append(self.create_subscription(Imu, '/g1/state/imu_raw', self.imus.append, qos_profile_sensor_data))
        self.timer = self.create_timer(.02, self.process)
        self.get_logger().info('Keep robot stationary; remove box before capturing the fixed floor reference.')

    def store(self, key, message):
        cache = self.cache[key]
        cache[ns(message.header)] = message
        while len(cache) > 20:
            cache.popitem(last=False)

    def process(self):
        common = set.intersection(*(set(c) for c in self.cache.values()))
        common = [s for s in common if s > self.last]
        if not common:
            return
        s = max(common)
        if not -50_000_000 <= self.get_clock().now().nanoseconds - s <= 350_000_000:
            return
        raw, marks, health = [self.cache[k][s] for k in ('raw', 'marks', 'health')]
        if (raw.header.frame_id != 'd435_depth_optical_frame'
                or marks.header.frame_id != 'pelvis' or health.header.frame_id != 'pelvis'
                or len(health.status) != 1 or health.status[0].level != DiagnosticStatus.OK
                or health.status[0].message != 'DIAGNOSTIC_CANDIDATE'):
            return
        try:
            tf = self.buffer.lookup_transform('pelvis', raw.header.frame_id, Time.from_msg(raw.header.stamp)).transform
            body_tf = self.buffer.lookup_transform('odom', 'pelvis', Time.from_msg(raw.header.stamp)).transform
        except TransformException:
            return
        v, q = tf.translation, tf.rotation
        bv, bq = body_tf.translation, body_tf.rotation
        body_translation = np.array([bv.x, bv.y, bv.z])
        body_quaternion = [bq.x, bq.y, bq.z, bq.w]
        xyz = read_points_numpy(raw, field_names=('x', 'y', 'z'), skip_nans=True)[::4]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        xyz = transform_points(xyz, [v.x, v.y, v.z], [q.x, q.y, q.z, q.w])
        if self.reference is None:
            if not self.imus:
                return
            imu = min(self.imus, key=lambda m: abs(ns(m.header)-s))
            if imu.header.frame_id != 'imu_in_pelvis' or abs(ns(imu.header)-s) > 50_000_000:
                return
            q = imu.orientation
            up = transform_points([[0., 0., 1.]], [0., 0., 0.], [-q.x, -q.y, -q.z, q.w])[0]
            forward = np.array([1., 0., 0.]); forward -= (forward @ up)*up
            forward /= np.linalg.norm(forward)
            side = np.cross(up, forward)
            patch = xyz[(xyz @ forward > .2) & (xyz @ forward < 1.2) & (np.abs(xyz @ side) < .4)]
            plane = select_floor_plane(patch, up, distance_bounds=(.65, .95))
            if plane is None:
                return
            normal = transform_points([plane[:3]], [0., 0., 0.], body_quaternion)[0]
            plane = np.r_[normal, plane[3] - normal @ body_translation]
            forward = np.array([1., 0., 0.]); forward -= (forward @ normal)*normal
            forward /= np.linalg.norm(forward)
            self.rotation = np.vstack((forward, np.cross(normal, forward), normal))
            self.offset = np.array([0., 0., plane[3]])
            # Inverse of ground <- odom: publish odom <- ground as a child.
            inverse = self.rotation.T
            qw = np.sqrt(1 + np.trace(inverse))/2
            assert qw > .9
            qxyz = np.array([inverse[2,1]-inverse[1,2], inverse[0,2]-inverse[2,0], inverse[1,0]-inverse[0,1]])/(4*qw)
            t = TransformStamped()
            t.header = Header(stamp=raw.header.stamp, frame_id='odom')
            t.child_frame_id = FRAME
            p = -inverse @ self.offset
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = map(float, qxyz)
            t.transform.rotation.w = float(qw)
            self.broadcaster.sendTransform(t)
            self.reference = plane
            log_dir = Path(os.environ.get('ROS_LOG_DIR') or Path.home() / '.ros/log').expanduser()
            out = log_dir / f'world-depth-costmap-{time.time_ns()}.json'
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({'scope': 'World-referenced diagnostic preview; initial camera-derived floor anchor, no calibration change', 'plane_in_odom': plane.tolist(), 'source_stamp_ns': s}, indent=2)+'\n')
            except OSError as error:
                self.get_logger().warning(f'Could not save floor reference: {error}')
            self.get_logger().info('WORLD_FLOOR_READY; await costmap checks before manual movement')
            return
        obstacle_xyz = read_points_numpy(marks, field_names=('x', 'y', 'z'), skip_nans=True)
        if not -50_000_000 <= self.get_clock().now().nanoseconds - s <= 350_000_000:
            return
        xyz = transform_points(xyz, body_translation, body_quaternion)
        obstacle_xyz = transform_points(obstacle_xyz, body_translation, body_quaternion)
        header = Header(stamp=raw.header.stamp, frame_id=FRAME)
        self.rays_pub.publish(create_cloud_xyz32(header, xyz @ self.rotation.T + self.offset))
        self.marks_pub.publish(create_cloud_xyz32(header, obstacle_xyz @ self.rotation.T + self.offset))
        self.last = s
        if time.monotonic() - self.last_log > 5:
            self.get_logger().info(f'LIVE_PAIR marks={len(obstacle_xyz)} clearing_returns={len(xyz)}')
            self.last_log = time.monotonic()

def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
