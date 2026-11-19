"""Isolated runtime checks for diagnostic depth filtering."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from sensor_msgs.msg import Imu
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, read_points_numpy
from std_msgs.msg import Header

from g1_navigation_perception.depth_geometry import transform_points
from g1_navigation_perception.depth_obstacle_preview import DepthObstaclePreview


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


@pytest.fixture
def preview():
    rclpy.init()
    node = DepthObstaclePreview()
    node.cloud_pub, node.status_pub = Recorder(), Recorder()
    clock = SimpleNamespace(ns=2_000_000_000)
    node.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=clock.ns))
    try:
        yield node, clock
    finally:
        node.destroy_node()
        rclpy.shutdown()


def inputs(node, clock, *, provide_tf=True, stamp=None):
    stamp = clock.ns if stamp is None else stamp
    header = Header(stamp=Time(nanoseconds=stamp).to_msg(),
                    frame_id="d435_depth_optical_frame")
    imu = Imu()
    imu.header = Header(stamp=header.stamp, frame_id="imu_in_pelvis")
    imu.orientation.w = 1.0
    node.imus.append(imu)
    if provide_tf:
        t = TransformStamped()
        t.header.frame_id = "pelvis"
        t.child_frame_id = header.frame_id
        t.transform.translation.z = 0.5
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = -0.5, 0.5, -0.5, 0.5
        node.buffer.set_transform_static(t, "test")
    x, y = np.meshgrid(np.linspace(0.3, 0.9, 20),
                       np.linspace(-0.3, 0.3, 20))
    floor = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -0.8)))
    box = floor.copy()
    box[:, 2] += 0.12
    points = transform_points(np.vstack((floor, box)) - [0., 0., 0.5],
                              [0., 0., 0.], [0.5, -0.5, 0.5, 0.5])
    return create_cloud_xyz32(header, points.astype(np.float32))


def test_candidate_cloud_preserves_stamp_and_watchdog_clears_it(preview):
    node, clock = preview
    message = inputs(node, clock)
    node.cloud_callback(message)
    node.process_latest()
    output = node.cloud_pub.messages[-1]
    assert node.status_pub.messages[-1].data == "DIAGNOSTIC_CANDIDATE"
    assert output.header.frame_id == "pelvis"
    assert output.header.stamp == message.header.stamp
    assert output.width > 0
    clock.ns += 350_000_001
    node.watchdog()
    assert node.status_pub.messages[-1].data == "CLOUD_STALE"
    assert node.cloud_pub.messages[-1].width == 0
    node.received = time.monotonic() - 0.36
    node.watchdog()
    assert node.status_pub.messages[-1].data == "CLOUD_TIMEOUT"
    assert node.cloud_pub.messages[-1].width == 0


@pytest.mark.parametrize("provide_tf, offset, expected", [
    (False, 0, "TF_UNAVAILABLE"),
    (True, -350_000_001, "CLOUD_STALE"),
])
def test_unusable_context_publishes_empty_cloud(preview, provide_tf, offset, expected):
    node, clock = preview
    message = inputs(node, clock, provide_tf=provide_tf, stamp=clock.ns + offset)
    node.cloud_callback(message)
    node.process_latest()
    assert node.status_pub.messages[-1].data == expected
    assert node.cloud_pub.messages[-1].width == 0


@pytest.mark.parametrize("fault, expected", [
    ("missing_imu", "IMU_MISSING"),
    ("bad_orientation", "GEOMETRY_INVALID"),
    ("sparse_cloud", "FLOOR_UNKNOWN"),
])
def test_unknown_inputs_clear_diagnostic_output(preview, fault, expected):
    node, clock = preview
    message = inputs(node, clock)
    if fault == "missing_imu":
        node.imus.clear()
    elif fault == "bad_orientation":
        node.imus[-1].orientation.w = 0.0
    else:
        message = create_cloud_xyz32(
            message.header, np.zeros((2, 3), dtype=np.float32))
    node.cloud_callback(message)
    node.process_latest()
    assert node.status_pub.messages[-1].data == expected
    assert node.cloud_pub.messages[-1].width == 0


def test_near_floor_fit_retains_obstacles_beyond_fit_region(preview):
    node, clock = preview
    message = inputs(node, clock)
    existing = read_points_numpy(message, field_names=("x", "y", "z"))
    clutter = np.column_stack((
        np.linspace(1.4, 1.8, 1600),
        np.linspace(-0.6, 0.6, 1600),
        np.linspace(-0.4, 0.1, 1600)))
    optical = transform_points(clutter - [0., 0., 0.5],
                               [0., 0., 0.], [0.5, -0.5, 0.5, 0.5])
    message = create_cloud_xyz32(
        message.header, np.vstack((existing, optical)).astype(np.float32))
    node.cloud_callback(message)
    node.process_latest()
    assert node.status_pub.messages[-1].data == "DIAGNOSTIC_CANDIDATE"
    output = read_points_numpy(
        node.cloud_pub.messages[-1], field_names=("x", "y", "z"))
    assert np.any(output[:, 0] > 1.3)


def test_only_newest_cloud_is_processed(preview):
    node, clock = preview
    older = inputs(node, clock, stamp=clock.ns - 66_000_000)
    newer = inputs(node, clock)
    node.cloud_callback(older)
    node.cloud_callback(newer)
    assert node.cloud_pub.messages == []
    node.process_latest()
    assert len(node.cloud_pub.messages) == 1
    assert node.cloud_pub.messages[-1].header.stamp == newer.header.stamp
    assert node.status_pub.messages[-1].data == "DIAGNOSTIC_CANDIDATE"


def test_pending_cloud_recovers_when_exact_tf_arrives(preview):
    node, clock = preview
    message = inputs(node, clock, provide_tf=False)
    node.cloud_callback(message)
    node.process_latest()
    assert node.pending is message
    assert node.previous is None
    inputs(node, clock, provide_tf=True)
    node.process_latest()
    assert node.pending is None
    assert node.status_pub.messages[-1].data == "DIAGNOSTIC_CANDIDATE"
    assert node.cloud_pub.messages[-1].header.stamp == message.header.stamp


def test_retry_grace_never_retains_expired_output(preview):
    node, clock = preview
    node.received = time.monotonic()
    node.active_stamp = clock.ns
    assert node.can_defer()
    node.received -= 0.061
    assert not node.can_defer()
    node.received = time.monotonic()
    node.active_stamp = clock.ns - 350_000_001
    assert not node.can_defer()
