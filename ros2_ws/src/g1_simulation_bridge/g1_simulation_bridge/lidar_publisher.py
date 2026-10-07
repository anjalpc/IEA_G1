"""Publish scheduled simulated MID-360 point clouds."""

import math

from sensor_msgs.msg import PointCloud2

from g1_simulation_bridge.lidar_messages import (
    build_lidar_point_cloud,
    lidar_qos,
)


LIDAR_POINTS_TOPIC = "/g1/sensors/lidar/points"


class SimulationLidarPublisher:
    """Compose, raycast, encode, and publish one due lidar state."""

    def __init__(
        self,
        *,
        node,
        ray_adapter,
        schedule,
        geometry,
        directions_sensor,
        frame_id,
        range_min,
        range_max,
    ):
        self._ray_adapter = ray_adapter
        self._schedule = schedule
        self._geometry = geometry
        self._directions_sensor = tuple(
            directions_sensor
        )
        self._frame_id = frame_id
        self._range_min = range_min
        self._range_max = range_max

        self._publisher = node.create_publisher(
            PointCloud2,
            LIDAR_POINTS_TOPIC,
            lidar_qos(),
        )

    def after_step(self, *, simulation_time):
        """Publish one cloud only when simulation time makes it due."""
        decision = self._schedule.after_step(
            simulation_time
        )
        if not decision.publish_lidar:
            return

        torso_pose = self._ray_adapter.torso_pose()
        sensor_pose = (
            self._geometry.compose_mid360_pose(
                torso_position_xyz=(
                    torso_pose.position_xyz
                ),
                torso_orientation_wxyz=(
                    torso_pose.orientation_wxyz
                ),
            )
        )
        directions_world = (
            self._geometry.sensor_directions_to_world(
                sensor_pose,
                self._directions_sensor,
            )
        )
        intersections = self._ray_adapter.cast(
            origin_world=sensor_pose.position_xyz,
            directions_world=directions_world,
        )
        distances = tuple(
            math.inf
            if intersection is None
            else intersection.distance
            for intersection in intersections
        )
        points_sensor = (
            self._geometry.points_from_ray_distances(
                directions_sensor=(
                    self._directions_sensor
                ),
                distances=distances,
                range_min=self._range_min,
                range_max=self._range_max,
            )
        )
        message = build_lidar_point_cloud(
            points_sensor=points_sensor,
            timestamp=decision.stamp,
            frame_id=self._frame_id,
        )
        self._publisher.publish(message)
