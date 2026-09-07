"""
RealSense depth-camera backend.

The Unitree RGB camera is owned by videohub_pc4. This backend therefore
opens only the D435i depth stream.

The camera bridge is the sole live-device owner.
"""


class DepthCamera:
    """Read RealSense depth frames in millimeters."""

    def __init__(self, bag_path=None):
        self.bag_path = (
            str(bag_path)
            if bag_path
            else None
        )

        self._pipeline = None
        self._depth_scale = None

    def start(self):
        import pyrealsense2 as rs

        if self._pipeline is not None:
            raise RuntimeError(
                "depth camera is already started"
            )

        pipeline = rs.pipeline()
        config = rs.config()

        if self.bag_path:
            config.enable_device_from_file(
                self.bag_path,
                repeat_playback=True,
            )
        else:
            config.enable_stream(
                rs.stream.depth,
                640,
                480,
                rs.format.z16,
                30,
            )

        profile = pipeline.start(config)

        if self.bag_path:
            profile.get_device().as_playback().set_real_time(
                False
            )

        depth_sensor = (
            profile.get_device()
            .first_depth_sensor()
        )

        self._depth_scale = (
            depth_sensor.get_depth_scale()
        )

        self._pipeline = pipeline

    def read_frame_mm(
        self,
        timeout_ms=2000,
    ):
        """Return the next depth frame in millimeters."""
        import numpy as np

        if self._pipeline is None:
            raise RuntimeError(
                "depth camera is not started"
            )

        frames = (
            self._pipeline.wait_for_frames(
                timeout_ms
            )
        )

        depth_frame = (
            frames.get_depth_frame()
        )

        if not depth_frame:
            raise RuntimeError(
                "no depth frame available"
            )

        raw_depth = np.asanyarray(
            depth_frame.get_data()
        )

        return (
            raw_depth.astype(np.float32)
            * (
                self._depth_scale
                * 1000.0
            )
        )

    def stop(self):
        if self._pipeline is not None:
            self._pipeline.stop()

        self._pipeline = None
        self._depth_scale = None
