"""Unitree RGB image source.

This module wraps Unitree's VideoClient.GetImageSample() API.

It depends on Unitree SDK2 and therefore must only be used by the standalone
video-helper process, never by the rclpy-facing hand_presence_node process.

ChannelFactoryInitialize(...) must be called by the owning process before
start().
"""


class UnitreeVideoSource:
    """Fetch and decode RGB frames from Unitree videohub."""

    def __init__(self, timeout_s=3.0):
        self.timeout_s = float(timeout_s)
        self._client = None

    def start(self):
        from unitree_sdk2py.go2.video.video_client import (
            VideoClient,
        )

        if self._client is not None:
            raise RuntimeError(
                "Unitree video source already started"
            )

        client = VideoClient()
        client.SetTimeout(self.timeout_s)
        client.Init()

        self._client = client

    def read_frame(self):
        """Fetch and return one BGR OpenCV frame.

        Return None if Unitree's video service reports an error or the JPEG
        cannot be decoded.
        """
        import cv2
        import numpy as np

        if self._client is None:
            raise RuntimeError(
                "Unitree video source is not started"
            )

        code, data = self._client.GetImageSample()

        if code != 0 or not data:
            return None

        encoded = np.frombuffer(
            bytes(data),
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        return frame

    def stop(self):
        self._client = None
