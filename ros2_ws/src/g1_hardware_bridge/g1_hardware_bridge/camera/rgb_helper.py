"""
Standalone Unitree RGB helper.

This process owns Unitree SDK2 / CycloneDDS and writes the newest decoded
1920x1080 BGR frame into shared memory.

There is deliberately no rclpy import in this module.
"""

import fcntl
import os
import struct
import time
from multiprocessing import shared_memory

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
)

from g1_hardware_bridge.camera.rgb_transport import (
    CHANNELS,
    FRAME_BYTES,
    HEADER_FORMAT,
    HEADER_SIZE,
    HEIGHT,
    LOCK_PATH,
    MAGIC,
    SHM_NAME,
    TOTAL_SIZE,
    VERSION,
    WIDTH,
)


TARGET_FPS = 30.0


class UnitreeVideoSource:
    """Fetch and decode RGB frames from the Unitree video service."""

    def __init__(self, timeout_s=3.0):
        self.timeout_s = float(timeout_s)
        self._client = None

    def start(self):
        """Initialize the Unitree video client."""
        from unitree_sdk2py.go2.video.video_client import VideoClient

        if self._client is not None:
            raise RuntimeError("Unitree video source already started")

        client = VideoClient()
        client.SetTimeout(self.timeout_s)
        client.Init()
        self._client = client

    def read_frame(self):
        """Return one decoded BGR frame, or None when unavailable."""
        import cv2
        import numpy as np

        if self._client is None:
            raise RuntimeError("Unitree video source is not started")

        code, data = self._client.GetImageSample()
        if code != 0 or not data:
            return None

        encoded = np.frombuffer(bytes(data), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def stop(self):
        """Release the Unitree video client."""
        self._client = None


def _pack_header(
    valid,
    sequence,
    timestamp_ns,
):
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        int(valid),
        int(sequence),
        int(timestamp_ns),
        WIDTH,
        HEIGHT,
        CHANNELS,
        FRAME_BYTES,
    )


def main():
    """Run the Unitree RGB shared-memory writer."""
    # This process exclusively owns Unitree SDK DDS.
    ChannelFactoryInitialize(
        0,
        "eth0",
    )

    camera = UnitreeVideoSource(
        timeout_s=3.0
    )
    camera.start()

    try:
        shm = shared_memory.SharedMemory(
            name=SHM_NAME,
            create=True,
            size=TOTAL_SIZE,
        )
    except FileExistsError:
        # Reuse a block left behind by an earlier interrupted helper.
        shm = shared_memory.SharedMemory(
            name=SHM_NAME,
            create=False,
        )

        if shm.size < TOTAL_SIZE:
            shm.close()
            raise RuntimeError(
                "existing RGB shared memory has "
                "unexpected size"
            )

    lock_fd = os.open(
        LOCK_PATH,
        os.O_CREAT | os.O_RDWR,
        0o666,
    )

    sequence = 0

    # Mark the stream invalid until the first complete frame is written.
    fcntl.flock(
        lock_fd,
        fcntl.LOCK_EX,
    )

    try:
        shm.buf[:HEADER_SIZE] = (
            _pack_header(
                valid=0,
                sequence=0,
                timestamp_ns=0,
            )
        )
    finally:
        fcntl.flock(
            lock_fd,
            fcntl.LOCK_UN,
        )

    print(
        "Unitree RGB helper running "
        f"({WIDTH}x{HEIGHT}, target {TARGET_FPS:.0f} FPS)"
    )

    period_s = 1.0 / TARGET_FPS

    try:
        while True:
            loop_start = time.monotonic()

            frame = camera.read_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            if frame.shape != (
                HEIGHT,
                WIDTH,
                CHANNELS,
            ):
                raise RuntimeError(
                    "unexpected Unitree RGB frame shape: "
                    f"{frame.shape}; expected "
                    f"({HEIGHT}, {WIDTH}, {CHANNELS})"
                )

            frame_bytes = frame.tobytes(
                order="C"
            )

            if len(frame_bytes) != FRAME_BYTES:
                raise RuntimeError(
                    "unexpected RGB frame byte size"
                )

            sequence += 1
            timestamp_ns = (
                time.monotonic_ns()
            )

            fcntl.flock(
                lock_fd,
                fcntl.LOCK_EX,
            )

            try:
                # Mark invalid while mutating the shared block.
                shm.buf[:HEADER_SIZE] = (
                    _pack_header(
                        valid=0,
                        sequence=sequence,
                        timestamp_ns=timestamp_ns,
                    )
                )

                shm.buf[
                    HEADER_SIZE:
                    HEADER_SIZE + FRAME_BYTES
                ] = frame_bytes

                # Publish metadata only after the full pixel copy is done.
                shm.buf[:HEADER_SIZE] = (
                    _pack_header(
                        valid=1,
                        sequence=sequence,
                        timestamp_ns=timestamp_ns,
                    )
                )

            finally:
                fcntl.flock(
                    lock_fd,
                    fcntl.LOCK_UN,
                )

            elapsed = (
                time.monotonic()
                - loop_start
            )

            remaining = period_s - elapsed

            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print(
            "\nStopping Unitree RGB helper"
        )

    finally:
        # Ensure readers immediately stop trusting the last image.
        try:
            fcntl.flock(
                lock_fd,
                fcntl.LOCK_EX,
            )

            try:
                shm.buf[:HEADER_SIZE] = (
                    _pack_header(
                        valid=0,
                        sequence=sequence,
                        timestamp_ns=0,
                    )
                )
            finally:
                fcntl.flock(
                    lock_fd,
                    fcntl.LOCK_UN,
                )
        except Exception:
            pass

        camera.stop()

        shm.close()

        try:
            shm.unlink()
        except FileNotFoundError:
            pass

        os.close(lock_fd)


if __name__ == "__main__":
    main()
