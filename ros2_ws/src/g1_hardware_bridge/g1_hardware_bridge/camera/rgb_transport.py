"""DDS-free shared-memory transport for Unitree RGB frames."""

import fcntl
import os
import struct
import time
from multiprocessing import shared_memory

import numpy as np


SHM_NAME = "g1_handshake_rgb"
LOCK_PATH = "/tmp/g1_handshake_rgb.lock"

MAGIC = 0x47314852
VERSION = 1

WIDTH = 1920
HEIGHT = 1080
CHANNELS = 3

FRAME_BYTES = (
    WIDTH
    * HEIGHT
    * CHANNELS
)

# magic
# version
# valid
# sequence
# timestamp_monotonic_ns
# width
# height
# channels
# payload_bytes
HEADER_FORMAT = "<IIIQQIIII"
HEADER_SIZE = struct.calcsize(
    HEADER_FORMAT
)

TOTAL_SIZE = HEADER_SIZE + FRAME_BYTES


class RgbFrameClient:
    """Read the newest Unitree RGB frame from shared memory."""

    def __init__(self):
        self._shm = None
        self._lock_fd = None

    def _open_lock(self):
        if self._lock_fd is None:
            self._lock_fd = os.open(
                LOCK_PATH,
                os.O_CREAT | os.O_RDWR,
                0o666,
            )

    def _ensure_shared_memory(self):
        if self._shm is not None:
            return True

        try:
            shm = shared_memory.SharedMemory(
                name=SHM_NAME,
                create=False,
            )
        except FileNotFoundError:
            return False

        if shm.size < TOTAL_SIZE:
            shm.close()
            return False

        # Python 3.8's multiprocessing resource tracker can otherwise try
        # to unlink shared memory that this reader does not own when the
        # reader exits.
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(
                shm._name,
                "shared_memory",
            )
        except Exception:
            pass

        self._shm = shm
        self._open_lock()

        return True

    def read_latest(self):
        """
        Atomically return (frame, age_seconds, sequence).

        Returns (None, None, None) when the helper has not created or
        initialized the shared-memory stream yet.
        """
        if not self._ensure_shared_memory():
            return None, None, None

        fcntl.flock(
            self._lock_fd,
            fcntl.LOCK_SH,
        )

        try:
            header_bytes = bytes(
                self._shm.buf[:HEADER_SIZE]
            )

            (
                magic,
                version,
                valid,
                sequence,
                timestamp_ns,
                width,
                height,
                channels,
                payload_bytes,
            ) = struct.unpack(
                HEADER_FORMAT,
                header_bytes,
            )

            if (
                magic != MAGIC
                or version != VERSION
                or valid != 1
                or timestamp_ns == 0
                or width != WIDTH
                or height != HEIGHT
                or channels != CHANNELS
                or payload_bytes != FRAME_BYTES
            ):
                return None, None, None

            # COPY the pixels while the lock is held.
            # Never return a view directly into mutable shared memory.
            raw = bytes(
                self._shm.buf[
                    HEADER_SIZE:
                    HEADER_SIZE + FRAME_BYTES
                ]
            )

        finally:
            fcntl.flock(
                self._lock_fd,
                fcntl.LOCK_UN,
            )

        frame = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).reshape(
            HEIGHT,
            WIDTH,
            CHANNELS,
        )

        age_seconds = max(
            0.0,
            (
                time.monotonic_ns()
                - timestamp_ns
            )
            / 1_000_000_000.0,
        )

        return (
            frame,
            age_seconds,
            sequence,
        )

    def get_latest_frame(self):
        """Return the newest frame or None."""
        frame, _age, _sequence = (
            self.read_latest()
        )
        return frame

    def frame_age(self):
        """Return current frame age in seconds, or None."""
        _frame, age, _sequence = (
            self.read_latest()
        )
        return age

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None

        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
