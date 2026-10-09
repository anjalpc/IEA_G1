"""Supervise the isolated native physical LowState helper."""

import json
import subprocess
import sys
import threading

from g1_hardware_bridge.low_state.protocol import (
    decode_sample,
)


class LowStateMonitor:
    """Forward validated helper samples to the application ROS node."""

    def __init__(
        self,
        *,
        interface,
        period_seconds,
        environment,
        sample_callback,
        availability_callback=None,
        process_factory=subprocess.Popen,
    ):
        self.interface = interface
        self.period_seconds = period_seconds
        self.environment = dict(environment)
        self.sample_callback = sample_callback
        self.availability_callback = availability_callback
        self.process_factory = process_factory

        self._lock = threading.Lock()
        self._process = None
        self._reader = None
        self._available = False

    def _set_available(self, available):
        available = bool(available)
        with self._lock:
            if available == self._available:
                return
            self._available = available

        if self.availability_callback is not None:
            self.availability_callback(available)

    def start(self):
        """Start exactly one helper process."""
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
            ):
                raise RuntimeError(
                    "LowState helper is already running"
                )

            process = self.process_factory(
                [
                    sys.executable,
                    "-m",
                    "g1_hardware_bridge.low_state.helper",
                    self.interface,
                    "--period-seconds",
                    str(self.period_seconds),
                ],
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )

            if process.stdout is None:
                process.terminate()
                raise RuntimeError(
                    "LowState helper has no output pipe"
                )

            self._process = process
            self._reader = threading.Thread(
                target=self._read_output,
                args=(process,),
                name="unitree-low-state-reader",
                daemon=True,
            )
            reader = self._reader

        self._set_available(True)
        reader.start()

    def _read_output(self, process):
        try:
            for line in process.stdout:
                self.consume_line(line)

            try:
                process.wait()
            except subprocess.TimeoutExpired:
                return
        finally:
            if process.poll() is not None:
                self._set_available(False)

    def consume_line(self, line):
        """Consume one helper protocol line."""
        try:
            sample = decode_sample(line)
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False

        if sample is None:
            return False

        self.sample_callback(sample)
        return True

    def stop(self, timeout=2.0):
        """Terminate and reap the helper process."""
        with self._lock:
            process = self._process
            reader = self._reader
            self._process = None
            self._reader = None

        if process is None:
            self._set_available(False)
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
        else:
            process.wait()

        if (
            reader is not None
            and reader is not threading.current_thread()
        ):
            reader.join(timeout=timeout)

        self._set_available(False)
