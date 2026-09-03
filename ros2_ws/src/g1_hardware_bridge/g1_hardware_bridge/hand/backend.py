"""BrainCo hand backend for the physical G1 bridge."""

import json
import subprocess
import sys
import threading

from g1_hardware_bridge.arm.backend import native_dds_environment


class BrainCoHand:
    """Send interruptible bounded position commands to either BrainCo hand."""

    def __init__(self, interface, timeout):
        self.interface = interface
        self.timeout = timeout
        self._lock = threading.Lock()
        self._active_process = None
        self._active_is_release = False
        self._cancelled = True

    def prepare_operation(self):
        with self._lock:
            if (self._active_process is not None
                    and self._active_process.poll() is None):
                raise RuntimeError("hand helper is still running")
            self._active_process = None
            self._cancelled = False

    def interrupt(self):
        """Stop an active normal command without stopping hand opening."""
        with self._lock:
            self._cancelled = True
            process = (None if self._active_is_release
                       else self._active_process)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def _run_helper(self, arguments, timeout, is_release=False):
        env = native_dds_environment()
        with self._lock:
            if self._cancelled and not is_release:
                raise RuntimeError("hand command cancelled")
            if (self._active_process is not None
                    and self._active_process.poll() is None):
                raise RuntimeError("another hand helper is already running")
            process = subprocess.Popen(
                [sys.executable, "-m",
                 "g1_hardware_bridge.hand.helper", *arguments],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True)
            self._active_process = process
            self._active_is_release = is_release
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError("BrainCo helper timed out")
        finally:
            with self._lock:
                if self._active_process is process:
                    self._active_process = None
                    self._active_is_release = False
        if process.returncode != 0:
            with self._lock:
                cancelled = self._cancelled
            if cancelled and not is_release:
                raise RuntimeError("hand command cancelled")
            raise RuntimeError(
                "BrainCo helper failed: "
                f"rc={process.returncode} stderr={stderr.strip()}")
        return stdout

    @staticmethod
    def _json_result(stdout, description):
        json_start = stdout.find("{")
        if json_start < 0:
            raise RuntimeError(f"{description} returned no result")
        return json.loads(stdout[json_start:])

    def command_position(self, hand, target, ramp_seconds):
        stdout = self._run_helper([
            "command-position", "--hand", str(hand), "--interface",
            self.interface, "--timeout", str(self.timeout), "--ramp-seconds",
            str(ramp_seconds), "--confirm",
            "CONFIRM_PHYSICAL_HAND_COMMAND", "--target",
            *[str(value) for value in target],
        ], self.timeout + ramp_seconds + 5.0)
        return self._json_result(stdout, "BrainCo position command")

    def neutral(self, hand, target, ramp_seconds):
        """Interrupt normal hand motion, then restore a calibrated posture."""
        self.interrupt()
        # A release must finish inside ros2 launch's five-second SIGINT grace
        # period. One second is ample to receive the 100 Hz hand state when
        # the BrainCo server is available, and fails promptly when it is not.
        state_timeout = min(self.timeout, 1.0)
        self._run_helper([
            "command-position", "--hand", str(hand), "--interface",
            self.interface, "--timeout", str(state_timeout), "--target",
            *[str(value) for value in target],
            "--ramp-seconds", str(ramp_seconds), "--confirm",
            "CONFIRM_PHYSICAL_HAND_COMMAND",
        ], state_timeout + 3.5, is_release=True)
