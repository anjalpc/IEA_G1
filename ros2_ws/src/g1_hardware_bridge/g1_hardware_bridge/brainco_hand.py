"""BrainCo hand backend for the physical G1 bridge."""

import json
import subprocess
import sys
import threading

from g1_hardware_bridge.unitree_arm import native_dds_environment


class BrainCoHand:
    """Send interruptible bounded commands to the BrainCo right hand."""

    def __init__(self, interface, timeout, neutral_target, grip_target,
                 grip_ramp_seconds, release_ramp_seconds):
        self.interface = interface
        self.timeout = timeout
        self.neutral_target = list(neutral_target)
        self.grip_target = list(grip_target)
        self.grip_ramp_seconds = grip_ramp_seconds
        self.release_ramp_seconds = release_ramp_seconds
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
                 "g1_hardware_bridge.brainco_hand_helper", *arguments],
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

    def grip(self):
        stdout = self._run_helper([
            "command-position", "--hand", "right", "--interface",
            self.interface, "--timeout", str(self.timeout), "--ramp-seconds",
            str(self.grip_ramp_seconds), "--confirm",
            "CONFIRM_PHYSICAL_HAND_COMMAND", "--target",
            *[str(value) for value in self.grip_target],
        ], self.timeout + self.grip_ramp_seconds + 5.0)
        return self._json_result(stdout, "BrainCo grip")

    def release(self):
        """Interrupt normal hand motion, then restore the neutral posture."""
        self.interrupt()
        # A release must finish inside ros2 launch's five-second SIGINT grace
        # period. One second is ample to receive the 100 Hz hand state when
        # the BrainCo server is available, and fails promptly when it is not.
        state_timeout = min(self.timeout, 1.0)
        self._run_helper([
            "command-position", "--hand", "right", "--interface",
            self.interface, "--timeout", str(state_timeout), "--target",
            *[str(value) for value in self.neutral_target],
            "--ramp-seconds", str(self.release_ramp_seconds), "--confirm",
            "CONFIRM_PHYSICAL_HAND_COMMAND",
        ], state_timeout + 3.5, is_release=True)
