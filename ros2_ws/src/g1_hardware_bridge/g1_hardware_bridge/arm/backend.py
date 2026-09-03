"""Unitree arm-action backend for the physical G1 bridge."""

import json
import os
import subprocess
import sys
import threading
import time


def native_dds_environment():
    """Return an environment using PC2's native CycloneDDS library."""
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = "0"
    library = env.get(
        "UNITREE_CYCLONEDDS_LIB",
        "/home/unitree/cyclonedds_ws/install/cyclonedds/lib")
    if os.path.isdir(library):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{library}:{existing}" if existing else library)
    return env


class UnitreeArm:
    """Execute interruptible Unitree actions in an isolated DDS process."""

    RELEASE_ACTION = 99
    ARM_SDK_OCCUPIED = 7400
    RELEASE_BUSY_ATTEMPTS = 3
    RELEASE_BUSY_DELAY_S = 0.5

    def __init__(self, interface, timeout, card_presentation_mode="disabled"):
        self.interface = interface
        self.timeout = timeout
        self.card_presentation_mode = str(card_presentation_mode)
        if self.card_presentation_mode not in ("disabled", "offline_shadow"):
            raise ValueError("unsupported card presentation mode")
        self._lock = threading.Lock()
        self._active_process = None
        self._active_is_release = False
        self._cancelled = True

    def prepare_operation(self):
        with self._lock:
            if (self._active_process is not None
                    and self._active_process.poll() is None):
                raise RuntimeError("arm helper is still running")
            self._active_process = None
            self._cancelled = False

    def prepare_presentation(self):
        """Reject physical card workflows unless read-only shadow is enabled."""
        if self.card_presentation_mode == "disabled":
            raise RuntimeError("physical card presentation is disabled")
        self.prepare_operation()

    def interrupt(self):
        """Stop an active normal action without stopping safe release."""
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

    def execute(self, action_id):
        return self._execute(action_id, is_release=False)

    def present(self, pose, fraction, move_seconds):
        """Calculate a left-card trajectory without publishing commands."""
        if pose != "left_card_presentation":
            raise ValueError("unsupported card presentation pose")
        if self.card_presentation_mode == "disabled":
            raise RuntimeError("physical card presentation is disabled")
        arguments = [
            sys.executable, "-m",
            "g1_hardware_bridge.experimental.arm_shadow_helper",
            self.interface,
            "--fraction", str(float(fraction)),
            "--move-seconds", str(float(move_seconds)),
        ]
        stdout = self._run_helper(
            arguments, float(move_seconds) + 8.0, is_release=False)
        marker = "SHADOW_RESULT="
        for line in reversed(stdout.splitlines()):
            if line.startswith(marker):
                return json.loads(line[len(marker):])
        raise RuntimeError("Unitree arm shadow helper returned no result")

    def release_presentation(self):
        """Release offline shadow state without sending a robot command."""
        self.interrupt()
        return {"mode": self.card_presentation_mode, "published": False}

    def release(self):
        self.interrupt()
        for attempt in range(self.RELEASE_BUSY_ATTEMPTS):
            result = self._execute(self.RELEASE_ACTION, is_release=True)
            if result != self.ARM_SDK_OCCUPIED:
                return result
            if attempt + 1 < self.RELEASE_BUSY_ATTEMPTS:
                time.sleep(self.RELEASE_BUSY_DELAY_S)
        return self.ARM_SDK_OCCUPIED

    def _execute(self, action_id, is_release):
        arguments = [
            sys.executable, "-m",
            "g1_hardware_bridge.arm.action_helper", self.interface,
            str(self.timeout), str(action_id)]
        stdout = self._run_helper(
            arguments, self.timeout + 3.0, is_release)
        marker = "ACTION_RESULT="
        for line in reversed(stdout.splitlines()):
            if line.startswith(marker):
                return int(line[len(marker):])
        raise RuntimeError("Unitree action helper returned no result")

    def _run_helper(self, arguments, timeout, is_release):
        env = native_dds_environment()
        with self._lock:
            if self._cancelled and not is_release:
                raise RuntimeError("arm action cancelled")
            if (self._active_process is not None
                    and self._active_process.poll() is None):
                raise RuntimeError("another arm helper is already running")
            process = subprocess.Popen(
                arguments, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._active_process = process
            self._active_is_release = is_release
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError("Unitree arm helper timed out")
        finally:
            with self._lock:
                if self._active_process is process:
                    self._active_process = None
                    self._active_is_release = False
        if process.returncode != 0:
            with self._lock:
                cancelled = self._cancelled
            if cancelled and not is_release:
                raise RuntimeError("arm action cancelled")
            raise RuntimeError(
                "Unitree action helper failed: "
                f"rc={process.returncode} stderr={stderr.strip()}")
        return stdout
