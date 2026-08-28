"""Unitree arm-action backend for the physical G1 bridge."""

import os
import subprocess
import sys
import threading
import time

from g1_robot_interfaces.constants import RELEASE_ARM_ACTION


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

    RELEASE_ACTION = RELEASE_ARM_ACTION
    ARM_SDK_OCCUPIED = 7400
    RELEASE_BUSY_ATTEMPTS = 3
    RELEASE_BUSY_DELAY_S = 0.5

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
                raise RuntimeError("arm helper is still running")
            self._active_process = None
            self._cancelled = False

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
        env = native_dds_environment()
        with self._lock:
            if self._cancelled and not is_release:
                raise RuntimeError("arm action cancelled")
            if (self._active_process is not None
                    and self._active_process.poll() is None):
                raise RuntimeError("another arm helper is already running")
            process = subprocess.Popen(
                [sys.executable, "-m",
                 "g1_hardware_bridge.unitree_action_helper", self.interface,
                 str(self.timeout), str(action_id)], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._active_process = process
            self._active_is_release = is_release
        try:
            stdout, stderr = process.communicate(timeout=self.timeout + 3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError(f"Unitree arm action {action_id} timed out")
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
        marker = "ACTION_RESULT="
        for line in reversed(stdout.splitlines()):
            if line.startswith(marker):
                return int(line[len(marker):])
        raise RuntimeError(
            "Unitree action helper returned no result: "
            f"stderr={stderr.strip()}")
