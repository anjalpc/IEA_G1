"""Unitree arm backend for the physical G1 bridge."""

import json
import math
from collections import deque
import subprocess
import sys
import threading
import time

from g1_robot_interfaces.arm_motion import ARM_JOINT_NAMES, validate_arm_motion
from g1_hardware_bridge.native_runtime import native_dds_environment


def validate_motion_envelope(motion):
    """
    Enforce the initial bounded Arm SDK capability, before starting DDS.

    Bounds come from the existing measured motion amplitudes,
    not vendor joint limits or permission for arbitrary trajectories. Version 1
    supports bilateral preparation and left-arm offsets only. Expansion needs
    separate physical validation. Requests cannot supply gains or safety policy.
    """
    motion = validate_arm_motion(motion)
    if motion['phase_seconds'] != [0.30, 0.90]:
        raise ValueError('unsupported phase timing for arm capability v1')
    if set(motion['target_offsets']) != set(ARM_JOINT_NAMES[:7]):
        raise ValueError('arm capability v1 requires the seven left-arm offsets')
    # Preserve the sign as well as the magnitude of the one physically tested
    # recipe. This is only a pre-DDS request gate. It is not a general physical
    # envelope; resolved displacement limits still require real robot traces.
    preparation_limits = (
        0.286, 0.218, -0.009, 0.978, 0.144, 0.006, -0.004,
        0.286, -0.218, 0.009, 0.978, -0.144, 0.006, 0.004,
    )
    offset_limits = (
        -0.348308, -0.206418, -0.015288, -0.619944,
        -0.164438, -0.347043, 0.065396,
    )
    for values, names, limits in (
            (motion['preparation'], ARM_JOINT_NAMES, preparation_limits),
            (motion['target_offsets'], ARM_JOINT_NAMES[:7], offset_limits)):
        for name, limit in zip(names, limits):
            value = values[name]
            low, high = sorted((0.0, limit))
            if value < low - 1e-9 or value > high + 1e-9:
                raise ValueError(f'motion exceeds capability envelope: {name}')
    return motion


class ArmMotionOutput:
    """Drain both pipes continuously and retain protocol results for waiters."""

    def __init__(self, process):
        self.process = process
        self.condition = threading.Condition()
        self.results = {}
        self.errors = deque(maxlen=20)
        self.stdout_done = False
        self.motion_cancelled = False
        for stream, is_stdout in ((process.stdout, True), (process.stderr, False)):
            threading.Thread(target=self._read, args=(stream, is_stdout),
                             daemon=True).start()

    def _read(self, stream, is_stdout):
        try:
            for line in stream:
                with self.condition:
                    if is_stdout:
                        for marker in ("PRESENTED=", "RELEASED="):
                            if line.startswith(marker):
                                self.results[marker] = line[len(marker):].strip()
                    else:
                        self.errors.append(line.strip())
                    self.condition.notify_all()
        finally:
            with self.condition:
                if is_stdout:
                    self.stdout_done = True
                self.condition.notify_all()

    def cancel_motion(self):
        """Wake a waiter whose outward motion was interrupted."""
        with self.condition:
            self.motion_cancelled = True
            self.condition.notify_all()

    def wait(self, marker, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while marker not in self.results:
                if marker == "PRESENTED=" and self.motion_cancelled:
                    raise RuntimeError(
                        "arm motion interrupted before completion")
                if self.stdout_done:
                    raise RuntimeError(
                        "Unitree arm helper closed stdout before "
                        f"{marker}: " + "\n".join(self.errors))
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Unitree arm helper timed out waiting for {marker}")
                self.condition.wait(remaining)
            return self.results[marker]


class UnitreeArm:
    """Own firmware actions and persistent restricted Arm SDK motion."""

    RELEASE_ACTION = 99
    ARM_SDK_OCCUPIED = 7400

    RELEASE_BUSY_ATTEMPTS = 3
    RELEASE_BUSY_DELAY_S = 0.5

    def __init__(
        self,
        interface,
        timeout,
        arm_motion_mode="disabled",
    ):
        self.interface = interface
        self.timeout = timeout

        self.arm_motion_mode = str(
            arm_motion_mode
        )

        if self.arm_motion_mode not in (
            "disabled",
            "live",
        ):
            raise ValueError(
                "arm_motion_mode must be "
                "'disabled' or 'live'"
            )

        self._lock = threading.Lock()
        self._motion_release_lock = threading.Lock()
        self._motion_output = None

        self._active_process = None
        self._active_is_release = False
        self._active_is_motion = False

        self._motion_timeout_seconds = None

        self._cancelled = True

    def prepare_operation(self):
        with self._lock:
            if (
                self._active_process is not None
                and self._active_process.poll() is None
            ):
                raise RuntimeError(
                    "arm helper is still running"
                )

            self._active_process = None
            self._active_is_release = False
            self._active_is_motion = False
            self._motion_timeout_seconds = None
            self._cancelled = False

    def prepare_motion(self):
        if self.arm_motion_mode != "live":
            raise RuntimeError(
                "physical Arm SDK motion is disabled"
            )

        self.prepare_operation()

    def interrupt(self):
        with self._lock:
            self._cancelled = True
            process = self._active_process
            is_release = self._active_is_release
            is_motion = self._active_is_motion

        if (
            process is None
            or process.poll() is not None
            or is_release
        ):
            return

        if is_motion:
            self._request_motion_release(process)
            return

        process.terminate()

        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def execute(self, action_id):
        return self._execute(
            action_id,
            is_release=False,
        )

    def execute_motion(self, motion, execution_timeout_s):
        motion = validate_motion_envelope(motion)
        if self.arm_motion_mode != "live":
            raise RuntimeError(
                "physical Arm SDK motion is disabled"
            )

        execution_timeout_s = float(execution_timeout_s)
        if not math.isfinite(execution_timeout_s) or execution_timeout_s <= 0.0:
            raise ValueError("execution_timeout_s must be finite and positive")

        arguments = [
            sys.executable,
            "-m",
            "g1_hardware_bridge.arm.arm_motion_helper",
            self.interface,
            "--motion-json",
            json.dumps(motion, allow_nan=False),
        ]

        env = native_dds_environment()

        with self._lock:
            if self._cancelled:
                raise RuntimeError(
                    "arm action cancelled"
                )

            if (
                self._active_process is not None
                and self._active_process.poll() is None
            ):
                raise RuntimeError(
                    "another arm helper is already running"
                )

            process = subprocess.Popen(
                arguments,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self._motion_output = ArmMotionOutput(process)
            self._active_process = process
            self._active_is_release = False
            self._active_is_motion = True
            self._motion_timeout_seconds = execution_timeout_s

        marker = "PRESENTED="

        try:
            payload = self._wait_for_marker(
                process,
                marker,
                execution_timeout_s + 10.0,
            )

            return json.loads(payload)

        except Exception:
            # A failed motion may still own authority. Wait for graceful
            # cleanup, preserving the process lock if cleanup cannot finish.
            try:
                self.release_motion()
            except Exception:
                pass
            raise

    def has_motion(self):
        """Include failed/exited sessions until their release is acknowledged."""
        with self._lock:
            return self._active_is_motion

    def release_motion(self):
        # Concurrent cancellation/shutdown callers share one physical cleanup.
        with self._motion_release_lock:
            return self._release_motion()

    def _release_motion(self):
        with self._lock:
            self._cancelled = True
            process = self._active_process
            is_motion = self._active_is_motion
            execution_timeout_s = self._motion_timeout_seconds
            output = self._motion_output
        if process is None or not is_motion:
            return {"mode": "live", "released": True, "already_idle": True}

        if output is not None:
            output.cancel_motion()
        self._request_motion_release(process)
        payload = self._wait_for_marker(
            process, "RELEASED=", float(execution_timeout_s or 8.0) + 8.0)
        result = json.loads(payload)
        if result.get("released") is not True:
            raise RuntimeError("arm helper did not confirm authority release")
        returncode = process.wait(timeout=2.0)
        if returncode != 0:
            raise RuntimeError(f"arm motion helper failed: rc={returncode}")
        with self._lock:
            if self._active_process is process:
                self._clear_active_locked()
        return result

    def release(self):
        if self.has_motion():
            self.release_motion()
            return 0
        self.interrupt()

        for attempt in range(
            self.RELEASE_BUSY_ATTEMPTS
        ):
            result = self._execute(
                self.RELEASE_ACTION,
                is_release=True,
            )

            if result != self.ARM_SDK_OCCUPIED:
                return result

            if (
                attempt + 1
                < self.RELEASE_BUSY_ATTEMPTS
            ):
                time.sleep(
                    self.RELEASE_BUSY_DELAY_S
                )

        return self.ARM_SDK_OCCUPIED

    def _request_motion_release(self, process):
        if process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                with self._lock:
                    process.stdin.write("RELEASE\n")
                    process.stdin.flush()
                return
        except (BrokenPipeError, OSError):
            pass
        # SIGTERM is a release request in the helper, never a kill-first path.
        process.terminate()

    def _wait_for_marker(self, process, marker, timeout):
        with self._lock:
            output = self._motion_output
        if output is None or output.process is not process:
            raise RuntimeError("arm motion output session is unavailable")
        return output.wait(marker, timeout)

    def _execute(
        self,
        action_id,
        is_release,
    ):
        arguments = [
            sys.executable,
            "-m",
            "g1_hardware_bridge.arm.action_helper",
            self.interface,
            str(self.timeout),
            str(action_id),
        ]

        stdout = self._run_helper(
            arguments,
            self.timeout + 3.0,
            is_release,
        )

        marker = "ACTION_RESULT="

        for line in reversed(
            stdout.splitlines()
        ):
            if line.startswith(marker):
                return int(
                    line[len(marker):]
                )

        raise RuntimeError(
            "Unitree action helper returned no result"
        )

    def _run_helper(
        self,
        arguments,
        timeout,
        is_release,
    ):
        env = native_dds_environment()

        with self._lock:
            if self._cancelled and not is_release:
                raise RuntimeError(
                    "arm action cancelled"
                )

            if (
                self._active_process is not None
                and self._active_process.poll() is None
            ):
                raise RuntimeError(
                    "another arm helper is already running"
                )

            process = subprocess.Popen(
                arguments,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self._active_process = process
            self._active_is_release = is_release
            self._active_is_motion = False

        try:
            stdout, stderr = process.communicate(
                timeout=timeout
            )

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

            raise TimeoutError(
                "Unitree arm helper timed out"
            )

        finally:
            with self._lock:
                if self._active_process is process:
                    self._clear_active_locked()

        if process.returncode != 0:
            with self._lock:
                cancelled = self._cancelled

            if cancelled and not is_release:
                raise RuntimeError(
                    "arm action cancelled"
                )

            raise RuntimeError(
                "Unitree action helper failed: "
                f"rc={process.returncode} "
                f"stderr={stderr.strip()}"
            )

        return stdout

    def _clear_active_locked(self):
        self._motion_output = None
        self._active_process = None
        self._active_is_release = False
        self._active_is_motion = False
        self._motion_timeout_seconds = None
