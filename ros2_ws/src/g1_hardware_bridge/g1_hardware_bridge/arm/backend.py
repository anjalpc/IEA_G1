"""Unitree arm backend for the physical G1 bridge."""

import json
from collections import deque
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
        "/home/unitree/cyclonedds_ws/install/cyclonedds/lib",
    )

    if os.path.isdir(library):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{library}:{existing}"
            if existing
            else library
        )

    return env


class PresentationOutput:
    """Drain both pipes continuously and retain protocol results for waiters."""

    def __init__(self, process):
        self.process = process
        self.condition = threading.Condition()
        self.results = {}
        self.errors = deque(maxlen=20)
        self.stdout_done = False
        self.present_cancelled = False
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

    def cancel_present(self):
        """Wake a waiter whose outward presentation was interrupted."""
        with self.condition:
            self.present_cancelled = True
            self.condition.notify_all()

    def wait(self, marker, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while marker not in self.results:
                if marker == "PRESENTED=" and self.present_cancelled:
                    raise RuntimeError(
                        "arm presentation interrupted before PRESENTED")
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
    """Own firmware actions and persistent live Arm-SDK presentations."""

    RELEASE_ACTION = 99
    ARM_SDK_OCCUPIED = 7400

    RELEASE_BUSY_ATTEMPTS = 3
    RELEASE_BUSY_DELAY_S = 0.5

    def __init__(
        self,
        interface,
        timeout,
        card_presentation_mode="disabled",
    ):
        self.interface = interface
        self.timeout = timeout

        self.card_presentation_mode = str(
            card_presentation_mode
        )

        if self.card_presentation_mode not in (
            "disabled",
            "live",
        ):
            raise ValueError(
                "card_presentation_mode must be "
                "'disabled' or 'live'"
            )

        self._lock = threading.Lock()
        self._presentation_release_lock = threading.Lock()
        self._presentation_output = None

        self._active_process = None
        self._active_is_release = False
        self._active_is_presentation = False

        self._presentation_move_seconds = None

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
            self._active_is_presentation = False
            self._presentation_move_seconds = None
            self._cancelled = False

    def prepare_presentation(self):
        if self.card_presentation_mode != "live":
            raise RuntimeError(
                "physical card presentation is disabled"
            )

        self.prepare_operation()

    def interrupt(self):
        with self._lock:
            self._cancelled = True
            process = self._active_process
            is_release = self._active_is_release
            is_presentation = self._active_is_presentation

        if (
            process is None
            or process.poll() is not None
            or is_release
        ):
            return

        if is_presentation:
            self._request_presentation_release(process)
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

    def present(
        self,
        pose,
        fraction,
        move_seconds,
    ):
        if pose != "left_card_presentation":
            raise ValueError(
                "unsupported card presentation pose"
            )

        if self.card_presentation_mode != "live":
            raise RuntimeError(
                "physical card presentation is disabled"
            )

        fraction = float(fraction)
        move_seconds = float(move_seconds)

        arguments = [
            sys.executable,
            "-m",
            "g1_hardware_bridge.arm.presentation_helper",
            self.interface,
            "--fraction",
            str(fraction),
            "--move-seconds",
            str(move_seconds),
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

            self._presentation_output = PresentationOutput(process)
            self._active_process = process
            self._active_is_release = False
            self._active_is_presentation = True
            self._presentation_move_seconds = move_seconds

        marker = "PRESENTED="

        try:
            payload = self._wait_for_marker(
                process,
                marker,
                move_seconds + 10.0,
            )

            return json.loads(payload)

        except Exception:
            # A failed present may still own authority. Wait for graceful
            # cleanup, preserving the process lock if cleanup cannot finish.
            try:
                self.release_presentation()
            except Exception:
                pass
            raise

    def has_presentation(self):
        """Include failed/exited sessions until their release is acknowledged."""
        with self._lock:
            return self._active_is_presentation

    def release_presentation(self):
        # Concurrent cancellation/shutdown callers share one physical cleanup.
        with self._presentation_release_lock:
            return self._release_presentation()

    def _release_presentation(self):
        with self._lock:
            self._cancelled = True
            process = self._active_process
            is_presentation = self._active_is_presentation
            move_seconds = self._presentation_move_seconds
            output = self._presentation_output
        if process is None or not is_presentation:
            return {"mode": "live", "released": True, "already_idle": True}

        if output is not None:
            output.cancel_present()
        self._request_presentation_release(process)
        payload = self._wait_for_marker(
            process, "RELEASED=", float(move_seconds or 8.0) + 8.0)
        result = json.loads(payload)
        if result.get("released") is not True:
            raise RuntimeError("arm helper did not confirm authority release")
        returncode = process.wait(timeout=2.0)
        if returncode != 0:
            raise RuntimeError(f"arm presentation helper failed: rc={returncode}")
        with self._lock:
            if self._active_process is process:
                self._clear_active_locked()
        return result

    def release(self):
        if self.has_presentation():
            self.release_presentation()
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

    def _request_presentation_release(self, process):
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
            output = self._presentation_output
        if output is None or output.process is not process:
            raise RuntimeError("presentation output session is unavailable")
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
            self._active_is_presentation = False

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
        self._presentation_output = None
        self._active_process = None
        self._active_is_release = False
        self._active_is_presentation = False
        self._presentation_move_seconds = None
