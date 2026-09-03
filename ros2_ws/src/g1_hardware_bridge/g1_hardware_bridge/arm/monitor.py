"""Supervise the isolated native Unitree arm-state reader process."""

import json
import subprocess
import sys
import threading

from g1_hardware_bridge.arm.state_helper import OUTPUT_PREFIX


class UnitreeArmMonitor:
    """Feed normalized helper output into an arm-ownership tracker."""

    def __init__(self, ownership, environment, process_factory=subprocess.Popen,
                 state_callback=None):
        self.ownership = ownership
        self.environment = dict(environment)
        self.process_factory = process_factory
        self.state_callback = state_callback
        self._lock = threading.Lock()
        self._process = None
        self._reader = None

    def start(self):
        """Start exactly one monitor helper and its output reader."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Unitree arm-state monitor is already running")
            process = self.process_factory(
                [sys.executable, "-m",
                 "g1_hardware_bridge.arm.state_helper"],
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                process.terminate()
                raise RuntimeError("Unitree arm-state monitor has no output pipe")
            self._process = process
            self.ownership.mark_monitor_available()
            self._reader = threading.Thread(
                target=self._read_output,
                args=(process,),
                name="unitree-arm-state-reader",
                daemon=True,
            )
            self._reader.start()

    def _read_output(self, process):
        try:
            for line in process.stdout:
                self.consume_line(line)
            # A real pipe reaches EOF after the child closes stdout, normally
            # at process exit. Test doubles can close it while still alive.
            try:
                process.wait()
            except subprocess.TimeoutExpired:
                return
        finally:
            if process.poll() is not None:
                self.ownership.mark_monitor_unavailable()
                self._notify()

    def _notify(self):
        """Publish the tracker's latest ownership classification."""
        if self.state_callback is not None:
            self.state_callback(self.ownership.snapshot())

    def consume_line(self, line):
        """Consume one helper line; return whether it contained valid state."""
        stripped = str(line).strip()
        if not stripped.startswith(OUTPUT_PREFIX):
            return False
        try:
            state = json.loads(stripped[len(OUTPUT_PREFIX):])
            action_id = int(state["id"])
            name = str(state.get("name", ""))
            holding = bool(state.get("holding", False))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        self.ownership.update(action_id, name, holding)
        self._notify()
        return True

    def stop(self, timeout=2.0):
        """Terminate, escalate if required, and reap the helper process."""
        with self._lock:
            process = self._process
            reader = self._reader
            self._process = None
            self._reader = None
        if process is None:
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
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=timeout)
        self.ownership.mark_monitor_unavailable()
