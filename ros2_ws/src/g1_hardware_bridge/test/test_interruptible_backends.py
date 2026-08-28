"""Tests for interruptible hardware-helper process ownership."""

import subprocess
import threading

import pytest

from g1_hardware_bridge.brainco_hand import BrainCoHand
from g1_hardware_bridge import unitree_arm
from g1_hardware_bridge.unitree_arm import UnitreeArm


class FakeProcess:
    """Controllable stand-in for a hardware helper subprocess."""

    def __init__(self, stdout="", finish_immediately=False):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = None
        self.started = threading.Event()
        self.finished = threading.Event()
        self.terminated = False
        self.killed = False
        if finish_immediately:
            self.complete()

    def complete(self, returncode=0):
        self.returncode = returncode
        self.finished.set()

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self.started.set()
        if not self.finished.wait(timeout):
            raise subprocess.TimeoutExpired("fake-helper", timeout)
        return self.stdout, self.stderr

    def terminate(self):
        self.terminated = True
        self.complete(-15)

    def kill(self):
        self.killed = True
        self.complete(-9)

    def wait(self, timeout=None):
        if not self.finished.wait(timeout):
            raise subprocess.TimeoutExpired("fake-helper", timeout)
        return self.returncode


def arm():
    return UnitreeArm("eth0", 0.1)


def hand():
    return BrainCoHand(
        "eth0", 0.1, [0.05] * 6, [0.2] * 6, 0.5, 0.75)


def test_hand_release_uses_configured_neutral_target(monkeypatch):
    backend = hand()
    process = FakeProcess(finish_immediately=True)
    captured = {}

    def popen(arguments, **kwargs):
        captured["arguments"] = arguments
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    backend.release()
    target_index = captured["arguments"].index("--target")
    assert captured["arguments"][target_index + 1:target_index + 7] == [
        "0.05"] * 6
    timeout_index = captured["arguments"].index("--timeout")
    assert captured["arguments"][timeout_index + 1] == "0.1"
    ramp_index = captured["arguments"].index("--ramp-seconds")
    assert captured["arguments"][ramp_index + 1] == "0.75"


def run_in_thread(command):
    errors = []

    def target():
        try:
            command()
        except Exception as exc:  # expected for an interrupted normal helper
            errors.append(exc)

    worker = threading.Thread(target=target)
    worker.start()
    return worker, errors


def test_arm_rejects_normal_action_until_prepared(monkeypatch):
    backend = arm()
    with pytest.raises(RuntimeError, match="cancelled"):
        backend.execute(27)
    process = FakeProcess("ACTION_RESULT=0\n", finish_immediately=True)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    backend.prepare_operation()
    assert backend.execute(27) == 0


def test_arm_interrupts_normal_helper(monkeypatch):
    backend = arm()
    process = FakeProcess("ACTION_RESULT=0\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    backend.prepare_operation()
    worker, errors = run_in_thread(lambda: backend.execute(27))
    assert process.started.wait(1.0)
    backend.interrupt()
    worker.join(1.0)
    assert process.terminated
    assert errors and "cancelled" in str(errors[0])


def test_arm_interrupt_does_not_stop_release(monkeypatch):
    backend = arm()
    process = FakeProcess("ACTION_RESULT=0\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    worker, errors = run_in_thread(backend.release)
    assert process.started.wait(1.0)
    backend.interrupt()
    assert not process.terminated
    process.complete()
    worker.join(1.0)
    assert not errors


def test_arm_release_retries_only_documented_busy_result(monkeypatch):
    processes = iter([
        FakeProcess("ACTION_RESULT=7400\n", finish_immediately=True),
        FakeProcess("ACTION_RESULT=0\n", finish_immediately=True),
    ])
    sleeps = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: next(processes))
    monkeypatch.setattr(unitree_arm.time, "sleep", sleeps.append)

    assert arm().release() == 0
    assert sleeps == [UnitreeArm.RELEASE_BUSY_DELAY_S]


def test_arm_release_reports_persistent_busy_result(monkeypatch):
    calls = []

    def popen(*args, **kwargs):
        calls.append(args)
        return FakeProcess("ACTION_RESULT=7400\n", finish_immediately=True)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(unitree_arm.time, "sleep", lambda _delay: None)

    assert arm().release() == UnitreeArm.ARM_SDK_OCCUPIED
    assert len(calls) == UnitreeArm.RELEASE_BUSY_ATTEMPTS


def test_hand_rejects_normal_command_until_prepared():
    backend = hand()
    with pytest.raises(RuntimeError, match="cancelled"):
        backend.grip()


def test_hand_interrupts_normal_helper(monkeypatch):
    backend = hand()
    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    backend.prepare_operation()
    worker, errors = run_in_thread(backend.grip)
    assert process.started.wait(1.0)
    backend.interrupt()
    worker.join(1.0)
    assert process.terminated
    assert errors and "cancelled" in str(errors[0])


def test_hand_interrupt_does_not_stop_opening(monkeypatch):
    backend = hand()
    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    worker, errors = run_in_thread(backend.release)
    assert process.started.wait(1.0)
    backend.interrupt()
    assert not process.terminated
    process.complete()
    worker.join(1.0)
    assert not errors
