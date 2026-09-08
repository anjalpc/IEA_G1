"""Tests for interruptible hardware-helper process ownership."""

import subprocess
import sys
import threading

import pytest

from g1_hardware_bridge.hand.backend import BrainCoHand
from g1_hardware_bridge.arm import backend as unitree_arm
from g1_hardware_bridge.arm.backend import UnitreeArm


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
    return BrainCoHand("eth0", 0.1)


def test_hand_release_uses_configured_neutral_target(monkeypatch):
    backend = hand()
    process = FakeProcess(finish_immediately=True)
    captured = {}

    def popen(arguments, **kwargs):
        captured["arguments"] = arguments
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    backend.neutral("right", [0.05] * 6, 0.75)
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


def test_card_presentation_is_disabled_by_default():
    backend = arm()
    with pytest.raises(RuntimeError, match="disabled"):
        backend.prepare_presentation()


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
        backend.command_position("right", [0.2] * 6, 0.5)


def test_hand_interrupts_normal_helper(monkeypatch):
    backend = hand()
    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    backend.prepare_operation()
    worker, errors = run_in_thread(
        lambda: backend.command_position("right", [0.2] * 6, 0.5))
    assert process.started.wait(1.0)
    backend.interrupt()
    worker.join(1.0)
    assert process.terminated
    assert errors and "cancelled" in str(errors[0])


def test_hand_interrupt_does_not_stop_opening(monkeypatch):
    backend = hand()
    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    worker, errors = run_in_thread(
        lambda: backend.neutral("right", [0.05] * 6, 0.75))
    assert process.started.wait(1.0)
    backend.interrupt()
    assert not process.terminated
    process.complete()
    worker.join(1.0)
    assert not errors


@pytest.fixture
def live_arm(monkeypatch):
    """Use real pipes and process exit, but never import or initialize DDS."""
    backend = UnitreeArm("lo", 0.1, "live")
    real_popen = subprocess.Popen
    processes = []

    def popen(arguments, **kwargs):
        assert "g1_hardware_bridge.arm.presentation_helper" in arguments
        script = (
            "import sys\n"
            "print('PRESENTED={\"mode\":\"live\"}', flush=True)\n"
            "assert sys.stdin.readline() == 'RELEASE\\n'\n"
            "print('RELEASED={\"mode\":\"live\",\"released\":true}', flush=True)\n"
        )
        process = real_popen([sys.executable, "-c", script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    backend.prepare_presentation()
    yield backend
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


def test_live_present_persists_and_release_uses_same_process(live_arm):
    assert live_arm.present("left_card_presentation", 0.25, 5.0) == {"mode": "live"}
    process = live_arm._active_process
    assert process.poll() is None
    assert live_arm.release_presentation()["released"] is True
    assert process.poll() == 0
    assert live_arm._active_process is None


def test_generic_release_waits_for_card_without_firmware_action(live_arm):
    live_arm.present("left_card_presentation", 1.0, 8.0)
    assert live_arm.release() == 0
    assert live_arm._active_process is None


def test_concurrent_release_waiters_share_cleanup(live_arm):
    live_arm.present("left_card_presentation", 1.0, 8.0)
    workers = [run_in_thread(live_arm.release_presentation) for _ in range(2)]
    for worker, errors in workers:
        worker.join(2)
        assert not worker.is_alive()
        assert not errors


def test_release_timeout_keeps_live_process_owned(live_arm, monkeypatch):
    live_arm.present("left_card_presentation", 1.0, 8.0)
    process = live_arm._active_process
    monkeypatch.setattr(live_arm, "_request_presentation_release", lambda _: None)

    def timeout(*args):
        raise TimeoutError("simulated release timeout")

    monkeypatch.setattr(live_arm, "_wait_for_marker", timeout)
    with pytest.raises(TimeoutError):
        live_arm.release_presentation()
    assert live_arm._active_process is process
    assert live_arm.has_presentation()
    assert process.poll() is None
    with pytest.raises(RuntimeError, match="still running"):
        live_arm.prepare_operation()


def test_marker_survives_immediate_exit_and_stderr_flood():
    process = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; sys.stderr.write('diagnostic\\n' * 10000); "
         "print('RELEASED={\"released\":true}', flush=True)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = unitree_arm.PresentationOutput(process)
    try:
        assert output.wait("RELEASED=", 3) == '{"released":true}'
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
