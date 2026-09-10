import io
import subprocess

from g1_hardware_bridge.arm.arm_state_monitor import UnitreeArmMonitor
from g1_hardware_bridge.arm.ownership import ArmOwnership


class FakeProcess:
    def __init__(self, lines="", running=True, resist=False):
        self.stdout = io.StringIO(lines)
        self.returncode = None if running else 0
        self.resist = resist
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.resist:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        self.waited = True
        if self.returncode is None:
            raise subprocess.TimeoutExpired("monitor", 1.0)
        return self.returncode


def ownership():
    return ArmOwnership(idle_stable_s=0.0, monotonic=lambda: 0.0)


def test_valid_line_updates_ownership_and_callback():
    states = []
    tracker = ownership()
    monitor = UnitreeArmMonitor(tracker, {}, state_callback=states.append)

    assert monitor.consume_line(
        'ARM_STATE={"id":552,"name":"vq_shake_hand",'
        '"holding":true}\n') is True
    assert tracker.snapshot()["status"] == ArmOwnership.EXTERNAL
    assert states[-1]["id"] == 552


def test_noise_and_malformed_state_are_ignored():
    monitor = UnitreeArmMonitor(ownership(), {})
    assert monitor.consume_line("startup log\n") is False
    assert monitor.consume_line("ARM_STATE=not-json\n") is False
    assert monitor.consume_line("ARM_STATE={}\n") is False


def test_start_uses_isolated_helper_command_and_environment():
    calls = []
    process = FakeProcess(running=False)

    def factory(command, **options):
        calls.append((command, options))
        return process

    monitor = UnitreeArmMonitor(
        ownership(), {"ROS_DOMAIN_ID": "0"}, process_factory=factory)
    monitor.start()
    monitor._reader.join(timeout=1.0)

    command, options = calls[0]
    assert command[-1] == "g1_hardware_bridge.arm.state_helper"
    assert options["env"]["ROS_DOMAIN_ID"] == "0"
    assert options["stdout"] == subprocess.PIPE
    assert process.waited is True
    assert monitor.ownership.snapshot()["status"] == ArmOwnership.UNAVAILABLE


def test_stop_terminates_and_reaps_running_process():
    process = FakeProcess()
    monitor = UnitreeArmMonitor(
        ownership(), {}, process_factory=lambda *_args, **_kwargs: process)
    monitor.start()
    monitor.stop()
    assert process.terminated is True
    assert process.killed is False
    assert process.waited is True


def test_stop_escalates_when_process_ignores_termination():
    process = FakeProcess(resist=True)
    monitor = UnitreeArmMonitor(
        ownership(), {}, process_factory=lambda *_args, **_kwargs: process)
    monitor.start()
    monitor.stop(timeout=0.01)
    assert process.terminated is True
    assert process.killed is True
    assert process.waited is True


def test_stop_is_idempotent_before_start():
    UnitreeArmMonitor(ownership(), {}).stop()


def test_running_monitor_allows_silent_idle_startup():
    process = FakeProcess()
    tracker = ownership()
    monitor = UnitreeArmMonitor(
        tracker, {}, process_factory=lambda *_args, **_kwargs: process)
    monitor.start()

    assert tracker.snapshot()["status"] == ArmOwnership.IDLE

    monitor.stop()
    assert tracker.snapshot()["status"] == ArmOwnership.UNAVAILABLE
