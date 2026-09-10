"""Offline arm-motion regressions with DDS replaced by deterministic doubles."""

import copy
import os
from types import SimpleNamespace

import pytest

from g1_hardware_bridge.arm import arm_motion_helper as helper
from motion_fixture import checkpoint_motion, LEFT_COMPANION, RIGHT_COMPANION


@pytest.fixture
def controller(monkeypatch):
    """Run the real lifecycle with a deterministic clock and advancing state."""
    clock = SimpleNamespace(now=0.0, advance_state=True, on_sleep=None)
    state = SimpleNamespace(
        tick=1, mode_machine=5,
        motor_state=[SimpleNamespace(q=0.01 * j) for j in range(35)],
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]))
    commands = []
    times = []

    class Subscriber:
        def __init__(self, *args):
            pass

        def Init(self, callback):
            callback(copy.deepcopy(state))

    class Publisher:
        def __init__(self, *args):
            pass

        def Init(self):
            pass

        def Write(self, command):
            commands.append(copy.deepcopy(command))
            times.append(clock.now)
            return True

    monkeypatch.setattr(helper, 'ChannelFactoryInitialize', lambda *args: None)
    monkeypatch.setattr(helper, 'ChannelSubscriber', Subscriber)
    monkeypatch.setattr(helper, 'ChannelPublisher', Publisher)
    monkeypatch.setattr(helper.time, 'monotonic', lambda: clock.now)
    instance = helper.ArmMotionController('lo', checkpoint_motion())
    monkeypatch.setattr(instance, 'poll_release', lambda: instance.release_requested)

    def sleep(seconds):
        clock.now += seconds
        if clock.on_sleep:
            clock.on_sleep()
        if clock.advance_state:
            state.tick += 1
            instance.receive_state(copy.deepcopy(state))

    monkeypatch.setattr(helper.time, 'sleep', sleep)
    instance.test_clock = clock
    instance.test_state = state
    instance.test_commands = commands
    instance.test_times = times
    return instance


def weights(controller):
    """Read the authority field from captured commands."""
    return [cmd.motor_cmd[29].q for cmd in controller.test_commands]


def test_initializes_entire_command_and_uses_post_acquisition_pose(controller):
    def shift():
        if controller.test_clock.now > helper.ACQUIRE_TIME / 2:
            controller.test_state.motor_state[15].q = 0.25

    controller.test_clock.on_sleep = shift
    assert controller.acquire()
    first = controller.test_commands[0]
    assert first.mode_pr == 0
    assert first.mode_machine == 5
    for joint in range(29):
        motor = first.motor_cmd[joint]
        assert motor.q == pytest.approx(0.01 * joint)
        assert motor.mode == 1
        assert motor.dq == motor.tau == 0.0
        expected_kp = (40.0 if joint in helper.WRIST_IDS else
                       80.0 if joint in helper.ARM_IDS else 300.0)
        assert motor.kp == expected_kp
    assert controller.q_start[15] == 0.25
    assert controller.target()[15] == pytest.approx(0.25 - 0.348308)


def test_safety_exception_returns_and_releases_authority(controller):
    def fault():
        if controller.test_clock.now > 3.2:
            controller.test_state.imu_state.rpy[0] = 0.2

    controller.test_clock.on_sleep = fault
    with pytest.raises(RuntimeError, match='roll safety limit'):
        controller.run()
    assert weights(controller)[-1] == 0.0
    assert controller.cmd.motor_cmd[15].q == controller.q_start[15]


def test_lost_telemetry_during_acquisition_releases_partial_weight(
        controller, monkeypatch):
    monkeypatch.setattr(
        helper, 'STATE_TIMEOUT', helper.ACQUIRE_TIME / 2)
    controller.test_clock.advance_state = False
    with pytest.raises(RuntimeError, match='stale'):
        controller.run()
    assert 0.0 < max(weights(controller)) < 1.0
    assert weights(controller)[-1] == 0.0


def test_cancel_during_acquisition_never_increases_to_full_weight(controller):
    def cancel():
        if controller.test_clock.now >= helper.ACQUIRE_TIME / 2:
            controller.request_release()

    controller.test_clock.on_sleep = cancel
    controller.run()
    assert 0.0 < max(weights(controller)) < 1.0
    assert weights(controller)[-1] == 0.0


def test_cancel_during_outward_motion_retracts(controller, capsys):
    cancel_at = (
        helper.ACQUIRE_TIME
        + helper.SETTLE_TIME
        + 0.30
        + 0.90 / 2
    )
    controller.test_clock.on_sleep = lambda: (
        controller.request_release()
        if controller.test_clock.now >= cancel_at
        else None
    )
    controller.run()
    output = capsys.readouterr().out
    assert 'PRESENTED=' not in output
    assert 'RELEASED=' in output
    assert weights(controller)[-1] == 0.0


def test_hold_keeps_publishing_until_release(controller, capsys):
    controller.test_clock.on_sleep = lambda: (
        controller.request_release() if controller.test_clock.now > 6.0 else None)
    controller.run()
    output = capsys.readouterr().out
    assert 'PRESENTED=' in output and 'RELEASED=' in output
    assert weights(controller).count(1.0) > 100
    assert weights(controller)[-1] == 0.0


def test_repeated_tick_does_not_refresh_freshness(controller):
    controller.test_clock.now = 1.0
    controller.receive_state(controller.test_state)
    with pytest.raises(RuntimeError, match='stale'):
        controller.latest()


def test_nonfinite_telemetry_rejected_before_command(controller):
    controller.test_state.motor_state[14].q = float('nan')
    controller.test_state.tick += 1
    controller.receive_state(controller.test_state)
    with pytest.raises(RuntimeError, match='invalid'):
        controller.run()
    assert not controller.test_commands


@pytest.mark.parametrize('chunk', [b'RELEASE\n', b''])
def test_stdin_release_and_parent_eof(controller, monkeypatch, chunk):
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, 'r') as stream:
        monkeypatch.setattr(helper.sys, 'stdin', stream)
        if chunk:
            os.write(write_fd, chunk)
        os.close(write_fd)
        assert helper.ArmMotionController.poll_release(controller)


def test_partial_command_does_not_block_publishing(controller, monkeypatch):
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, 'r') as stream:
            monkeypatch.setattr(helper.sys, 'stdin', stream)
            os.write(write_fd, b'REL')
            assert not helper.ArmMotionController.poll_release(controller)
            os.write(write_fd, b'EASE\n')
            assert helper.ArmMotionController.poll_release(controller)
    finally:
        os.close(write_fd)


def arm_pose(command, joints):
    return [command.motor_cmd[j].q for j in joints]


@pytest.mark.parametrize('duration', [1.2, 4.0])
def test_bilateral_phases_and_reverse_timing(controller, duration):
    controller.trajectory_duration = duration
    controller.prepare_seconds = duration / 4
    controller.extend_seconds = duration * 3 / 4
    assert controller.acquire()
    start = controller.test_clock.now
    offset = len(controller.test_commands)
    assert controller.execute_outward_motion()
    assert controller.test_clock.now - start == pytest.approx(duration)
    companion_time = start + duration / 4
    for t, cmd in zip(controller.test_times[offset:], controller.test_commands[offset:]):
        for j in range(15):
            assert cmd.motor_cmd[j].q == controller.q_start[j]
        if t >= companion_time:
            assert arm_pose(cmd, helper.RIGHT_IDS) == pytest.approx(RIGHT_COMPANION)
        if t == pytest.approx(companion_time):
            assert arm_pose(cmd, helper.LEFT_IDS) == pytest.approx(LEFT_COMPANION)
    target = controller.target()
    assert arm_pose(controller.cmd, helper.LEFT_IDS) == pytest.approx(
        [target[j] for j in helper.LEFT_IDS])
    # Fresh measured release pose, including a small left tracking error.
    for j in helper.ARM_IDS:
        controller.test_state.motor_state[j].q = target[j]
    controller.test_state.motor_state[15].q += 0.01
    controller.test_state.tick += 1
    controller.receive_state(copy.deepcopy(controller.test_state))
    start = controller.test_clock.now
    offset = len(controller.test_commands)
    controller.safe_release()
    first = controller.test_commands[offset]
    assert first.motor_cmd[15].q == pytest.approx(target[15] + 0.01)
    for t, cmd in zip(controller.test_times[offset:], controller.test_commands[offset:]):
        if t <= start + duration * 3 / 4:
            assert arm_pose(cmd, helper.RIGHT_IDS) == pytest.approx(RIGHT_COMPANION)
        if t == pytest.approx(start + duration * 3 / 4):
            assert arm_pose(cmd, helper.LEFT_IDS) == pytest.approx(LEFT_COMPANION)
        if t >= start + duration:
            assert arm_pose(cmd, range(29)) == pytest.approx(list(controller.q_start.values()))
        if t < start + duration + helper.HOME_SETTLE_TIME:
            assert cmd.motor_cmd[29].q == 1.0
    assert weights(controller)[-1] == 0.0


@pytest.mark.parametrize('elapsed', [0.1, 0.5])
def test_release_in_each_phase_uses_correct_reverse_sequence(controller, elapsed):
    assert controller.acquire()
    start = controller.test_clock.now
    controller.test_clock.on_sleep = lambda: (
        controller.request_release() if controller.test_clock.now >= start + elapsed else None)
    assert not controller.execute_outward_motion()
    assert controller.extension_started == (elapsed > 0.3)
    start = controller.test_clock.now
    controller.return_home()
    expected = 1.2 if elapsed > 0.3 else 0.3
    assert controller.test_clock.now - start == pytest.approx(
        expected + helper.HOME_SETTLE_TIME, abs=helper.DT)


def test_trajectory_duration_comes_from_motion_phases(controller):
    instance = helper.ArmMotionController('lo', checkpoint_motion())
    assert instance.prepare_seconds == 0.30
    assert instance.extend_seconds == 0.90
    assert instance.trajectory_duration == pytest.approx(
        0.30 + 0.90)
