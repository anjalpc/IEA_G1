from types import SimpleNamespace

import torch

from simulation.isaac_lab.g1_stairs.runtime_snapshot import (
    capture_temporal_runtime,
    restore_temporal_runtime,
)


class FakeCircularBuffer:
    def __init__(self):
        self._buffer = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
        self._pointer = 2
        self._num_pushes = torch.tensor([7], dtype=torch.long)


class FakeGait:
    def __init__(self):
        self._raw = torch.tensor([[0.4]])
        self._frequency = torch.tensor([[1.12]])
        self.phase = torch.tensor([0.37])
        self.active = torch.tensor([True])
        self.aligned = torch.tensor([True])
        self.reward_phase = torch.tensor([0.35])
        self.reward_active = torch.tensor([True])
        self.contact_loss = torch.tensor([0.2])
        self.swing_started = torch.tensor([True])
        self.reference_expired = torch.tensor([False])
        self.endpoint_hold_s = torch.tensor([0.04])
        self.selection_wait_s = torch.tensor([0.02])
        self.approach_unload_dwell_s = torch.tensor([[0.06, 0.0]])
        self.transition_motion_miss = torch.tensor([False])
        self.transition_motion_miss_event = torch.tensor([False])
        self.transition_motion_miss_reason = torch.tensor([0])
        self.approach_simultaneous_unload = torch.tensor([False])
        self.approach_committed = torch.tensor([False])
        self.approach_clear_pending = torch.tensor([False])
        self.approach_preferred_foot = torch.tensor([-1])
        self.last_finished_step = 123
        self._checked_contract = True


class FakeActionManager:
    def __init__(self):
        self._action = torch.arange(30, dtype=torch.float32).reshape(1, 30)
        self._prev_action = self._action - 1.0
        self.gait = FakeGait()

    @property
    def action(self):
        return self._action

    @property
    def prev_action(self):
        return self._prev_action

    def get_term(self, name):
        assert name == "gait_frequency"
        return self.gait


class FakeCommand:
    def __init__(self):
        self.vel_command_b = torch.tensor([[0.35, 0.0, -0.1]])
        self.time_left = torch.tensor([1.7])
        self.command_counter = torch.tensor([4], dtype=torch.long)
        self.heading_target = torch.tensor([0.2])
        self.is_heading_env = torch.tensor([False])
        self.is_standing_env = torch.tensor([False])
        self.metrics = {"ignored_nested_tensor": torch.tensor([9.0])}


def make_env():
    return SimpleNamespace(
        action_manager=FakeActionManager(),
        observation_manager=SimpleNamespace(
            _group_obs_term_history_buffer={
                "policy": {
                    "base_ang_vel": FakeCircularBuffer(),
                    "last_action": FakeCircularBuffer(),
                }
            }
        ),
        command_manager=SimpleNamespace(_terms={"base_velocity": FakeCommand()}),
    )


def test_capture_restore_recovers_temporal_runtime_state():
    env = make_env()
    snapshot = capture_temporal_runtime(env)

    env.action_manager._action.zero_()
    env.action_manager._prev_action.zero_()

    gait = env.action_manager.gait
    for value in vars(gait).values():
        if isinstance(value, torch.Tensor):
            value.zero_()
    gait.last_finished_step = -1
    gait._checked_contract = False

    history = env.observation_manager._group_obs_term_history_buffer["policy"]
    for buffer in history.values():
        buffer._buffer.zero_()
        buffer._pointer = 0
        buffer._num_pushes.zero_()

    command = env.command_manager._terms["base_velocity"]
    for value in vars(command).values():
        if isinstance(value, torch.Tensor):
            value.zero_()

    restore_temporal_runtime(env, snapshot)

    torch.testing.assert_close(env.action_manager.action, snapshot.action)
    torch.testing.assert_close(env.action_manager.prev_action, snapshot.prev_action)
    assert gait.last_finished_step == 123
    assert gait._checked_contract is True
    torch.testing.assert_close(gait.phase, torch.tensor([0.37]))
    torch.testing.assert_close(gait._frequency, torch.tensor([[1.12]]))

    for key, saved in snapshot.observation_history.items():
        restored = history[key[1]]
        assert restored._pointer == saved.pointer
        torch.testing.assert_close(restored._num_pushes, saved.num_pushes)
        torch.testing.assert_close(restored._buffer, saved.buffer)

    torch.testing.assert_close(command.vel_command_b, torch.tensor([[0.35, 0.0, -0.1]]))
    torch.testing.assert_close(command.time_left, torch.tensor([1.7]))
    torch.testing.assert_close(command.command_counter, torch.tensor([4]))


def test_snapshot_is_deep_copy():
    env = make_env()
    snapshot = capture_temporal_runtime(env)

    env.action_manager._action.add_(100.0)
    env.action_manager.gait.phase.add_(1.0)
    env.command_manager._terms["base_velocity"].vel_command_b.add_(2.0)

    torch.testing.assert_close(
        snapshot.action,
        torch.arange(30, dtype=torch.float32).reshape(1, 30),
    )
    torch.testing.assert_close(snapshot.gait.tensors["phase"], torch.tensor([0.37]))
    torch.testing.assert_close(
        snapshot.command_terms["base_velocity"].tensors["vel_command_b"],
        torch.tensor([[0.35, 0.0, -0.1]]),
    )
