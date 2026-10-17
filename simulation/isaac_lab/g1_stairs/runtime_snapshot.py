"""Exact temporal-state capture/restore helpers for C7 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CircularBufferSnapshot:
    buffer: torch.Tensor | None
    pointer: int
    num_pushes: torch.Tensor


@dataclass
class TensorObjectSnapshot:
    tensors: dict[str, torch.Tensor]
    scalars: dict[str, object]


@dataclass
class PhysicsRuntimeSnapshot:
    root_link_state: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_pos_target: torch.Tensor


@dataclass
class TemporalRuntimeSnapshot:
    action: torch.Tensor
    prev_action: torch.Tensor
    observation_history: dict[tuple[str, str], CircularBufferSnapshot]
    gait: TensorObjectSnapshot
    command_terms: dict[str, TensorObjectSnapshot]


def _clone_tensor_attributes(obj) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in vars(obj).items()
        if isinstance(value, torch.Tensor)
    }


def _restore_tensor_attributes(obj, values: dict[str, torch.Tensor]) -> None:
    for name, saved in values.items():
        current = getattr(obj, name)
        if not isinstance(current, torch.Tensor):
            raise RuntimeError(f"{type(obj).__name__}.{name} stopped being a tensor")
        if current.shape != saved.shape:
            raise RuntimeError(
                f"{type(obj).__name__}.{name} shape changed: "
                f"{tuple(current.shape)} != {tuple(saved.shape)}"
            )
        current.copy_(saved)


def _capture_object(obj, *, scalar_names: tuple[str, ...] = ()) -> TensorObjectSnapshot:
    return TensorObjectSnapshot(
        tensors=_clone_tensor_attributes(obj),
        scalars={name: getattr(obj, name) for name in scalar_names},
    )


def _restore_object(obj, snapshot: TensorObjectSnapshot) -> None:
    _restore_tensor_attributes(obj, snapshot.tensors)
    for name, value in snapshot.scalars.items():
        setattr(obj, name, value)



def capture_physics_runtime(env, env_ids=None) -> PhysicsRuntimeSnapshot:
    """Capture articulation state required for a dynamically consistent reset."""

    robot = env.scene["robot"]
    index = slice(None) if env_ids is None else env_ids
    target = robot.data.joint_pos_target
    if target is None:
        raise RuntimeError("robot.data.joint_pos_target is required for runtime snapshots")
    return PhysicsRuntimeSnapshot(
        root_link_state=robot.data.root_link_state_w[index].detach().clone(),
        joint_pos=robot.data.joint_pos[index].detach().clone(),
        joint_vel=robot.data.joint_vel[index].detach().clone(),
        joint_pos_target=target[index].detach().clone(),
    )


def restore_physics_runtime(env, snapshot: PhysicsRuntimeSnapshot, env_ids=None) -> None:
    """Restore an articulation snapshot through Isaac Lab's simulation write API."""

    robot = env.scene["robot"]
    robot.write_root_link_state_to_sim(snapshot.root_link_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        snapshot.joint_pos,
        snapshot.joint_vel,
        env_ids=env_ids,
    )
    robot.set_joint_position_target(snapshot.joint_pos_target, env_ids=env_ids)
    env.scene.write_data_to_sim()

def capture_temporal_runtime(env) -> TemporalRuntimeSnapshot:
    """Capture manager/runtime state destroyed by an Isaac Lab environment reset."""

    action_manager = env.action_manager
    observation_manager = env.observation_manager
    gait = action_manager.get_term("gait_frequency")

    histories: dict[tuple[str, str], CircularBufferSnapshot] = {}
    for group_name, terms in observation_manager._group_obs_term_history_buffer.items():
        for term_name, buffer in terms.items():
            histories[(group_name, term_name)] = CircularBufferSnapshot(
                buffer=None if buffer._buffer is None else buffer._buffer.detach().clone(),
                pointer=int(buffer._pointer),
                num_pushes=buffer._num_pushes.detach().clone(),
            )

    return TemporalRuntimeSnapshot(
        action=action_manager.action.detach().clone(),
        prev_action=action_manager.prev_action.detach().clone(),
        observation_history=histories,
        gait=_capture_object(
            gait,
            scalar_names=("last_finished_step", "_checked_contract"),
        ),
        command_terms={
            name: _capture_object(term)
            for name, term in env.command_manager._terms.items()
        },
    )


def restore_temporal_runtime(env, snapshot: TemporalRuntimeSnapshot) -> None:
    """Restore manager/runtime state captured by :func:`capture_temporal_runtime`."""

    action_manager = env.action_manager
    if action_manager.action.shape != snapshot.action.shape:
        raise RuntimeError("action shape changed across snapshot restore")
    if action_manager.prev_action.shape != snapshot.prev_action.shape:
        raise RuntimeError("previous-action shape changed across snapshot restore")

    action_manager._action.copy_(snapshot.action)
    action_manager._prev_action.copy_(snapshot.prev_action)

    buffers = env.observation_manager._group_obs_term_history_buffer
    expected_groups = {group for group, _ in snapshot.observation_history}
    if set(buffers) != expected_groups:
        raise RuntimeError("observation history groups changed across restore")

    for (group_name, term_name), saved in snapshot.observation_history.items():
        if term_name not in buffers[group_name]:
            raise RuntimeError(
                f"observation history term disappeared: {group_name}/{term_name}"
            )
        buffer = buffers[group_name][term_name]
        buffer._pointer = saved.pointer
        buffer._num_pushes.copy_(saved.num_pushes)
        if saved.buffer is None:
            buffer._buffer = None
        elif buffer._buffer is None or buffer._buffer.shape != saved.buffer.shape:
            buffer._buffer = saved.buffer.detach().clone()
        else:
            buffer._buffer.copy_(saved.buffer)

    _restore_object(action_manager.get_term("gait_frequency"), snapshot.gait)

    if set(env.command_manager._terms) != set(snapshot.command_terms):
        raise RuntimeError("command terms changed across snapshot restore")
    for name, saved in snapshot.command_terms.items():
        _restore_object(env.command_manager._terms[name], saved)
