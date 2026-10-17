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



def _single_env_tensor(value: torch.Tensor, env_id: int, num_envs: int, name: str) -> torch.Tensor:
    if value.ndim == 0 or value.shape[0] != num_envs:
        raise RuntimeError(
            f"snapshot tensor {name} is not environment-batched: shape={tuple(value.shape)}"
        )
    return value[env_id : env_id + 1].detach().cpu().clone()


def build_single_env_snapshot_artifact(env, env_id: int = 0) -> dict:
    """Build a portable one-environment near-riser snapshot artifact."""

    if not 0 <= env_id < env.num_envs:
        raise ValueError(f"env_id {env_id} outside [0, {env.num_envs})")

    physics = capture_physics_runtime(
        env,
        env_ids=torch.tensor([env_id], dtype=torch.long, device=env.device),
    )
    temporal = capture_temporal_runtime(env)

    histories = {}
    buffers = env.observation_manager._group_obs_term_history_buffer
    for (group_name, term_name), saved in temporal.observation_history.items():
        current = buffers[group_name][term_name]
        if current._buffer is None:
            raise RuntimeError(f"history buffer is uninitialized: {group_name}/{term_name}")
        histories[f"{group_name}/{term_name}"] = {
            "ordered": current.buffer[env_id : env_id + 1].detach().cpu().clone(),
            "num_pushes": saved.num_pushes[env_id : env_id + 1].detach().cpu().clone(),
        }

    gait_tensors = {
        name: _single_env_tensor(value, env_id, env.num_envs, f"gait.{name}")
        for name, value in temporal.gait.tensors.items()
    }
    command_terms = {}
    for term_name, saved in temporal.command_terms.items():
        command_terms[term_name] = {
            name: _single_env_tensor(
                value, env_id, env.num_envs, f"command.{term_name}.{name}"
            )
            for name, value in saved.tensors.items()
            if value.ndim > 0 and value.shape[0] == env.num_envs
        }

    acquisition = getattr(env, "_g1_v04_acquisition", None)
    acquisition_tensors = {}
    if acquisition is not None:
        acquisition_tensors = {
            name: _single_env_tensor(
                value, env_id, env.num_envs, f"acquisition.{name}"
            )
            for name, value in vars(acquisition).items()
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == env.num_envs
        }

    return {
        "format": "iea_g1_c7_near_riser_snapshot",
        "version": 1,
        "source_env_origin": env.scene.env_origins[env_id : env_id + 1].detach().cpu().clone(),
        "episode_length": env.episode_length_buf[env_id : env_id + 1].detach().cpu().clone(),
        "physics": {
            "root_link_state": physics.root_link_state.detach().cpu().clone(),
            "joint_pos": physics.joint_pos.detach().cpu().clone(),
            "joint_vel": physics.joint_vel.detach().cpu().clone(),
            "joint_pos_target": physics.joint_pos_target.detach().cpu().clone(),
        },
        "temporal": {
            "action": temporal.action[env_id : env_id + 1].detach().cpu().clone(),
            "prev_action": temporal.prev_action[env_id : env_id + 1].detach().cpu().clone(),
            "observation_history": histories,
            "gait": gait_tensors,
            "command_terms": command_terms,
        },
        "acquisition": acquisition_tensors,
    }


def save_single_env_snapshot_artifact(path, env, env_id: int = 0) -> None:
    """Write a portable snapshot artifact using tensor-only torch serialization."""

    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_single_env_snapshot_artifact(env, env_id=env_id), destination)


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
