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


def load_single_env_snapshot_artifact(path) -> dict:
    """Load and validate a portable C7 near-riser snapshot artifact."""

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    _validate_single_env_snapshot_artifact(artifact)
    return artifact


def _validate_single_env_snapshot_artifact(artifact: dict) -> None:
    if not isinstance(artifact, dict):
        raise RuntimeError("near-riser snapshot artifact must be a dictionary")
    if artifact.get("format") != "iea_g1_c7_near_riser_snapshot":
        raise RuntimeError(f"unexpected near-riser snapshot format: {artifact.get('format')!r}")
    if artifact.get("version") != 1:
        raise RuntimeError(f"unsupported near-riser snapshot version: {artifact.get('version')!r}")
    for key in ("source_env_origin", "physics", "temporal", "acquisition"):
        if key not in artifact:
            raise RuntimeError(f"near-riser snapshot is missing {key!r}")


def _target_env_ids(env, env_ids) -> torch.Tensor:
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)
    if ids.numel() == 0:
        raise ValueError("near-riser snapshot restore requires at least one environment")
    if torch.any(ids < 0) or torch.any(ids >= env.num_envs):
        raise ValueError(f"near-riser snapshot env_ids out of range: {ids.tolist()}")
    return ids


def _broadcast_saved(saved: torch.Tensor, count: int, device, name: str) -> torch.Tensor:
    if not isinstance(saved, torch.Tensor) or saved.ndim == 0 or saved.shape[0] != 1:
        shape = None if not isinstance(saved, torch.Tensor) else tuple(saved.shape)
        raise RuntimeError(f"snapshot tensor {name} must have leading singleton env dimension, got {shape}")
    return saved.to(device=device).expand(count, *saved.shape[1:]).clone()


def _restore_rows(current: torch.Tensor, env_ids: torch.Tensor, saved: torch.Tensor, name: str) -> None:
    values = _broadcast_saved(saved, env_ids.numel(), current.device, name)
    if tuple(values.shape[1:]) != tuple(current.shape[1:]):
        raise RuntimeError(
            f"snapshot tensor {name} shape changed: {tuple(values.shape[1:])} != {tuple(current.shape[1:])}"
        )
    current[env_ids] = values


def _prepare_history_for_next_append(buffer, env_ids: torch.Tensor, ordered: torch.Tensor, name: str) -> None:
    """Seed reset rows so the next ObservationManager append reconstructs saved history."""

    if buffer._buffer is None:
        raise RuntimeError(f"history buffer is uninitialized: {name}")
    if not isinstance(ordered, torch.Tensor) or ordered.ndim < 3 or ordered.shape[0] != 1:
        raise RuntimeError(f"snapshot history {name} must be [1, history, ...]")
    if ordered.shape[1] != buffer.max_length:
        raise RuntimeError(
            f"snapshot history length changed for {name}: {ordered.shape[1]} != {buffer.max_length}"
        )
    if tuple(ordered.shape[2:]) != tuple(buffer._buffer.shape[2:]):
        raise RuntimeError(
            f"snapshot history shape changed for {name}: "
            f"{tuple(ordered.shape[2:])} != {tuple(buffer._buffer.shape[2:])}"
        )

    # ObservationManager.compute(update_history=True) performs one append after
    # reset. Preload the oldest N-1 samples around the buffer's current global
    # pointer, leaving the final saved sample to be produced from restored state.
    next_pointer = (int(buffer._pointer) + 1) % buffer.max_length
    ordered_device = ordered.to(device=buffer._buffer.device)
    for index in range(buffer.max_length - 1):
        raw_index = (next_pointer + 1 + index) % buffer.max_length
        value = ordered_device[:, index].expand(env_ids.numel(), *ordered.shape[2:])
        buffer._buffer[raw_index, env_ids] = value
    buffer._num_pushes[env_ids] = buffer.max_length - 1


def restore_single_env_snapshot_artifact(env, artifact: dict, env_ids) -> None:
    """Restore one saved near-riser state into selected reset environments.

    This function must run after normal manager reset has cleared temporal state
    and before ObservationManager.compute(update_history=True). Histories are
    intentionally prepared for that mandatory next append.
    """

    _validate_single_env_snapshot_artifact(artifact)
    ids = _target_env_ids(env, env_ids)
    count = ids.numel()

    source_origin = artifact["source_env_origin"]
    if not isinstance(source_origin, torch.Tensor) or tuple(source_origin.shape) != (1, 3):
        raise RuntimeError("snapshot source_env_origin must have shape (1, 3)")
    target_origins = env.scene.env_origins[ids]

    physics = artifact["physics"]
    root = _broadcast_saved(physics["root_link_state"], count, env.device, "physics.root_link_state")
    source_origin_device = source_origin.to(device=env.device)
    root[:, :3] += target_origins - source_origin_device
    physics_snapshot = PhysicsRuntimeSnapshot(
        root_link_state=root,
        joint_pos=_broadcast_saved(physics["joint_pos"], count, env.device, "physics.joint_pos"),
        joint_vel=_broadcast_saved(physics["joint_vel"], count, env.device, "physics.joint_vel"),
        joint_pos_target=_broadcast_saved(
            physics["joint_pos_target"], count, env.device, "physics.joint_pos_target"
        ),
    )
    restore_physics_runtime(env, physics_snapshot, env_ids=ids)

    temporal = artifact["temporal"]
    _restore_rows(env.action_manager._action, ids, temporal["action"], "temporal.action")
    _restore_rows(env.action_manager._prev_action, ids, temporal["prev_action"], "temporal.prev_action")

    buffers = env.observation_manager._group_obs_term_history_buffer
    expected_history = temporal["observation_history"]
    actual_names = {f"{group}/{term}" for group, terms in buffers.items() for term in terms}
    if set(expected_history) != actual_names:
        raise RuntimeError("observation history terms changed across near-riser snapshot restore")
    for full_name, saved in expected_history.items():
        group_name, term_name = full_name.split("/", 1)
        _prepare_history_for_next_append(
            buffers[group_name][term_name], ids, saved["ordered"], full_name
        )

    gait = env.action_manager.get_term("gait_frequency")
    for name, saved in temporal["gait"].items():
        current = getattr(gait, name, None)
        if not isinstance(current, torch.Tensor):
            raise RuntimeError(f"gait_frequency.{name} stopped being a tensor")
        _restore_rows(current, ids, saved, f"gait.{name}")
    gait.last_finished_step = int(env.common_step_counter)

    command_terms = temporal["command_terms"]
    if set(command_terms) != set(env.command_manager._terms):
        raise RuntimeError("command terms changed across near-riser snapshot restore")
    for term_name, saved_term in command_terms.items():
        term = env.command_manager._terms[term_name]
        for name, saved in saved_term.items():
            current = getattr(term, name, None)
            if not isinstance(current, torch.Tensor):
                raise RuntimeError(f"command.{term_name}.{name} stopped being a tensor")
            _restore_rows(current, ids, saved, f"command.{term_name}.{name}")

    acquisition = getattr(env, "_g1_v04_acquisition", None)
    if artifact["acquisition"] and acquisition is None:
        raise RuntimeError("snapshot contains acquisition state but runtime is unavailable")
    if acquisition is not None:
        for name, saved in artifact["acquisition"].items():
            current = getattr(acquisition, name, None)
            if not isinstance(current, torch.Tensor):
                raise RuntimeError(f"acquisition.{name} stopped being a tensor")
            _restore_rows(current, ids, saved, f"acquisition.{name}")
