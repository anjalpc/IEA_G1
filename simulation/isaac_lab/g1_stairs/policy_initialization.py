"""Initialization of the stair actor from the Unitree velocity ONNX policy."""

from __future__ import annotations

import hashlib
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper


LOCOMOTION_OBS_DIM = 480
TERRAIN_OBS_DIM = 187
EXPLICIT_GEOMETRY_OBS_DIM = 5

STAIR_OBS_DIM = (
    LOCOMOTION_OBS_DIM
    + TERRAIN_OBS_DIM
    + EXPLICIT_GEOMETRY_OBS_DIM
)
ACTION_DIM = 29

EXPECTED_ONNX_SHAPES = {
    "actor.0.weight": (512, 480),
    "actor.0.bias": (512,),
    "actor.2.weight": (256, 512),
    "actor.2.bias": (256,),
    "actor.4.weight": (128, 256),
    "actor.4.bias": (128,),
    "actor.6.weight": (29, 128),
    "actor.6.bias": (29,),
}


def load_frozen_unitree_actor(
    onnx_path: Path,
    device: str,
) -> torch.nn.Sequential:
    """Load the original 480-to-29 Unitree actor as a frozen module."""
    onnx_path = onnx_path.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    model = onnx.load(str(onnx_path))
    weights = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }
    for name, expected_shape in EXPECTED_ONNX_SHAPES.items():
        if name not in weights:
            raise RuntimeError(f"Missing ONNX tensor: {name}")
        if tuple(weights[name].shape) != expected_shape:
            raise RuntimeError(
                f"{name}: expected {expected_shape}, "
                f"got {tuple(weights[name].shape)}"
            )

    actor = torch.nn.Sequential(
        torch.nn.Linear(LOCOMOTION_OBS_DIM, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, ACTION_DIM),
    ).to(device)

    with torch.no_grad():
        for index in (0, 2, 4, 6):
            actor[index].weight.copy_(
                torch.as_tensor(weights[f"actor.{index}.weight"], device=device)
            )
            actor[index].bias.copy_(
                torch.as_tensor(weights[f"actor.{index}.bias"], device=device)
            )

    actor.eval()
    actor.requires_grad_(False)
    return actor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_actor_from_unitree_onnx(
    runner,
    onnx_path: Path,
) -> dict[str, object]:
    """Transfer the Unitree velocity actor into the terrain-aware stair actor."""

    onnx_path = onnx_path.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    model = onnx.load(str(onnx_path))
    weights = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }

    for name, expected_shape in EXPECTED_ONNX_SHAPES.items():
        if name not in weights:
            raise RuntimeError(f"Missing ONNX tensor: {name}")

        actual_shape = tuple(weights[name].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{name}: expected {expected_shape}, got {actual_shape}"
            )

    policy = runner.alg.policy
    actor = policy.actor

    if len(actor) != 7:
        raise RuntimeError(f"Expected 7 actor modules, got {len(actor)}")

    runtime_input_dim = int(actor[0].weight.shape[1])

    if tuple(actor[0].weight.shape) != (
        512,
        runtime_input_dim,
    ):
        raise RuntimeError(
            "Unexpected first actor layer shape: "
            f"{tuple(actor[0].weight.shape)}"
        )

    if runtime_input_dim < LOCOMOTION_OBS_DIM:
        raise RuntimeError(
            "Runtime actor has fewer observations than the "
            f"{LOCOMOTION_OBS_DIM}-D pretrained locomotion actor: "
            f"{runtime_input_dim}"
        )

    expected_runtime_shapes = {
        0: ((512, runtime_input_dim), (512,)),
        2: ((256, 512), (256,)),
        4: ((128, 256), (128,)),
        6: ((29, 128), (29,)),
    }

    for index, (weight_shape, bias_shape) in expected_runtime_shapes.items():
        layer = actor[index]

        if tuple(layer.weight.shape) != weight_shape:
            raise RuntimeError(
                f"actor[{index}].weight: expected {weight_shape}, "
                f"got {tuple(layer.weight.shape)}"
            )

        if tuple(layer.bias.shape) != bias_shape:
            raise RuntimeError(
                f"actor[{index}].bias: expected {bias_shape}, "
                f"got {tuple(layer.bias.shape)}"
            )

    def copy_tensor(target, source) -> None:
        value = torch.tensor(
            source,
            dtype=target.dtype,
            device=target.device,
        )
        target.copy_(value)

    with torch.no_grad():
        # First layer:
        # Copy the exact 480-D Unitree locomotion actor.
        # Every new terrain/geometry column begins at zero influence.
        actor[0].weight.zero_()
        copy_tensor(
            actor[0].weight[:, :LOCOMOTION_OBS_DIM],
            weights["actor.0.weight"],
        )
        copy_tensor(actor[0].bias, weights["actor.0.bias"])

        # Remaining actor layers are architecture-identical.
        for index in (2, 4, 6):
            copy_tensor(
                actor[index].weight,
                weights[f"actor.{index}.weight"],
            )
            copy_tensor(
                actor[index].bias,
                weights[f"actor.{index}.bias"],
            )

    # Hard check that the new terrain columns really start at zero.
    new_columns_max_abs = float(
        actor[0].weight[:, LOCOMOTION_OBS_DIM:]
        .detach()
        .abs()
        .max()
        .cpu()
    )

    if new_columns_max_abs != 0.0:
        raise RuntimeError(
            "New observation columns were not zero initialized: "
            f"{new_columns_max_abs}"
        )

    explicit_geometry_observations = (
        EXPLICIT_GEOMETRY_OBS_DIM
        if runtime_input_dim
        >= LOCOMOTION_OBS_DIM + EXPLICIT_GEOMETRY_OBS_DIM
        else 0
    )

    actor_height_scan_observations = max(
        0,
        runtime_input_dim
        - LOCOMOTION_OBS_DIM
        - explicit_geometry_observations,
    )

    metadata = {
        "type": "unitree_velocity_onnx_transfer",
        "source": str(onnx_path),
        "source_sha256": _sha256(onnx_path),
        "locomotion_observations": LOCOMOTION_OBS_DIM,
        "actor_height_scan_observations": (
            actor_height_scan_observations
        ),
        "explicit_geometry_observations": (
            explicit_geometry_observations
        ),
        "stair_observations": runtime_input_dim,
        "actions": ACTION_DIM,
        "runtime_observations": runtime_input_dim,
        "new_observations": (
            runtime_input_dim - LOCOMOTION_OBS_DIM
        ),
        "new_columns_max_abs": new_columns_max_abs,
    }

    print("[INFO] Policy initialization:", metadata, flush=True)

    return metadata
