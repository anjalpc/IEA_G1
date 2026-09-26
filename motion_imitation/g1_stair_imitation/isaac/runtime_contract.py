"""Read-only integration contract with the validated pure-RL Isaac plant.

The imitation stack deliberately reuses the validated G1 plant and stair terrain
as a dependency.  This module mutates only a parsed in-memory environment config;
it never edits the pure-RL source/configuration on disk.
"""

from __future__ import annotations

from typing import Any


BASE_TASK_ID = "Unitree-IEA-G1-29dof-Stairs-v0"
TERRAIN_KEY = "three_step_ascent"

RISER_HEIGHT_M = 0.20
TREAD_DEPTH_M = 0.25
STEP_COUNT = 3
APPROACH_LENGTH_M = 2.0
PLATFORM_LENGTH_M = 3.0
FIRST_RISER_COURSE_X_M = 1.0

EXPECTED_ACTION_SCALE = 0.25
EXPECTED_USE_DEFAULT_OFFSET = True
EXPECTED_ACTION_DIM = 29


def _stair_cfg(env_cfg: Any) -> Any:
    generator = env_cfg.scene.terrain.terrain_generator

    if generator is None:
        raise RuntimeError("Isaac environment has no terrain generator")

    sub_terrains = generator.sub_terrains

    if TERRAIN_KEY not in sub_terrains:
        raise RuntimeError(
            f"terrain generator is missing {TERRAIN_KEY!r}; "
            f"available={sorted(sub_terrains)}"
        )

    return sub_terrains[TERRAIN_KEY]


def configure_imitation_preflight_env(env_cfg: Any) -> Any:
    """Apply the frozen 20 cm / 3-step contract to one parsed env config.

    The caller owns ``env_cfg``.  These changes are process-local and do not
    modify the pure-RL configuration files or an already-running environment.
    """

    generator = env_cfg.scene.terrain.terrain_generator

    if generator is None:
        raise RuntimeError("Isaac environment has no terrain generator")

    generator.num_rows = 1
    generator.num_cols = 1

    stair = _stair_cfg(env_cfg)
    stair.riser_height_range = (RISER_HEIGHT_M, RISER_HEIGHT_M)
    stair.tread_depth = TREAD_DEPTH_M
    stair.step_count = STEP_COUNT
    stair.approach_length = APPROACH_LENGTH_M
    stair.platform_length = PLATFORM_LENGTH_M
    stair.start_clearance = FIRST_RISER_COURSE_X_M

    return stair


def validate_action_cfg(env_cfg: Any) -> tuple[float, bool, int]:
    """Validate the inherited 29-DoF Unitree position-action contract."""

    action = env_cfg.actions.JointPositionAction

    scale = float(action.scale)
    use_default_offset = bool(action.use_default_offset)
    action_dim = len(action.joint_names)

    if abs(scale - EXPECTED_ACTION_SCALE) > 1.0e-12:
        raise RuntimeError(
            f"unexpected JointPositionAction scale {scale}; "
            f"expected {EXPECTED_ACTION_SCALE}"
        )

    if use_default_offset is not EXPECTED_USE_DEFAULT_OFFSET:
        raise RuntimeError(
            "JointPositionAction use_default_offset must remain True"
        )

    if action_dim != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            f"JointPositionAction controls {action_dim} joints; "
            f"expected {EXPECTED_ACTION_DIM}"
        )

    return scale, use_default_offset, action_dim
