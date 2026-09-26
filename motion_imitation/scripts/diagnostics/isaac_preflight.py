#!/usr/bin/env python3
"""One-environment Isaac plant/alignment preflight for imitation work.

This script does not train, does not call env.step(), and does not write to the
pure-RL configuration tree.  It parses a fresh environment config, applies the
20 cm / 3-step geometry in memory, creates one Isaac environment, verifies the
plant/action/reference contract, then exits.
"""

from __future__ import annotations

from pathlib import Path
import sys

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


MOTION_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MOTION_ROOT.parent

if str(MOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_ROOT))


def main() -> int:
    import gymnasium as gym
    import numpy as np

    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from g1_stair_imitation.isaac.runtime_contract import (
        BASE_TASK_ID,
        EXPECTED_ACTION_DIM,
        FIRST_RISER_COURSE_X_M,
        RISER_HEIGHT_M,
        STEP_COUNT,
        TREAD_DEPTH_M,
        configure_imitation_preflight_env,
        validate_action_cfg,
    )
    from g1_stair_imitation.paths import REFERENCE_FILE
    from g1_stair_imitation.reference import (
        load_g1_stair_imitation_reference,
    )

    print("=" * 78)
    print(" G1 IMITATION ISAAC PREFLIGHT")
    print("=" * 78)
    print("mode      : one environment / no PPO / no env.step")
    print("base task :", BASE_TASK_ID)
    print("reference :", REFERENCE_FILE)

    reference = load_g1_stair_imitation_reference(REFERENCE_FILE)

    env_cfg = parse_env_cfg(
        BASE_TASK_ID,
        device="cuda:0",
        num_envs=1,
    )

    stair_cfg = configure_imitation_preflight_env(env_cfg)
    action_scale, use_default_offset, configured_action_dim = (
        validate_action_cfg(env_cfg)
    )

    print()
    print("LOCAL GEOMETRY OVERRIDE")
    print("  riser       :", stair_cfg.riser_height_range)
    print("  tread       :", stair_cfg.tread_depth)
    print("  steps       :", stair_cfg.step_count)
    print("  first riser :", stair_cfg.start_clearance)
    print("  terrain rows:", env_cfg.scene.terrain.terrain_generator.num_rows)
    print("  terrain cols:", env_cfg.scene.terrain.terrain_generator.num_cols)

    assert tuple(stair_cfg.riser_height_range) == (
        RISER_HEIGHT_M,
        RISER_HEIGHT_M,
    )
    assert abs(float(stair_cfg.tread_depth) - TREAD_DEPTH_M) < 1.0e-12
    assert int(stair_cfg.step_count) == STEP_COUNT
    assert (
        abs(float(stair_cfg.start_clearance) - FIRST_RISER_COURSE_X_M)
        < 1.0e-12
    )

    print()
    print("ACTION CONFIG")
    print("  dimension          :", configured_action_dim)
    print("  scale              :", action_scale)
    print("  use_default_offset :", use_default_offset)

    env = None

    try:
        env = gym.make(
            BASE_TASK_ID,
            cfg=env_cfg,
            render_mode=None,
        )
        env.reset(seed=42)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        runtime_joint_names = tuple(robot.joint_names)

        if runtime_joint_names != reference.joint_names:
            mismatches = [
                (i, got, expected)
                for i, (got, expected) in enumerate(
                    zip(runtime_joint_names, reference.joint_names)
                )
                if got != expected
            ]
            raise RuntimeError(
                "runtime/reference joint-order mismatch: "
                f"{mismatches[:8]}"
            )

        q_default = (
            robot.data.default_joint_pos[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        max_default_error = float(
            np.max(np.abs(q_default - reference.q_default))
        )

        if max_default_error > 1.0e-6:
            raise RuntimeError(
                "runtime/reference default-pose mismatch: "
                f"{max_default_error:.3e} rad"
            )

        runtime_action_dim = int(base_env.action_manager.total_action_dim)

        if runtime_action_dim != EXPECTED_ACTION_DIM:
            raise RuntimeError(
                f"runtime action dimension {runtime_action_dim}; "
                f"expected {EXPECTED_ACTION_DIM}"
            )

        physics_dt = float(base_env.cfg.sim.dt)
        control_dt = float(base_env.step_dt)

        if abs(physics_dt - 0.005) > 1.0e-12:
            raise RuntimeError(
                f"physics dt changed: {physics_dt}"
            )

        if abs(control_dt - reference.dt) > 1.0e-12:
            raise RuntimeError(
                f"control/reference dt mismatch: "
                f"runtime={control_dt} reference={reference.dt}"
            )

        env_origin = (
            base_env.scene.env_origins[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        root_world = (
            robot.data.root_pos_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        root_course = root_world - env_origin

        # Frame 152 is the first task-space-valid sample.  The reference uses
        # first-riser-local X=0, while this Isaac course uses first riser X=1.
        i152 = reference.reference_index_for_source_frame(152)
        left_spheres = reference.left_sole_points[i152].copy()
        left_spheres[:, 0] += FIRST_RISER_COURSE_X_M
        left_bottom_z = left_spheres[:, 2] - 0.005

        if not bool(reference.left_contact[i152]):
            raise RuntimeError("frame 152 left foot must be support")

        if int(reference.left_step[i152]) != 0:
            raise RuntimeError("frame 152 left support must be on floor")

        max_floor_error = float(np.max(np.abs(left_bottom_z)))

        if max_floor_error > 0.002:
            raise RuntimeError(
                "frame 152 transformed support does not align with floor: "
                f"{max_floor_error * 1000.0:.3f} mm"
            )

        print()
        print("RUNTIME PLANT")
        print("  runtime joint order : PASS")
        print("  max default error   :", f"{max_default_error:.3e} rad")
        print("  runtime action dim  :", runtime_action_dim)
        print("  physics dt          :", physics_dt)
        print("  control dt          :", control_dt)

        print()
        print("ISAAC COURSE FRAME")
        print("  env origin world :", env_origin.tolist())
        print("  reset root world :", root_world.tolist())
        print("  reset root course:", root_course.tolist())
        print("  first riser x    :", FIRST_RISER_COURSE_X_M)

        print()
        print("REFERENCE ALIGNMENT")
        print("  frame 151 task-space valid:", bool(reference.taskspace_valid[0]))
        print("  frame 152 task-space valid:", bool(reference.taskspace_valid[i152]))
        print(
            "  frame 152 left sole floor error:",
            f"{max_floor_error * 1000.0:.6f} mm",
        )

        print()
        print("SAFETY")
        print("  env.step calls      : 0")
        print("  PPO/training        : disabled")
        print("  imitation rewards   : not installed in Patch 002")
        print("  pure-RL files edited: no")

        print()
        print("=" * 78)
        print(" IMITATION ISAAC PREFLIGHT PASS")
        print("=" * 78)
        return 0

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
