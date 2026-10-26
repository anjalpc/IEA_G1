#!/usr/bin/env python3
"""Short canonical-reference physics probe for the G1 imitation stack.

Patch 003 is diagnostic only.  It writes the audited frame-151 state into one
Isaac environment, verifies the write, then advances only frames 152..160 using
q_ref as a temporary position target.  This is a mechanical-feasibility probe,
not a training action, policy, reward target, or deployment controller.
"""

from __future__ import annotations

from pathlib import Path
import sys

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


MOTION_ROOT = Path(__file__).resolve().parents[2]
if str(MOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_ROOT))

START_SOURCE_FRAME = 151
END_SOURCE_FRAME = 160
CONTACT_THRESHOLD_N = 1.0


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy().astype("float64")


def main() -> int:
    import gymnasium as gym
    import numpy as np
    import torch

    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from simulation.isaac_lab.g1_stairs.contact_sensor_contract import (
        FOOT_BODY_NAMES,
        FORBIDDEN_CONTACT_BODY_NAMES,
    )
    from g1_stair_imitation.isaac.runtime_contract import (
        BASE_TASK_ID,
        configure_imitation_preflight_env,
    )
    from g1_stair_imitation.isaac.runtime_diagnostics import (
        LATERAL_DIAGNOSTIC_JOINTS,
        SOLE_RADIUS_M,
        WEAK_SAGITTAL_JOINTS,
        body_angular_velocity_to_world,
        named_joint_indices,
        quaternion_angle_error_rad,
        reference_position_world,
        reference_sole_world,
        rmse,
        sole_points_world_from_link_pose,
        xyzw_to_wxyz,
    )
    from g1_stair_imitation.paths import REFERENCE_FILE
    from g1_stair_imitation.reference import load_g1_stair_imitation_reference

    reference = load_g1_stair_imitation_reference(REFERENCE_FILE)
    if reference.source_frame_start != START_SOURCE_FRAME:
        raise RuntimeError(
            f"Patch 003 expects reference start {START_SOURCE_FRAME}, "
            f"got {reference.source_frame_start}"
        )
    if END_SOURCE_FRAME > reference.source_frame_end:
        raise RuntimeError("runtime probe end exceeds canonical reference")

    env_cfg = parse_env_cfg(BASE_TASK_ID, device="cuda:0", num_envs=1)
    configure_imitation_preflight_env(env_cfg)

    env = None
    try:
        env = gym.make(BASE_TASK_ID, cfg=env_cfg, render_mode=None)
        env.reset(seed=42)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        sim = base_env.sim
        sensor = base_env.scene.sensors.get("contact_forces")
        if sensor is None:
            raise RuntimeError("required contact_forces sensor is missing")

        runtime_joint_names = tuple(robot.joint_names)
        if runtime_joint_names != reference.joint_names:
            raise RuntimeError("runtime/reference joint order changed")

        joint_ids = list(range(len(runtime_joint_names)))
        weak_ids = named_joint_indices(runtime_joint_names, WEAK_SAGITTAL_JOINTS)
        lateral_ids = named_joint_indices(runtime_joint_names, LATERAL_DIAGNOSTIC_JOINTS)

        body_index = {name: i for i, name in enumerate(tuple(robot.body_names))}
        sensor_index = {name: i for i, name in enumerate(tuple(sensor.body_names))}
        for name in FOOT_BODY_NAMES:
            if name not in body_index or name not in sensor_index:
                raise RuntimeError(f"runtime foot body missing: {name}")

        env_origin = _as_numpy(base_env.scene.env_origins[0])
        device = robot.data.joint_pos.device
        dtype = robot.data.joint_pos.dtype

        i0 = reference.reference_index_for_source_frame(START_SOURCE_FRAME)
        root_position0 = reference_position_world(reference.root_pos[i0], env_origin)
        root_quat0 = xyzw_to_wxyz(reference.root_quat_xyzw[i0])
        root_lin_vel0 = np.asarray(reference.root_lin_vel[i0], dtype=np.float64)
        root_ang_vel0 = body_angular_velocity_to_world(
            reference.root_quat_xyzw[i0],
            reference.root_ang_vel_body[i0],
        )

        root_pose = torch.as_tensor(
            np.concatenate((root_position0, root_quat0)),
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        root_velocity = torch.as_tensor(
            np.concatenate((root_lin_vel0, root_ang_vel0)),
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        q0 = torch.as_tensor(reference.q_ref[i0], device=device, dtype=dtype).unsqueeze(0)
        dq0 = torch.as_tensor(reference.dq_ref[i0], device=device, dtype=dtype).unsqueeze(0)

        # Use the same audited root/joint write APIs already used by the validated
        # stair reference executors.  No env.step() and no policy are involved.
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_link_velocity_to_sim(root_velocity)
        robot.write_joint_state_to_sim(q0, dq0, joint_ids=joint_ids)
        robot.set_joint_position_target(q0, joint_ids=joint_ids)
        robot.reset()
        base_env.scene.write_data_to_sim()

        actual_root0 = _as_numpy(robot.data.root_pos_w[0])
        actual_quat0 = _as_numpy(robot.data.root_quat_w[0])
        actual_q0 = _as_numpy(robot.data.joint_pos[0, joint_ids])
        actual_dq0 = _as_numpy(robot.data.joint_vel[0, joint_ids])

        init_root_max = float(np.max(np.abs(actual_root0 - root_position0)))
        init_quat_err = quaternion_angle_error_rad(actual_quat0, root_quat0)
        init_q_max = float(np.max(np.abs(actual_q0 - reference.q_ref[i0])))
        init_dq_max = float(np.max(np.abs(actual_dq0 - reference.dq_ref[i0])))

        if max(init_root_max, init_q_max, init_dq_max) > 1.0e-5 or init_quat_err > 1.0e-5:
            raise RuntimeError(
                "frame-151 simulator state write did not reproduce the canonical state: "
                f"root={init_root_max:.3e} q={init_q_max:.3e} "
                f"dq={init_dq_max:.3e} quat={init_quat_err:.3e}"
            )

        physics_dt = float(sim.get_physics_dt())
        physics_steps_per_reference = int(round(reference.dt / physics_dt))
        if physics_steps_per_reference < 1:
            raise RuntimeError("reference dt is smaller than physics dt")
        if abs(physics_steps_per_reference * physics_dt - reference.dt) > 1.0e-9:
            raise RuntimeError(
                "reference/physics timestep ratio is not integral: "
                f"reference={reference.dt} physics={physics_dt}"
            )

        print("=" * 78)
        print(" G1 IMITATION PATCH 003 RUNTIME PROBE")
        print("=" * 78)
        print("mode             : diagnostic direct-q playback / no PPO / no env.step")
        print("reference        :", REFERENCE_FILE)
        print("source frames    :", f"{START_SOURCE_FRAME}->{END_SOURCE_FRAME}")
        print("physics dt       :", physics_dt)
        print("reference dt     :", reference.dt)
        print("physics/ref      :", physics_steps_per_reference)
        print()
        print("FRAME 151 STATE WRITE")
        print("  root max error :", f"{init_root_max:.3e} m")
        print("  quat angle err :", f"{init_quat_err:.3e} rad")
        print("  q max error    :", f"{init_q_max:.3e} rad")
        print("  dq max error   :", f"{init_dq_max:.3e} rad/s")
        print("  result         : PASS")
        print()
        print("SHORT PHYSICS WINDOW")
        print(
            "frame root_m qweak_rad qlat_rad Lsole_m Rsole_m "
            "supportZ_mm swingDelta_mm contacts tau_Nm power_W targetRate_radps forbidden"
        )

        max_root_error = 0.0
        max_weak_q_rmse = 0.0
        max_lateral_q_rmse = 0.0
        max_support_z_error = 0.0
        max_support_penetration = 0.0
        min_swing_clearance_delta = float("inf")
        max_torque = 0.0
        max_joint_acc = 0.0
        max_power = 0.0
        contact_matches = 0
        contact_checks = 0
        forbidden_contact_frames = 0

        previous_q_target = np.asarray(reference.q_ref[i0], dtype=np.float64)

        for source_frame in range(START_SOURCE_FRAME + 1, END_SOURCE_FRAME + 1):
            ref_i = reference.reference_index_for_source_frame(source_frame)
            q_target_np = np.asarray(reference.q_ref[ref_i], dtype=np.float64)
            q_target = torch.as_tensor(q_target_np, device=device, dtype=dtype).unsqueeze(0)

            for _ in range(physics_steps_per_reference):
                robot.set_joint_position_target(q_target, joint_ids=joint_ids)
                base_env.scene.write_data_to_sim()
                sim.step()
                base_env.scene.update(physics_dt)

            data = robot.data
            values = (
                data.root_pos_w[0],
                data.root_quat_w[0],
                data.joint_pos[0],
                data.joint_vel[0],
                data.joint_acc[0],
                data.applied_torque[0],
                sensor.data.net_forces_w[0],
            )
            if not all(bool(torch.isfinite(value).all().item()) for value in values):
                raise RuntimeError(f"non-finite runtime state at source frame {source_frame}")

            actual_root = _as_numpy(data.root_pos_w[0])
            expected_root = reference_position_world(reference.root_pos[ref_i], env_origin)
            root_error = float(np.linalg.norm(actual_root - expected_root))

            actual_q = _as_numpy(data.joint_pos[0, joint_ids])
            weak_q_rmse = rmse(actual_q[list(weak_ids)], reference.q_ref[ref_i, list(weak_ids)])
            lateral_q_rmse = rmse(
                actual_q[list(lateral_ids)], reference.q_ref[ref_i, list(lateral_ids)]
            )

            actual_sole = {}
            expected_sole = {
                "left_ankle_roll_link": reference_sole_world(reference.left_sole_points[ref_i], env_origin),
                "right_ankle_roll_link": reference_sole_world(reference.right_sole_points[ref_i], env_origin),
            }
            for name in FOOT_BODY_NAMES:
                b = body_index[name]
                actual_sole[name] = sole_points_world_from_link_pose(
                    _as_numpy(data.body_pos_w[0, b]),
                    _as_numpy(data.body_quat_w[0, b]),
                )

            left_error = float(
                np.mean(
                    np.linalg.norm(
                        actual_sole["left_ankle_roll_link"] - expected_sole["left_ankle_roll_link"],
                        axis=1,
                    )
                )
            )
            right_error = float(
                np.mean(
                    np.linalg.norm(
                        actual_sole["right_ankle_roll_link"] - expected_sole["right_ankle_roll_link"],
                        axis=1,
                    )
                )
            )

            expected_contact = {
                "left_ankle_roll_link": bool(reference.left_contact[ref_i]),
                "right_ankle_roll_link": bool(reference.right_contact[ref_i]),
            }
            actual_contact = {}
            support_z_errors = []
            support_penetrations = []
            swing_deltas = []
            for name in FOOT_BODY_NAMES:
                force = _as_numpy(sensor.data.net_forces_w[0, sensor_index[name]])
                actual_contact[name] = bool(np.linalg.norm(force) > CONTACT_THRESHOLD_N)
                contact_checks += 1
                contact_matches += int(actual_contact[name] == expected_contact[name])

                actual_bottom = actual_sole[name][:, 2] - SOLE_RADIUS_M
                expected_bottom = expected_sole[name][:, 2] - SOLE_RADIUS_M
                if expected_contact[name]:
                    support_z_errors.append(float(np.max(np.abs(actual_bottom - expected_bottom))))
                    support_penetrations.append(
                        float(np.max(np.maximum(expected_bottom - actual_bottom, 0.0)))
                    )
                else:
                    swing_deltas.append(float(np.min(actual_bottom - expected_bottom)))

            support_z_error = max(support_z_errors, default=0.0)
            support_penetration = max(support_penetrations, default=0.0)
            swing_clearance_delta = min(swing_deltas, default=float("nan"))

            forbidden = []
            for name in FORBIDDEN_CONTACT_BODY_NAMES:
                s = sensor_index.get(name)
                if s is None:
                    continue
                force = _as_numpy(sensor.data.net_forces_w[0, s])
                if np.linalg.norm(force) > CONTACT_THRESHOLD_N:
                    forbidden.append(name)
            forbidden_contact_frames += int(bool(forbidden))

            torque = _as_numpy(data.applied_torque[0, joint_ids])
            joint_velocity = _as_numpy(data.joint_vel[0, joint_ids])
            joint_acc = _as_numpy(data.joint_acc[0, joint_ids])
            torque_max = float(np.max(np.abs(torque)))
            joint_acc_max = float(np.max(np.abs(joint_acc)))
            power = float(np.sum(np.abs(torque * joint_velocity)))
            target_rate = float(np.mean(np.abs(q_target_np - previous_q_target)) / reference.dt)
            previous_q_target = q_target_np

            max_root_error = max(max_root_error, root_error)
            max_weak_q_rmse = max(max_weak_q_rmse, weak_q_rmse)
            max_lateral_q_rmse = max(max_lateral_q_rmse, lateral_q_rmse)
            max_support_z_error = max(max_support_z_error, support_z_error)
            max_support_penetration = max(max_support_penetration, support_penetration)
            if np.isfinite(swing_clearance_delta):
                min_swing_clearance_delta = min(min_swing_clearance_delta, swing_clearance_delta)
            max_torque = max(max_torque, torque_max)
            max_joint_acc = max(max_joint_acc, joint_acc_max)
            max_power = max(max_power, power)

            contact_text = (
                f"L{int(actual_contact['left_ankle_roll_link'])}/{int(expected_contact['left_ankle_roll_link'])} "
                f"R{int(actual_contact['right_ankle_roll_link'])}/{int(expected_contact['right_ankle_roll_link'])}"
            )
            swing_mm = (
                f"{swing_clearance_delta * 1000.0:+.1f}"
                if np.isfinite(swing_clearance_delta)
                else "n/a"
            )
            print(
                f"{source_frame:3d} "
                f"{root_error:7.4f} "
                f"{weak_q_rmse:9.4f} "
                f"{lateral_q_rmse:8.4f} "
                f"{left_error:7.4f} "
                f"{right_error:7.4f} "
                f"{support_z_error * 1000.0:10.2f} "
                f"{swing_mm:>13} "
                f"{contact_text:11s} "
                f"{torque_max:7.2f} "
                f"{power:8.2f} "
                f"{target_rate:14.3f} "
                f"{','.join(forbidden) if forbidden else '-'}"
            )

        contact_agreement = float(contact_matches) / float(contact_checks)
        if min_swing_clearance_delta == float("inf"):
            min_swing_clearance_delta = float("nan")

        print()
        print("SUMMARY (diagnostic; no reward thresholds selected)")
        print("  max root position error       :", f"{max_root_error:.6f} m")
        print("  max weak sagittal q RMSE      :", f"{max_weak_q_rmse:.6f} rad")
        print("  max lateral diagnostic q RMSE :", f"{max_lateral_q_rmse:.6f} rad")
        print("  max support sole Z error      :", f"{max_support_z_error * 1000.0:.3f} mm")
        print("  max support penetration proxy :", f"{max_support_penetration * 1000.0:.3f} mm")
        print(
            "  min swing clearance delta     :",
            "n/a" if not np.isfinite(min_swing_clearance_delta) else f"{min_swing_clearance_delta * 1000.0:+.3f} mm",
        )
        print("  contact agreement             :", f"{contact_matches}/{contact_checks} ({contact_agreement:.1%})")
        print("  forbidden-contact frames      :", forbidden_contact_frames)
        print("  max |applied torque|          :", f"{max_torque:.3f} Nm")
        print("  max |joint acceleration|      :", f"{max_joint_acc:.3f} rad/s^2")
        print("  max abs mechanical power sum  :", f"{max_power:.3f} W")
        print()
        print("SAFETY")
        print("  env.step calls      : 0")
        print("  PPO/training        : disabled")
        print("  imitation rewards   : disabled")
        print("  action_ref29 used   : no")
        print("  q_ref direct target : diagnostic only, frames 152->160")
        print("  pure-RL files edited: no")
        print()
        print("=" * 78)
        print(" PATCH 003 RUNTIME PROBE COMPLETE")
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
