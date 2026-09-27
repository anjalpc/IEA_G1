#!/usr/bin/env python3
"""Deterministic, policy-free G1 right-foot single-support qualification.

This diagnostic intentionally executes no PPO policy and disables reward and
termination terms. It drives the exact specialist JointPositionAction contract
with a bounded bank of scripted raw actions, then records physical mechanics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNITREE_RL_ROOT = Path(
    os.environ.get("UNITREE_RL_LAB_ROOT", "/workspace/unitree_rl_lab")
)

from simulation.isaac_lab.g1_stairs.unitree_bootstrap import (  # noqa: E402
    bootstrap_unitree_rl_lab,
)

bootstrap_unitree_rl_lab(UNITREE_RL_ROOT)


parser = argparse.ArgumentParser(
    description="Qualify G1 single-support mechanics without PPO or rewards."
)
parser.add_argument("--task", default="Unitree-IEA-G1-29dof-Stairs-v0")
parser.add_argument("--resolved-config", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


SETTLE_STEPS = 100
SHIFT_RAMP_STEPS = 75
SHIFT_HOLD_STEPS = 75
LIFT_RAMP_STEPS = 75
LIFT_HOLD_STEPS = 100
TOTAL_STEPS = (
    SETTLE_STEPS
    + SHIFT_RAMP_STEPS
    + SHIFT_HOLD_STEPS
    + LIFT_RAMP_STEPS
    + LIFT_HOLD_STEPS
)


def _disable_terms(group) -> tuple[str, ...]:
    disabled = []
    for name, value in vars(group).items():
        if name.startswith("_") or value is None:
            continue
        setattr(group, name, None)
        disabled.append(name)
    return tuple(sorted(disabled))


def _quat_roll_wxyz(q) -> float:
    w, x, y, z = [float(v) for v in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(sinr_cosp, cosr_cosp)


def main() -> None:
    import gymnasium as gym
    import torch

    import unitree_rl_lab.tasks  # noqa: F401
    import simulation.isaac_lab.g1_stairs.register  # noqa: F401

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS
    from simulation.isaac_lab.g1_stairs.evaluation.step_up_checkpoint_validation import (
        apply_resolved_replay_config,
    )
    from simulation.isaac_lab.g1_stairs.mechanics_qualification import (
        CONTROL_TRACKING_JOINTS,
        classify_candidate,
        linear_ramp,
        overall_verdict,
        scripted_candidates,
    )

    resolved_path = Path(args.resolved_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)

    candidates = scripted_candidates()
    num_envs = 1 + len(candidates)  # env 0 is the neutral control.

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    env_cfg.commands.base_velocity.debug_vis = False

    replay = apply_resolved_replay_config(env_cfg, resolved_path)

    # The mechanics qualification must have no reward-driven side effects and
    # no reset that can hide a failure halfway through the scripted sequence.
    disabled_rewards = _disable_terms(env_cfg.rewards)
    disabled_terminations = _disable_terms(env_cfg.terminations)

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = num_envs

    material = env_cfg.scene.terrain.physics_material
    physics_material = {}
    for key in (
        "static_friction",
        "dynamic_friction",
        "restitution",
        "friction_combine_mode",
        "restitution_combine_mode",
    ):
        value = getattr(material, key, None)
        if value is None or isinstance(value, (str, int, float, bool)):
            physics_material[key] = value
        else:
            physics_material[key] = str(value)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    sensor = base_env.scene.sensors["contact_forces"]

    if base_env.action_manager.total_action_dim != len(BODY_JOINTS):
        raise RuntimeError(
            "mechanics qualification requires exactly 29 body actions; got "
            f"{base_env.action_manager.total_action_dim}"
        )

    joint_names = tuple(robot.joint_names)
    body_names = tuple(robot.body_names)
    sensor_names = tuple(sensor.body_names)

    joint_ids = []
    for name in BODY_JOINTS:
        if name not in joint_names:
            raise RuntimeError(f"missing required action joint: {name}")
        joint_ids.append(joint_names.index(name))

    joint_action_index = {name: i for i, name in enumerate(BODY_JOINTS)}
    joint_robot_index = {name: joint_names.index(name) for name in BODY_JOINTS}

    required_feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_body_index = {}
    foot_sensor_index = {}
    for name in required_feet:
        if name not in body_names or name not in sensor_names:
            raise RuntimeError(f"missing required foot body/contact sensor: {name}")
        foot_body_index[name] = body_names.index(name)
        foot_sensor_index[name] = sensor_names.index(name)

    obs, _ = env.reset()
    del obs

    device = base_env.device
    actions = torch.zeros((num_envs, len(BODY_JOINTS)), device=device)
    shift_actions = torch.zeros_like(actions)
    lift_actions = torch.zeros_like(actions)

    for env_index, candidate in enumerate(candidates, start=1):
        for name, value in candidate.values.items():
            index = joint_action_index[name]
            if name in (
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
            ):
                lift_actions[env_index, index] = float(value)
            else:
                shift_actions[env_index, index] = float(value)

    env_origins = base_env.scene.env_origins

    def foot_xy(env_ids, name: str):
        p = robot.data.body_pos_w[env_ids, foot_body_index[name], :2]
        return p - env_origins[env_ids, :2]

    def foot_z(env_ids, name: str):
        p = robot.data.body_pos_w[env_ids, foot_body_index[name], 2]
        return p - env_origins[env_ids, 2]

    def foot_speed_xy(name: str):
        velocity = robot.data.body_lin_vel_w[:, foot_body_index[name], :2]
        return torch.linalg.vector_norm(velocity, dim=1)

    def foot_fz(name: str):
        values = sensor.data.net_forces_w[:, foot_sensor_index[name], 2]
        return torch.clamp(values, min=0.0)

    # Capture the settled stance anchor at the end of the neutral phase.
    neutral_anchor_left = None
    neutral_anchor_right = None
    neutral_height_left = None
    neutral_height_right = None
    stance_total_force_ref = None

    trace_rows: list[dict] = []
    candidate_samples: dict[int, list[dict]] = {
        i: [] for i in range(1, num_envs)
    }
    neutral_samples: list[dict] = []

    default_joint = robot.data.default_joint_pos[:, joint_ids]

    for step in range(TOTAL_STEPS):
        shift_alpha = linear_ramp(step, SETTLE_STEPS, SHIFT_RAMP_STEPS)
        lift_start = SETTLE_STEPS + SHIFT_RAMP_STEPS + SHIFT_HOLD_STEPS
        lift_alpha = linear_ramp(step, lift_start, LIFT_RAMP_STEPS)

        actions.zero_()
        actions += shift_actions * shift_alpha
        actions += lift_actions * lift_alpha

        env.step(actions)

        left_xy = foot_xy(slice(None), "left_ankle_roll_link")
        right_xy = foot_xy(slice(None), "right_ankle_roll_link")
        left_z = foot_z(slice(None), "left_ankle_roll_link")
        right_z = foot_z(slice(None), "right_ankle_roll_link")
        left_speed = foot_speed_xy("left_ankle_roll_link")
        right_speed = foot_speed_xy("right_ankle_roll_link")
        left_force = foot_fz("left_ankle_roll_link")
        right_force = foot_fz("right_ankle_roll_link")
        total_force = torch.clamp(left_force + right_force, min=1.0)
        left_share = left_force / total_force
        right_share = right_force / total_force
        upright = -robot.data.projected_gravity_b[:, 2]

        if step == SETTLE_STEPS - 1:
            neutral_anchor_left = left_xy.clone()
            neutral_anchor_right = right_xy.clone()
            neutral_height_left = left_z.clone()
            neutral_height_right = right_z.clone()
            stance_total_force_ref = total_force.clone()

        if neutral_anchor_left is None:
            continue

        left_drift = torch.linalg.vector_norm(left_xy - neutral_anchor_left, dim=1)
        right_drift = torch.linalg.vector_norm(right_xy - neutral_anchor_right, dim=1)
        left_height_gain = left_z - neutral_height_left

        desired_joint = default_joint + 0.25 * actions
        actual_joint = robot.data.joint_pos[:, joint_ids]
        tracking_error = desired_joint - actual_joint

        critical_indices = [
            joint_action_index[name] for name in CONTROL_TRACKING_JOINTS
        ]
        critical_tracking = torch.sqrt(
            torch.mean(tracking_error[:, critical_indices] ** 2, dim=1)
        )

        applied_torque = robot.data.applied_torque[:, joint_ids]
        effort_limits = torch.clamp(robot.data.joint_effort_limits[:, joint_ids], min=1e-6)
        torque_util = torch.max(torch.abs(applied_torque) / effort_limits, dim=1).values

        root_local_y = robot.data.root_pos_w[:, 1] - env_origins[:, 1]
        root_to_right_y = root_local_y - right_xy[:, 1]
        q = robot.data.root_quat_w
        sinr_cosp = 2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
        cosr_cosp = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        root_roll = torch.atan2(sinr_cosp, cosr_cosp)

        if stance_total_force_ref is None:
            raise RuntimeError("stance force reference was not captured")

        # Stable support is intentionally physical and independent of rewards.
        support_flags = (
            (left_share <= 0.15)
            & (right_share >= 0.70)
            & (right_force >= 0.50 * stance_total_force_ref)
            & (left_height_gain >= 0.030)
            & (right_drift <= 0.030)
            & (right_speed <= 0.050)
            & (upright >= 0.80)
            & (torch.abs(root_roll) <= 0.35)
        )

        phase = (
            "neutral"
            if step < SETTLE_STEPS
            else "shift_ramp"
            if step < SETTLE_STEPS + SHIFT_RAMP_STEPS
            else "shift_hold"
            if step < lift_start
            else "lift_ramp"
            if step < lift_start + LIFT_RAMP_STEPS
            else "lift_hold"
        )

        # Store neutral control and candidate summaries; write a compact trace
        # for env0 and the best candidates later rather than 49 full matrices.
        for env_index in range(num_envs):
            sample = {
                "step": step,
                "time_s": step * float(base_env.step_dt),
                "phase": phase,
                "env_index": env_index,
                "right_anchor_drift_m": float(right_drift[env_index].item()),
                "left_anchor_drift_m": float(left_drift[env_index].item()),
                "right_planar_speed_mps": float(right_speed[env_index].item()),
                "left_planar_speed_mps": float(left_speed[env_index].item()),
                "right_force_n": float(right_force[env_index].item()),
                "left_force_n": float(left_force[env_index].item()),
                "right_force_share": float(right_share[env_index].item()),
                "left_force_share": float(left_share[env_index].item()),
                "left_foot_height_gain_m": float(left_height_gain[env_index].item()),
                "upright_gravity_z": float(upright[env_index].item()),
                "root_roll_rad": float(root_roll[env_index].item()),
                "root_to_right_foot_y_m": float(root_to_right_y[env_index].item()),
                "critical_tracking_rms_rad": float(critical_tracking[env_index].item()),
                "torque_utilization": float(torque_util[env_index].item()),
                "single_support": bool(support_flags[env_index].item()),
            }
            if env_index == 0:
                neutral_samples.append(sample)
            else:
                candidate_samples[env_index].append(sample)

    step_dt = float(base_env.step_dt)

    # Neutral summary uses the post-settle interval only.
    neutral_eval = neutral_samples[-min(50, len(neutral_samples)) :]
    neutral_summary = {
        "right_anchor_drift_max_m": max(row["right_anchor_drift_m"] for row in neutral_eval),
        "left_anchor_drift_max_m": max(row["left_anchor_drift_m"] for row in neutral_eval),
        "right_planar_speed_mean_mps": sum(row["right_planar_speed_mps"] for row in neutral_eval) / len(neutral_eval),
        "left_planar_speed_mean_mps": sum(row["left_planar_speed_mps"] for row in neutral_eval) / len(neutral_eval),
        "upright_min": min(row["upright_gravity_z"] for row in neutral_eval),
    }

    candidate_summaries = []
    for env_index, candidate in enumerate(candidates, start=1):
        samples = candidate_samples[env_index]
        lift_hold = [row for row in samples if row["phase"] == "lift_hold"]
        if not lift_hold:
            raise RuntimeError("lift-hold samples missing")

        summary = {
            "candidate": candidate.name,
            "family": candidate.family,
            "sign": candidate.sign,
            "shift_raw": candidate.shift_raw,
            "lift_scale": candidate.lift_scale,
            "right_anchor_drift_max_m": max(row["right_anchor_drift_m"] for row in lift_hold),
            "right_planar_speed_mean_mps": sum(row["right_planar_speed_mps"] for row in lift_hold) / len(lift_hold),
            "right_planar_speed_max_mps": max(row["right_planar_speed_mps"] for row in lift_hold),
            "left_force_share_min": min(row["left_force_share"] for row in lift_hold),
            "left_force_share_mean": sum(row["left_force_share"] for row in lift_hold) / len(lift_hold),
            "right_force_share_mean": sum(row["right_force_share"] for row in lift_hold) / len(lift_hold),
            "left_foot_height_gain_max_m": max(row["left_foot_height_gain_m"] for row in lift_hold),
            "upright_min": min(row["upright_gravity_z"] for row in lift_hold),
            "root_roll_abs_max_rad": max(abs(row["root_roll_rad"]) for row in lift_hold),
            "root_to_right_foot_y_abs_max_m": max(abs(row["root_to_right_foot_y_m"]) for row in lift_hold),
            "critical_tracking_rms_rad": sum(row["critical_tracking_rms_rad"] for row in lift_hold) / len(lift_hold),
            "torque_utilization_max": max(row["torque_utilization"] for row in lift_hold),
            "single_support_flags": [row["single_support"] for row in lift_hold],
        }
        candidate_summaries.append(classify_candidate(summary, step_dt=step_dt))

    verdict = overall_verdict(neutral_summary, candidate_summaries)

    # Rank traces by the same mechanics we care about. Always save neutral and
    # the top five candidates to keep the artifact small and inspectable.
    ranked = sorted(
        candidate_summaries,
        key=lambda row: (
            not row["single_support_pass"],
            row["left_force_share_min"],
            row["right_anchor_drift_max_m"],
            -row["left_foot_height_gain_max_m"],
        ),
    )
    trace_envs = {0}
    candidate_name_to_env = {candidate.name: i for i, candidate in enumerate(candidates, start=1)}
    for row in ranked[:5]:
        trace_envs.add(candidate_name_to_env[row["candidate"]])

    selected_rows = []
    for row in neutral_samples:
        if row["env_index"] == 0:
            selected_rows.append(row)
    for env_index in sorted(trace_envs - {0}):
        selected_rows.extend(candidate_samples[env_index])

    trace_path = output_dir / "single_support_trace.csv"
    if selected_rows:
        with trace_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(selected_rows)

    ranking_path = output_dir / "candidate_summary.csv"
    rows_for_csv = []
    for row in ranked:
        copy = dict(row)
        copy.pop("single_support_flags", None)
        rows_for_csv.append(copy)
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_for_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_csv)

    metadata = {
        "policy_used": False,
        "rewards_disabled": list(disabled_rewards),
        "terminations_disabled": list(disabled_terminations),
        "task": args.task,
        "seed": args.seed,
        "resolved_config": str(resolved_path),
        "num_envs": num_envs,
        "candidate_count": len(candidates),
        "step_dt_s": step_dt,
        "total_steps": TOTAL_STEPS,
        "action_contract": "q_target = q_default + 0.25 * raw_action",
        "first_riser_x_m": replay.first_riser_x_m,
        "riser_height_m": replay.riser_height_m,
        "tread_depth_m": replay.tread_depth_m,
        "physics_material": physics_material,
        "single_support_contract": {
            "minimum_duration_s": 0.30,
            "left_force_share_max": 0.15,
            "right_force_share_min": 0.70,
            "left_foot_height_gain_min_m": 0.030,
            "right_anchor_drift_max_m": 0.030,
            "right_planar_speed_max_mps": 0.050,
            "upright_gravity_z_min": 0.80,
            "root_roll_abs_max_rad": 0.35,
        },
    }

    summary = {
        "metadata": metadata,
        "neutral": neutral_summary,
        "verdict": verdict,
        "top_candidates": [
            {k: v for k, v in row.items() if k != "single_support_flags"}
            for row in ranked[:10]
        ],
    }

    (output_dir / "mechanics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "========================================",
        " G1 SINGLE-SUPPORT MECHANICS QUALIFICATION",
        "========================================",
        "Policy used:          NO",
        "Reward terms active:  NO",
        f"Candidates:           {len(candidates)}",
        f"Physics dt/control:   step_dt={step_dt:.4f} s",
        "",
        "NEUTRAL STANCE",
        f"  right drift max      {neutral_summary['right_anchor_drift_max_m']:.4f} m",
        f"  left drift max       {neutral_summary['left_anchor_drift_max_m']:.4f} m",
        f"  right speed mean     {neutral_summary['right_planar_speed_mean_mps']:.4f} m/s",
        f"  upright min          {neutral_summary['upright_min']:.4f}",
        "",
        "VERDICT",
        f"  {verdict['verdict']}",
        f"  {verdict['interpretation']}",
    ]
    if verdict.get("best_candidate"):
        lines.append(f"  best candidate: {verdict['best_candidate']}")

    lines += ["", "TOP CANDIDATES"]
    for row in ranked[:5]:
        lines.append(
            "  "
            f"{row['candidate']}: pass={row['single_support_pass']} "
            f"stable={row['stable_single_support_duration_s']:.2f}s "
            f"Rdrift={row['right_anchor_drift_max_m']:.3f}m "
            f"Rspeed={row['right_planar_speed_mean_mps']:.3f}m/s "
            f"Lshare_min={row['left_force_share_min']:.3f} "
            f"Llift={row['left_foot_height_gain_max_m']:.3f}m "
            f"track={row['critical_tracking_rms_rad']:.3f}rad "
            f"torque={row['torque_utilization_max']:.2f}x"
        )

    lines += [
        "",
        "OUTPUTS",
        f"  {output_dir / 'mechanics_summary.json'}",
        f"  {output_dir / 'candidate_summary.csv'}",
        f"  {output_dir / 'single_support_trace.csv'}",
        "========================================",
    ]

    report = "\n".join(lines) + "\n"
    (output_dir / "mechanics_report.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
