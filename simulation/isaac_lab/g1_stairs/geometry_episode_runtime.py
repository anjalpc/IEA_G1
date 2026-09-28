"""Opt-in Isaac adapter for the shared V03 geometry episode contract."""

from dataclasses import asdict
import torch

from .contact_sensor_contract import FOOT_BODY_NAMES, FORBIDDEN_CONTACT_BODY_NAMES
from .contracts import G1_SOLE_SPHERE_CENTRES_M, G1_SOLE_SPHERE_RADIUS_M
from .geometry_episode import EpisodeContract, EpisodeTracker, Result
from .terrain_state import riser_height_per_env, stair_cfg


class GeometryEpisodeRuntime:
    def __init__(self, env, contract):
        cfg = stair_cfg(env)
        self.robot = env.scene["robot"]
        self.sensor = env.scene.sensors["contact_forces"]
        self.contract = EpisodeContract(**contract)
        self.robot_feet = [list(self.robot.body_names).index(n) for n in FOOT_BODY_NAMES]
        names = list(self.sensor.body_names)
        self.sensor_feet = [names.index(n) for n in FOOT_BODY_NAMES]
        # These links must be physically observable. Fixed visual hand links
        # may be absent from the active USD sensor inventory.
        required = ("pelvis", "torso_link", "left_knee_link", "right_knee_link")
        missing = set(required) - set(names)
        if missing:
            raise RuntimeError(f"missing required outcome contact bodies: {sorted(missing)}")
        self.forbidden_names = [n for n in FORBIDDEN_CONTACT_BODY_NAMES if n in names]
        self.forbidden_ids = [names.index(n) for n in self.forbidden_names]
        print("[INFO] Geometry outcome contact coverage:", self.forbidden_names, flush=True)
        start_x_in_tile = cfg.approach_length - cfg.start_clearance
        self.tracker = EpisodeTracker(
            env.num_envs, env.device, self.contract,
            first_riser=cfg.start_clearance, tread=cfg.tread_depth,
            step_count=cfg.step_count, course_start=-start_x_in_tile,
            course_end=cfg.size[0] - start_x_in_tile,
            half_width=cfg.size[1] / 2, dt=env.step_dt,
        )
        self.heights = tuple(cfg.riser_heights or (cfg.riser_height_range[0],))
        self.counts = torch.zeros((len(self.heights), len(Result)), dtype=torch.long, device=env.device)
        self.phase_failures = torch.zeros((len(self.heights), 3), dtype=torch.long, device=env.device)
        self.platform_counts = torch.zeros(len(self.heights), dtype=torch.long, device=env.device)
        self.last_step = -1
        self.last_completed = None
        self.sole_centres = torch.tensor(G1_SOLE_SPHERE_CENTRES_M, device=env.device)

    def update(self, env):
        # TerminationManager invokes both termination and timeout functions.
        # Compute physical outcomes exactly once before automatic resets.
        step = int(env.common_step_counter)
        if self.last_step == step:
            return self.tracker.reason
        self.last_step = step
        data = self.robot.data
        position = data.root_pos_w - env.scene.env_origins
        q = data.root_quat_w
        norm = torch.linalg.vector_norm(q, dim=-1)
        q = q / norm.clamp_min(1.e-8)[:, None]
        w, x, y, z = q.unbind(-1)
        roll = torch.atan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
        feet = data.body_pos_w[:, self.robot_feet] - env.scene.env_origins[:, None, :]
        fq = data.body_quat_w[:, self.robot_feet]
        v = self.sole_centres[None, None].expand(env.num_envs, 2, -1, -1)
        qv = fq[:, :, None, 1:].expand_as(v)
        cross = 2 * torch.cross(qv, v, dim=-1)
        rotated = v + fq[:, :, None, :1] * cross + torch.cross(qv, cross, dim=-1)
        feet = feet.clone()
        feet[:, :, 2] += rotated[:, :, :, 2].min(-1).values - G1_SOLE_SPHERE_RADIUS_M
        forces = self.sensor.data.net_forces_w
        forbidden = (torch.linalg.vector_norm(forces[:, self.forbidden_ids], dim=-1)
                     > self.contract.forbidden_contact_force_n).any(-1)
        invalid = norm <= 1.e-8
        for value in (position, data.root_quat_w, data.joint_pos, data.joint_vel,
                      data.applied_torque, forces, feet, fq):
            invalid |= ~torch.isfinite(value).flatten(1).all(-1)
        limits = data.joint_pos_limits
        return self.tracker.update(
            position=position, roll_pitch=torch.stack((roll, pitch), -1),
            riser_height=riser_height_per_env(env), foot_position=feet,
            foot_force_z=forces[:, self.sensor_feet, 2],
            forbidden_contact=forbidden, invalid_state=invalid,
            joint_limit=((data.joint_pos < limits[:, :, 0] - 1.e-5)
                         | (data.joint_pos > limits[:, :, 1] + 1.e-5)).any(-1),
            velocity_limit=(data.joint_vel.abs() > data.soft_joint_vel_limits + 1.e-5).any(-1),
            torque_limit=(data.applied_torque.abs() > data.joint_effort_limits + 1.e-3).any(-1),
            timed_out=env.episode_length_buf >= env.max_episode_length,
        )

    def metrics(self):
        counts = self.counts.detach().cpu().tolist()
        phases = self.phase_failures.detach().cpu().tolist()
        platforms = self.platform_counts.detach().cpu().tolist()
        result = {}
        stair_attempts = stair_successes = flat_attempts = flat_successes = 0
        for level, height in enumerate(self.heights):
            prefix = f"outcome_h{round(height * 100):02d}cm"
            attempts = sum(counts[level])
            successes = counts[level][Result.SUCCESS]
            result[f"{prefix}_attempts"] = attempts
            result[f"{prefix}_successes"] = successes
            result[f"{prefix}_success_rate"] = successes / attempts if attempts else None
            result[f"{prefix}_supported_platform_episodes"] = platforms[level]
            for reason in Result:
                if reason not in (Result.RUNNING, Result.SUCCESS):
                    result[f"{prefix}_{reason.name.lower()}"] = counts[level][reason]
            for phase, label in enumerate(("approach", "ascent", "platform")):
                result[f"{prefix}_{label}_failures"] = phases[level][phase]
            if height > 0:
                stair_attempts += attempts
                stair_successes += successes
            else:
                flat_attempts += attempts
                flat_successes += successes
        result.update(
            verified_stair_attempts=stair_attempts,
            verified_stair_successes=stair_successes,
            verified_stair_success_rate=stair_successes / stair_attempts if stair_attempts else None,
            flat_traversal_attempts=flat_attempts,
            flat_traversal_successes=flat_successes,
            flat_traversal_success_rate=flat_successes / flat_attempts if flat_attempts else None,
        )
        return result


def _runtime(env, contract):
    runtime = getattr(env, "_g1_geometry_episode", None)
    if runtime is None:
        runtime = GeometryEpisodeRuntime(env, contract)
        env._g1_geometry_episode = runtime
    return runtime


def geometry_terminal(env, contract):
    reason = _runtime(env, contract).update(env)
    return (reason != Result.RUNNING) & (reason != Result.TIMEOUT)


def geometry_timeout(env, contract):
    return _runtime(env, contract).update(env) == Result.TIMEOUT


def finish_geometry_episodes(env, env_ids, promote=False):
    """Count the terminal snapshot before scene reset or terrain reassignment."""
    runtime = getattr(env, "_g1_geometry_episode", None)
    if runtime is None or len(env_ids) == 0:
        return 0.0
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[env.episode_length_buf[ids] > 0]
    if ids.numel() == 0:
        return 0.0
    tracker = runtime.tracker
    reason = tracker.reason[ids].clone()
    if (reason == Result.RUNNING).any():
        raise RuntimeError("V03 received a reset without a terminal outcome")
    terrain = env.scene.terrain
    levels = (terrain.terrain_levels[ids].clone() if len(runtime.heights) > 1
              else torch.zeros_like(ids))
    success = reason == Result.SUCCESS
    runtime.last_completed = {
        "step": int(env.common_step_counter), "env_ids": ids.clone(),
        "level": levels.clone(), "reason": reason,
        "episode_steps": env.episode_length_buf[ids].clone(),
        "supported_platform": tracker.platform_seen[ids].clone(),
        "entered_stairs": tracker.entered_stairs[ids].clone(),
        "phase": tracker.phase[ids].clone(),
    }
    flat_indices = levels * len(Result) + reason
    runtime.counts.view(-1).scatter_add_(0, flat_indices, torch.ones_like(ids))
    runtime.platform_counts.scatter_add_(0, levels, tracker.platform_seen[ids].long())
    runtime.phase_failures.view(-1).scatter_add_(
        0, levels * 3 + tracker.phase[ids], (~success).long(),
    )
    # Preserve the existing dashboard's curriculum counter keys, but source
    # them from completed episodes with the shared physical outcome contract.
    env._g1_stair_curriculum_attempts = runtime.counts.sum(-1)
    env._g1_stair_curriculum_successes = runtime.counts[:, Result.SUCCESS].clone()
    if promote:
        terrain.update_env_origins(
            ids, move_up=success & (levels < len(runtime.heights) - 1),
            move_down=~success & (levels > 0),
        )
    tracker.reset(ids)
    return float(success.float().mean().item())


def apply_geometry_episode_contract(env_cfg, resolved, *, promote=None):
    """Enable only explicitly selected geometry experiments; retain old runs."""
    settings = resolved.get("episode_contract", {})
    if not settings.get("enabled", False):
        return False
    if not str(resolved["reward_experiment"]["active_stage"]).startswith("geometry_locomotion"):
        raise ValueError("geometry episode contract requires a geometry locomotion stage")
    contract = asdict(EpisodeContract(**{k: v for k, v in settings.items() if k != "enabled"}))
    from isaaclab.managers import TerminationTermCfg, CurriculumTermCfg
    from isaaclab.utils import configclass

    @configclass
    class GeometryTerminations:
        geometry_terminal = TerminationTermCfg(func=geometry_terminal, params={"contract": contract})
        time_out = TerminationTermCfg(func=geometry_timeout, params={"contract": contract}, time_out=True)

    @configclass
    class GeometryCurriculum:
        episode_outcomes = CurriculumTermCfg(
            func=finish_geometry_episodes,
            params={"promote": bool(resolved.get("terrain", {}).get("curriculum", {}).get("enabled", False)
                                    if promote is None else promote)},
        )

    env_cfg.terminations = GeometryTerminations()
    env_cfg.curriculum = GeometryCurriculum()
    # Training and replay receive identical spawn randomization.
    reset = resolved.get("reset_randomization", {})
    if reset.get("enabled", False):
        clearance = env_cfg.scene.terrain.terrain_generator.sub_terrains["three_step_ascent"].start_clearance
        near, far = reset["distance_before_first_riser_m"]
        lateral = tuple(reset["lateral_offset_m"])
        yaw = tuple(reset["yaw_offset_rad"])
        if not (0 < near <= far <= clearance and lateral[0] <= lateral[1] and yaw[0] <= yaw[1]):
            raise ValueError("invalid V03 spawn randomization")
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (clearance - far, clearance - near), "y": lateral, "yaw": yaw,
        }
    return True
