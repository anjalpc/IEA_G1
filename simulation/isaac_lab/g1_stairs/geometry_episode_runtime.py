"""Opt-in Isaac adapter for the shared V03 geometry episode contract."""

from dataclasses import asdict
import math
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
        self.platform_contact_counts = torch.zeros(
            len(self.heights), dtype=torch.long, device=env.device
        )
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
        finish_state = {}
        if self.contract.stable_finish:
            centres = feet[:, :, None, :] + rotated
            radius = G1_SOLE_SPHERE_RADIUS_M
            finish_state = {
                "sole_bounds": torch.stack((
                    centres[..., 0].amin(-1) - radius,
                    centres[..., 0].amax(-1) + radius,
                    centres[..., 1].amin(-1) - radius,
                    centres[..., 1].amax(-1) + radius,
                    centres[..., 2].amin(-1) - radius,
                    centres[..., 2].amax(-1) - radius,
                ), dim=-1),
                "body_linear_velocity": data.root_lin_vel_w,
                "body_angular_velocity": data.root_ang_vel_w,
                "foot_linear_velocity": data.body_lin_vel_w[:, self.robot_feet],
            }
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

        # Diagnostic only: physical joint limits and reward remain unchanged.
        overshoot = torch.maximum(
            limits[:, :, 0] - data.joint_pos,
            data.joint_pos - limits[:, :, 1],
        ).clamp_min(0)
        overshoot = torch.where(
            invalid[:, None], torch.zeros_like(overshoot), overshoot,
        ).amax(dim=1)
        if not hasattr(self, "_joint_limit_peak_rad"):
            self._joint_limit_peak_rad = torch.zeros(
                len(self.heights), device=env.device,
            )
        levels = (
            env.scene.terrain.terrain_levels
            if len(self.heights) > 1
            else torch.zeros_like(env.episode_length_buf)
        )
        for level in range(len(self.heights)):
            peak = torch.where(levels == level, overshoot, 0.0).amax()
            self._joint_limit_peak_rad[level] = torch.maximum(
                self._joint_limit_peak_rad[level], peak,
            )
        reason = self.tracker.update(
            position=position, roll_pitch=torch.stack((roll, pitch), -1),
            riser_height=riser_height_per_env(env), foot_position=feet,
            foot_force_z=forces[:, self.sensor_feet, 2],
            forbidden_contact=forbidden, invalid_state=invalid,
            joint_limit=torch.zeros_like(invalid),
            velocity_limit=(data.joint_vel.abs() > data.soft_joint_vel_limits + 1.e-5).any(-1),
            torque_limit=(data.applied_torque.abs() > data.joint_effort_limits + 1.e-3).any(-1),
            timed_out=env.episode_length_buf >= env.max_episode_length,
            **finish_state,
        )
        self._record_joint_limit_diagnostics(env, data, limits, reason)
        return reason

    def _record_joint_limit_diagnostics(self, env, data, limits, reason):
        # Diagnostic only: capture the pre-reset state of episodes whose
        # primary terminal reason is JOINT_LIMIT. No policy inputs change.
        terminal = reason == Result.JOINT_LIMIT
        if not bool(terminal.any().item()):
            return

        if not hasattr(self, "_joint_limit_stats"):
            self._joint_limit_stats = {}

        levels = (
            env.scene.terrain.terrain_levels
            if len(self.heights) > 1
            else torch.zeros_like(reason)
        )

        for level, height in enumerate(self.heights):
            selected = terminal & (levels == level)
            if not bool(selected.any().item()):
                continue

            positions = data.joint_pos[selected]
            bounds = limits[selected]
            lower_hit = positions < bounds[:, :, 0] - 1.e-5
            upper_hit = positions > bounds[:, :, 1] + 1.e-5

            lower_excess = torch.where(
                lower_hit,
                (bounds[:, :, 0] - positions).clamp_min(0),
                torch.zeros_like(positions),
            )
            upper_excess = torch.where(
                upper_hit,
                (positions - bounds[:, :, 1]).clamp_min(0),
                torch.zeros_like(positions),
            )

            lower_counts = lower_hit.sum(0).detach().cpu().tolist()
            upper_counts = upper_hit.sum(0).detach().cpu().tolist()
            lower_max = lower_excess.max(0).values.detach().cpu().tolist()
            upper_max = upper_excess.max(0).values.detach().cpu().tolist()

            for joint, name in enumerate(self.robot.joint_names):
                if lower_counts[joint] == 0 and upper_counts[joint] == 0:
                    continue

                prefix = f"joint_limit_h{round(height * 100):02d}cm_{name}"
                for side, counts, maxima in (
                    ("lower", lower_counts, lower_max),
                    ("upper", upper_counts, upper_max),
                ):
                    count_key = f"{prefix}_{side}_count"
                    max_key = f"{prefix}_{side}_max_overshoot_rad"
                    self._joint_limit_stats[count_key] = (
                        self._joint_limit_stats.get(count_key, 0)
                        + int(counts[joint])
                    )
                    self._joint_limit_stats[max_key] = max(
                        self._joint_limit_stats.get(max_key, 0.0),
                        float(maxima[joint]),
                    )

    def metrics(self):
        counts = self.counts.detach().cpu().tolist()
        phases = self.phase_failures.detach().cpu().tolist()
        platform_contacts = self.platform_contact_counts.detach().cpu().tolist()
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
            result[f"{prefix}_upper_platform_contact_episodes"] = platform_contacts[level]
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
        result.update(getattr(self, "_joint_limit_stats", {}))

        peaks = getattr(self, "_joint_limit_peak_rad", None)
        if peaks is not None:
            for height, peak in zip(self.heights, peaks.detach().cpu().tolist()):
                result[
                    f"observed_joint_limit_h{round(height * 100):02d}cm_max_overshoot_rad"
                ] = peak
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


def finish_geometry_episodes(
    env,
    env_ids,
    promote=False,
    minimum_nonflat_env_fraction=0.0,
):
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
    # Curriculum progression is intentionally independent of the final
    # one-second finish dwell. A physically valid supported platform pose
    # qualifies the stair height for promotion.
    curriculum_success = tracker.platform_seen[ids].clone()

    runtime.last_completed = {
        "step": int(env.common_step_counter), "env_ids": ids.clone(),
        "level": levels.clone(), "reason": reason,
        "episode_steps": env.episode_length_buf[ids].clone(),
        "upper_platform_contact": tracker.platform_contact_seen[ids].clone(),
        "supported_platform": tracker.platform_seen[ids].clone(),
        "entered_stairs": tracker.entered_stairs[ids].clone(),
        "phase": tracker.phase[ids].clone(),
    }
    flat_indices = levels * len(Result) + reason
    runtime.counts.view(-1).scatter_add_(0, flat_indices, torch.ones_like(ids))
    runtime.platform_contact_counts.scatter_add_(
        0, levels, tracker.platform_contact_seen[ids].long()
    )
    runtime.platform_counts.scatter_add_(0, levels, tracker.platform_seen[ids].long())
    runtime.phase_failures.view(-1).scatter_add_(
        0, levels * 3 + tracker.phase[ids], (~success).long(),
    )
    # Preserve the existing dashboard's curriculum counter keys, but source
    # them from completed episodes with the shared physical outcome contract.
    env._g1_stair_curriculum_attempts = runtime.counts.sum(-1)
    env._g1_stair_curriculum_successes = runtime.platform_counts.clone()
    if promote:
        move_up = curriculum_success & (levels < len(runtime.heights) - 1)
        move_down = ~curriculum_success & (levels > 0)

        fraction = float(minimum_nonflat_env_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "minimum_nonflat_env_fraction must be in [0, 1]"
            )

        if fraction > 0.0:
            all_levels = terrain.terrain_levels
            minimum_nonflat = int(
                math.ceil(all_levels.numel() * fraction)
            )
            current_nonflat = int(
                torch.count_nonzero(all_levels > 0).item()
            )
            incoming_nonflat = int(
                torch.count_nonzero(move_up & (levels == 0)).item()
            )
            allowed_level1_demotions = max(
                current_nonflat + incoming_nonflat - minimum_nonflat,
                0,
            )
            level1_candidates = torch.nonzero(
                move_down & (levels == 1),
                as_tuple=False,
            ).flatten()

            if level1_candidates.numel() > allowed_level1_demotions:
                bounded_level1 = torch.zeros_like(
                    move_down, dtype=torch.bool
                )
                if allowed_level1_demotions > 0:
                    bounded_level1[
                        level1_candidates[:allowed_level1_demotions]
                    ] = True
                # Higher levels may still demote because they remain non-flat.
                move_down = (move_down & (levels > 1)) | bounded_level1

        terrain.update_env_origins(
            ids, move_up=move_up, move_down=move_down
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
    curriculum_cfg = resolved.get("terrain", {}).get("curriculum", {})
    if curriculum_cfg is None:
        curriculum_cfg = {}
    if not isinstance(curriculum_cfg, dict):
        raise ValueError("terrain.curriculum must be a mapping")
    minimum_nonflat_env_fraction = float(
        curriculum_cfg.get("minimum_nonflat_env_fraction", 0.0)
    )
    if not 0.0 <= minimum_nonflat_env_fraction <= 1.0:
        raise ValueError(
            "terrain.curriculum.minimum_nonflat_env_fraction must be in [0, 1]"
        )
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
            params={
                "promote": bool(
                    curriculum_cfg.get("enabled", False)
                    if promote is None
                    else promote
                ),
                "minimum_nonflat_env_fraction": (
                    minimum_nonflat_env_fraction
                ),
            },
        )

    acquisition = resolved.get("acquisition_credit", {}) or {}
    if acquisition.get("verified_insertion", False):
        dwell = float(acquisition.get("support_dwell_s", 0.10))
        if not math.isfinite(dwell) or dwell <= 0:
            raise ValueError("support_dwell_s must be finite and positive")
        env_cfg.verified_insertion_dwell_s = dwell
        env_cfg.require_first_touchdown_placement = bool(
            acquisition.get("require_first_touchdown_placement", False)
        )

        shaping_start = float(
            acquisition.get(
                "first_touchdown_shaping_start_margin_m",
                -0.150,
            )
        )
        if not math.isfinite(shaping_start):
            raise ValueError(
                "first_touchdown_shaping_start_margin_m "
                "must be finite"
            )
        env_cfg.first_touchdown_shaping_start_margin_m = shaping_start

    if contract["stable_finish"]:
        from .platform_finish_command import apply_platform_finish_command
        apply_platform_finish_command(env_cfg, resolved, contract)
        heights = resolved.get("terrain", {}).get("riser_heights_m")
        if heights and "initial_max_level" in curriculum_cfg:
            initial = int(curriculum_cfg["initial_max_level"])
            if not 0 <= initial < len(heights):
                raise ValueError("initial_max_level outside configured heights")
            env_cfg.scene.terrain.max_init_terrain_level = initial

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
