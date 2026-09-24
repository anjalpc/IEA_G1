"""043 physical single-step snapshot, shared by terminations and rewards.

State is allocated only on the first simulated step. No policy architecture,
historical reward, or historical detector is modified. Four transformed source
URDF collision spheres test support geometry; this is not a centre-of-pressure
measurement. Event/bounded returns are divided by dt exactly once because
Isaac Lab's RewardManager multiplies every reward by dt.
"""

from dataclasses import asdict

from .physical_step_config import PhysicalStepConfig, RATE_COMPONENTS
from .step_up_isaac import FOOT_BODY_NAMES, FORBIDDEN_CONTACT_BODY_NAMES


# Pinned g1_29dof_rev_1_0.urdf ankle-roll collision spheres, in link frame.
SOLE_CENTRES = ((-0.05, -0.025, -0.03), (-0.05, 0.025, -0.03),
                (0.12, -0.03, -0.03), (0.12, 0.03, -0.03))
SOLE_RADIUS = 0.005


def rotate(q, vectors):
    import torch
    xyz = q[..., 1:]
    uv = torch.cross(xyz, vectors, dim=-1)
    return vectors + 2 * (q[..., :1] * uv + torch.cross(xyz, uv, dim=-1))


class PhysicalStepState:
    def __init__(self, num_envs, device, config):
        import torch
        self.config = config
        self.last_step = None
        self.snapshot = None
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.base_reference = torch.zeros(num_envs, device=device)
        self.lower_age = torch.zeros((num_envs, 2), device=device)
        self.upper_age = torch.zeros((num_envs, 2), device=device)
        self.seen_upper = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        self.best = torch.zeros((num_envs, 3), device=device)
        self.previous_velocity = torch.zeros((num_envs, 2, 3), device=device)
        self.previous_contact = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        self.has_previous = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.totals = {}

    def reset(self, ids):
        # Keep the last immutable snapshot for post-autoreset metrics.
        for value in (self.stage, self.ready, self.base_reference, self.lower_age,
                      self.upper_age, self.seen_upper, self.best,
                      self.previous_velocity, self.previous_contact, self.has_previous):
            value[ids] = 0

    def update(self, step, dt, *, root, gravity, foot_pos, foot_quat,
               foot_velocity, foot_angular_velocity, forces, forbidden, bodyweight):
        import torch
        if self.last_step == step:
            return self.snapshot
        if dt <= 0:
            raise ValueError("control timestep must be positive")
        c = self.config
        values = (root, gravity, foot_pos, foot_quat, foot_velocity,
                  foot_angular_velocity, forces, bodyweight)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise RuntimeError("043 non-finite physical detector input")
        if bool((bodyweight <= 0).any()):
            raise RuntimeError("043 invalid articulation mass")
        if bool((torch.linalg.vector_norm(foot_quat, dim=2) - 1).abs().gt(1e-3).any()):
            raise RuntimeError("043 foot orientation is not a unit quaternion")
        old_stage = self.stage.clone()
        centres = torch.tensor(SOLE_CENTRES, device=root.device, dtype=root.dtype)
        offsets = centres.expand(root.shape[0], 2, -1, -1)
        q = foot_quat.unsqueeze(2).expand(-1, -1, 4, -1)
        points = foot_pos.unsqueeze(2) + rotate(q, offsets)
        bottom = points[..., 2] - SOLE_RADIUS
        x_min = points[..., 0].amin(dim=2) - SOLE_RADIUS
        x_max = points[..., 0].amax(dim=2) + SOLE_RADIUS
        y_max = points[..., 1].abs().amax(dim=2) + SOLE_RADIUS
        normals = torch.zeros_like(foot_pos)
        normals[..., 2] = 1
        normal_z = rotate(foot_quat, normals)[..., 2]
        upright = -gravity[:, 2] >= c.upright_cos
        safe = upright & ~forbidden & (root[:, 2] >= c.minimum_base_height_m)
        loaded = forces[..., 2] >= bodyweight[:, None] * c.support_force_bodyweight_fraction
        slow = torch.linalg.vector_norm(foot_velocity, dim=2) <= c.support_speed_mps
        slow_rotation = torch.linalg.vector_norm(foot_angular_velocity, dim=2) <= c.support_angular_speed_radps
        lateral = y_max <= c.width_m / 2 - c.footprint_margin_m
        quality = loaded & slow & slow_rotation & lateral & (normal_z >= c.foot_normal_cos) & safe[:, None]
        lower_geometry = ((bottom.abs() <= c.sole_height_tolerance_m).all(dim=2)
                          & (x_max <= c.first_riser_m - c.footprint_margin_m))
        upper_geometry = (((bottom - c.height_m).abs() <= c.sole_height_tolerance_m).all(dim=2)
                          & (x_min >= c.first_riser_m + c.footprint_margin_m))
        landing_rectangle = x_max <= c.first_riser_m + c.tread_m - c.footprint_margin_m
        lower = lower_geometry & quality
        upper = upper_geometry & quality
        self.lower_age = torch.where(lower, self.lower_age + dt, 0.0)
        self.upper_age = torch.where(upper, self.upper_age + dt, 0.0)
        stable_lower = self.lower_age >= c.support_dwell_s - 1e-6
        stable_upper = self.upper_age >= c.support_dwell_s - 1e-6
        # Ground reference is captured during actual loaded double support.
        # Upper starts/reset teleporting can never mint an achievement.
        calibrate = ~self.ready & lower.all(dim=1)
        self.base_reference = torch.where(calibrate, root[:, 2], self.base_reference)
        self.ready |= calibrate
        first = (old_stage == 0) & self.ready & (stable_upper & landing_rectangle).any(dim=1) & safe
        transfer = ((old_stage == 1) & stable_upper.any(dim=1) & safe
                    & (root[:, 0] >= c.first_riser_m + 0.02)
                    & (root[:, 2] >= self.base_reference + c.height_m * c.base_height_gain_fraction))
        self.seen_upper |= stable_upper & self.ready[:, None]
        # Both feet must each establish upper support, but not necessarily be
        # stationary simultaneously: do not impose a standing-only gait.
        both = ((old_stage == 2) & self.seen_upper.all(dim=1) & upper_geometry.all(dim=1)
                & stable_upper.any(dim=1) & safe)
        complete = ((old_stage == 3) & upper_geometry.all(dim=1) & upper.any(dim=1) & safe
                    & (root[:, 2] >= self.base_reference + c.height_m * c.base_height_gain_fraction)
                    & (root[:, 0] >= c.first_riser_m + c.tread_m + c.continuation_m))
        for mask, stage in ((first, 1), (transfer, 2), (both, 3), (complete, 4)):
            self.stage[mask] = stage

        # Bounded shaping: only improvements over the episode's previous best
        # receive credit. Hovering, touchdown cycling and bouncing cannot repay.
        approach = (root[:, 0] / (c.first_riser_m - c.takeoff_distance_m)).clamp(0, 1)
        approach *= (safe & lower.any(dim=1) & self.ready).to(root.dtype)
        swing_zone = ((x_max >= c.first_riser_m - c.takeoff_distance_m)
                      & (x_min <= c.first_riser_m + c.tread_m))
        stance = stable_lower.flip(dims=(1,))
        swing_ok = swing_zone & stance & ~loaded & safe[:, None] & self.ready[:, None]
        clearance_value = (bottom.amin(dim=2) / (c.height_m + c.clearance_margin_m)).clamp(0, 1)
        clearance = torch.where(swing_ok, clearance_value, 0.0).amax(dim=1)
        rise = ((root[:, 2] - self.base_reference) / c.height_m).clamp(0, 1)
        rise *= (safe & stable_upper.any(dim=1) & self.ready).to(root.dtype)
        potentials = torch.stack((approach, clearance, rise), dim=1)
        increments = (potentials - self.best).clamp_min(0)
        self.best = torch.maximum(self.best, potentials)

        # Early high kicks are penalized throughout the approach, not just once
        # the pelvis happens to enter the old clearance window.
        early = x_max < c.first_riser_m - c.takeoff_distance_m
        early_lift = torch.where(early, (bottom.amax(dim=2) - 0.08).clamp_min(0), 0.0).sum(dim=1)
        distance_to_face = (c.first_riser_m - x_max).clamp_min(0)
        approaching_face = (x_max <= c.first_riser_m + 0.02) & (x_max >= c.first_riser_m - 0.12)
        below_riser = bottom.amin(dim=2) < c.height_m + 0.005
        risk = (approaching_face & below_riser).to(root.dtype) * (1 - distance_to_face / 0.12).clamp(0, 1)
        riser_risk = (risk * foot_velocity[..., 0].clamp_min(0)).sum(dim=1)
        contact = forces[..., 2] > 5.0
        onset = contact & ~self.previous_contact & self.has_previous[:, None]
        near_upper = x_max >= c.first_riser_m
        # Preserve the same pre-contact velocity used by the touchdown
        # penalty before advancing the detector's history state. Diagnostics
        # must report this value, not the current post-step foot velocity.
        precontact_velocity = self.previous_velocity.clone()

        downward_excess = (
            -precontact_velocity[..., 2]
            - c.touchdown_downward_speed_mps
        ).clamp_min(0)
        horizontal_excess = (
            torch.linalg.vector_norm(
                precontact_velocity[..., :2],
                dim=2,
            )
            - c.touchdown_horizontal_speed_mps
        ).clamp_min(0)
        touchdown = (
            (
                downward_excess.square()
                + horizontal_excess.square()
            )
            * (onset & near_upper).to(root.dtype)
        ).sum(dim=1)

        self.previous_velocity = foot_velocity.clone()
        self.previous_contact = contact.clone()
        self.has_previous[:] = True
        failure = forbidden | (-gravity[:, 2] < 0.70) | (root[:, 2] < c.minimum_base_height_m)
        # Physical completion is never credited on a simultaneously failed step.
        complete &= ~failure
        components = dict(first_support=first.to(root.dtype), base_transfer=transfer.to(root.dtype),
                          both_feet=both.to(root.dtype), complete=complete.to(root.dtype),
                          approach=increments[:, 0], clearance=increments[:, 1],
                          supported_rise=increments[:, 2], early_lift=early_lift,
                          riser_risk=riser_risk, touchdown=touchdown)
        measurements = dict(first_support_count=first, base_transfer_count=transfer,
                            both_feet_count=both, completion_count=complete,
                            failure_env_steps=failure, forbidden_env_steps=forbidden,
                            upright_env_steps=upright, supported_env_steps=stable_upper.any(dim=1),
                            calibrated_env_steps=self.ready.clone())
        for name, value in measurements.items():
            if name not in self.totals:
                self.totals[name] = value.to(torch.long).sum()
            else:
                self.totals[name] += value.to(torch.long).sum()
        self.totals["environment_steps"] = self.totals.get("environment_steps", 0) + root.shape[0]

        # Diagnostic payload only.
        #
        # Every tensor is cloned so downstream instrumentation cannot mutate
        # reward, termination, success, stage, or physical detector state.
        # Nothing in this payload is consumed by the task itself.
        diagnostics = dict(
            root=root.clone(),
            foot_x_min=x_min.clone(),
            foot_x_max=x_max.clone(),
            sole_bottom_min=bottom.amin(dim=2).clone(),
            sole_bottom_max=bottom.amax(dim=2).clone(),
            foot_normal_z=normal_z.clone(),
            vertical_force=forces[..., 2].clone(),
            bodyweight=bodyweight.clone(),
            loaded=loaded.clone(),
            slow_translation=slow.clone(),
            slow_rotation=slow_rotation.clone(),
            lateral_contained=lateral.clone(),
            support_quality=quality.clone(),
            lower_geometry=lower_geometry.clone(),
            upper_geometry=upper_geometry.clone(),
            landing_rectangle=landing_rectangle.clone(),
            stable_lower=stable_lower.clone(),
            stable_upper=stable_upper.clone(),
            swing_zone=swing_zone.clone(),
            swing_ok=swing_ok.clone(),
            clearance_value=clearance_value.clone(),
            distance_to_riser_face=distance_to_face.clone(),
            contact=contact.clone(),
            contact_onset=onset.clone(),
            near_upper=near_upper.clone(),
            precontact_velocity=precontact_velocity.clone(),
            foot_velocity=foot_velocity.clone(),
            foot_angular_velocity=foot_angular_velocity.clone(),
            upright=upright.clone(),
            safe=safe.clone(),
            forbidden=forbidden.clone(),
        )

        self.snapshot = dict(
            components=components,
            failure=failure,
            success=complete,
            measurements=measurements,
            stage=self.stage.clone(),
            diagnostics=diagnostics,
        )
        self.last_step = step
        return self.snapshot


def reset(env, env_ids):
    runtime = getattr(env, "_physical_step043", None)
    if runtime is not None:
        runtime.reset(slice(None) if env_ids is None else env_ids)


def snapshot(env, config):
    import torch
    runtime = getattr(env, "_physical_step043", None)
    if runtime is not None and runtime.last_step == env.common_step_counter:
        return runtime.snapshot
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    if runtime is None:
        runtime = PhysicalStepState(env.num_envs, env.device, PhysicalStepConfig(**config).validate())
        runtime.foot_ids = [robot.body_names.index(name) for name in FOOT_BODY_NAMES]
        runtime.contact_ids = [sensor.body_names.index(name) for name in FOOT_BODY_NAMES]
        runtime.forbidden_ids = [sensor.body_names.index(name) for name in FORBIDDEN_CONTACT_BODY_NAMES]
        masses = robot.root_physx_view.get_masses().to(device=env.device)
        runtime.bodyweight = masses.sum(dim=1) * 9.81
        env._physical_step043 = runtime
    data = robot.data
    ids = runtime.foot_ids
    # All samples are from the link frame, not the body's offset COM frame.
    positions = data.body_link_pos_w[:, ids] - env.scene.env_origins[:, None, :]
    quaternions = data.body_link_quat_w[:, ids]
    velocities = data.body_link_lin_vel_w[:, ids]
    angular = data.body_link_ang_vel_w[:, ids]
    forces = sensor.data.net_forces_w
    forbidden = torch.linalg.vector_norm(forces[:, runtime.forbidden_ids], dim=2).gt(config["forbidden_force_n"]).any(dim=1)
    return runtime.update(env.common_step_counter, float(env.step_dt),
                          root=data.root_pos_w - env.scene.env_origins,
                          gravity=data.projected_gravity_b, foot_pos=positions, foot_quat=quaternions,
                          foot_velocity=velocities, foot_angular_velocity=angular,
                          forces=forces[:, runtime.contact_ids], forbidden=forbidden,
                          bodyweight=runtime.bodyweight)


def reward(env, config, component):
    value = snapshot(env, config)["components"][component]
    return value if component in RATE_COMPONENTS else value / float(env.step_dt)


def failure(env, config):
    return snapshot(env, config)["failure"]


def success(env, config):
    return snapshot(env, config)["success"]


def summary(env):
    runtime = getattr(env, "_physical_step043", None)
    if runtime is None:
        raise RuntimeError("043 physical snapshot was never evaluated")
    result = {name: int(value) for name, value in runtime.totals.items()}
    result["config"] = asdict(runtime.config)
    result["meaning"] = "event counts and environment-step denominators; not evaluation success rates"
    return result
