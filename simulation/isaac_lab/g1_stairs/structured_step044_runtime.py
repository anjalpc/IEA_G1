"""Pure structured state-machine core for experiment 044.

This module deliberately does not modify or depend on PhysicalStepState.
Isaac scene extraction is added only after the tensor/state contract passes.
"""

from dataclasses import asdict

from .structured_step044_config import (
    RATE_COMPONENTS,
    StructuredStep044Config,
)
from .step_up_isaac import (
    FOOT_BODY_NAMES,
    FORBIDDEN_CONTACT_BODY_NAMES,
)


LOWER = 0
SWING = 1
LIFTED = 2
RISER_READY = 3
CROSSED = 4
TOUCHDOWN = 5
UPPER_SUPPORT = 6

STAGE_NAMES = (
    "LOWER",
    "SWING",
    "LIFTED",
    "RISER_READY",
    "CROSSED",
    "TOUCHDOWN",
    "UPPER_SUPPORT",
)


# Pinned g1_29dof_rev_1_0 ankle-roll collision spheres.
SOLE_CENTRES = (
    (-0.05, -0.025, -0.03),
    (-0.05,  0.025, -0.03),
    ( 0.12, -0.03,  -0.03),
    ( 0.12,  0.03,  -0.03),
)

SOLE_RADIUS = 0.005


def rotate(q, vectors):
    import torch

    xyz = q[..., 1:]
    uv = torch.cross(
        xyz,
        vectors,
        dim=-1,
    )

    return vectors + 2 * (
        q[..., :1] * uv
        + torch.cross(
            xyz,
            uv,
            dim=-1,
        )
    )


class StructuredStep044State:
    """Per-foot ordered stair-transition state."""

    def __init__(self, num_envs, device, config):
        import torch

        self.config = config
        self.last_step = None
        self.snapshot = None
        self.totals = {}

        self.foot_stage = torch.zeros(
            (num_envs, 2),
            dtype=torch.long,
            device=device,
        )

        self.ready = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

        self.base_reference = torch.zeros(
            num_envs,
            device=device,
        )

        self.lower_age = torch.zeros(
            (num_envs, 2),
            device=device,
        )

        self.upper_age = torch.zeros(
            (num_envs, 2),
            device=device,
        )

        self.seen_upper = torch.zeros(
            (num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self.approach_best = torch.zeros(
            num_envs,
            device=device,
        )

        self.lift_advance_best = torch.zeros(
            (num_envs, 2),
            device=device,
        )

        self.insertion_best = torch.zeros(
            (num_envs, 2),
            device=device,
        )

        self.body_rise_best = torch.zeros(
            num_envs,
            device=device,
        )

        self.previous_x_max = torch.zeros(
            (num_envs, 2),
            device=device,
        )

        self.previous_velocity = torch.zeros(
            (num_envs, 2, 3),
            device=device,
        )

        self.previous_contact = torch.zeros(
            (num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self.has_previous = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

        self.low_crossing_seen = torch.zeros(
            (num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self.clearance_loss_seen = torch.zeros(
            (num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self.body_transfer_seen = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

        self.both_feet_seen = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

        self.complete_seen = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

    def reset(self, ids):
        values = (
            self.foot_stage,
            self.ready,
            self.base_reference,
            self.lower_age,
            self.upper_age,
            self.seen_upper,
            self.approach_best,
            self.lift_advance_best,
            self.insertion_best,
            self.body_rise_best,
            self.previous_x_max,
            self.previous_velocity,
            self.previous_contact,
            self.has_previous,
            self.low_crossing_seen,
            self.clearance_loss_seen,
            self.body_transfer_seen,
            self.both_feet_seen,
            self.complete_seen,
        )

        for value in values:
            value[ids] = 0

    def update(
        self,
        dt,
        *,
        root,
        gravity,
        x_min,
        x_max,
        sole_bottom,
        normal_z,
        loaded,
        stable_lower,
        stable_upper,
        upper_geometry,
        landing_rectangle,
        contact,
        foot_velocity,
        forbidden,
    ):
        import torch

        if dt <= 0:
            raise ValueError(
                "044 control timestep must be positive"
            )

        values = (
            root,
            gravity,
            x_min,
            x_max,
            sole_bottom,
            normal_z,
            foot_velocity,
        )

        if not all(
            bool(torch.isfinite(value).all())
            for value in values
        ):
            raise RuntimeError(
                "044 non-finite detector input"
            )

        c = self.config
        dtype = root.dtype

        upright = (
            -gravity[:, 2]
            >= c.upright_cos
        )

        safe = (
            upright
            & ~forbidden
            & (
                root[:, 2]
                >= c.minimum_base_height_m
            )
        )

        # Calibrate base height only from real double lower support.
        calibrate = (
            ~self.ready
            & stable_lower.all(dim=1)
            & safe
        )

        self.base_reference = torch.where(
            calibrate,
            root[:, 2],
            self.base_reference,
        )

        self.ready |= calibrate

        opposite_support = (
            stable_lower.flip(dims=(1,))
            | stable_upper.flip(dims=(1,))
        )

        in_lift_zone = (
            x_max
            >= (
                c.first_riser_m
                - c.lift_zone_distance_m
            )
        ) & (
            x_max
            < c.first_riser_m
        )

        swing_eligible = (
            self.ready[:, None]
            & safe[:, None]
            & opposite_support
            & ~loaded
            & in_lift_zone
        )

        previous_stage = (
            self.foot_stage.clone()
        )

        # --------------------------------------------------------
        # LOWER -> SWING
        # --------------------------------------------------------
        enter_swing = (
            (previous_stage == LOWER)
            & swing_eligible
        )

        self.foot_stage = torch.where(
            enter_swing,
            torch.full_like(
                self.foot_stage,
                SWING,
            ),
            self.foot_stage,
        )

        # --------------------------------------------------------
        # SWING -> LIFTED
        # --------------------------------------------------------
        meaningful_lift_condition = (
            sole_bottom
            >= (
                c.meaningful_lift_m
                - c.geometry_tolerance_m
            )
        )

        meaningful_lift_event = (
            (previous_stage == SWING)
            & swing_eligible
            & meaningful_lift_condition
        )

        self.foot_stage = torch.where(
            meaningful_lift_event,
            torch.full_like(
                self.foot_stage,
                LIFTED,
            ),
            self.foot_stage,
        )

        full_clearance = (
            sole_bottom
            >= (
                c.required_clearance_m
                - c.geometry_tolerance_m
            )
        )

        good_normal = (
            normal_z
            >= c.foot_normal_cos
        )

        # --------------------------------------------------------
        # LIFTED -> RISER_READY
        # --------------------------------------------------------
        ready_geometry = (
            x_max
            >= (
                c.first_riser_m
                - c.riser_ready_distance_m
            )
        ) & (
            x_max
            < c.first_riser_m
        )

        riser_ready_event = (
            (previous_stage == LIFTED)
            & ready_geometry
            & full_clearance
            & good_normal
            & opposite_support
            & safe[:, None]
        )

        self.foot_stage = torch.where(
            riser_ready_event,
            torch.full_like(
                self.foot_stage,
                RISER_READY,
            ),
            self.foot_stage,
        )

        crossing_onset = (
            self.has_previous[:, None]
            & (
                self.previous_x_max
                < c.first_riser_m
            )
            & (
                x_max
                >= c.first_riser_m
            )
        )

        # --------------------------------------------------------
        # Dominant historical failure:
        # front edge reaches plane without physical clearance.
        # --------------------------------------------------------
        low_crossing_raw = (
            crossing_onset
            & ~full_clearance
            & (
                self.foot_stage
                < CROSSED
            )
            & safe[:, None]
        )

        low_crossing_event = (
            low_crossing_raw
            & ~self.low_crossing_seen
        )

        self.low_crossing_seen |= (
            low_crossing_event
        )

        # --------------------------------------------------------
        # RISER_READY -> CROSSED
        # --------------------------------------------------------
        valid_crossing_event = (
            (previous_stage == RISER_READY)
            & crossing_onset
            & full_clearance
            & good_normal
            & opposite_support
            & safe[:, None]
        )

        self.foot_stage = torch.where(
            valid_crossing_event,
            torch.full_like(
                self.foot_stage,
                CROSSED,
            ),
            self.foot_stage,
        )

        # Lift-then-drop exploit prevention.
        clearance_loss_raw = (
            (previous_stage == RISER_READY)
            & ~full_clearance
            & ~valid_crossing_event
        )

        clearance_loss_event = (
            clearance_loss_raw
            & ~self.clearance_loss_seen
        )

        self.clearance_loss_seen |= (
            clearance_loss_event
        )

        # --------------------------------------------------------
        # Coupled lift + advance bounded potential.
        # --------------------------------------------------------
        height_fraction = (
            sole_bottom
            / c.required_clearance_m
        ).clamp(0.0, 1.0)

        forward_fraction = (
            (
                x_max
                - (
                    c.first_riser_m
                    - c.lift_zone_distance_m
                )
            )
            / c.lift_zone_distance_m
        ).clamp(0.0, 1.0)

        lift_advance_value = (
            height_fraction
            * forward_fraction
        )

        lift_advance_value = torch.where(
            swing_eligible
            | (
                (self.foot_stage >= LIFTED)
                & (
                    self.foot_stage
                    <= RISER_READY
                )
                & opposite_support
                & safe[:, None]
            ),
            lift_advance_value,
            0.0,
        )

        lift_advance = (
            lift_advance_value
            - self.lift_advance_best
        ).clamp_min(0.0)

        self.lift_advance_best = torch.maximum(
            self.lift_advance_best,
            lift_advance_value,
        )

        # --------------------------------------------------------
        # Tread insertion after VALID crossing only.
        # --------------------------------------------------------
        nominal_foot_length_m = 0.18

        insertion_start = (
            c.first_riser_m
            - nominal_foot_length_m
        )

        insertion_fraction = (
            (
                x_min
                - insertion_start
            )
            / nominal_foot_length_m
        ).clamp(0.0, 1.0)

        insertion_gate = (
            (self.foot_stage >= CROSSED)
            & full_clearance
            & good_normal
            & safe[:, None]
        )

        insertion_value = torch.where(
            insertion_gate,
            insertion_fraction,
            0.0,
        )

        tread_insertion = (
            insertion_value
            - self.insertion_best
        ).clamp_min(0.0)

        self.insertion_best = torch.maximum(
            self.insertion_best,
            insertion_value,
        )

        # --------------------------------------------------------
        # Contact onset + PRE-contact velocity semantics.
        # --------------------------------------------------------
        contact_onset = (
            contact
            & ~self.previous_contact
            & self.has_previous[:, None]
        )

        precontact_velocity = (
            self.previous_velocity.clone()
        )

        downward_excess = (
            -precontact_velocity[..., 2]
            - c.touchdown_downward_speed_mps
        ).clamp_min(0.0)

        horizontal_excess = (
            torch.linalg.vector_norm(
                precontact_velocity[..., :2],
                dim=2,
            )
            - c.touchdown_horizontal_speed_mps
        ).clamp_min(0.0)

        bad_touchdown_value = (
            downward_excess.square()
            + horizontal_excess.square()
        )

        touchdown_candidate = (
            contact_onset
            & (
                previous_stage == CROSSED
            )
        )

        good_touchdown_event = (
            touchdown_candidate
            & upper_geometry
            & landing_rectangle
            & good_normal
            & (
                downward_excess
                <= c.geometry_tolerance_m
            )
            & (
                horizontal_excess
                <= c.geometry_tolerance_m
            )
            & safe[:, None]
        )

        bad_touchdown = torch.where(
            touchdown_candidate
            & ~good_touchdown_event,
            bad_touchdown_value,
            0.0,
        )

        self.foot_stage = torch.where(
            good_touchdown_event,
            torch.full_like(
                self.foot_stage,
                TOUCHDOWN,
            ),
            self.foot_stage,
        )

        # --------------------------------------------------------
        # TOUCHDOWN -> UPPER_SUPPORT
        # --------------------------------------------------------
        stable_upper_event = (
            (previous_stage == TOUCHDOWN)
            & stable_upper
            & landing_rectangle
            & safe[:, None]
        )

        self.foot_stage = torch.where(
            stable_upper_event,
            torch.full_like(
                self.foot_stage,
                UPPER_SUPPORT,
            ),
            self.foot_stage,
        )

        self.seen_upper |= (
            self.foot_stage == UPPER_SUPPORT
        )

        # --------------------------------------------------------
        # Approach bounded potential.
        # --------------------------------------------------------
        approach_value = (
            root[:, 0]
            / (
                c.first_riser_m
                - c.lift_zone_distance_m
            )
        ).clamp(0.0, 1.0)

        approach_value *= (
            self.ready
            & safe
            & stable_lower.any(dim=1)
        ).to(dtype)

        approach = (
            approach_value
            - self.approach_best
        ).clamp_min(0.0)

        self.approach_best = torch.maximum(
            self.approach_best,
            approach_value,
        )

        # --------------------------------------------------------
        # Body rise only after genuine upper support.
        # --------------------------------------------------------
        supported_body_rise_value = (
            (
                root[:, 2]
                - self.base_reference
            )
            / c.height_m
        ).clamp(0.0, 1.0)

        supported_body_rise_value *= (
            self.ready
            & safe
            & stable_upper.any(dim=1)
        ).to(dtype)

        supported_body_rise = (
            supported_body_rise_value
            - self.body_rise_best
        ).clamp_min(0.0)

        self.body_rise_best = torch.maximum(
            self.body_rise_best,
            supported_body_rise_value,
        )

        body_transfer_condition = (
            self.ready
            & safe
            & stable_upper.any(dim=1)
            & (
                root[:, 0]
                >= c.first_riser_m + 0.02
            )
            & (
                root[:, 2]
                >= (
                    self.base_reference
                    + c.height_m
                    * c.base_height_gain_fraction
                )
            )
        )

        body_transfer_event = (
            body_transfer_condition
            & ~self.body_transfer_seen
        )

        self.body_transfer_seen |= (
            body_transfer_event
        )

        both_feet_condition = (
            self.body_transfer_seen
            & self.seen_upper.all(dim=1)
            & upper_geometry.all(dim=1)
            & stable_upper.any(dim=1)
            & safe
        )

        both_feet_supported_event = (
            both_feet_condition
            & ~self.both_feet_seen
        )

        self.both_feet_seen |= (
            both_feet_supported_event
        )

        complete_condition = (
            self.both_feet_seen
            & upper_geometry.all(dim=1)
            & safe
            & (
                root[:, 2]
                >= (
                    self.base_reference
                    + c.height_m
                    * c.base_height_gain_fraction
                )
            )
            & (
                root[:, 0]
                >= (
                    c.first_riser_m
                    + c.tread_m
                    + c.continuation_m
                )
            )
        )

        complete_event = (
            complete_condition
            & ~self.complete_seen
        )

        self.complete_seen |= (
            complete_event
        )

        # --------------------------------------------------------
        # Failure-aware rate terms.
        # --------------------------------------------------------
        premature_zone = (
            x_max
            < (
                c.first_riser_m
                - c.lift_zone_distance_m
            )
        )

        premature_lift = torch.where(
            premature_zone,
            (
                sole_bottom
                - c.premature_lift_height_m
            ).clamp_min(0.0),
            0.0,
        ).sum(dim=1)

        distance_to_face = (
            c.first_riser_m
            - x_max
        ).clamp_min(0.0)

        face_zone = (
            x_max
            >= c.first_riser_m - 0.12
        ) & (
            x_max
            <= c.first_riser_m + 0.02
        )

        below_required = (
            sole_bottom
            < c.required_clearance_m
        )

        riser_risk = (
            face_zone
            & below_required
        ).to(dtype)

        riser_risk *= (
            1.0
            - distance_to_face / 0.12
        ).clamp(0.0, 1.0)

        riser_risk = (
            riser_risk
            * foot_velocity[..., 0].clamp_min(0.0)
        ).sum(dim=1)

        failure = (
            forbidden
            | (
                -gravity[:, 2]
                < 0.70
            )
            | (
                root[:, 2]
                < c.minimum_base_height_m
            )
        )

        complete_event &= ~failure

        # Advance history only after all transition tests.
        self.previous_x_max = x_max.clone()
        self.previous_velocity = (
            foot_velocity.clone()
        )
        self.previous_contact = (
            contact.clone()
        )
        self.has_previous[:] = True

        components = {
            "approach": approach,
            "lift_advance": (
                lift_advance.sum(dim=1)
            ),
            "meaningful_lift": (
                meaningful_lift_event
                .to(dtype)
                .sum(dim=1)
            ),
            "riser_ready": (
                riser_ready_event
                .to(dtype)
                .sum(dim=1)
            ),
            "valid_crossing": (
                valid_crossing_event
                .to(dtype)
                .sum(dim=1)
            ),
            "tread_insertion": (
                tread_insertion.sum(dim=1)
            ),
            "good_touchdown": (
                good_touchdown_event
                .to(dtype)
                .sum(dim=1)
            ),
            "stable_upper_support": (
                stable_upper_event
                .to(dtype)
                .sum(dim=1)
            ),
            "supported_body_rise":
                supported_body_rise,
            "body_transfer":
                body_transfer_event.to(dtype),
            "both_feet_supported":
                both_feet_supported_event.to(dtype),
            "complete":
                complete_event.to(dtype),
            "premature_lift":
                premature_lift,
            "riser_risk":
                riser_risk,
            "low_crossing": (
                low_crossing_event
                .to(dtype)
                .sum(dim=1)
            ),
            "clearance_loss": (
                clearance_loss_event
                .to(dtype)
                .sum(dim=1)
            ),
            "bad_touchdown": (
                bad_touchdown.sum(dim=1)
            ),
        }

        diagnostics = {
            "foot_stage":
                self.foot_stage.clone(),
            "height_fraction":
                height_fraction.clone(),
            "forward_fraction":
                forward_fraction.clone(),
            "lift_advance_value":
                lift_advance_value.clone(),
            "meaningful_lift_event":
                meaningful_lift_event.clone(),
            "riser_ready_event":
                riser_ready_event.clone(),
            "crossing_onset":
                crossing_onset.clone(),
            "valid_crossing_event":
                valid_crossing_event.clone(),
            "low_crossing_event":
                low_crossing_event.clone(),
            "clearance_loss_event":
                clearance_loss_event.clone(),
            "insertion_fraction":
                insertion_fraction.clone(),
            "good_touchdown_event":
                good_touchdown_event.clone(),
            "bad_touchdown_value":
                bad_touchdown.clone(),
            "stable_upper_entry":
                stable_upper_event.clone(),
            "supported_body_rise":
                supported_body_rise.clone(),
            "body_transfer_event":
                body_transfer_event.clone(),
            "both_feet_supported_event":
                both_feet_supported_event.clone(),
            "completion_event":
                complete_event.clone(),
            "failure":
                failure.clone(),
        }

        return {
            "components": components,
            "diagnostics": diagnostics,
            "failure": failure,
            "success": complete_event,
            "foot_stage":
                self.foot_stage.clone(),
        }


def reset(env, env_ids):
    runtime = getattr(
        env,
        "_structured_step044",
        None,
    )

    if runtime is not None:
        runtime.reset(
            slice(None)
            if env_ids is None
            else env_ids
        )


def snapshot(env, config):
    """Build the physical snapshot once per control step."""
    import torch

    runtime = getattr(
        env,
        "_structured_step044",
        None,
    )

    if (
        runtime is not None
        and runtime.last_step
        == env.common_step_counter
    ):
        return runtime.snapshot

    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]

    if runtime is None:
        cfg = StructuredStep044Config(
            **config
        ).validate()

        runtime = StructuredStep044State(
            env.num_envs,
            env.device,
            cfg,
        )

        runtime.foot_ids = [
            robot.body_names.index(name)
            for name in FOOT_BODY_NAMES
        ]

        runtime.contact_ids = [
            sensor.body_names.index(name)
            for name in FOOT_BODY_NAMES
        ]

        runtime.forbidden_ids = [
            sensor.body_names.index(name)
            for name in FORBIDDEN_CONTACT_BODY_NAMES
        ]

        masses = (
            robot.root_physx_view
            .get_masses()
            .to(device=env.device)
        )

        runtime.bodyweight = (
            masses.sum(dim=1) * 9.81
        )

        if bool(
            (runtime.bodyweight <= 0).any()
        ):
            raise RuntimeError(
                "044 invalid articulation mass"
            )

        env._structured_step044 = runtime

    c = runtime.config
    data = robot.data

    ids = runtime.foot_ids

    foot_pos = (
        data.body_link_pos_w[:, ids]
        - env.scene.env_origins[:, None, :]
    )

    foot_quat = (
        data.body_link_quat_w[:, ids]
    )

    foot_velocity = (
        data.body_link_lin_vel_w[:, ids]
    )

    foot_angular_velocity = (
        data.body_link_ang_vel_w[:, ids]
    )

    if bool(
        (
            torch.linalg.vector_norm(
                foot_quat,
                dim=2,
            )
            - 1
        )
        .abs()
        .gt(1.0e-3)
        .any()
    ):
        raise RuntimeError(
            "044 foot orientation is not a unit quaternion"
        )

    centres = torch.tensor(
        SOLE_CENTRES,
        device=env.device,
        dtype=foot_pos.dtype,
    )

    offsets = centres.expand(
        env.num_envs,
        2,
        -1,
        -1,
    )

    q = foot_quat.unsqueeze(2).expand(
        -1,
        -1,
        4,
        -1,
    )

    points = (
        foot_pos.unsqueeze(2)
        + rotate(q, offsets)
    )

    bottom = (
        points[..., 2]
        - SOLE_RADIUS
    )

    x_min = (
        points[..., 0].amin(dim=2)
        - SOLE_RADIUS
    )

    x_max = (
        points[..., 0].amax(dim=2)
        + SOLE_RADIUS
    )

    y_max = (
        points[..., 1]
        .abs()
        .amax(dim=2)
        + SOLE_RADIUS
    )

    sole_bottom = bottom.amin(dim=2)

    normals = torch.zeros_like(
        foot_pos
    )
    normals[..., 2] = 1.0

    normal_z = rotate(
        foot_quat,
        normals,
    )[..., 2]

    forces = (
        sensor.data.net_forces_w[
            :,
            runtime.contact_ids,
        ]
    )

    all_forces = (
        sensor.data.net_forces_w
    )

    forbidden = (
        torch.linalg.vector_norm(
            all_forces[
                :,
                runtime.forbidden_ids,
            ],
            dim=2,
        )
        .gt(c.forbidden_force_n)
        .any(dim=1)
    )

    root = (
        data.root_pos_w
        - env.scene.env_origins
    )

    gravity = (
        data.projected_gravity_b
    )

    upright = (
        -gravity[:, 2]
        >= c.upright_cos
    )

    safe = (
        upright
        & ~forbidden
        & (
            root[:, 2]
            >= c.minimum_base_height_m
        )
    )

    loaded = (
        forces[..., 2]
        >= (
            runtime.bodyweight[:, None]
            * c.support_force_bodyweight_fraction
        )
    )

    slow_translation = (
        torch.linalg.vector_norm(
            foot_velocity,
            dim=2,
        )
        <= c.support_speed_mps
    )

    slow_rotation = (
        torch.linalg.vector_norm(
            foot_angular_velocity,
            dim=2,
        )
        <= c.support_angular_speed_radps
    )

    lateral_contained = (
        y_max
        <= (
            c.width_m / 2
            - c.footprint_margin_m
        )
    )

    support_quality = (
        loaded
        & slow_translation
        & slow_rotation
        & lateral_contained
        & (
            normal_z
            >= c.foot_normal_cos
        )
        & safe[:, None]
    )

    lower_geometry = (
        (
            bottom.abs()
            <= c.sole_height_tolerance_m
        ).all(dim=2)
        & (
            x_max
            <= (
                c.first_riser_m
                - c.footprint_margin_m
            )
        )
    )

    upper_geometry = (
        (
            (
                bottom
                - c.height_m
            ).abs()
            <= c.sole_height_tolerance_m
        ).all(dim=2)
        & (
            x_min
            >= (
                c.first_riser_m
                + c.footprint_margin_m
            )
        )
    )

    landing_rectangle = (
        x_max
        <= (
            c.first_riser_m
            + c.tread_m
            - c.footprint_margin_m
        )
    )

    lower = (
        lower_geometry
        & support_quality
    )

    upper = (
        upper_geometry
        & support_quality
    )

    dt = float(env.step_dt)

    runtime.lower_age = torch.where(
        lower,
        runtime.lower_age + dt,
        0.0,
    )

    runtime.upper_age = torch.where(
        upper,
        runtime.upper_age + dt,
        0.0,
    )

    stable_lower = (
        runtime.lower_age
        >= (
            c.support_dwell_s
            - 1.0e-6
        )
    )

    stable_upper = (
        runtime.upper_age
        >= (
            c.support_dwell_s
            - 1.0e-6
        )
    )

    contact = (
        forces[..., 2] > 5.0
    )

    result = runtime.update(
        dt,
        root=root,
        gravity=gravity,
        x_min=x_min,
        x_max=x_max,
        sole_bottom=sole_bottom,
        normal_z=normal_z,
        loaded=loaded,
        stable_lower=stable_lower,
        stable_upper=stable_upper,
        upper_geometry=upper_geometry,
        landing_rectangle=landing_rectangle,
        contact=contact,
        foot_velocity=foot_velocity,
        forbidden=forbidden,
    )

    diagnostics = result[
        "diagnostics"
    ]

    # Preserve the established physical diagnostic vocabulary.
    diagnostics.update(
        {
            "root":
                root.clone(),
            "foot_x_min":
                x_min.clone(),
            "foot_x_max":
                x_max.clone(),
            "sole_bottom_min":
                sole_bottom.clone(),
            "foot_normal_z":
                normal_z.clone(),
            "vertical_force":
                forces[..., 2].clone(),
            "bodyweight":
                runtime.bodyweight.clone(),
            "loaded":
                loaded.clone(),
            "slow_translation":
                slow_translation.clone(),
            "slow_rotation":
                slow_rotation.clone(),
            "lateral_contained":
                lateral_contained.clone(),
            "support_quality":
                support_quality.clone(),
            "lower_geometry":
                lower_geometry.clone(),
            "upper_geometry":
                upper_geometry.clone(),
            "landing_rectangle":
                landing_rectangle.clone(),
            "stable_lower":
                stable_lower.clone(),
            "stable_upper":
                stable_upper.clone(),
            "contact":
                contact.clone(),
            "precontact_velocity":
                runtime.previous_velocity.clone(),
            "foot_velocity":
                foot_velocity.clone(),
            "foot_angular_velocity":
                foot_angular_velocity.clone(),
            "upright":
                upright.clone(),
            "safe":
                safe.clone(),
            "forbidden":
                forbidden.clone(),
        }
    )

    measurements = {
        "meaningful_lift_count":
            diagnostics[
                "meaningful_lift_event"
            ].any(dim=1),

        "riser_ready_count":
            diagnostics[
                "riser_ready_event"
            ].any(dim=1),

        "valid_crossing_count":
            diagnostics[
                "valid_crossing_event"
            ].any(dim=1),

        "low_crossing_count":
            diagnostics[
                "low_crossing_event"
            ].any(dim=1),

        "clearance_loss_count":
            diagnostics[
                "clearance_loss_event"
            ].any(dim=1),

        "good_touchdown_count":
            diagnostics[
                "good_touchdown_event"
            ].any(dim=1),

        "stable_upper_count":
            diagnostics[
                "stable_upper_entry"
            ].any(dim=1),

        "body_transfer_count":
            diagnostics[
                "body_transfer_event"
            ],

        "both_feet_count":
            diagnostics[
                "both_feet_supported_event"
            ],

        "completion_count":
            diagnostics[
                "completion_event"
            ],

        "failure_env_steps":
            result["failure"],

        "calibrated_env_steps":
            runtime.ready.clone(),
    }

    result["measurements"] = (
        measurements
    )

    for name, value in (
        measurements.items()
    ):
        amount = int(
            value.to(torch.long)
            .sum()
            .item()
        )

        runtime.totals[name] = (
            runtime.totals.get(
                name,
                0,
            )
            + amount
        )

    runtime.totals[
        "environment_steps"
    ] = (
        runtime.totals.get(
            "environment_steps",
            0,
        )
        + env.num_envs
    )

    runtime.snapshot = result
    runtime.last_step = (
        env.common_step_counter
    )

    return result


def reward(
    env,
    config,
    component,
):
    value = snapshot(
        env,
        config,
    )["components"][component]

    # Isaac RewardManager multiplies every term by dt.
    # Continuous penalties are true rates.
    # Events and bounded potentials must therefore be divided by dt.
    if component in RATE_COMPONENTS:
        return value

    return (
        value
        / float(env.step_dt)
    )


def failure(env, config):
    return snapshot(
        env,
        config,
    )["failure"]


def success(env, config):
    return snapshot(
        env,
        config,
    )["success"]


def summary(env):
    runtime = getattr(
        env,
        "_structured_step044",
        None,
    )

    if runtime is None:
        raise RuntimeError(
            "044 structured snapshot "
            "was never evaluated"
        )

    result = {
        name: int(value)
        for name, value
        in runtime.totals.items()
    }

    result["config"] = (
        asdict(runtime.config)
    )

    result["meaning"] = (
        "044 physical event counts and "
        "environment-step denominators; "
        "video remains authoritative"
    )

    return result


def summary_config(config):
    return asdict(config)
