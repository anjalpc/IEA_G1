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

    def __init__(self, num_envs, device, config, reward_weights=None):
        import torch

        self.config = config
        self.reward_weights = dict(reward_weights or {})
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

        # 047 anti-exploit state. These are dormant for reward modes 1/2.
        self.ascent_started = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

        self.previous_full_clearance = torch.zeros(
            (num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        self.qualified_x_best = torch.zeros(
            (num_envs, 2),
            device=device,
        )

        self.maintained_advance_total = torch.zeros(
            (num_envs, 2),
            device=device,
        )

        # 048 terminal stability dwell. Dormant in earlier reward modes.
        self.terminal_stability_age = torch.zeros(
            num_envs,
            device=device,
        )

        self.prelaunch_age = torch.zeros(
            num_envs,
            device=device,
        )

        self.ascent_progress_age = torch.zeros(
            num_envs,
            device=device,
        )

        self.positive_stair_credit = torch.zeros(
            num_envs,
            device=device,
        )

        # 049 clearance-acquisition state. The baseline is captured exactly
        # when the body enters the valid ascent phase, so a pre-launch high
        # foot cannot be back-credited as stair-learning progress.
        self.clearance_baseline_fraction = torch.zeros(
            (num_envs, 2),
            device=device,
        )
        self.clearance_acquired_seen = torch.zeros(
            (num_envs, 2),
            dtype=torch.bool,
            device=device,
        )

        if config.reward_mode in (4, 5, 6):
            required_weights = {
                "step044_approach",
                "step044_lift_advance",
                "step044_maintained_clearance_advance",
                "step044_meaningful_lift",
                "step044_riser_ready",
                "step044_valid_crossing",
                "step044_tread_insertion",
                "step044_good_touchdown",
                "step044_stable_upper_support",
                "step044_supported_body_rise",
                "step044_body_transfer",
                "step044_both_feet_supported",
                "step044_complete",
                "step044_failure",
            }
            missing = sorted(
                required_weights - set(self.reward_weights)
            )
            if missing:
                raise RuntimeError(
                    "048/049 requires reward weights for failure clawback: "
                    f"{missing}"
                )
            positive_names = required_weights - {"step044_failure"}
            nonpositive = sorted(
                name
                for name in positive_names
                if self.reward_weights[name] <= 0.0
            )
            if nonpositive:
                raise RuntimeError(
                    "048/049 positive milestone weights must be positive: "
                    f"{nonpositive}"
                )
            if self.reward_weights["step044_failure"] >= 0.0:
                raise RuntimeError(
                    "048/049 failure reward weight must be negative"
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
            self.ascent_started,
            self.previous_full_clearance,
            self.qualified_x_best,
            self.maintained_advance_total,
            self.terminal_stability_age,
            self.prelaunch_age,
            self.ascent_progress_age,
            self.positive_stair_credit,
            self.clearance_baseline_fraction,
            self.clearance_acquired_seen,
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
        forbidden_contacts=None,
        root_linear_velocity=None,
        root_angular_velocity=None,
        root_yaw=None,
        foot_lateral_max=None,
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

        if c.reward_mode in (4, 5, 6):
            mode4_values = (
                root_linear_velocity,
                root_angular_velocity,
                root_yaw,
                foot_lateral_max,
            )
            if any(value is None for value in mode4_values):
                raise RuntimeError(
                    "048/049 requires root velocity, yaw, and foot lateral geometry"
                )
            if not all(
                bool(torch.isfinite(value).all())
                for value in mode4_values
            ):
                raise RuntimeError(
                    "048/049 non-finite alignment/stability input"
                )

        if forbidden_contacts is None:
            forbidden_contacts = torch.zeros(
                (root.shape[0], len(FORBIDDEN_CONTACT_BODY_NAMES)),
                dtype=torch.bool,
                device=root.device,
            )
        elif forbidden_contacts.shape != (
            root.shape[0], len(FORBIDDEN_CONTACT_BODY_NAMES)
        ):
            raise RuntimeError(
                "049 forbidden-contact attribution shape mismatch"
            )

        forbidden_index = {
            name: index
            for index, name in enumerate(FORBIDDEN_CONTACT_BODY_NAMES)
        }
        forbidden_pelvis = forbidden_contacts[:, forbidden_index["pelvis"]]
        forbidden_torso = forbidden_contacts[:, forbidden_index["torso_link"]]
        forbidden_knee = forbidden_contacts[:, [
            forbidden_index["left_knee_link"],
            forbidden_index["right_knee_link"],
        ]].any(dim=1)
        forbidden_elbow = forbidden_contacts[:, [
            forbidden_index["left_elbow_link"],
            forbidden_index["right_elbow_link"],
        ]].any(dim=1)
        forbidden_hand = forbidden_contacts[:, [
            forbidden_index["left_rubber_hand"],
            forbidden_index["right_rubber_hand"],
        ]].any(dim=1)

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

        # Historical modes calibrate only from real double lower support.
        # 048/049 instead initialize once from the safe post-reset body pose.
        # Requiring a 0.12 s simultaneous double-support dwell here prevented
        # continuous walking episodes from ever arming the stair state machine.
        if c.reward_mode in (4, 5, 6):
            calibrate = (
                ~self.ready
                & safe
                & (
                    root[:, 0]
                    < c.first_riser_m - c.ascent_launch_distance_m
                )
            )
        else:
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

        root_centered = torch.ones_like(safe)
        feet_centered = torch.ones_like(safe)
        yaw_aligned = torch.ones_like(safe)
        launch_speed_ok = torch.ones_like(safe)
        terminal_motion_ok = torch.ones_like(safe)

        if c.reward_mode in (4, 5, 6):
            root_centered = (
                torch.abs(root[:, 1])
                <= c.alignment_root_y_tolerance_m
            )
            feet_centered = (
                foot_lateral_max
                <= c.alignment_foot_y_tolerance_m
            ).all(dim=1)
            yaw_aligned = (
                torch.abs(root_yaw)
                <= c.alignment_yaw_tolerance_rad
            )
            launch_speed_ok = (
                root_linear_velocity[:, 0]
                >= -c.geometry_tolerance_m
            ) & (
                root_linear_velocity[:, 0]
                <= c.ascent_max_forward_speed_mps
            ) & (
                torch.abs(root_linear_velocity[:, 1])
                <= c.ascent_max_lateral_speed_mps
            )
            terminal_motion_ok = (
                torch.linalg.vector_norm(
                    root_linear_velocity[:, :2],
                    dim=1,
                )
                <= c.terminal_linear_speed_mps
            ) & (
                torch.linalg.vector_norm(
                    root_angular_velocity,
                    dim=1,
                )
                <= c.terminal_angular_speed_radps
            )

        opposite_support = (
            stable_lower.flip(dims=(1,))
            | stable_upper.flip(dims=(1,))
        )

        # 050 distinguishes validated swing ENTRY from dynamic swing
        # CONTINUATION. A 0.12 s stable-support dwell is appropriate for
        # entering a supported swing, but re-demanding that dwell on every
        # later control frame can suppress legitimate gait dynamics.
        # The opposite foot must remain physically loaded, so two-foot
        # jumping still cannot earn clearance progress.
        opposite_loaded = loaded.flip(dims=(1,))

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

        # --------------------------------------------------------
        # Optional 045 body-gated ascent.
        #
        # Historical 044 behavior remains unchanged in reward_mode 1.
        #
        # In reward_mode 2 the foot cannot unlock the stair state
        # machine by reaching forward while the pelvis is still far
        # from the riser.  The root/pelvis must first enter the final
        # approach corridor.
        # --------------------------------------------------------
        distance_to_riser = (
            c.first_riser_m
            - root[:, 0]
        )

        ascent_active = (
            self.ready
            & safe
            & (
                distance_to_riser
                <= c.ascent_launch_distance_m
            )
            & (
                distance_to_riser
                > -c.geometry_tolerance_m
            )
        )

        if c.reward_mode in (4, 5, 6):
            # 048/049 launch requires actual task preparation, not merely X
            # proximity: centered body and feet, correct heading, controlled
            # approach speed, and at least one loaded foot.
            ascent_active &= (
                root_centered
                & feet_centered
                & yaw_aligned
                & launch_speed_ok
                & loaded.any(dim=1)
            )

        ascent_start_event = torch.zeros_like(
            ascent_active
        )

        if c.reward_mode in (3, 4, 5, 6):
            # 047/048/049: entering the launch corridor is a one-way episode phase
            # transition. It remains active after the pelvis crosses the riser
            # so the trailing foot is never locked out.
            ascent_start_event = (
                ascent_active
                & ~self.ascent_started
            )
            self.ascent_started |= ascent_active

        swing_eligible = (
            self.ready[:, None]
            & safe[:, None]
            & opposite_support
            & ~loaded
            & in_lift_zone
        )

        if c.reward_mode == 2:
            swing_eligible = (
                swing_eligible
                & ascent_active[:, None]
            )
        elif c.reward_mode in (3, 4, 5, 6):
            # Once ascent is latched, either foot may become the swing foot.
            # Do not retain the lower x-bound of the first-foot lift zone: the
            # trailing foot can legitimately still be behind it after body
            # transfer. The riser plane remains a hard pre-crossing bound.
            swing_eligible = (
                self.ready[:, None]
                & safe[:, None]
                & opposite_support
                & ~loaded
                & ~contact
                & self.ascent_started[:, None]
                & (x_max < c.first_riser_m)
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

        if c.reward_mode == 6:
            swing_progress_eligible = (
                self.ready[:, None]
                & safe[:, None]
                & opposite_loaded
                & ~loaded
                & ~contact
                & self.ascent_started[:, None]
                & (x_max < c.first_riser_m)
            )
        else:
            swing_progress_eligible = swing_eligible

        meaningful_lift_event = (
            (previous_stage == SWING)
            & swing_progress_eligible
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

        if c.reward_mode in (3, 4, 5, 6):
            riser_ready_event &= ~contact

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
        if c.reward_mode in (3, 4, 5, 6):
            # Any premature/invalid plane crossing is a failure of the ordered
            # pre-crossing contract, not only a geometrically low crossing.
            valid_crossing_geometry = (
                (previous_stage == RISER_READY)
                & full_clearance
                & good_normal
                & opposite_support
                & ~contact
                & safe[:, None]
            )
            low_crossing_raw = (
                crossing_onset
                & ~valid_crossing_geometry
                & (self.foot_stage < CROSSED)
                & safe[:, None]
            )
        else:
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

        if c.reward_mode in (3, 4, 5, 6):
            valid_crossing_event &= ~contact

        self.foot_stage = torch.where(
            valid_crossing_event,
            torch.full_like(
                self.foot_stage,
                CROSSED,
            ),
            self.foot_stage,
        )

        # Lift-then-drop exploit prevention. Historical modes only
        # penalized a drop after RISER_READY. Mode 3 starts protection as soon
        # as a LIFTED/RISER_READY foot has actually held full clearance.
        if c.reward_mode in (3, 4, 5, 6):
            clearance_loss_raw = (
                (previous_stage >= LIFTED)
                & (previous_stage <= RISER_READY)
                & self.previous_full_clearance
                & ~full_clearance
                & ~valid_crossing_event
            )
        else:
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

        if c.reward_mode == 1:
            # Historical 044C behavior.
            lift_advance_value = (
                height_fraction
                * forward_fraction
            )

            lift_advance_gate = (
                (x_max < c.first_riser_m)
                & safe[:, None]
            )

            lift_advance_value = torch.where(
                lift_advance_gate,
                lift_advance_value,
                0.0,
            )

        elif c.reward_mode == 2:
            # ----------------------------------------------------
            # 045 body-gated vertical-clearance shaping.
            # ----------------------------------------------------
            clearance_potential = (
                height_fraction.square()
            )

            clearance_gate = (
                ascent_active[:, None]
                & opposite_support
                & ~loaded
                & (
                    x_max
                    < c.first_riser_m
                )
            )

            lift_advance_value = torch.where(
                clearance_gate,
                clearance_potential,
                0.0,
            )
        elif c.reward_mode == 6:
            # ----------------------------------------------------
            # 050 locomotion-preserving clearance acquisition.
            #
            # Capture the height baseline at the exact LOWER->SWING
            # transition, not at body-level ascent start. This makes the
            # shaping phase-correct for whichever foot becomes the swing foot
            # and avoids penalizing a foot that happened to be mid-gait when
            # the pelvis entered the launch corridor. Each foot can enter
            # SWING only once per episode, so this baseline cannot be reset to
            # farm the bounded potential.
            # ----------------------------------------------------
            self.clearance_baseline_fraction = torch.where(
                enter_swing,
                height_fraction,
                self.clearance_baseline_fraction,
            )

            clearance_potential = (
                height_fraction
                - self.clearance_baseline_fraction
            ).clamp(0.0, 1.0)

            clearance_stage = (
                (self.foot_stage >= SWING)
                & (self.foot_stage <= RISER_READY)
            )

            clearance_gate = (
                self.ascent_started[:, None]
                & safe[:, None]
                & opposite_loaded
                & ~loaded
                & ~contact
                & clearance_stage
                & (x_max < c.first_riser_m)
            )

            lift_advance_value = torch.where(
                clearance_gate,
                clearance_potential,
                0.0,
            )

        elif c.reward_mode == 5:
            # ----------------------------------------------------
            # 049 clearance acquisition.
            #
            # Capture the foot-height state at the exact launch transition and
            # reward only NEW vertical clearance acquired afterwards. This
            # prevents a pre-launch high kick from being back-credited. The
            # potential is linear rather than squared so small useful vertical
            # improvements are not numerically suppressed.
            # ----------------------------------------------------
            self.clearance_baseline_fraction = torch.where(
                ascent_start_event[:, None],
                height_fraction,
                self.clearance_baseline_fraction,
            )

            clearance_potential = (
                height_fraction
                - self.clearance_baseline_fraction
            ).clamp(0.0, 1.0)

            clearance_stage = (
                (self.foot_stage >= SWING)
                & (self.foot_stage <= RISER_READY)
            )

            clearance_gate = (
                self.ascent_started[:, None]
                & safe[:, None]
                & opposite_support
                & ~loaded
                & ~contact
                & clearance_stage
                & (x_max < c.first_riser_m)
            )

            lift_advance_value = torch.where(
                clearance_gate,
                clearance_potential,
                0.0,
            )
        else:
            # ----------------------------------------------------
            # 047/048 bounded lift discovery.
            # ----------------------------------------------------
            clearance_potential = (
                height_fraction.square()
            )

            clearance_gate = (
                self.ascent_started[:, None]
                & safe[:, None]
                & opposite_support
                & ~loaded
                & ~contact
                & (x_max < c.first_riser_m)
            )

            lift_advance_value = torch.where(
                clearance_gate,
                clearance_potential,
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

        # 049 explicit clearance-acquisition milestone. It is diagnostic
        # only; reward still comes from the bounded lift potential above.
        if c.reward_mode in (5, 6):
            clearance_support = (
                opposite_loaded
                if c.reward_mode == 6
                else opposite_support
            )
            clearance_acquired_raw = (
                self.ascent_started[:, None]
                & safe[:, None]
                & clearance_support
                & ~loaded
                & ~contact
                & full_clearance
                & (self.foot_stage >= LIFTED)
                & (self.foot_stage <= RISER_READY)
                & (x_max < c.first_riser_m)
            )
            clearance_acquired_event = (
                clearance_acquired_raw
                & ~self.clearance_acquired_seen
            )
            self.clearance_acquired_seen |= clearance_acquired_event
        else:
            clearance_acquired_event = torch.zeros_like(full_clearance)

        # --------------------------------------------------------
        # 047/048/049 maintained-clearance forward advance.
        #
        # Forward credit is available only on consecutive frames of full
        # clearance. The first high-clearance frame merely establishes the
        # frontier and pays nothing. If the foot advances while low and then
        # re-lifts, the frontier jumps forward without back-credit. Therefore
        # lift/drop/re-lift and backward/forward oscillations cannot farm this
        # potential. Total credit is bounded to 1 per foot per episode.
        # --------------------------------------------------------
        maintained_clearance_advance = torch.zeros_like(
            sole_bottom
        )
        qualified_clearance_now = torch.zeros_like(
            full_clearance
        )

        if c.reward_mode in (3, 4, 5, 6):
            pre_cross_stage = (
                (previous_stage >= LIFTED)
                & (previous_stage <= RISER_READY)
            ) | meaningful_lift_event

            advance_support = (
                opposite_loaded
                if c.reward_mode == 6
                else opposite_support
            )
            qualified_now = (
                self.ascent_started[:, None]
                & safe[:, None]
                & advance_support
                & ~loaded
                & ~contact
                & pre_cross_stage
                & full_clearance
                & good_normal
                & landing_rectangle
            )
            qualified_clearance_now = qualified_now

            # Never shape beyond the physical riser plane.
            riser_x = torch.full_like(
                x_max,
                c.first_riser_m,
            )
            qualified_x = torch.minimum(
                x_max,
                riser_x,
            )

            rising_clearance = (
                qualified_now
                & ~self.previous_full_clearance
            )

            # Establish/re-establish the frontier without paying for distance
            # covered while clearance was absent.
            self.qualified_x_best = torch.where(
                rising_clearance,
                torch.maximum(
                    self.qualified_x_best,
                    qualified_x,
                ),
                self.qualified_x_best,
            )

            continuous_clearance = (
                qualified_now
                & self.previous_full_clearance
            )

            delta = (
                qualified_x
                - self.qualified_x_best
            ).clamp_min(0.0)

            normalized_delta = (
                delta
                / c.lift_zone_distance_m
            )

            remaining = (
                1.0
                - self.maintained_advance_total
            ).clamp_min(0.0)

            maintained_clearance_advance = torch.where(
                continuous_clearance,
                torch.minimum(
                    normalized_delta,
                    remaining,
                ),
                0.0,
            )

            self.maintained_advance_total += (
                maintained_clearance_advance
            )

            self.qualified_x_best = torch.where(
                qualified_now,
                torch.maximum(
                    self.qualified_x_best,
                    qualified_x,
                ),
                self.qualified_x_best,
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

        if c.reward_mode in (3, 4, 5, 6):
            insertion_gate &= landing_rectangle

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

        if c.reward_mode in (3, 4, 5, 6):
            # Invalid geometry/normal is still a bad touchdown even when the
            # contact speed is small. The +1 makes that event non-zero.
            bad_touchdown_penalty = (
                1.0 + bad_touchdown_value
            )
        else:
            bad_touchdown_penalty = (
                bad_touchdown_value
            )

        bad_touchdown = torch.where(
            touchdown_candidate
            & ~good_touchdown_event,
            bad_touchdown_penalty,
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

        approach_gate = (
            self.ready
            & safe
            & stable_lower.any(dim=1)
        )
        if c.reward_mode in (5, 6):
            # After launch, only clearance acquisition and ordered stair
            # milestones may provide positive task credit. Root forward motion
            # itself is not a stair objective.
            approach_gate &= ~self.ascent_started

        approach_value *= approach_gate.to(dtype)

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

        body_rise_gate = (
            self.ready
            & safe
            & stable_upper.any(dim=1)
        )

        if c.reward_mode in (3, 4, 5, 6):
            # Physical upper contact alone is not enough. The SAME currently
            # stable foot must already have passed crossing -> touchdown ->
            # UPPER_SUPPORT in the ordered state machine.
            body_rise_gate &= (
                self.seen_upper
                & stable_upper
            ).any(dim=1)

        supported_body_rise_value *= (
            body_rise_gate.to(dtype)
        )

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

        if c.reward_mode in (3, 4, 5, 6):
            body_transfer_condition &= (
                self.seen_upper
                & stable_upper
            ).any(dim=1)

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

        if c.reward_mode in (3, 4, 5, 6):
            both_feet_condition &= (
                stable_upper.all(dim=1)
            )

        both_feet_supported_event = (
            both_feet_condition
            & ~self.both_feet_seen
        )

        self.both_feet_seen |= (
            both_feet_supported_event
        )

        if c.reward_mode in (4, 5, 6):
            # 048/049 success is CLIMB + STABILIZE, not "keep moving in +X".
            # Both feet must be genuinely stable on the upper surface, the
            # transferred body must remain elevated/aligned, and the whole body
            # must stay quiet continuously for the terminal dwell.
            terminal_stable = (
                self.both_feet_seen
                & self.body_transfer_seen
                & upper_geometry.all(dim=1)
                & stable_upper.all(dim=1)
                & safe
                & root_centered
                & feet_centered
                & yaw_aligned
                & terminal_motion_ok
                & (
                    -gravity[:, 2]
                    >= c.terminal_upright_cos
                )
                & (
                    root[:, 2]
                    >= (
                        self.base_reference
                        + c.height_m
                        * c.terminal_base_height_gain_fraction
                    )
                )
            )

            self.terminal_stability_age = torch.where(
                terminal_stable,
                self.terminal_stability_age + dt,
                0.0,
            )

            complete_condition = (
                terminal_stable
                & (
                    self.terminal_stability_age
                    >= (
                        c.terminal_stability_dwell_s
                        - c.geometry_tolerance_m
                    )
                )
            )
        else:
            terminal_stable = torch.zeros_like(safe)
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

        # The completion latch is updated only after simultaneous failures
        # have been masked below.

        # --------------------------------------------------------
        # Failure-aware rate terms.
        # --------------------------------------------------------
        if c.reward_mode in (3, 4, 5, 6):
            # 047 defines "premature" by the body-level approach/ascent phase,
            # not by an individual foot's x-position. After launch, the
            # trailing foot may legitimately still be far behind the riser.
            premature_zone = (
                ~self.ascent_started[:, None]
            ).expand_as(x_max)
        else:
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

        # 048/049: alignment/approach cannot be farmed by camping indefinitely.
        # From the first safe calibration frame, the policy has a bounded time
        # to reach the valid launch contract. This is deliberately a phase
        # deadline, not a positive forward-speed reward.
        if c.reward_mode in (4, 5, 6):
            self.prelaunch_age = torch.where(
                self.ready & ~self.ascent_started,
                self.prelaunch_age + dt,
                torch.zeros_like(self.prelaunch_age),
            )
            failure_prelaunch_timeout = (
                self.ready
                & ~self.ascent_started
                & (self.prelaunch_age >= c.prelaunch_timeout_s)
            )
        else:
            failure_prelaunch_timeout = torch.zeros_like(safe)

        # 048/049: once ascent has begun, refusing to make physical progress is not
        # a viable timeout strategy. Any ordered milestone or bounded physical
        # progress rearms the timer; otherwise a prolonged stall terminates.
        if c.reward_mode in (4, 5, 6):
            progress_event = (
                ascent_start_event
                | enter_swing.any(dim=1)
                | meaningful_lift_event.any(dim=1)
                | clearance_acquired_event.any(dim=1)
                | (maintained_clearance_advance > c.geometry_tolerance_m).any(dim=1)
                | riser_ready_event.any(dim=1)
                | valid_crossing_event.any(dim=1)
                | (tread_insertion > c.geometry_tolerance_m).any(dim=1)
                | good_touchdown_event.any(dim=1)
                | stable_upper_event.any(dim=1)
                | (supported_body_rise > c.geometry_tolerance_m)
                | body_transfer_event
                | both_feet_supported_event
            )

            self.ascent_progress_age = torch.where(
                ~self.ascent_started,
                0.0,
                torch.where(
                    progress_event,
                    0.0,
                    self.ascent_progress_age + dt,
                ),
            )

            failure_stall = (
                self.ascent_started
                & (
                    self.ascent_progress_age
                    >= c.ascent_stall_timeout_s
                )
                & ~complete_condition
            )
        else:
            progress_event = torch.zeros_like(safe)
            failure_stall = torch.zeros_like(safe)

        failure_forbidden = forbidden
        failure_tilt = (
            -gravity[:, 2]
            < 0.70
        )
        failure_low_base = (
            root[:, 2]
            < c.minimum_base_height_m
        )
        failure_low_crossing = (
            low_crossing_event.any(dim=1)
            if c.reward_mode in (5, 6)
            else torch.zeros_like(safe)
        )

        failure = (
            failure_forbidden
            | failure_tilt
            | failure_low_base
            | failure_prelaunch_timeout
            | failure_stall
            | failure_low_crossing
        )

        complete_event &= ~failure
        self.complete_seen |= complete_event

        if c.reward_mode in (4, 5, 6):
            positive_step_credit = (
                approach
                * self.reward_weights["step044_approach"]
                + lift_advance.sum(dim=1)
                * self.reward_weights["step044_lift_advance"]
                + maintained_clearance_advance.sum(dim=1)
                * self.reward_weights["step044_maintained_clearance_advance"]
                + meaningful_lift_event.to(dtype).sum(dim=1)
                * self.reward_weights["step044_meaningful_lift"]
                + riser_ready_event.to(dtype).sum(dim=1)
                * self.reward_weights["step044_riser_ready"]
                + valid_crossing_event.to(dtype).sum(dim=1)
                * self.reward_weights["step044_valid_crossing"]
                + tread_insertion.sum(dim=1)
                * self.reward_weights["step044_tread_insertion"]
                + good_touchdown_event.to(dtype).sum(dim=1)
                * self.reward_weights["step044_good_touchdown"]
                + stable_upper_event.to(dtype).sum(dim=1)
                * self.reward_weights["step044_stable_upper_support"]
                + supported_body_rise
                * self.reward_weights["step044_supported_body_rise"]
                + body_transfer_event.to(dtype)
                * self.reward_weights["step044_body_transfer"]
                + both_feet_supported_event.to(dtype)
                * self.reward_weights["step044_both_feet_supported"]
                + complete_event.to(dtype)
                * self.reward_weights["step044_complete"]
            )

            self.positive_stair_credit += positive_step_credit

            failure_weight = abs(
                self.reward_weights["step044_failure"]
            )
            failure_component = torch.where(
                failure,
                1.0
                + c.failure_clawback_scale
                * self.positive_stair_credit
                / failure_weight,
                0.0,
            )
        else:
            positive_step_credit = torch.zeros_like(root[:, 0])
            failure_component = failure.to(dtype)

        # Advance history only after all transition tests.
        self.previous_x_max = x_max.clone()
        self.previous_velocity = (
            foot_velocity.clone()
        )
        self.previous_contact = (
            contact.clone()
        )
        self.previous_full_clearance = (
            qualified_clearance_now.clone()
            if c.reward_mode in (3, 4, 5, 6)
            else full_clearance.clone()
        )
        self.has_previous[:] = True

        components = {
            "approach": approach,
            "lift_advance": (
                lift_advance.sum(dim=1)
            ),
            "maintained_clearance_advance": (
                maintained_clearance_advance
                .sum(dim=1)
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
            "failure": failure_component,
        }

        diagnostics = {
            "foot_stage":
                self.foot_stage.clone(),
            "distance_to_riser":
                distance_to_riser.clone(),
            "ascent_started":
                self.ascent_started.clone(),
            "ascent_start_event":
                ascent_start_event.clone(),
            "full_clearance":
                full_clearance.clone(),
            "maintained_clearance_advance":
                maintained_clearance_advance.clone(),
            "maintained_advance_total":
                self.maintained_advance_total.clone(),
            "qualified_x_best":
                self.qualified_x_best.clone(),
            "seen_upper":
                self.seen_upper.clone(),
            "body_transfer_seen":
                self.body_transfer_seen.clone(),
            "both_feet_seen":
                self.both_feet_seen.clone(),
            "complete_seen":
                self.complete_seen.clone(),
            "low_crossing_seen":
                self.low_crossing_seen.clone(),
            "clearance_loss_seen":
                self.clearance_loss_seen.clone(),
            "height_fraction":
                height_fraction.clone(),
            "forward_fraction":
                forward_fraction.clone(),
            "lift_advance_value":
                lift_advance_value.clone(),
            "clearance_baseline_fraction":
                self.clearance_baseline_fraction.clone(),
            "clearance_acquired_seen":
                self.clearance_acquired_seen.clone(),
            "clearance_acquired_event":
                clearance_acquired_event.clone(),
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
            "root_centered":
                root_centered.clone(),
            "feet_centered":
                feet_centered.clone(),
            "yaw_aligned":
                yaw_aligned.clone(),
            "launch_speed_ok":
                launch_speed_ok.clone(),
            "terminal_motion_ok":
                terminal_motion_ok.clone(),
            "terminal_stable":
                terminal_stable.clone(),
            "terminal_stability_age_s":
                self.terminal_stability_age.clone(),
            "prelaunch_age_s":
                self.prelaunch_age.clone(),
            "failure_forbidden":
                failure_forbidden.clone(),
            "failure_forbidden_pelvis":
                forbidden_pelvis.clone(),
            "failure_forbidden_torso":
                forbidden_torso.clone(),
            "failure_forbidden_knee":
                forbidden_knee.clone(),
            "failure_forbidden_elbow":
                forbidden_elbow.clone(),
            "failure_forbidden_hand":
                forbidden_hand.clone(),
            "failure_low_crossing":
                failure_low_crossing.clone(),
            "failure_tilt":
                failure_tilt.clone(),
            "failure_low_base":
                failure_low_base.clone(),
            "failure_prelaunch_timeout":
                failure_prelaunch_timeout.clone(),
            "failure_stall":
                failure_stall.clone(),
            "progress_event":
                progress_event.clone(),
            "ascent_progress_age_s":
                self.ascent_progress_age.clone(),
            "positive_stair_credit":
                self.positive_stair_credit.clone(),
            "positive_step_credit":
                positive_step_credit.clone(),
            "failure_penalty_value":
                failure_component.clone(),
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


def snapshot(env, config, reward_weights=None):
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
            reward_weights=reward_weights,
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

    forbidden_contacts = (
        torch.linalg.vector_norm(
            all_forces[
                :,
                runtime.forbidden_ids,
            ],
            dim=2,
        )
        .gt(c.forbidden_force_n)
    )
    forbidden = forbidden_contacts.any(dim=1)

    root = (
        data.root_pos_w
        - env.scene.env_origins
    )

    gravity = (
        data.projected_gravity_b
    )

    root_linear_velocity = (
        data.root_lin_vel_w
    )
    root_angular_velocity = (
        data.root_ang_vel_w
    )

    qw, qx, qy, qz = (
        data.root_quat_w.unbind(dim=1)
    )
    root_yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    root_yaw = torch.atan2(
        torch.sin(root_yaw),
        torch.cos(root_yaw),
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

    if c.reward_mode in (3, 4, 5, 6):
        # In 047 the landing rectangle is genuinely 2-D: forward tread/end
        # containment plus lateral stair-width containment.
        landing_rectangle &= lateral_contained

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
        forbidden_contacts=forbidden_contacts,
        root_linear_velocity=root_linear_velocity,
        root_angular_velocity=root_angular_velocity,
        root_yaw=root_yaw,
        foot_lateral_max=y_max,
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
            "root_linear_velocity":
                root_linear_velocity.clone(),
            "root_angular_velocity":
                root_angular_velocity.clone(),
            "root_yaw":
                root_yaw.clone(),
            "foot_lateral_max":
                y_max.clone(),
            "forbidden":
                forbidden.clone(),
            "forbidden_contacts":
                forbidden_contacts.clone(),
        }
    )

    measurements = {
        "ascent_start_count":
            diagnostics[
                "ascent_start_event"
            ],

        "maintained_advance_count":
            (
                diagnostics[
                    "maintained_clearance_advance"
                ]
                > 0
            ).any(dim=1),

        "clearance_acquired_count":
            diagnostics[
                "clearance_acquired_event"
            ].any(dim=1),

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

        "failure_forbidden_count":
            diagnostics["failure_forbidden"],

        "failure_forbidden_pelvis_count":
            diagnostics["failure_forbidden_pelvis"],

        "failure_forbidden_torso_count":
            diagnostics["failure_forbidden_torso"],

        "failure_forbidden_knee_count":
            diagnostics["failure_forbidden_knee"],

        "failure_forbidden_elbow_count":
            diagnostics["failure_forbidden_elbow"],

        "failure_forbidden_hand_count":
            diagnostics["failure_forbidden_hand"],

        "failure_low_crossing_count":
            diagnostics["failure_low_crossing"],

        "failure_tilt_count":
            diagnostics["failure_tilt"],

        "failure_low_base_count":
            diagnostics["failure_low_base"],

        "failure_prelaunch_timeout_count":
            diagnostics["failure_prelaunch_timeout"],

        "failure_stall_count":
            diagnostics["failure_stall"],

        "aligned_launch_env_steps": (
            diagnostics["root_centered"]
            & diagnostics["feet_centered"]
            & diagnostics["yaw_aligned"]
            & diagnostics["launch_speed_ok"]
        ),

        "terminal_stable_env_steps":
            diagnostics["terminal_stable"],

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
    reward_weights,
    component,
):
    value = snapshot(
        env,
        config,
        reward_weights=reward_weights,
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


def failure(env, config, reward_weights):
    return snapshot(
        env,
        config,
        reward_weights=reward_weights,
    )["failure"]


def success(env, config, reward_weights):
    return snapshot(
        env,
        config,
        reward_weights=reward_weights,
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
