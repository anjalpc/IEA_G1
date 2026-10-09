"""Pure reward math helpers with no Isaac runtime dependency."""

from __future__ import annotations

import torch


def exp_tracking_score(
    squared_error: torch.Tensor,
    variance: float,
    *,
    zero_centered: bool,
) -> torch.Tensor:
    """Return exponential tracking score, optionally with zero maximum."""

    if variance <= 0.0:
        raise ValueError("tracking variance must be positive")
    if not torch.isfinite(squared_error).all():
        raise RuntimeError("tracking error must be finite")

    score = torch.exp(
        -squared_error / float(variance)
    )
    return score - 1.0 if zero_centered else score


def finish_transition_gate(
    command_x_mps: torch.Tensor,
    *,
    cruise_speed_mps: float,
) -> torch.Tensor:
    """Smoothly activate finish shaping as the forward command slows."""

    if cruise_speed_mps <= 0.0:
        raise ValueError("cruise_speed_mps must be positive")
    return (
        (float(cruise_speed_mps) - command_x_mps)
        / float(cruise_speed_mps)
    ).clamp(0.0, 1.0)


def phase_gated_posture_scale(
    platform_progress: torch.Tensor,
    command_x_mps: torch.Tensor,
    *,
    cruise_speed_mps: float,
    weak_scale: float = 0.25,
    platform_gain: float = 0.75,
    finish_gain: float = 1.0,
    maximum_scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound posture pressure by platform progress and the finish command.

    Returns ``(phase_scale, finish_gate)``.  The finish gate is explicitly
    multiplied by platform progress so a low command away from the platform
    cannot activate finish posture pressure.
    """
    if weak_scale <= 0.0:
        raise ValueError("weak_scale must be positive")
    if platform_gain < 0.0:
        raise ValueError("platform_gain must be non-negative")
    if finish_gain < 0.0:
        raise ValueError("finish_gain must be non-negative")
    if maximum_scale < weak_scale:
        raise ValueError("maximum_scale must be >= weak_scale")

    progress = platform_progress.clamp(0.0, 1.0)
    raw_finish_gate = finish_transition_gate(
        command_x_mps,
        cruise_speed_mps=cruise_speed_mps,
    )
    finish_gate = progress * raw_finish_gate

    phase_scale = (
        float(weak_scale)
        + float(platform_gain) * progress
        + float(finish_gain) * finish_gate
    ).clamp(max=float(maximum_scale))

    return phase_scale, finish_gate


def normalized_planar_velocity_error(
    velocity_xy_mps: torch.Tensor,
    target_x_mps: torch.Tensor,
    *,
    scale_mps: float,
    maximum_severity: float = 4.0,
) -> torch.Tensor:
    """Bounded planar tracking-error severity for dense finish braking credit."""

    if velocity_xy_mps.ndim != 2 or velocity_xy_mps.shape[1] != 2:
        raise ValueError("velocity_xy_mps must have shape [N, 2]")
    if target_x_mps.ndim != 1 or target_x_mps.shape[0] != velocity_xy_mps.shape[0]:
        raise ValueError("target_x_mps must have shape [N]")
    if scale_mps <= 0.0:
        raise ValueError("scale_mps must be positive")
    if maximum_severity <= 0.0:
        raise ValueError("maximum_severity must be positive")
    squared_error = (
        torch.square(velocity_xy_mps[:, 0] - target_x_mps)
        + torch.square(velocity_xy_mps[:, 1])
    )
    return (
        squared_error / float(scale_mps * scale_mps)
    ).clamp(0.0, float(maximum_severity))


def normalized_excess_forward_velocity(
    forward_velocity_mps: torch.Tensor,
    *,
    free_velocity_mps: float,
    full_penalty_velocity_mps: float,
) -> torch.Tensor:
    """Return a bounded severity for excessive positive forward velocity."""

    if free_velocity_mps < 0.0:
        raise ValueError("free_velocity_mps cannot be negative")
    if full_penalty_velocity_mps <= free_velocity_mps:
        raise ValueError(
            "full_penalty_velocity_mps must exceed free_velocity_mps"
        )
    return (
        (forward_velocity_mps.clamp_min(0.0) - float(free_velocity_mps))
        / float(full_penalty_velocity_mps - free_velocity_mps)
    ).clamp(0.0, 1.0)


def tread_containment_score(
    rear_x_m: torch.Tensor,
    front_x_m: torch.Tensor,
    *,
    riser_x_m: float,
    tread_depth_m: float,
    margin_m: float,
) -> torch.Tensor:
    """Smooth [0, 1] score for keeping the complete foot inside a tread."""

    if tread_depth_m <= 0.0:
        raise ValueError("tread_depth_m must be positive")
    if margin_m <= 0.0 or 2.0 * margin_m >= tread_depth_m:
        raise ValueError("margin_m must be positive and smaller than half the tread")

    rear_score = (
        (rear_x_m - float(riser_x_m)) / float(margin_m)
    ).clamp(0.0, 1.0)
    front_score = (
        (
            float(riser_x_m + tread_depth_m)
            - front_x_m
        )
        / float(margin_m)
    ).clamp(0.0, 1.0)
    return rear_score * front_score


def support_quality(
    *,
    containment: torch.Tensor,
    height_score: torch.Tensor,
    vertical_force_n: torch.Tensor,
    speed_mps: torch.Tensor,
    full_support_force_n: float | torch.Tensor,
    maximum_supported_speed_mps: float,
) -> torch.Tensor:
    """Bounded upper-foot support quality from geometry, load, and slip speed."""

    if isinstance(full_support_force_n, torch.Tensor):
        if bool((full_support_force_n <= 0.0).any()):
            raise ValueError("full_support_force_n must be positive")
        force_scale = full_support_force_n
    else:
        if full_support_force_n <= 0.0:
            raise ValueError("full_support_force_n must be positive")
        force_scale = float(full_support_force_n)
    if maximum_supported_speed_mps <= 0.0:
        raise ValueError("maximum_supported_speed_mps must be positive")

    force_score = (
        vertical_force_n / force_scale
    ).clamp(0.0, 1.0)
    speed_score = (
        1.0 - speed_mps / float(maximum_supported_speed_mps)
    ).clamp(0.0, 1.0)
    return (
        containment.clamp(0.0, 1.0)
        * height_score.clamp(0.0, 1.0)
        * force_score
        * speed_score
    )





def bottleneck_score(*scores: torch.Tensor) -> torch.Tensor:
    """Return the minimum bounded component score.

    This is intentionally harsher than an arithmetic mean: one broken
    prerequisite cannot be compensated by unrelated good components.
    """

    if not scores:
        raise ValueError("bottleneck_score requires at least one score")
    shape = scores[0].shape
    if any(score.shape != shape for score in scores[1:]):
        raise ValueError("all bottleneck scores must have the same shape")
    return torch.stack(tuple(score.clamp(0.0, 1.0) for score in scores), dim=0).amin(dim=0)


def deadzone_score(
    absolute_error: torch.Tensor,
    *,
    free_tolerance: float,
    zero_at: float,
) -> torch.Tensor:
    """Return 1 inside a dead-zone and linearly fall to 0 at ``zero_at``."""

    if free_tolerance < 0.0:
        raise ValueError("free_tolerance cannot be negative")
    if zero_at <= free_tolerance:
        raise ValueError("zero_at must be greater than free_tolerance")
    return (
        1.0
        - (absolute_error - float(free_tolerance))
        / float(zero_at - free_tolerance)
    ).clamp(0.0, 1.0)


def target_band_score(
    value: torch.Tensor,
    *,
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
    falloff: float,
) -> torch.Tensor:
    """Return 1 inside [lower, upper] with linear falloff outside the band."""

    if falloff <= 0.0:
        raise ValueError("falloff must be positive")
    lower_t = torch.as_tensor(lower, dtype=value.dtype, device=value.device)
    upper_t = torch.as_tensor(upper, dtype=value.dtype, device=value.device)
    if bool((upper_t < lower_t).any()):
        raise ValueError("upper must be >= lower")

    below = (value - (lower_t - float(falloff))) / float(falloff)
    above = ((upper_t + float(falloff)) - value) / float(falloff)
    return torch.minimum(below, above).clamp(0.0, 1.0)

def supported_single_foot_lift_score(
    *,
    sole_bottom_m: torch.Tensor,
    vertical_force_n: torch.Tensor,
    foot_speed_mps: torch.Tensor,
    bodyweight_n: torch.Tensor,
    riser_height_m: torch.Tensor,
    lower_support_height_tolerance_m: float,
    lower_support_force_bodyweight_fraction: float,
    maximum_lower_support_foot_speed_mps: float,
) -> torch.Tensor:
    """Return a bounded single-foot lift score with opposite-foot support.

    ``sole_bottom_m``, ``vertical_force_n`` and ``foot_speed_mps`` must be
    shaped ``[N, 2]`` in left/right foot order.  A candidate swing foot earns
    height progress only while the opposite foot remains loaded, near the
    lower floor, and sufficiently slow.
    """

    if sole_bottom_m.ndim != 2 or sole_bottom_m.shape[1] != 2:
        raise ValueError("sole_bottom_m must have shape [N, 2]")
    if vertical_force_n.shape != sole_bottom_m.shape:
        raise ValueError("vertical_force_n must match sole_bottom_m")
    if foot_speed_mps.shape != sole_bottom_m.shape:
        raise ValueError("foot_speed_mps must match sole_bottom_m")
    if bodyweight_n.ndim != 1 or bodyweight_n.shape[0] != sole_bottom_m.shape[0]:
        raise ValueError("bodyweight_n must have shape [N]")
    if riser_height_m.ndim != 1 or riser_height_m.shape[0] != sole_bottom_m.shape[0]:
        raise ValueError("riser_height_m must have shape [N]")
    if lower_support_height_tolerance_m <= 0.0:
        raise ValueError("lower_support_height_tolerance_m must be positive")
    if not 0.0 < lower_support_force_bodyweight_fraction <= 1.0:
        raise ValueError(
            "lower_support_force_bodyweight_fraction must be in (0, 1]"
        )
    if maximum_lower_support_foot_speed_mps <= 0.0:
        raise ValueError(
            "maximum_lower_support_foot_speed_mps must be positive"
        )

    target_height = riser_height_m.unsqueeze(1).clamp_min(1.0e-4)
    lift_fraction = (sole_bottom_m / target_height).clamp(0.0, 1.0)

    lower_height_score = (
        1.0
        - torch.abs(sole_bottom_m) / float(lower_support_height_tolerance_m)
    ).clamp(0.0, 1.0)
    full_lower_force = (
        bodyweight_n.unsqueeze(1)
        * float(lower_support_force_bodyweight_fraction)
    ).clamp_min(1.0e-6)
    lower_force_score = (
        vertical_force_n / full_lower_force
    ).clamp(0.0, 1.0)
    lower_speed_score = (
        1.0
        - foot_speed_mps / float(maximum_lower_support_foot_speed_mps)
    ).clamp(0.0, 1.0)
    lower_support = (
        lower_height_score * lower_force_score * lower_speed_score
    )

    return torch.stack(
        (
            lift_fraction[:, 0] * lower_support[:, 1],
            lift_fraction[:, 1] * lower_support[:, 0],
        ),
        dim=1,
    ).amax(dim=1)


def coupled_swing_score(
    *,
    sole_bottom_m: torch.Tensor,
    front_x_m: torch.Tensor,
    riser_x_m: float,
    riser_height_m: torch.Tensor,
    activation_distance_m: float,
    insertion_target_m: float,
    clearance_margin_m: float,
) -> torch.Tensor:
    """Couple foot lift and advance so neither earns full credit alone."""

    if activation_distance_m <= 0.0:
        raise ValueError("activation_distance_m must be positive")
    if insertion_target_m <= 0.0:
        raise ValueError("insertion_target_m must be positive")
    if clearance_margin_m < 0.0:
        raise ValueError("clearance_margin_m cannot be negative")

    target_height = (
        riser_height_m + float(clearance_margin_m)
    ).clamp_min(1.0e-4)
    height_fraction = (
        sole_bottom_m / target_height
    ).clamp(0.0, 1.0)

    launch_x = float(riser_x_m) - float(activation_distance_m)
    target_x = float(riser_x_m) + float(insertion_target_m)
    forward_fraction = (
        (front_x_m - launch_x)
        / max(target_x - launch_x, 1.0e-6)
    ).clamp(0.0, 1.0)

    return height_fraction * forward_fraction



def support_relative_forward_progress(
    *,
    root_x_m: torch.Tensor,
    support_anchor_x_m: torch.Tensor,
    activation_root_x_m: torch.Tensor,
    target_forward_from_support_m: float,
) -> torch.Tensor:
    """Return bounded sagittal body progress relative to the planted support foot.

    The target is expressed from the support-foot anchor rather than world X,
    so the term cannot be earned by sliding the support foot forward.  Progress
    begins at the root position recorded when the clearance/hover state first
    becomes physically valid.
    """

    if root_x_m.ndim != 1:
        raise ValueError("root_x_m must have shape [N]")
    if support_anchor_x_m.shape != root_x_m.shape:
        raise ValueError("support_anchor_x_m must match root_x_m")
    if activation_root_x_m.shape != root_x_m.shape:
        raise ValueError("activation_root_x_m must match root_x_m")
    if target_forward_from_support_m <= 0.0:
        raise ValueError("target_forward_from_support_m must be positive")

    nominal_target_x = (
        support_anchor_x_m + float(target_forward_from_support_m)
    )
    # Always require some new motion after activation even if the body is
    # already unusually far forward when the term turns on.
    target_x = torch.maximum(
        nominal_target_x,
        activation_root_x_m + 0.020,
    )
    denominator = (target_x - activation_root_x_m).clamp_min(0.020)
    return (
        (root_x_m - activation_root_x_m) / denominator
    ).clamp(0.0, 1.0)



def support_anchor_slip_penalty(
    *,
    anchor_error_m: torch.Tensor,
    planar_speed_mps: torch.Tensor,
    active_gate: torch.Tensor,
    anchor_free_m: float,
    anchor_zero_m: float,
    speed_free_mps: float,
    speed_zero_mps: float,
    anchor_gate: torch.Tensor | None = None,
    speed_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a bounded penalty severity for a loaded support foot that drifts or slips.

    The anchor term measures accumulated XY displacement from the support-foot
    location captured at reset.  The speed term measures current planar slip.
    The larger violation wins, so either a slow accumulated shuffle or a fast
    transient slip is visible to the policy.  The caller supplies the physical
    single-support gate so ordinary two-foot stance is not penalized.
    """

    if anchor_error_m.shape != planar_speed_mps.shape:
        raise ValueError("anchor_error_m and planar_speed_mps must match")
    if active_gate.shape != anchor_error_m.shape:
        raise ValueError("active_gate must match anchor_error_m")
    if anchor_gate is not None and anchor_gate.shape != anchor_error_m.shape:
        raise ValueError("anchor_gate must match anchor_error_m")
    if speed_gate is not None and speed_gate.shape != anchor_error_m.shape:
        raise ValueError("speed_gate must match anchor_error_m")
    if anchor_free_m < 0.0 or anchor_zero_m <= anchor_free_m:
        raise ValueError("anchor_zero_m must exceed anchor_free_m >= 0")
    if speed_free_mps < 0.0 or speed_zero_mps <= speed_free_mps:
        raise ValueError("speed_zero_mps must exceed speed_free_mps >= 0")

    anchor_bad = 1.0 - deadzone_score(
        anchor_error_m,
        free_tolerance=anchor_free_m,
        zero_at=anchor_zero_m,
    )
    speed_bad = 1.0 - deadzone_score(
        planar_speed_mps,
        free_tolerance=speed_free_mps,
        zero_at=speed_zero_mps,
    )
    legacy_gate = active_gate.to(dtype=torch.bool)
    anchor_gate = legacy_gate if anchor_gate is None else anchor_gate.to(dtype=torch.bool)
    speed_gate = legacy_gate if speed_gate is None else speed_gate.to(dtype=torch.bool)
    anchor_term = torch.where(anchor_gate, anchor_bad, torch.zeros_like(anchor_bad))
    speed_term = torch.where(speed_gate, speed_bad, torch.zeros_like(speed_bad))
    return torch.maximum(anchor_term, speed_term)


def continuous_touchdown_potential(
    *,
    entry_potential: torch.Tensor,
    commitment_score: torch.Tensor,
    upper_support_score: torch.Tensor,
    commitment_full_at: float = 0.75,
    commitment_zero_at: float = 0.25,
) -> torch.Tensor:
    """Continue phase-3 potential through touchdown without a reward cliff.

    A valid phase-3 -> phase-4 transition records the potential that was already
    earned.  While current placement remains committed, upper support can only
    increase that potential toward 1.0.  If placement retracts or the support
    geometry breaks, the commitment factor falls and the potential is removed
    reversibly instead of remaining banked by phase memory.
    """

    if entry_potential.shape != commitment_score.shape:
        raise ValueError("entry_potential and commitment_score must match")
    if upper_support_score.shape != entry_potential.shape:
        raise ValueError("upper_support_score must match entry_potential")
    if not 0.0 <= commitment_zero_at < commitment_full_at <= 1.0:
        raise ValueError(
            "commitment thresholds must satisfy 0 <= zero < full <= 1"
        )

    commitment = (
        (commitment_score - float(commitment_zero_at))
        / float(commitment_full_at - commitment_zero_at)
    ).clamp(0.0, 1.0)
    entry = entry_potential.clamp(0.0, 1.0)
    support = upper_support_score.clamp(0.0, 1.0)
    supported_progress = entry + (1.0 - entry) * support
    return commitment * supported_progress

def swing_phase_potential(
    *,
    phase: torch.Tensor,
    active_score: torch.Tensor,
    phase_count: int,
    reversible_touchdown_credit: bool,
) -> torch.Tensor:
    """Return bounded waypoint potential with optional reversible touchdown credit.

    Historical v04-v06 semantics bank the integer phase index directly.  With
    five phases this means merely entering phase 4 guarantees 4/5 potential,
    even before a physical upper-tread foothold exists.  The opt-in v07 mode
    intentionally holds phase-4 banked credit at the phase-3 baseline; current
    touchdown quality can raise it, and only completion earns the final 1.0.
    """

    if phase.shape != active_score.shape:
        raise ValueError("phase and active_score must have the same shape")
    if phase_count < 2:
        raise ValueError("phase_count must be at least 2")

    phase_credit = phase.to(dtype=active_score.dtype).clamp(
        min=0,
        max=phase_count,
    )
    if bool(reversible_touchdown_credit):
        touchdown_phase = phase_count - 1
        phase_credit = torch.where(
            phase == touchdown_phase,
            torch.full_like(phase_credit, float(touchdown_phase - 1)),
            phase_credit,
        )

    incomplete = phase < phase_count
    score_credit = torch.where(
        incomplete,
        active_score.clamp(0.0, 1.0),
        torch.zeros_like(active_score),
    )
    return (
        (phase_credit + score_credit) / float(phase_count)
    ).clamp(0.0, 1.0)


def body_advance_gate(
    *,
    phase: torch.Tensor,
    live_clearance_gate: torch.Tensor,
    touchdown_placement_gate: torch.Tensor,
    upper_support_gate: torch.Tensor,
    require_live_touchdown_commitment: bool,
    touchdown_phase_index: int = 4,
) -> torch.Tensor:
    """Gate body-advance potential using current physical commitment.

    The historical v06 contract keeps the potential active for every state
    with ``phase >= 4``.  The opt-in v07 contract instead requires either a
    currently valid tread-placement state or genuine upper-foot support after
    phase 4 is reached, so retracting the swing foot removes the potential.
    """

    if touchdown_phase_index < 0:
        raise ValueError("touchdown_phase_index cannot be negative")
    if phase.shape != live_clearance_gate.shape:
        raise ValueError("phase and live_clearance_gate must have the same shape")
    if touchdown_placement_gate.shape != phase.shape:
        raise ValueError("touchdown_placement_gate must match phase shape")
    if upper_support_gate.shape != phase.shape:
        raise ValueError("upper_support_gate must match phase shape")

    touchdown_or_later = phase >= int(touchdown_phase_index)
    if not bool(require_live_touchdown_commitment):
        return live_clearance_gate | touchdown_or_later

    return live_clearance_gate | (
        touchdown_or_later
        & (touchdown_placement_gate | upper_support_gate)
    )

def swing_arc_waypoints(
    *,
    start_x_m: torch.Tensor,
    riser_x_m: float,
    tread_depth_m: float,
    riser_height_m: torch.Tensor,
    early_x_fraction: float,
    clearance_x_fraction: float,
    early_lift_riser_fraction: float,
    clearance_margin_m: float,
    hover_margin_m: float,
) -> dict[str, torch.Tensor]:
    """Return geometry-scaled waypoints for the v04 single-step swing arc."""

    if start_x_m.ndim != 1:
        raise ValueError("start_x_m must have shape [N]")
    if riser_height_m.ndim != 1 or riser_height_m.shape != start_x_m.shape:
        raise ValueError("riser_height_m must match start_x_m shape")
    if tread_depth_m <= 0.0:
        raise ValueError("tread_depth_m must be positive")
    if not 0.0 < early_x_fraction < clearance_x_fraction < 1.0:
        raise ValueError("waypoint x fractions must satisfy 0 < early < clearance < 1")
    if not 0.0 < early_lift_riser_fraction < 1.0:
        raise ValueError("early_lift_riser_fraction must be in (0, 1)")
    if clearance_margin_m <= 0.0 or hover_margin_m <= 0.0:
        raise ValueError("clearance margins must be positive")

    landing_x = torch.full_like(
        start_x_m,
        float(riser_x_m) + 0.5 * float(tread_depth_m),
    )
    path = (landing_x - start_x_m).clamp_min(0.05)
    return {
        "landing_x": landing_x,
        "early_x": start_x_m + float(early_x_fraction) * path,
        "clearance_x": start_x_m + float(clearance_x_fraction) * path,
        "hover_x": landing_x,
        "early_z": riser_height_m * float(early_lift_riser_fraction),
        "clearance_z": riser_height_m + float(clearance_margin_m),
        "hover_z": riser_height_m + float(hover_margin_m),
        "touchdown_z": riser_height_m,
    }


def phase_arm_tolerance(
    entered_stairs: torch.Tensor,
    safe_platform: torch.Tensor,
    command_x_mps: torch.Tensor,
    *,
    approach_tolerance_rad: float,
    ascent_tolerance_rad: float,
    finish_tolerance_rad: float,
    cruise_speed_mps: float,
) -> torch.Tensor:
    """Phase-aware arm tolerance with smooth tightening during final stop."""

    if (
        entered_stairs.ndim != 1
        or safe_platform.ndim != 1
        or command_x_mps.ndim != 1
    ):
        raise ValueError("phase-arm inputs must have shape [N]")
    if not (
        entered_stairs.shape
        == safe_platform.shape
        == command_x_mps.shape
    ):
        raise ValueError("phase-arm inputs must have matching shapes")

    values = (
        approach_tolerance_rad,
        ascent_tolerance_rad,
        finish_tolerance_rad,
    )
    if any(value < 0.0 for value in values):
        raise ValueError("arm tolerances must be non-negative")
    if (
        ascent_tolerance_rad < approach_tolerance_rad
        or ascent_tolerance_rad < finish_tolerance_rad
    ):
        raise ValueError(
            "ascent tolerance must be at least as permissive "
            "as approach and finish"
        )

    tolerance = torch.full_like(
        command_x_mps,
        float(approach_tolerance_rad),
    )

    tolerance = torch.where(
        entered_stairs,
        torch.full_like(tolerance, float(ascent_tolerance_rad)),
        tolerance,
    )

    finish_gate = finish_transition_gate(
        command_x_mps,
        cruise_speed_mps=cruise_speed_mps,
    )
    finish_gate = finish_gate * safe_platform.to(tolerance.dtype)

    finish_value = torch.full_like(
        tolerance,
        float(finish_tolerance_rad),
    )

    return tolerance + finish_gate * (finish_value - tolerance)



def alternating_foot_gate(
    preferred_foot: torch.Tensor,
) -> torch.Tensor:
    """Return [num_envs, 2] reward eligibility for dynamic alternation.

    preferred_foot values: -1 means either foot, 0 left, 1 right.
    """
    if preferred_foot.ndim != 1:
        raise ValueError("preferred_foot must have shape [num_envs]")
    if bool(((preferred_foot < -1) | (preferred_foot > 1)).any()):
        raise ValueError("preferred_foot values must be -1, 0, or 1")

    foot_index = torch.arange(
        2,
        dtype=preferred_foot.dtype,
        device=preferred_foot.device,
    )[None, :]
    return (
        (preferred_foot < 0)[:, None]
        | (foot_index == preferred_foot[:, None])
    )


def update_alternating_preferred_foot(
    preferred_foot: torch.Tensor,
    physical_touchdown: torch.Tensor,
    acquired: torch.Tensor,
) -> torch.Tensor:
    """Prefer the opposite foot after a unique physical acquisition."""
    if preferred_foot.ndim != 1:
        raise ValueError("preferred_foot must have shape [num_envs]")
    if physical_touchdown.shape != (preferred_foot.shape[0], 2):
        raise ValueError(
            "physical_touchdown must have shape [num_envs, 2]"
        )
    if acquired.shape != preferred_foot.shape:
        raise ValueError("acquired must match preferred_foot shape")

    count = physical_touchdown.to(torch.long).sum(dim=1)
    unique = acquired & (count == 1)
    acquirer = physical_touchdown.to(torch.long).argmax(dim=1)
    opposite = 1 - acquirer

    # Simultaneous acquisition stays unconstrained rather than
    # hard-coding a left/right tie break.
    after_acquisition = torch.where(
        unique,
        opposite.to(preferred_foot.dtype),
        torch.full_like(preferred_foot, -1),
    )
    return torch.where(
        acquired,
        after_acquisition,
        preferred_foot,
    )




def anticipatory_clearance_target(
    distance_to_edge_m: torch.Tensor,
    full_clearance_m: torch.Tensor,
    *,
    start_distance_m: float,
    full_distance_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a smooth pre-riser clearance target and eligibility window.

    Horizontal progress never appears in the returned reward itself; it only
    changes the vertical target. The window closes once the sole front crosses
    the riser, so a single oversized forward stride cannot earn anticipatory
    credit merely by reaching/passing the stair.
    """
    if not start_distance_m > full_distance_m >= 0.0:
        raise ValueError(
            "anticipatory distances require start_distance_m > "
            "full_distance_m >= 0"
        )
    if distance_to_edge_m.shape != full_clearance_m.shape:
        raise ValueError(
            "distance_to_edge_m and full_clearance_m must match"
        )

    progress = (
        (float(start_distance_m) - distance_to_edge_m)
        / float(start_distance_m - full_distance_m)
    ).clamp(0.0, 1.0)
    # Smoothstep avoids a slope discontinuity at the start/full boundaries.
    progress = progress * progress * (3.0 - 2.0 * progress)

    eligible = (
        (distance_to_edge_m >= 0.0)
        & (distance_to_edge_m <= float(start_distance_m))
    )
    target = full_clearance_m.clamp_min(0.0) * progress
    return target, eligible


def bounded_vertical_progress_credit(
    current_height_m: torch.Tensor,
    previous_height_m: torch.Tensor,
    target_height_m: torch.Tensor,
    full_clearance_m: torch.Tensor,
    *,
    max_credit_vertical_speed_mps: float,
    dt_s: float,
) -> torch.Tensor:
    """Normalized positive vertical progress with a hard credit-speed cap.

    Moving horizontally cannot create credit because only the measured change
    in vertical height is used. Raising faster than the configured speed does
    not earn more reward, which removes an incentive for abrupt leg throwing.
    """
    if current_height_m.shape != previous_height_m.shape:
        raise ValueError("current_height_m and previous_height_m must match")
    if target_height_m.shape != current_height_m.shape:
        raise ValueError("target_height_m must match height tensors")
    if full_clearance_m.shape != current_height_m.shape:
        raise ValueError("full_clearance_m must match height tensors")
    if max_credit_vertical_speed_mps <= 0.0:
        raise ValueError("max_credit_vertical_speed_mps must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    vertical_delta = (current_height_m - previous_height_m).clamp_min(0.0)
    remaining = (target_height_m - previous_height_m).clamp_min(0.0)
    credited = torch.minimum(vertical_delta, remaining)
    credited = torch.minimum(
        credited,
        torch.full_like(
            credited,
            float(max_credit_vertical_speed_mps) * float(dt_s),
        ),
    )
    denominator = full_clearance_m.clamp_min(1.0e-6)
    return credited / denominator


def bounded_monotonic_vertical_progress_credit(
    current_height_m: torch.Tensor,
    best_height_m: torch.Tensor,
    target_height_m: torch.Tensor,
    full_clearance_m: torch.Tensor,
    *,
    max_credit_vertical_speed_mps: float,
    dt_s: float,
    suppress_credit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward only newly achieved vertical peak height toward the target.

    The physical peak is monotonic for the current stair target. Dropping and
    lifting back to an already reached height therefore earns nothing. The
    peak still advances on abrupt motion so excess speed-capped displacement
    cannot be cashed on later steps.
    """
    shape = current_height_m.shape
    for name, value in (
        ("best_height_m", best_height_m),
        ("target_height_m", target_height_m),
        ("full_clearance_m", full_clearance_m),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must match current_height_m")
    if max_credit_vertical_speed_mps <= 0.0:
        raise ValueError("max_credit_vertical_speed_mps must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if suppress_credit is None:
        suppress_credit = torch.zeros(
            shape, dtype=torch.bool, device=current_height_m.device
        )
    elif suppress_credit.shape != shape or suppress_credit.dtype != torch.bool:
        raise ValueError("suppress_credit must be a boolean tensor matching height")

    previous_peak = best_height_m.clamp_min(0.0)
    current = current_height_m.clamp_min(0.0)
    new_peak = torch.maximum(previous_peak, current)

    peak_delta = (current - previous_peak).clamp_min(0.0)
    remaining = (target_height_m - previous_peak).clamp_min(0.0)
    credited = torch.minimum(peak_delta, remaining)
    credited = torch.minimum(
        credited,
        torch.full_like(
            credited,
            float(max_credit_vertical_speed_mps) * float(dt_s),
        ),
    )
    denominator = full_clearance_m.clamp_min(1.0e-6)
    credit = credited / denominator
    credit = torch.where(suppress_credit, torch.zeros_like(credit), credit)
    return new_peak, credit


def bounded_anticipatory_attempt_credit(
    current_height_m: torch.Tensor,
    best_height_m: torch.Tensor,
    target_height_m: torch.Tensor,
    full_clearance_m: torch.Tensor,
    attempt_active: torch.Tensor,
    attempt_spent: torch.Tensor,
    eligible: torch.Tensor,
    invalidate_attempt: torch.Tensor,
    *,
    max_credit_vertical_speed_mps: float,
    dt_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One bounded anticipatory lift attempt per foot and stair target.

    An attempt starts only on an eligible sample. Its first sample establishes
    the local height baseline and pays nothing. Only new vertical peak height
    reached while the same attempt remains eligible can earn credit. A strike,
    lower-surface replant, or crossing of the riser invalidates the attempt and
    marks it spent until the stair target changes/reset. This prevents ordinary
    approach swings, strike bounce, or repeated lift/replant cycles from
    recycling clearance reward.
    """
    shape = current_height_m.shape
    for name, value in (
        ("best_height_m", best_height_m),
        ("target_height_m", target_height_m),
        ("full_clearance_m", full_clearance_m),
        ("attempt_active", attempt_active),
        ("attempt_spent", attempt_spent),
        ("eligible", eligible),
        ("invalidate_attempt", invalidate_attempt),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must match current_height_m")
    for name, value in (
        ("attempt_active", attempt_active),
        ("attempt_spent", attempt_spent),
        ("eligible", eligible),
        ("invalidate_attempt", invalidate_attempt),
    ):
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")

    # Invalidation has priority: collision/replant cannot simultaneously start
    # or receive credit.
    new_attempt = eligible & ~attempt_active & ~attempt_spent & ~invalidate_attempt
    continuing = attempt_active & eligible & ~invalidate_attempt

    candidate_best, credit = bounded_monotonic_vertical_progress_credit(
        current_height_m,
        best_height_m,
        target_height_m,
        full_clearance_m,
        max_credit_vertical_speed_mps=max_credit_vertical_speed_mps,
        dt_s=dt_s,
    )
    credit = torch.where(continuing, credit, torch.zeros_like(credit))

    best_now = torch.where(continuing, candidate_best, best_height_m)
    # Starting the local attempt establishes a physical baseline but never pays
    # an entry bonus for height achieved during the ordinary approach gait.
    best_now = torch.where(new_attempt, current_height_m.clamp_min(0.0), best_now)

    ended = attempt_active & invalidate_attempt
    spent_now = attempt_spent | ended
    active_now = (attempt_active | new_attempt) & ~invalidate_attempt

    return best_now, active_now, spent_now, credit


def first_touchdown_reward_event(
    new_attempt: torch.Tensor,
    preferred_gate: torch.Tensor,
    reward_seen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Globally one-shot first-contact shaping for each stair target."""
    if new_attempt.shape != preferred_gate.shape or new_attempt.ndim != 2:
        raise ValueError("new_attempt and preferred_gate must match [N, 2]")
    if new_attempt.dtype != torch.bool or preferred_gate.dtype != torch.bool:
        raise ValueError("first-touchdown gates must be boolean")
    if reward_seen.shape != (new_attempt.shape[0],) or reward_seen.dtype != torch.bool:
        raise ValueError("reward_seen must be boolean [N]")

    eligible = new_attempt & preferred_gate
    unique = eligible.sum(dim=1) == 1
    event = eligible & unique[:, None] & ~reward_seen[:, None]
    seen_now = reward_seen | new_attempt.any(dim=1)
    return event, seen_now


def nonpreferred_acquisition_event(
    preferred_foot: torch.Tensor,
    physical_touchdown: torch.Tensor,
) -> torch.Tensor:
    """One event when a stair is acquired without the preferred foot.

    A preferred value of -1 leaves the first acquisition unconstrained. The
    physical acquisition itself is never blocked; this function only reports
    whether reward shaping should treat the acquisition as nonpreferred.
    """
    gate = alternating_foot_gate(preferred_foot)
    if physical_touchdown.shape != gate.shape:
        raise ValueError(
            "physical_touchdown must have shape [num_envs, 2]"
        )
    acquired = physical_touchdown.any(dim=1)
    preferred_acquired = (physical_touchdown & gate).any(dim=1)
    constrained = preferred_foot >= 0
    return constrained & acquired & ~preferred_acquired



def preferred_transition_contact_update(
    armed: torch.Tensor,
    resolved: torch.Tensor,
    clear_lift: torch.Tensor,
    old_tread_contact: torch.Tensor,
    next_tread_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update one preferred-foot transition without prescribing a trajectory.

    A transition becomes eligible only after clear lift. The first loaded
    top-surface contact then resolves as either DIRECT_NEXT or REPLANT_OLD.
    Next-tread contact takes priority if numerical overlap is ambiguous.
    """
    shape = armed.shape
    for name, value in (
        ("resolved", resolved),
        ("clear_lift", clear_lift),
        ("old_tread_contact", old_tread_contact),
        ("next_tread_contact", next_tread_contact),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must match armed shape")

    armed_now = armed | (clear_lift & ~resolved)
    unresolved = armed_now & ~resolved
    direct_next = unresolved & next_tread_contact
    replant_old = unresolved & old_tread_contact & ~next_tread_contact
    resolved_now = resolved | direct_next | replant_old
    return armed_now, resolved_now, replant_old, direct_next


def bounded_impact_severity(
    opposing_force_n: torch.Tensor,
    event: torch.Tensor,
    *,
    minimum_force_n: float,
    full_severity_force_n: float,
    minimum_event_fraction: float = 0.25,
) -> torch.Tensor:
    """Bounded strike-event severity in [0, 1].

    The minimum detected impact still has a small cost, while harder impacts
    scale up to a bounded maximum. Extremely large collision forces therefore
    cannot create unbounded gradients, but are no longer equivalent to mild
    threshold-crossing contacts.
    """
    if not full_severity_force_n > minimum_force_n > 0.0:
        raise ValueError(
            "impact severity requires full_severity_force_n > "
            "minimum_force_n > 0"
        )
    if not 0.0 <= minimum_event_fraction <= 1.0:
        raise ValueError("minimum_event_fraction must be in [0, 1]")
    if opposing_force_n.shape != event.shape:
        raise ValueError("opposing_force_n and event must match")

    magnitude = (-opposing_force_n).clamp_min(0.0)
    scaled = (
        (magnitude - float(minimum_force_n))
        / float(full_severity_force_n - minimum_force_n)
    ).clamp(0.0, 1.0)
    severity = (
        float(minimum_event_fraction)
        + (1.0 - float(minimum_event_fraction)) * scaled
    )
    return torch.where(event, severity, torch.zeros_like(severity))



def step_over_transition_update(
    active: torch.Tensor,
    lifted: torch.Tensor,
    cleared: torch.Tensor,
    next_contact_seen: torch.Tensor,
    resolved: torch.Tensor,
    *,
    clear_lift: torch.Tensor,
    clear_riser: torch.Tensor,
    old_tread_contact: torch.Tensor,
    next_tread_contact: torch.Tensor,
    strike_event: torch.Tensor,
    forward_excursion_violation: torch.Tensor,
    acquisition_event: torch.Tensor,
    preferred_acquisition: torch.Tensor,
    single_acquisition: torch.Tensor,
    strike_severity: torch.Tensor,
    prelift_old_tread_support: torch.Tensor | None = None,
    success_value: float = 1.0,
    failure_value: float = -1.0,
    strike_failure_base: float = -0.5,
    strike_failure_severity_scale: float = -0.5,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Advance an explicit clean step-over transition contract.

    A transition must lift, clear the target riser, make its first loaded
    top-surface contact on the target tread, and then physically acquire that
    tread with the committed foot. Replant, strike, excessive forward swing,
    nonpreferred acquisition, or acquisition without the required sequence
    resolves the transition as a failure. Each transition emits at most one
    signed event reward.
    """
    shape = active.shape

    if prelift_old_tread_support is None:
        prelift_old_tread_support = torch.zeros_like(active)

    tensors = (
        ("lifted", lifted),
        ("cleared", cleared),
        ("next_contact_seen", next_contact_seen),
        ("resolved", resolved),
        ("clear_lift", clear_lift),
        ("clear_riser", clear_riser),
        ("old_tread_contact", old_tread_contact),
        ("next_tread_contact", next_tread_contact),
        ("strike_event", strike_event),
        ("forward_excursion_violation", forward_excursion_violation),
        ("acquisition_event", acquisition_event),
        ("preferred_acquisition", preferred_acquisition),
        ("single_acquisition", single_acquisition),
        ("strike_severity", strike_severity),
        ("prelift_old_tread_support", prelift_old_tread_support),
    )
    for name, value in tensors:
        if value.shape != shape:
            raise ValueError(f"{name} must match active shape")

    eligible = active & ~resolved
    lifted_now = lifted | (eligible & clear_lift)
    cleared_now = cleared | (eligible & lifted_now & clear_riser)

    strike_failure = eligible & strike_event
    excursion_failure = (
        eligible
        & lifted_now
        & forward_excursion_violation
        & ~strike_failure
    )

    # Stair-to-stair only: once the previous stair has been acquired, the
    # committed opposite foot must remain on the lower support until lift.
    # Loading the acquired tread first is the observed step-to exploit.
    consolidation_failure = (
        eligible
        & ~lifted_now
        & prelift_old_tread_support
        & ~strike_failure
        & ~excursion_failure
    )

    replant_failure = (
        eligible
        & lifted_now
        & old_tread_contact
        & ~strike_failure
        & ~excursion_failure
        & ~consolidation_failure
    )

    # A target-tread contact before verified riser clearance does not count as
    # a clean step-over.
    premature_next_failure = (
        eligible
        & lifted_now
        & next_tread_contact
        & ~cleared_now
        & ~strike_failure
        & ~excursion_failure
        & ~consolidation_failure
        & ~replant_failure
    )

    direct_contact = (
        eligible
        & lifted_now
        & cleared_now
        & next_tread_contact
        & ~strike_failure
        & ~excursion_failure
        & ~consolidation_failure
        & ~replant_failure
    )
    next_seen_now = next_contact_seen | direct_contact

    already_failed = (
        strike_failure
        | excursion_failure
        | consolidation_failure
        | replant_failure
        | premature_next_failure
    )
    acquisition_pending = eligible & acquisition_event & ~already_failed

    nonpreferred_failure = (
        acquisition_pending
        & (~preferred_acquisition | ~single_acquisition)
    )
    unclean_acquisition_failure = (
        acquisition_pending
        & preferred_acquisition
        & single_acquisition
        & ~next_seen_now
    )
    success = (
        acquisition_pending
        & preferred_acquisition
        & single_acquisition
        & next_seen_now
    )

    failure = (
        strike_failure
        | excursion_failure
        | consolidation_failure
        | replant_failure
        | premature_next_failure
        | nonpreferred_failure
        | unclean_acquisition_failure
    )
    resolved_now = resolved | success | failure

    reward = torch.zeros_like(strike_severity)
    reward = torch.where(
        success,
        torch.full_like(reward, float(success_value)),
        reward,
    )
    reward = torch.where(
        failure & ~strike_failure,
        torch.full_like(reward, float(failure_value)),
        reward,
    )
    strike_value = (
        float(strike_failure_base)
        + float(strike_failure_severity_scale)
        * strike_severity.clamp(0.0, 1.0)
    )
    reward = torch.where(strike_failure, strike_value, reward)

    # Event codes are diagnostic only:
    # 0 none, 1 success, 2 replant, 3 strike, 4 nonpreferred,
    # 5 sequence/clearance failure, 6 excessive forward excursion,
    # 7 pre-lift old-tread consolidation.
    event_code = torch.zeros(shape, dtype=torch.long, device=active.device)
    event_code = torch.where(success, torch.ones_like(event_code), event_code)
    event_code = torch.where(
        replant_failure,
        torch.full_like(event_code, 2),
        event_code,
    )
    event_code = torch.where(
        consolidation_failure,
        torch.full_like(event_code, 7),
        event_code,
    )
    event_code = torch.where(
        strike_failure,
        torch.full_like(event_code, 3),
        event_code,
    )
    event_code = torch.where(
        nonpreferred_failure,
        torch.full_like(event_code, 4),
        event_code,
    )
    event_code = torch.where(
        premature_next_failure | unclean_acquisition_failure,
        torch.full_like(event_code, 5),
        event_code,
    )
    event_code = torch.where(
        excursion_failure,
        torch.full_like(event_code, 6),
        event_code,
    )

    return (
        lifted_now,
        cleared_now,
        next_seen_now,
        resolved_now,
        reward,
        event_code,
    )


def step_over_hard_failure_mask(
    event_code: torch.Tensor,
    target: torch.Tensor,
    acquisition_event: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard-fail invalid acquisitions plus stair-to-stair violations.

    Ground -> stair1 remains free-choice and keeps approach strike/replant
    failures soft. Once a physical stair1 acquisition occurs, however, it
    must be a single clean acquisition; simultaneous/uncommitted or unclean
    acquisition cannot disable the later alternating contract.
    """
    if event_code.shape != target.shape:
        raise ValueError("event_code and target must have matching shapes")

    if acquisition_event is None:
        acquisition_event = torch.zeros_like(
            event_code,
            dtype=torch.bool,
        )

    if acquisition_event.shape != target.shape:
        raise ValueError(
            "acquisition_event and target must have matching shapes"
        )

    if acquisition_event.dtype != torch.bool:
        raise ValueError("acquisition_event must be boolean")

    first_acquisition_failure = (
        (target == 0)
        & acquisition_event
        & (event_code >= 2)
    )

    stair_to_stair_failure = (
        (target > 0)
        & (event_code >= 2)
    )

    return first_acquisition_failure | stair_to_stair_failure


def gate_physical_touchdown_by_first_placement(
    physical_touchdown: torch.Tensor,
    first_placement_gate: torch.Tensor,
) -> torch.Tensor:
    """Reject continuous-support shuffles after a poor first loaded contact."""
    if physical_touchdown.shape != first_placement_gate.shape:
        raise ValueError(
            "physical_touchdown and first_placement_gate must match"
        )

    if (
        physical_touchdown.dtype != torch.bool
        or first_placement_gate.dtype != torch.bool
    ):
        raise ValueError(
            "touchdown placement gates must be boolean"
        )

    return physical_touchdown & first_placement_gate




def discounted_potential_shaping(
    previous_potential: torch.Tensor,
    current_potential: torch.Tensor,
    terminal: torch.Tensor,
    *,
    discount_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gamma*Phi(s') - Phi(s), forcing terminal Phi(s') to zero."""
    if previous_potential.shape != current_potential.shape:
        raise ValueError(
            "previous_potential and current_potential must match"
        )
    if terminal.shape != current_potential.shape:
        raise ValueError("terminal must match potential tensors")
    if terminal.dtype != torch.bool:
        raise ValueError("terminal must be boolean")
    if not 0.0 < discount_gamma <= 1.0:
        raise ValueError("discount_gamma must be in (0, 1]")

    next_potential = torch.where(
        terminal,
        torch.zeros_like(current_potential),
        current_potential,
    )
    shaping = (
        float(discount_gamma) * next_potential
        - previous_potential
    )
    return shaping, next_potential


def monotonic_progress_credit(
    current_fraction: torch.Tensor,
    best_fraction: torch.Tensor,
    eligible: torch.Tensor,
    *,
    total_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit only previously unseen bounded progress.

    The potential is clamped to [0, 1]. Moving backward or repeating the
    same height emits zero additional credit.
    """
    shape = current_fraction.shape
    for name, value in (
        ("best_fraction", best_fraction),
        ("eligible", eligible),
    ):
        if value.shape != shape:
            raise ValueError(
                f"{name} must match current_fraction shape"
            )
    if total_value < 0.0:
        raise ValueError("total_value cannot be negative")

    current = current_fraction.clamp(0.0, 1.0)
    previous = best_fraction.clamp(0.0, 1.0)

    candidate = torch.where(
        eligible,
        current,
        previous,
    )
    best_now = torch.maximum(previous, candidate)

    delta = (best_now - previous).clamp_min(0.0)
    credit = delta * float(total_value)

    return best_now, credit


def escrowed_transition_reward(
    base_event_reward: torch.Tensor,
    event_code: torch.Tensor,
    new_lift: torch.Tensor,
    new_clearance: torch.Tensor,
    credit_paid: torch.Tensor,
    *,
    lift_milestone_value: float = 0.20,
    clearance_milestone_value: float = 0.30,
    progress_credit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit non-farmable milestone credit for a step-over transition.

    Lift and verified-riser-clearance milestones are paid at most once.  A
    terminal event repays all previously emitted milestone credit by subtracting
    ``credit_paid`` from the terminal value.  Therefore a clean transition
    still totals exactly its terminal success value, while any failed
    transition totals exactly its bounded failure value regardless of how much
    partial progress occurred first.  Horizontal displacement never enters this
    calculation.
    """
    shape = base_event_reward.shape
    if progress_credit is None:
        progress_credit = torch.zeros_like(base_event_reward)

    tensors = (
        ("event_code", event_code),
        ("new_lift", new_lift),
        ("new_clearance", new_clearance),
        ("credit_paid", credit_paid),
        ("progress_credit", progress_credit),
    )
    for name, value in tensors:
        if value.shape != shape:
            raise ValueError(f"{name} must match base_event_reward shape")
    if lift_milestone_value < 0.0:
        raise ValueError("lift_milestone_value cannot be negative")
    if clearance_milestone_value < 0.0:
        raise ValueError("clearance_milestone_value cannot be negative")

    milestone = (
        new_lift.to(base_event_reward.dtype) * float(lift_milestone_value)
        + new_clearance.to(base_event_reward.dtype)
        * float(clearance_milestone_value)
        + progress_credit.to(base_event_reward.dtype)
    )
    terminal = event_code != 0

    # On a terminal step, do not pay a new milestone first.  The emitted value
    # makes the complete transition return equal to the terminal target after
    # accounting for milestones paid on earlier steps.
    reward = torch.where(
        terminal,
        base_event_reward - credit_paid,
        milestone,
    )
    paid_now = torch.where(
        terminal,
        credit_paid,
        credit_paid + milestone,
    )
    return reward, paid_now


def global_stair_target_update(
    target_step: torch.Tensor,
    physical_touchdown: torch.Tensor,
    step_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one shared stair target when either foot acquires it."""
    if target_step.shape != physical_touchdown.shape:
        raise ValueError(
            "target_step and physical_touchdown must have matching shapes"
        )
    if target_step.ndim != 2:
        raise ValueError(
            "target_step must have shape [num_envs, num_feet]"
        )

    acquired = physical_touchdown.any(dim=1)
    old_target = target_step[:, 0].clone()

    next_target = (
        old_target + acquired.to(dtype=old_target.dtype)
    ).clamp(max=int(step_count))

    synchronized = next_target[:, None].expand_as(target_step)
    return old_target, acquired, synchronized


def first_touchdown_placement_score(
    rear_margin_m: torch.Tensor,
    front_margin_m: torch.Tensor,
    *,
    shaping_start_margin_m: float,
    full_rear_margin_m: float,
    front_safe_margin_m: float,
) -> torch.Tensor:
    """Smooth [0, 1] score for first loaded tread placement."""

    if full_rear_margin_m <= shaping_start_margin_m:
        raise ValueError(
            "full_rear_margin_m must exceed shaping_start_margin_m"
        )
    if front_safe_margin_m <= 0.0:
        raise ValueError("front_safe_margin_m must be positive")

    rear_score = (
        (
            rear_margin_m - float(shaping_start_margin_m)
        )
        / float(full_rear_margin_m - shaping_start_margin_m)
    ).clamp(0.0, 1.0)

    front_score = (
        front_margin_m / float(front_safe_margin_m)
    ).clamp(0.0, 1.0)

    return rear_score * front_score


def weighted_arm_family_excess_cost(
    family_errors: tuple[torch.Tensor, ...],
    family_tolerances: tuple[torch.Tensor, ...],
    family_weights: tuple[float, ...],
) -> torch.Tensor:
    """Mean tolerance-normalized excess cost per arm family.

    Each family is normalized by its active tolerance before weighting,
    so family costs represent relative envelope violation rather than
    absolute radians. Family joint count does not affect weighting.
    """

    if not (
        len(family_errors)
        == len(family_tolerances)
        == len(family_weights)
    ):
        raise ValueError("arm family inputs must have matching lengths")
    if not family_errors:
        raise ValueError("at least one arm family is required")

    num_envs = family_errors[0].shape[0]
    cost = torch.zeros(
        num_envs,
        device=family_errors[0].device,
        dtype=family_errors[0].dtype,
    )
    denominator = 0.0

    for error, tolerance, weight in zip(
        family_errors,
        family_tolerances,
        family_weights,
        strict=True,
    ):
        if error.ndim != 2 or error.shape[0] != num_envs:
            raise ValueError("family error must have shape [N, J]")
        if tolerance.shape != (num_envs,):
            raise ValueError("family tolerance must have shape [N]")
        if weight < 0.0:
            raise ValueError("family weights must be non-negative")

        if torch.any(tolerance <= 0.0):
            raise ValueError("family tolerances must be positive")

        excess = (
            error.abs() - tolerance[:, None]
        ).clamp_min(0.0)
        normalized_excess = excess / tolerance[:, None]

        cost = cost + float(weight) * torch.mean(
            normalized_excess.square(),
            dim=1,
        )
        denominator += float(weight)

    if denominator <= 0.0:
        raise ValueError("arm family weights must sum positive")

    return cost / denominator
