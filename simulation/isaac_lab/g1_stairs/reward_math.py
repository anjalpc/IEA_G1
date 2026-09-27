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
    severity = torch.maximum(anchor_bad, speed_bad)
    return torch.where(
        active_gate.to(dtype=torch.bool),
        severity,
        torch.zeros_like(severity),
    )


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
