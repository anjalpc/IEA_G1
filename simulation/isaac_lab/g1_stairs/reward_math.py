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
