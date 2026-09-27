"""Pure contracts for deterministic G1 single-support mechanics qualification."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScriptCandidate:
    """One deterministic raw-action trajectory for right-foot support."""

    name: str
    family: str
    sign: int
    shift_raw: float
    lift_scale: float
    values: dict[str, float]


CONTROL_TRACKING_JOINTS = (
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
)


def _shift_pattern(family: str, sign: int, magnitude: float) -> dict[str, float]:
    """Return conservative raw-action roll offsets for a lateral weight shift.

    Both signs are always tested at runtime, so the diagnostic does not assume
    a particular G1 joint-frame sign convention.
    """

    s = float(sign)
    a = float(magnitude)

    if family == "hip_ankle_counter":
        return {
            "left_hip_roll_joint": s * a,
            "right_hip_roll_joint": s * a,
            "left_ankle_roll_joint": -s * 0.70 * a,
            "right_ankle_roll_joint": -s * 0.70 * a,
            "waist_roll_joint": -s * 0.25 * a,
        }
    if family == "hip_ankle_same":
        return {
            "left_hip_roll_joint": s * a,
            "right_hip_roll_joint": s * a,
            "left_ankle_roll_joint": s * 0.50 * a,
            "right_ankle_roll_joint": s * 0.50 * a,
            "waist_roll_joint": -s * 0.25 * a,
        }
    if family == "asymmetric_support":
        return {
            "left_hip_roll_joint": s * 1.20 * a,
            "right_hip_roll_joint": s * 0.80 * a,
            "left_ankle_roll_joint": -s * 0.60 * a,
            "right_ankle_roll_joint": -s * 0.85 * a,
            "waist_roll_joint": -s * 0.30 * a,
        }
    if family == "waist_counter":
        return {
            "left_hip_roll_joint": s * a,
            "right_hip_roll_joint": s * a,
            "left_ankle_roll_joint": -s * a,
            "right_ankle_roll_joint": -s * a,
            "waist_roll_joint": -s * 0.50 * a,
        }
    raise ValueError(f"unknown shift family: {family}")


def scripted_candidates() -> tuple[ScriptCandidate, ...]:
    """Deterministic, bounded trajectory bank used by the qualification."""

    result: list[ScriptCandidate] = []
    families = (
        "hip_ankle_counter",
        "hip_ankle_same",
        "asymmetric_support",
        "waist_counter",
    )

    for family in families:
        for sign in (-1, 1):
            for magnitude in (0.25, 0.50, 0.75):
                for lift_scale in (0.85, 1.15):
                    values = _shift_pattern(family, sign, magnitude)
                    # Conservative left-leg flexion. These are raw actions;
                    # the active environment converts them with scale=0.25.
                    values.update(
                        {
                            "left_hip_pitch_joint": -0.80 * lift_scale,
                            "left_knee_joint": 1.60 * lift_scale,
                            "left_ankle_pitch_joint": -0.80 * lift_scale,
                        }
                    )
                    result.append(
                        ScriptCandidate(
                            name=(
                                f"{family}_s{sign:+d}_a{magnitude:.2f}"
                                f"_l{lift_scale:.2f}"
                            ),
                            family=family,
                            sign=sign,
                            shift_raw=magnitude,
                            lift_scale=lift_scale,
                            values=values,
                        )
                    )

    if len(result) != 48:
        raise RuntimeError("qualification candidate bank must contain 48 trajectories")
    return tuple(result)


def linear_ramp(step: int, start: int, duration: int) -> float:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if step < start:
        return 0.0
    return max(0.0, min(1.0, (step - start + 1) / duration))


def longest_true_run(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def classify_candidate(summary: dict, *, step_dt: float) -> dict:
    """Classify one scripted trajectory using physical, not reward, criteria."""

    stable_flags = list(summary.get("single_support_flags", []))
    stable_steps = longest_true_run(stable_flags)
    stable_duration_s = stable_steps * float(step_dt)

    result = dict(summary)
    result["stable_single_support_steps"] = stable_steps
    result["stable_single_support_duration_s"] = stable_duration_s
    result["single_support_pass"] = stable_duration_s >= 0.30
    return result


def overall_verdict(neutral: dict, candidates: list[dict]) -> dict:
    """Return a bounded diagnostic verdict with an explicit interpretation."""

    if (
        neutral["right_anchor_drift_max_m"] > 0.020
        or neutral["left_anchor_drift_max_m"] > 0.020
        or neutral["upright_min"] < 0.90
    ):
        return {
            "verdict": "NEUTRAL_STANCE_FAIL",
            "interpretation": (
                "The plant drifts or loses posture before any scripted weight shift. "
                "Inspect contact/friction/reset/PD before RL."
            ),
        }

    passed = [row for row in candidates if row.get("single_support_pass")]
    if passed:
        best = min(
            passed,
            key=lambda row: (
                row["right_anchor_drift_max_m"],
                row["right_planar_speed_mean_mps"],
            ),
        )
        return {
            "verdict": "PLANT_CAPABLE_SINGLE_SUPPORT",
            "best_candidate": best["candidate"],
            "interpretation": (
                "At least one deterministic non-PPO trajectory maintained stable "
                "right-foot single support. The simulation plant can satisfy the "
                "prerequisite; remaining stair failure is policy/control/reward-side."
            ),
        }

    tracking_bad = [
        row
        for row in candidates
        if row["critical_tracking_rms_rad"] > 0.15
        or row["torque_utilization_max"] >= 0.98
    ]
    unload_attempts = [
        row
        for row in candidates
        if row["left_force_share_min"] <= 0.25
        and row["left_foot_height_gain_max_m"] >= 0.020
    ]
    slipping_unload = [
        row
        for row in unload_attempts
        if row["right_anchor_drift_max_m"] >= 0.060
        or row["right_planar_speed_mean_mps"] >= 0.15
    ]

    if slipping_unload:
        best = min(slipping_unload, key=lambda row: row["right_anchor_drift_max_m"])
        return {
            "verdict": "SUPPORT_SLIP_LIMITING",
            "best_candidate": best["candidate"],
            "interpretation": (
                "The scripted controller can unload/lift the left foot, but the "
                "loaded right support foot translates too far or too fast. Inspect "
                "foot-ground friction/contact geometry and lateral balance mechanics."
            ),
        }

    if tracking_bad and len(tracking_bad) >= max(1, len(candidates) // 3):
        return {
            "verdict": "ACTUATOR_TRACKING_LIMITING",
            "interpretation": (
                "A large fraction of deterministic candidates show excessive joint "
                "tracking error or torque saturation. Inspect gains, effort limits, "
                "action scaling and commanded pose feasibility before PPO."
            ),
        }

    best = min(
        candidates,
        key=lambda row: (
            row["left_force_share_min"],
            -row["left_foot_height_gain_max_m"],
            row["right_anchor_drift_max_m"],
        ),
    )
    return {
        "verdict": "SCRIPTED_TRANSFER_NOT_ACHIEVED",
        "best_candidate": best["candidate"],
        "interpretation": (
            "Neutral stance is stable, but this bounded deterministic trajectory bank "
            "did not establish a 0.30 s stable right-foot single-support interval. "
            "This is not proof that the plant is incapable; inspect the best trace "
            "before changing RL rewards."
        ),
    }


def finite_number(value: float) -> bool:
    return math.isfinite(float(value))
