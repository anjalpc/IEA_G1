"""C3 phase math. No Isaac imports or changes to existing task behavior."""

import math
import torch


def gait_frequency(raw_action, *, learned=True):
    """Unfiltered frequency; zero mean action gives 1 Hz."""
    if not learned:
        return torch.ones_like(raw_action)
    return 1.0 + 0.3 * torch.tanh(raw_action)


def advance_gait_phase(phase, frequency, active, acquired_foot, dt):
    """Advance once per control step, then realign on unique acquisition.

    acquired_foot: -1 = no unique acquisition, 0 = left, 1 = right.
    active must describe the post-acquisition transition state.
    Score the just-executed step BEFORE calling this function.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    advanced = torch.remainder(phase + dt * frequency, 1.0)
    aligned = torch.where(
        acquired_foot == 0,
        torch.zeros_like(phase),
        torch.where(
            acquired_foot == 1,
            torch.full_like(phase, 0.5),
            advanced,
        ),
    )
    return torch.where(active, aligned, torch.zeros_like(phase))


def gait_phase_observation(phase, active):
    """Global left-foot phase encoded as [sin, cos]; inactive = [0, 0]."""
    angle = 2.0 * math.pi * phase
    features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    return torch.where(active.unsqueeze(-1), features, torch.zeros_like(features))


def phase_swing_contact_loss(
    phase, active, vertical_foot_force, *, touchdown_force_n
):
    """Continuous, bounded contact cost using the PRE-ADVANCE phase.

    vertical_foot_force is canonical upward contact force, ordered [L, R].
    No /dt conversion. RewardManager supplies the negative weight and dt.
    """
    if not math.isfinite(touchdown_force_n) or touchdown_force_n <= 0:
        raise ValueError("touchdown_force_n must be finite and positive")

    left_clock = torch.sin(2.0 * math.pi * phase)
    # Exact boundaries should have zero gate despite floating-point sin(pi).
    boundary = (phase == 0.0) | (phase == 0.5)
    left_clock = torch.where(boundary, torch.zeros_like(left_clock), left_clock)
    clocks = torch.stack((left_clock, -left_clock), dim=-1)
    swing_gate = (-clocks).clamp(0.0, 1.0)
    loaded = vertical_foot_force >= touchdown_force_n
    loss = (swing_gate * loaded.to(swing_gate.dtype)).sum(dim=-1)
    return torch.where(active, loss, torch.zeros_like(loss))
