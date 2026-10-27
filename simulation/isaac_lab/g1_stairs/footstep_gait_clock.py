"""Reference-inspired swing / double-support phase for planned G1 footsteps.

Paper: Learning Bipedal Walking on Planned Footsteps, SS/DS alternation and
2-D periodic clock. Durations are inputs, NOT calibrated G1 parameters.
No Isaac dependency, reward activation, or contact inference.
"""
from __future__ import annotations

import math
import torch


class FootstepGaitClock:
    """Per-environment stance/swing clock synchronized to target assignments.

    Interval order for each assigned foot:
        first double support -> assigned-foot swing -> final double support.
    A new verified touchdown can synchronize the clock to the next target.
    A missed target cannot advance the planner solely because time expires.
    """

    def __init__(self, *, num_envs: int, device: str | torch.device,
                 single_support_s: float, double_support_s: float):
        for name, value in (("single_support_s", single_support_s),
                            ("double_support_s", double_support_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.single_support_s = float(single_support_s)
        self.double_support_s = float(double_support_s)
        self.elapsed_s = torch.zeros(num_envs, device=device)

    @property
    def step_period_s(self) -> float:
        return self.single_support_s + self.double_support_s

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.elapsed_s[slice(None) if env_ids is None else env_ids] = 0.0

    def update(self, *, dt: float, acquired: torch.Tensor | None = None) -> None:
        """Advance once per policy step; acquisition resets corresponding phase.

        On target miss the clock cycles but the target index remains unchanged.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if acquired is not None:
            if acquired.shape != (self.num_envs,) or acquired.dtype != torch.bool:
                raise ValueError("acquired must be per-environment bool")
            if acquired.device != self.device:
                raise ValueError("acquired device mismatch")
        self.elapsed_s = torch.remainder(self.elapsed_s + dt, self.step_period_s)
        if acquired is not None:
            self.elapsed_s = torch.where(acquired, 0.0, self.elapsed_s)

    def features(self, *, target_foot: torch.Tensor,
                 valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return [N,2] clock, [N,2] desired swing, [N,2] desired stance.

        Foot columns are [left, right]; double support requests both in stance.
        Clock is 2-D sine/cosine. Opposite target-foot assignment shifts it
        by half a gait cycle, preserving the alternating-foot convention.
        """
        if target_foot.shape != (self.num_envs,) or valid.shape != (self.num_envs,):
            raise ValueError("target_foot and valid must be per-environment vectors")
        if valid.dtype != torch.bool or target_foot.device != self.device or valid.device != self.device:
            raise ValueError("invalid validity dtype or device")
        if torch.any(valid & ((target_foot != 0) & (target_foot != 1))):
            raise ValueError("valid target foot must be left (0) or right (1)")

        phase = self.elapsed_s / self.step_period_s
        # Full cycle includes alternating left/right targets.
        cyclic_phase = torch.remainder((phase + 0.5 * target_foot), 1.0)
        angle = 2 * torch.pi * cyclic_phase
        clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        # A configurable half-DS interval at either boundary.
        half_ds = self.double_support_s / 2
        in_swing = ((self.elapsed_s >= half_ds)
                    & (self.elapsed_s < half_ds + self.single_support_s))
        feet = torch.arange(2, device=self.device)[None, :]
        swing = in_swing[:, None] & (feet == target_foot[:, None]) & valid[:, None]
        stance = valid[:, None] & ~swing
        return (torch.where(valid[:, None], clock, torch.zeros_like(clock)),
                swing.to(self.elapsed_s.dtype),
                stance.to(self.elapsed_s.dtype))
