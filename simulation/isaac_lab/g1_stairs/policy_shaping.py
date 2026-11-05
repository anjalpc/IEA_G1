"""Training-only shaping of the residual policy's action distribution.

Both act inside ``ActorCritic.update_distribution`` (rsl-rl-lib 3.0.1), so
rollouts and PPO updates see the same distribution. Neither changes the
action mean the robot executes, the saved checkpoint layout or replay.

1. Mean bound. The residual is clipped to +-residual_clip before execution
   (balance_residual.compose_balance_residual_action). A mean beyond the clip
   sits in a dead zone: every reward sees the clipped value, so no reward can
   pull it back. Runs 20261102-20261104: ~70% of all outputs beyond the clip
   (waist, shoulders, wrists, hip roll/yaw 88-100%); the pinned hip yaw
   (2 x 0.3 x 0.25 rad = 8.6 deg) matches the measured 8-9 deg toe-out. The
   bound adds ``coeff * sum_j relu(|mean_j| - clip)`` (mean over the batch)
   to the PPO loss through the mean's gradient (identity forward pass). It is
   L1, so the pull per output is the same however far beyond the clip it
   sits. Until a mean reaches the clip, executed actions do not change.

2. Std floor. With entropy_coef 0 the exploration std of the leg pitch
   joints collapsed (hip pitch 0.26 -> 0.075 -> 0.02 over patches 43-45).
   The distribution uses max(std, floor); the std parameter itself is left
   as it is.
"""
from __future__ import annotations

import torch
from torch.distributions import Normal


class _MeanBound(torch.autograd.Function):
    """Identity forward; backward adds d/dmean of coeff * sum relu(|mean| - clip)."""

    @staticmethod
    def forward(ctx, mean, clip, coeff):
        ctx.save_for_backward(mean)
        ctx.clip, ctx.coeff = clip, coeff
        return mean.view_as(mean)

    @staticmethod
    def backward(ctx, grad):
        (mean,) = ctx.saved_tensors
        rows = mean.shape[0] if mean.ndim > 1 else 1
        outside = (mean.abs() > ctx.clip).to(grad.dtype)
        return grad + ctx.coeff * outside * torch.sign(mean) / rows, None, None


def bound_loss(mean: torch.Tensor, clip: float, coeff: float) -> torch.Tensor:
    """The loss whose gradient _MeanBound adds (for tests and logging)."""
    rows = mean.shape[0] if mean.ndim > 1 else 1
    return coeff * torch.relu(mean.abs() - clip).sum() / rows


class PolicyShaping:
    """Wraps ``policy.update_distribution``; keeps simple statistics."""

    def __init__(self, policy, *, residual_clip: float, bound_coeff: float = 0.0,
                 std_floor: float = 0.0, log_every: int = 0):
        if residual_clip <= 0.0 or bound_coeff < 0.0 or std_floor < 0.0:
            raise ValueError("residual_clip > 0, bound_coeff >= 0, std_floor >= 0 required")
        self.clip, self.coeff, self.floor = float(residual_clip), float(bound_coeff), float(std_floor)
        self.log_every = int(log_every)
        self.calls = 0
        self._outside = 0.0
        self._excess = 0.0
        self._original = policy.update_distribution
        self._policy = policy
        policy.update_distribution = self.update_distribution

    def update_distribution(self, obs):
        self._original(obs)
        dist = self._policy.distribution
        mean, std = dist.mean, dist.stddev
        if self.floor > 0.0:
            std = std.clamp(min=self.floor)
        if self.coeff > 0.0 and torch.is_grad_enabled() and mean.requires_grad:
            with torch.no_grad():
                excess = torch.relu(mean.abs() - self.clip)
                self._outside += float((excess > 0).float().mean())
                self._excess += float(excess.sum(dim=-1).mean())
            self.calls += 1
            mean = _MeanBound.apply(mean, self.clip, self.coeff)
            if self.log_every and self.calls % self.log_every == 0:
                n = self.log_every
                print(f"[POLICY_SHAPING] means beyond clip {self._outside / n:.3f}, "
                      f"summed excess per state {self._excess / n:.2f}, "
                      f"bound loss {self.coeff * self._excess / n:.4f}", flush=True)
                self._outside = self._excess = 0.0
        self._policy.distribution = Normal(mean, std)
