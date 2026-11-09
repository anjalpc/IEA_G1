"""Training-only shaping of the residual policy's action distribution.

1 and 2 act inside ``ActorCritic.update_distribution`` (rsl-rl-lib 3.0.1),
so rollouts and PPO updates see the same distribution; 3 wraps
``act_inference`` during the update. None of them changes the action mean
the robot executes, the saved checkpoint layout or replay.

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

   Paired mode (patch 48). Plain L1 pulls a left/right pair at the same
   rate, so the side with less excess reaches the clip first and the
   executed posture turns asymmetric while its partner is still pinned.
   Run 20261105_124659 (patch 47): right hip yaw saturation 0.89 -> 0.73
   while the left stayed at 0.90, right toe-out 10.6 -> 9.8 deg vs left
   12.8, flat yaw drift -0.65 -> +1.76 deg/s. In paired mode each output's
   pull is scaled by excess / max(excess, partner excess) in the same state
   (partner = mirror_actions), so the larger side keeps the full pull and
   the smaller side slows down: both excesses shrink in proportion and
   reach the clip together. Joints that mirror onto themselves (waist)
   keep the plain L1 pull.

2. Std floor. With entropy_coef 0 the exploration std of the leg pitch
   joints collapsed (hip pitch 0.26 -> 0.075 -> 0.02 over patches 43-45).
   The distribution uses max(std, floor); the std parameter itself is left
   as it is.

3. Clipped mirror loss (patch 49). rsl-rl 3.0.1 computes the mirror loss
   as MSE on ``act_inference`` means, i.e. on raw outputs. ~95% of that
   error sits in outputs beyond the clip (waist, hip roll/yaw, arms; run
   20261105_153312 replays), where it makes no physical difference, while
   the executed asymmetry that turns the robot (hip yaw/roll, knee) gets
   little of the gradient. With ``clip_mirror_means`` act_inference
   returns clamp(mean, +-clip) while gradients are enabled (the PPO update
   only); rollouts use ``act`` and play/export run without gradients, so
   neither changes. Outputs beyond the clip get no mirror gradient; the
   bound handles them.

4. Symmetric actor (patch 54, policy_symmetry.py). The actor's mean is
   clamped and averaged with its mirror, so the mean the distribution sees
   never exceeds the clip and a bound there would be inert. With
   ``symmetric=True`` the bound acts on the raw MLP outputs of both the state
   and its mirror through ``raw_hook`` (plain L1: the executed action is
   symmetric anyway, pairing is not needed).
"""
from __future__ import annotations

import torch
from torch.distributions import Normal


class _AddGrad(torch.autograd.Function):
    """Identity forward; backward adds a fixed tensor to the mean's gradient."""

    @staticmethod
    def forward(ctx, mean, extra):
        ctx.save_for_backward(extra)
        return mean.view_as(mean)

    @staticmethod
    def backward(ctx, grad):
        (extra,) = ctx.saved_tensors
        return grad + extra, None


def bound_gradient(mean: torch.Tensor, clip: float, coeff: float, partner=None) -> torch.Tensor:
    """Gradient the bound adds to the mean. partner=None: d/dmean of
    bound_loss (plain L1). partner=callable (mirror_actions): each output's
    pull scaled by excess / max(excess, partner excess)."""
    rows = mean.shape[0] if mean.ndim > 1 else 1
    excess = torch.relu(mean.abs() - clip)
    if partner is None:
        weight = (excess > 0).to(mean.dtype)
    else:
        other = torch.relu(partner(mean).abs() - clip)
        weight = excess / torch.maximum(excess, other).clamp(min=1e-12)
    return coeff * weight * torch.sign(mean) / rows


def bound_loss(mean: torch.Tensor, clip: float, coeff: float) -> torch.Tensor:
    """Plain L1 bound loss; its gradient is bound_gradient(partner=None)."""
    rows = mean.shape[0] if mean.ndim > 1 else 1
    return coeff * torch.relu(mean.abs() - clip).sum() / rows


class PolicyShaping:
    """Wraps ``policy.update_distribution``; keeps simple statistics."""

    def __init__(self, policy, *, residual_clip: float, bound_coeff: float = 0.0,
                 std_floor: float = 0.0, log_every: int = 0, partner=None,
                 clip_mirror_means: bool = False, symmetric: bool = False):
        if residual_clip <= 0.0 or bound_coeff < 0.0 or std_floor < 0.0:
            raise ValueError("residual_clip > 0, bound_coeff >= 0, std_floor >= 0 required")
        self.clip, self.coeff, self.floor = float(residual_clip), float(bound_coeff), float(std_floor)
        self.log_every = int(log_every)
        self.partner = None if symmetric else partner
        self.symmetric = bool(symmetric)
        self._hook_calls = 0
        self.calls = 0
        self._outside = 0.0
        self._excess = 0.0
        self._original = policy.update_distribution
        self._policy = policy
        policy.update_distribution = self.update_distribution
        if clip_mirror_means:
            self._act_inference = policy.act_inference
            policy.act_inference = self.act_inference

    def act_inference(self, obs):
        mean = self._act_inference(obs)
        if torch.is_grad_enabled():
            return mean.clamp(-self.clip, self.clip)
        return mean

    def raw_hook(self, raw):
        """Bound on raw actor outputs (symmetric actor, training only)."""
        if self.coeff <= 0.0 or not raw.requires_grad:
            return raw
        with torch.no_grad():
            excess = torch.relu(raw.abs() - self.clip)
            # Tensors, read only when printed: float() here waited for the
            # GPU on every minibatch (5% of the training thread, py-spy p57).
            self._outside = self._outside + (excess > 0).float().mean()
            self._excess = self._excess + excess.sum(dim=-1).mean()
            extra = bound_gradient(raw, self.clip, self.coeff)
        self._hook_calls += 1
        return _AddGrad.apply(raw, extra)

    def update_distribution(self, obs):
        self._original(obs)
        dist = self._policy.distribution
        mean, std = dist.mean, dist.stddev
        if self.floor > 0.0:
            std = std.clamp(min=self.floor)
        if self.symmetric:
            if self.coeff > 0.0 and torch.is_grad_enabled():
                self.calls += 1
                if self.log_every and self.calls % self.log_every == 0 and self._hook_calls:
                    n = self._hook_calls
                    print(f"[POLICY_SHAPING] means beyond clip {self._outside / n:.3f}, "
                          f"summed excess per state {self._excess / n:.2f}, "
                          f"bound loss {self.coeff * self._excess / n:.4f}", flush=True)
                    self._outside = self._excess = 0.0
                    self._hook_calls = 0
        elif self.coeff > 0.0 and torch.is_grad_enabled() and mean.requires_grad:
            with torch.no_grad():
                excess = torch.relu(mean.abs() - self.clip)
                self._outside += float((excess > 0).float().mean())
                self._excess += float(excess.sum(dim=-1).mean())
                extra = bound_gradient(mean, self.clip, self.coeff, self.partner)
            self.calls += 1
            mean = _AddGrad.apply(mean, extra)
            if self.log_every and self.calls % self.log_every == 0:
                n = self.log_every
                print(f"[POLICY_SHAPING] means beyond clip {self._outside / n:.3f}, "
                      f"summed excess per state {self._excess / n:.2f}, "
                      f"bound loss {self.coeff * self._excess / n:.4f}", flush=True)
                self._outside = self._excess = 0.0
        self._policy.distribution = Normal(mean, std)
