"""Mirror-symmetric residual actor (patch 54).

The actor's mean becomes

    a(s) = (clip(f(s)) + M clip(f(M s))) / 2

with f the trained MLP, M the left/right mirror (footstep_symmetry) and clip
the residual clip of balance_residual_control. a(M s) = M a(s) holds exactly,
so the residual cannot carry a left/right bias. Same construction as the
teacher averaging of patch 44, here for the residual.

Evidence (run 20261106_151951 model_14200, 10 robots per height, replayed with
record_view.py --symmetry average, which drives exactly this action): flat
turn +0.71 -> +0.08 deg/s, sideways 34 -> 7 cm, stance knee 0.28/0.32 ->
0.30/0.30, landing 190/270 -> 198/205 N; 4 cm stairs still 10/10 climbed but
tread 2 short in 5/10 (the policy was not trained under averaging).

The patch replaces ``policy.actor.forward`` on the instance, so act, PPO's
update_distribution, act_inference, replays and evaluation all see the
symmetric mean and checkpoints keep their layout (the MLP weights are
unchanged). Every loader of a policy trained with ``ppo.symmetric_actor``
must call ``apply_from_config``; deployment must evaluate the policy twice
per step, as for the teacher.

Outputs beyond the clip get no gradient through the clamp; in training the
mean bound (policy_shaping) acts on the raw outputs through ``raw_hook``.
"""
from __future__ import annotations

import torch

from .footstep_symmetry import mirror_actions, mirror_policy_obs


def make_actor_symmetric(policy, residual_clip: float, raw_hook=None) -> None:
    """Make ``policy.actor`` mirror-symmetric in place (idempotent)."""
    if residual_clip <= 0.0:
        raise ValueError("residual_clip must be positive")
    if getattr(policy, "actor_obs_normalization", False):
        # The mirror acts on raw observations; a normaliser in front of the
        # actor would need its own mirrored statistics.
        raise ValueError("symmetric actor needs actor_obs_normalization=False")
    actor = policy.actor
    if getattr(actor, "_g1_symmetric", False):
        return
    forward = actor.forward
    clip = float(residual_clip)

    def symmetric_forward(x):
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        raw, raw_m = forward(torch.cat([x, mirror_policy_obs(x)], dim=0)).split(x.shape[0])
        if raw_hook is not None and torch.is_grad_enabled():
            raw, raw_m = raw_hook(raw), raw_hook(raw_m)
        out = 0.5 * (raw.clamp(-clip, clip) + mirror_actions(raw_m.clamp(-clip, clip)))
        return out.squeeze(0) if single else out

    actor.forward = symmetric_forward
    actor._g1_symmetric = True


def enabled(resolved_config) -> bool:
    return bool(((resolved_config or {}).get("ppo") or {}).get("symmetric_actor", False))


def apply_from_config(policy, resolved_config, raw_hook=None) -> bool:
    """Apply the symmetric actor when the run was trained with it."""
    if not enabled(resolved_config):
        return False
    residual = resolved_config.get("balance_residual_control") or {}
    if "residual_clip" not in residual:
        raise ValueError("ppo.symmetric_actor needs balance_residual_control.residual_clip")
    make_actor_symmetric(policy, float(residual["residual_clip"]), raw_hook=raw_hook)
    print("[INFO] Symmetric actor: (clip(f(s)) + M clip(f(M s))) / 2, clip",
          float(residual["residual_clip"]), flush=True)
    return True
