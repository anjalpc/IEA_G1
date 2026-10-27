"""Paper-inspired, tensor-only footstep objectives; no old stair reward dependencies.

Learning Bipedal Walking on Planned Footsteps:
step acquisition/progress, foot force/velocity phase, posture/height.
These are diagnostic raw scores, NOT registered reward terms or calibrated weights.
"""
import torch


def footstep_target_score(sole_xyz, targets, feet, valid, acquired, *, sigma_m=0.25):
    """[N] positive ONLY for externally qualified acquisition of assigned foot."""
    n = sole_xyz.shape[0]
    if sole_xyz.shape != (n, 2, 3) or targets.shape != (n, 2, 4):
        raise ValueError("invalid footstep geometry")
    if feet.shape != (n, 2) or valid.shape != (n, 2) or acquired.shape != (n,):
        raise ValueError("invalid target state")
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    idx = feet[:, 0].long()
    assigned = sole_xyz[torch.arange(n,device=sole_xyz.device),idx]
    error = torch.linalg.vector_norm(assigned-targets[:,0,:3],dim=-1)
    return torch.exp(-error/sigma_m) * (acquired & valid[:,0]).to(error.dtype)


def clock_contact_scores(force_z, foot_speed, swing, stance, *, threshold_n=30.0, speed_scale=0.3):
    """Return [N] stance-support and swing-motion scores (normalized by active feet).

    Force score measures expected stance contact; swing score measures foot motion.
    No causal reward assertion until force/speed traces and timing are validated.
    """
    n=force_z.shape[0]
    if force_z.shape != (n,2) or foot_speed.shape != (n,2):
        raise ValueError("foot forces and speeds must be [N,2]")
    if swing.shape != (n,2) or stance.shape != (n,2):
        raise ValueError("clock labels must be [N,2]")
    if threshold_n <= 0 or speed_scale <= 0:
        raise ValueError("positive scaling parameters required")
    stance_score = ((force_z >= threshold_n).to(force_z.dtype)*stance).sum(1)/stance.sum(1).clamp(min=1)
    swing_score = (torch.clamp(foot_speed/speed_scale,max=1.0)*swing).sum(1)/swing.sum(1).clamp(min=1)
    return stance_score, swing_score


# Stage 19 intentionally does not change PPO reward registration, losses, or training.

"""Isolation contract for a NEW paper-inspired task (not legacy reward retuning).

Registration is intentionally blocked until physics touchdown and reward-activation
diagnostics are verified. No existing 76-term rewards may be inherited implicitly.
"""
REQUIRED_NEW_FAMILIES = (
    "footstep_acquisition",
    "footstep_target_progress",
    "clock_stance_support",
    "clock_swing_motion",
    "body_orientation_height",
    "upper_body_alignment",
)

def assert_paper_only_rewards(active_names):
    """Fail closed if any active reward has not been explicitly redesigned."""
    active = tuple(active_names)
    unexpected = sorted(set(active) - set(REQUIRED_NEW_FAMILIES))
    if unexpected:
        raise RuntimeError("Legacy or unreviewed reward terms enabled: "+", ".join(unexpected))
    return True
