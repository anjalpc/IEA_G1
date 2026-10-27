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


def clock_contact_scores(force_z, sole_xyz, sole_vel_xyz, targets, feet, valid,
                         swing, stance, *, threshold_n=30.0, speed_scale=0.3):
    # Reward directed movement of the assigned swing foot, not arbitrary speed.
    n = force_z.shape[0]
    if (force_z.shape != (n, 2) or sole_xyz.shape != (n, 2, 3)
            or sole_vel_xyz.shape != (n, 2, 3) or targets.shape != (n, 2, 4)
            or feet.shape != (n, 2) or valid.shape != (n, 2)
            or swing.shape != (n, 2) or stance.shape != (n, 2)):
        raise ValueError('invalid contact/footstep tensor shapes')
    if threshold_n <= 0 or speed_scale <= 0:
        raise ValueError('threshold_n and speed_scale must be positive')
    stance_score = ((force_z >= threshold_n).to(force_z.dtype) * stance).sum(1)
    stance_score /= stance.sum(1).clamp(min=1)
    rows = torch.arange(n, device=feet.device)
    assigned = feet[:, 0].long()
    delta = targets[:, 0, :3] - sole_xyz[rows, assigned]
    dist = torch.linalg.vector_norm(delta, dim=-1)
    direction = delta / dist.clamp(min=1.0e-6)[:, None]
    toward_velocity = (sole_vel_xyz[rows, assigned] * direction).sum(-1)
    score = torch.clamp(toward_velocity / speed_scale, min=0, max=1)
    active = swing[rows, assigned].bool() & valid[:, 0].bool() & (dist > 0.001)
    return stance_score, torch.where(active, score, torch.zeros_like(score))



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


def cached_acquisition(env):
    """Read qualified assigned-foot touchdown from post-physics hook."""
    value = getattr(env, "_g1_footstep_last_advanced", None)
    if value is None:
        raise RuntimeError("footstep acquisition reward evaluated before physics hook")
    if value.shape != (env.num_envs,):
        raise RuntimeError("invalid cached acquisition shape")
    return value.float()


def cached_directed_swing(env):
    """Read target-directed swing measurement; never update coordinator."""
    value = getattr(env, "_g1_footstep_directed_swing", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing post-physics swing measurement")
    return value


def new_target_progress(previous_distance, current_distance, best_distance, valid, *, scale_m=0.10):
    """One-way improvement against best distance reached for the CURRENT target.

    Returns (score, updated_best).  Distance increases never pay, and
    retracing previously rewarded ground cannot pay again. The caller must
    reset best distance only when the target changes or an episode resets.
    """
    if (previous_distance.shape != current_distance.shape
            or previous_distance.shape != best_distance.shape
            or previous_distance.shape != valid.shape):
        raise ValueError("progress tensors have different shapes")
    if scale_m <= 0:
        raise ValueError("scale_m must be positive")
    if not torch.isfinite(previous_distance).all() or not torch.isfinite(current_distance).all():
        raise ValueError("nonfinite target distance")
    if not torch.isfinite(best_distance).all():
        raise ValueError("nonfinite best distance")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    new_best = torch.minimum(best_distance, current_distance)
    improvement = (best_distance - new_best).clamp(min=0)
    reward = torch.where(valid, improvement / scale_m, torch.zeros_like(improvement))
    return reward, torch.where(valid, new_best, best_distance)


def cached_net_progress(env):
    """Non-recycling progress computed once after physics."""
    value = getattr(env, "_g1_footstep_net_progress", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached footstep net progress")
    return value


def cached_stance_support(env):
    """Observed support during assigned gait-clock stance."""
    value = getattr(env, "_g1_footstep_stance_support", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached stance support")
    return value


def cached_clearance_progress(env):
    """New peak assigned-foot clearance with opposite-foot support."""
    value = getattr(env, "_g1_footstep_clearance_progress", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached footstep clearance")
    return value


def cached_failure_event(env):
    """Non-timeout termination event."""
    value = getattr(env, "_g1_footstep_failure_event", None)
    if value is None or value.shape != (env.num_envs,):
        raise RuntimeError("missing cached failure event")
    return value
