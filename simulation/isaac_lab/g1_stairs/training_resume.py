"""Restore saved learning state into a new run, without modifying the source."""
import hashlib
from pathlib import Path
import torch


def resume_training(runner, path, expected_sha256):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Resume checkpoint SHA256 mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict", "iter"):
        if key not in saved:
            raise ValueError(f"Checkpoint missing {key}")
    if not saved["optimizer_state_dict"].get("state"):
        raise ValueError("Checkpoint has no populated optimizer state")

    runner.load(str(path), load_optimizer=True, map_location=runner.device)

    # Verify restoration instead of relying solely on runner.load's return value.
    restored = runner.alg.policy.state_dict()
    original = saved["model_state_dict"]
    if set(restored) != set(original):
        raise ValueError("Model-state keys differ after resume")
    for key in original:
        if not torch.equal(restored[key].detach().cpu(), original[key].cpu()):
            raise ValueError(f"Model state was not restored: {key}")

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(
                equal(a[k], b[k]) for k in a
            )
        if isinstance(a, (list, tuple)):
            return isinstance(b, (list, tuple)) and len(a) == len(b) and all(
                equal(x, y) for x, y in zip(a, b)
            )
        return a == b

    if not equal(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"]):
        raise ValueError("Optimizer state was not fully restored")

    rates = {float(g["lr"]) for g in runner.alg.optimizer.param_groups}
    if len(rates) != 1:
        raise ValueError("Resume currently requires one shared optimizer learning rate")
    runner.alg.learning_rate = rates.pop()

    # Installed runner saves iter after completing that update.
    runner.current_learning_iteration = int(saved["iter"]) + 1
    metadata = {
        "type": "rsl_rl_training_resume",
        "source": str(path),
        "source_sha256": digest,
        "saved_iteration": int(saved["iter"]),
        "next_iteration": runner.current_learning_iteration,
        "learning_rate": runner.alg.learning_rate,
        "actor_critic_optimizer_verified": True,
        "simulator_and_rng_restored": False,
    }
    print("[RESUME]", metadata, flush=True)
    return metadata


def cap_action_noise_std(policy, cap, partner):
    """Cap the restored exploration std at ``cap`` and make it left/right
    symmetric (mean of each joint and its mirror partner).

    The std is a state-independent parameter that the entropy bonus keeps
    growing and that neither mirror augmentation nor the mirror loss
    constrains. Run 20261103_133659 model_8200: 20 of 29 joints at 3.3-9.4
    against a residual clip of 2 (random full-range commands while
    training), left hip yaw 27% above the right. ``partner[i]`` is the
    mirror joint of action i. Returns (old, new) std lists.
    """
    if cap <= 0.0:
        raise ValueError("action noise std cap must be > 0")
    if hasattr(policy, "std"):
        param, to_std, from_std = policy.std, (lambda p: p), (lambda s: s)
    elif hasattr(policy, "log_std"):
        param, to_std, from_std = policy.log_std, torch.exp, torch.log
    else:
        raise ValueError("policy has neither std nor log_std")
    if len(partner) != param.numel():
        raise ValueError("partner map does not match the action width")
    with torch.no_grad():
        old = to_std(param.data).clone()
        new = old.clamp(max=cap)
        new = 0.5 * (new + new[torch.as_tensor(partner, device=new.device)])
        param.data.copy_(from_std(new))
    return old.cpu().tolist(), new.cpu().tolist()


def raise_action_noise_std(policy, minimum, partner):
    """Raise the restored exploration std to at least ``minimum[i]`` per
    action, once, on resume (it stays a free parameter afterwards).

    With no entropy bonus the std only shrinks. Run 20261116_091452
    model_84200: hip pitch 0.005, knee 0.007, ankle pitch 0.010 raw units
    (about 0.002 rad on the joint), so the policy tried almost nothing new on
    the joints that lift and place the foot: after 3000 iterations with a
    knee clip of 3 the knee correction peaked at +2.1, a crouch cost of
    1.7/s moved the flat stance knee 0.58 -> 0.54 rad, and 19 cm needed over
    10,000 iterations for its first climbs. Not a floor: patch 46's std
    floor (policy_shaping) was removed in patch 47 (forward lean, riser
    stall). ``minimum`` must be left/right symmetric. Returns (old, new).
    """
    if hasattr(policy, "std"):
        param, to_std, from_std = policy.std, (lambda p: p), (lambda s: s)
    elif hasattr(policy, "log_std"):
        param, to_std, from_std = policy.log_std, torch.exp, torch.log
    else:
        raise ValueError("policy has neither std nor log_std")
    low = torch.as_tensor(minimum, dtype=param.dtype, device=param.device)
    if low.shape != param.shape or len(partner) != param.numel():
        raise ValueError("std minimum does not match the action width")
    if bool((low < 0.0).any()) or not torch.equal(low, low[torch.as_tensor(partner, device=low.device)]):
        raise ValueError("std minimum must be >= 0 and left/right symmetric")
    with torch.no_grad():
        old = to_std(param.data).clone()
        new = torch.maximum(old, low)
        param.data.copy_(from_std(new))
    return old.cpu().tolist(), new.cpu().tolist()
