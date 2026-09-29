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
