"""CPU-testable sustained-support timing."""
import math
import torch


def sustained_support(previous, supported, dt, required):
    if not (math.isfinite(required) and required > 0 and dt > 0):
        raise ValueError("support timing must be positive and finite")
    elapsed = torch.where(supported, previous + dt, torch.zeros_like(previous))
    return elapsed, supported & (elapsed + 1e-6 >= required)
