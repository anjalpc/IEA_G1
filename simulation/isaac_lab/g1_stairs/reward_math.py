"""Pure reward math helpers with no Isaac runtime dependency."""

from __future__ import annotations

import torch


def exp_tracking_score(
    squared_error: torch.Tensor,
    variance: float,
    *,
    zero_centered: bool,
) -> torch.Tensor:
    """Return exponential tracking score, optionally with zero maximum."""

    if variance <= 0.0:
        raise ValueError("tracking variance must be positive")
    if not torch.isfinite(squared_error).all():
        raise RuntimeError("tracking error must be finite")

    score = torch.exp(
        -squared_error / float(variance)
    )
    return score - 1.0 if zero_centered else score
