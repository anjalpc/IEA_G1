"""Opt-in target/contact/gait coordinator; no Isaac lifecycle side effects.

An external policy-step hook supplies fresh physical measurements exactly once.
Target progression is conditional on an externally verified tread-contact mask,
rather than raw force alone. No rewards or PPO modifications.
"""
from __future__ import annotations

import math
import torch

from .footstep_runtime import FootstepRuntime
from .footstep_gait_clock import FootstepGaitClock


class FootstepCoordinator:
    def __init__(self, *, runtime: FootstepRuntime, clock: FootstepGaitClock):
        if runtime.num_envs != clock.num_envs or runtime.device != clock.device:
            raise ValueError("runtime/clock batch and device must agree")
        self.runtime = runtime
        self.clock = clock
        self.last_step = -1
        self._primed = torch.zeros(runtime.num_envs, device=runtime.device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | None = None,
              first_foot: torch.Tensor | None = None):
        self.runtime.reset(env_ids, first_foot=first_foot)
        self.clock.reset(env_ids)
        self._primed[slice(None) if env_ids is None else env_ids] = False

    def step(self, *, step_id: int, dt: float, targets: torch.Tensor,
             feet: torch.Tensor, valid: torch.Tensor, sole_xyz: torch.Tensor,
             supported: torch.Tensor, qualified_tread: torch.Tensor,
             xy_tolerance: float, z_tolerance: float) -> torch.Tensor:
        """Return successful target transitions [N]; idempotent for same step.

        qualified_tread must be computed independently from collider/sole bounds
        and collision-free tread support. Raw foot force is NOT sufficient.
        """
        if not isinstance(step_id, int) or step_id < 0 or step_id < self.last_step:
            raise ValueError("policy step id must be monotonically increasing")
        n = self.runtime.num_envs
        if supported.shape != (n, 2) or supported.dtype != torch.bool:
            raise ValueError("supported must be [N,2] bool")
        if qualified_tread.shape != (n, 2) or qualified_tread.dtype != torch.bool:
            raise ValueError("qualified_tread must be [N,2] bool")
        if supported.device != self.runtime.device or qualified_tread.device != self.runtime.device:
            raise ValueError("contact device mismatch")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if step_id == self.last_step:
            return torch.zeros(n, device=self.runtime.device, dtype=torch.bool)

        # Prime initial support per environment after reset: no artificial
        # unsupported->supported transition can become an acquisition.
        unprimed = ~self._primed
        previous = self.runtime.was_supported.clone()
        self.runtime.was_supported[unprimed] = supported[unprimed]
        self._primed[:] = True
        foot_id = feet[:, 0].long()
        qualified_current = qualified_tread.gather(1, foot_id[:, None]).squeeze(1)
        qualified_valid = valid.clone()
        qualified_valid[:, 0] &= qualified_current
        acquired = self.runtime.update(
            targets=targets, feet=feet, valid=qualified_valid,
            sole_xyz=sole_xyz, supported=supported,
            xy_tolerance=xy_tolerance, z_tolerance=z_tolerance,
        )
        # Track RAW support edges. A contact which becomes geometrically
        # qualified later without unloading is NOT a new touchdown.
        self.clock.update(dt=dt, acquired=acquired)
        self.last_step = step_id
        return acquired
