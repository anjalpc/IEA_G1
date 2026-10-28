"""Opt-in batched footstep state, independent of stair reward/evaluator runtimes.

Contract: caller supplies course-local poses and externally measured per-foot
support flags. ``update`` must be called once per *policy step*, after physics
contacts are updated. A transition to supported is a landing candidate; a foot
that remains supported cannot trigger repeated target completions.
"""
from __future__ import annotations

import torch

from .footstep_conditioning import encode_next_two
from .footstep_tensor import planned_target_batch


class FootstepRuntime:
    """Per-env alternating planned-footstep progress; no Isaac imports."""

    def __init__(self, *, num_envs: int, device: torch.device | str,
                 step_count: int, first_foot: torch.Tensor,
                 approach_steps: int = 0):
        if num_envs < 1 or step_count < 1 or first_foot.shape != (num_envs,):
            raise ValueError("invalid environment count, step count or first-foot shape")
        if approach_steps < 0:
            raise ValueError("approach_steps must be non-negative")
        if first_foot.device != torch.device(device) or torch.any((first_foot != 0) & (first_foot != 1)):
            raise ValueError("first_foot must be 0=L / 1=R on runtime device")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_count = step_count
        # Stair plans only; flat plans never use approach targets.
        self.approach_steps = int(approach_steps)
        self.first_foot = first_foot.to(torch.long).clone()
        self.target_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.was_supported = torch.zeros((num_envs, 2), dtype=torch.bool, device=self.device)
        self.completed = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.attempted = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None, *, first_foot: torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        if first_foot is not None:
            if first_foot.shape != self.first_foot[ids].shape or first_foot.device != self.device:
                raise ValueError("first_foot reset shape/device mismatch")
            if torch.any((first_foot != 0) & (first_foot != 1)):
                raise ValueError("first_foot must be 0 or 1")
            self.first_foot[ids] = first_foot
        self.target_index[ids] = 0
        self.was_supported[ids] = False
        self.completed[ids] = 0
        self.attempted[ids] = 0

    def approach_for(self, flat_ground: bool) -> int:
        return 0 if flat_ground else self.approach_steps

    def plan_length(self, flat_ground: bool) -> int:
        """Total targets: approach (stairs only) + treads + two platform."""
        return self.approach_for(flat_ground) + self.step_count + 2

    def targets(self, *, first_riser_x: torch.Tensor, tread_depth: torch.Tensor,
                riser_height: torch.Tensor, centre_y: torch.Tensor,
                foot_separation: torch.Tensor, flat_ground: bool = False):
        return planned_target_batch(
            first_riser_x=first_riser_x, tread_depth=tread_depth,
            riser_height=riser_height, step_count=self.step_count,
            first_foot=self.first_foot, target_index=self.target_index,
            centre_y=centre_y, foot_separation=foot_separation,
            flat_ground=flat_ground,
            approach_steps=self.approach_for(flat_ground),
        )

    def observation(self, *, targets: torch.Tensor, feet: torch.Tensor,
                    valid: torch.Tensor, root_xyz: torch.Tensor,
                    root_yaw: torch.Tensor, position_scale: float):
        return encode_next_two(targets=targets, feet=feet, valid=valid,
                               root_xyz=root_xyz, root_yaw=root_yaw,
                               position_scale=position_scale)

    def update(self, *, targets: torch.Tensor, feet: torch.Tensor,
               valid: torch.Tensor, sole_xyz: torch.Tensor,
               supported: torch.Tensor, xy_tolerance: float,
               z_tolerance: float) -> torch.Tensor:
        """Advance only at newly supported target-foot landing within XYZ tolerance.

        Caller is responsible for contact semantics, course frame, collision
        validity and optionally sustained-support gating. Returns [N] advances.
        """
        n = self.num_envs
        if (targets.shape != (n, 2, 4) or feet.shape != (n, 2)
                or valid.shape != (n, 2) or sole_xyz.shape != (n, 2, 3)
                or supported.shape != (n, 2) or supported.dtype != torch.bool):
            raise ValueError("invalid update tensor shapes or support dtype")
        if (not 0 < xy_tolerance or not 0 < z_tolerance):
            raise ValueError("tolerances must be positive")
        if any(x.device != self.device for x in (targets, feet, valid, sole_xyz, supported)):
            raise ValueError("update tensors must be on runtime device")
        rising = supported & ~self.was_supported
        target_foot = feet[:, 0].long()
        row = torch.arange(n, device=self.device)
        observed = sole_xyz[row, target_foot]
        delta = observed - targets[:, 0, :3]
        close_xy = torch.linalg.vector_norm(delta[:, :2], dim=-1) <= xy_tolerance
        close_z = delta[:, 2].abs() <= z_tolerance
        candidate = rising[row, target_foot] & valid[:, 0]
        success = candidate & close_xy & close_z
        self.attempted += candidate.long()
        self.completed += success.long()
        self.target_index += success.long()
        self.was_supported.copy_(supported)
        return success


def evaluate_footstep_sequence(
    *, required_feet, landings, destination_verified=None,
    terminal_failure=None,
):
    """Assess externally measured landings, without inferring physics.

    Each landing supplies target_index, foot, placement_valid,
    support_valid, slip_valid and clearance_valid.
    Values must be actual evaluator measurements: True/False/None.
    None denotes missing evidence, not failure or success.

    No contact, persistence or collision measurement is made here.
    """
    if not required_feet or any(f not in (0, 1) for f in required_feet):
        raise ValueError("invalid required foot sequence")
    if any(a == b for a, b in zip(required_feet, required_feet[1:])):
        raise ValueError("required foot sequence must alternate")
    if destination_verified not in (None, True, False):
        raise ValueError("invalid destination evidence")
    if terminal_failure is not None and (
        not isinstance(terminal_failure, str) or not terminal_failure
    ):
        raise ValueError("invalid terminal failure")

    required = (
        "placement_valid", "support_valid",
        "slip_valid", "clearance_valid",
    )
    verified = 0
    reason = terminal_failure
    missing = False

    for event in landings:
        if not isinstance(event, dict) or any(k not in event for k in (
            "target_index", "foot", *required
        )):
            raise ValueError("incomplete landing event")
        if any(event[k] not in (None, True, False) for k in required):
            raise ValueError("invalid physical evidence")
        if not isinstance(event["target_index"], int) or (
            type(event["foot"]) is not int or event["foot"] not in (0, 1)
        ):
            raise ValueError("invalid landing identity")

        if reason is not None or missing:
            break
        if verified >= len(required_feet):
            reason = "extra_landing"
            break
        if event["target_index"] != verified:
            reason = "wrong_target_order"
            break
        if event["foot"] != required_feet[verified]:
            reason = "wrong_foot"
            break
        if any(event[k] is False for k in required):
            reason = next(k for k in required if event[k] is False)
            break
        if any(event[k] is None for k in required):
            missing = True
            break
        verified += 1

    if reason is not None:
        outcome = "FAILURE"
    elif missing or verified < len(required_feet) or destination_verified is None:
        outcome = "NOT_VERIFIED"
    elif destination_verified is False:
        outcome = "FAILURE"
        reason = "destination_invalid"
    else:
        outcome = "SUCCESS"

    return {
        "contract": "planned_footsteps_v1",
        "outcome": outcome,
        "verified_landings": verified,
        "required_landings": len(required_feet),
        "failure_reason": reason,
    }
