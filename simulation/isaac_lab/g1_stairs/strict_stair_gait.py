"""Reward-independent physical contract for alternating stair gait."""

from __future__ import annotations

import torch


class StrictStairGait:
    """Require one alternating, fully supported landing per tread.

    The contract uses the same physical sole bounds as the geometry evaluator.
    It intentionally does not grant reward; it only records physical validity.
    """

    TREAD_MARGIN_M = 0.025
    LANE_TOLERANCE_M = 0.050
    CLEARANCE_M = 0.015
    UNLOAD_FORCE_N = 15.0
    SUPPORT_FORCE_N = 30.0
    TOUCHDOWN_SPEED_MPS = 0.20
    MAX_TILT_RAD = 0.15
    MAX_YAW_RAD = 0.20
    SUPPORT_DWELL_S = 0.10

    # Diagnostic failure codes.
    NONE = 0
    RISER_STRIKE = 1
    EDGE_BRIDGE = 2
    BAD_LANDING = 3
    LOST_SUPPORT = 4

    def __init__(
        self,
        n: int,
        device,
        *,
        first_riser: float,
        tread: float,
        steps: int,
        half_width: float,
        dt: float,
    ):
        self.first = float(first_riser)
        self.tread = float(tread)
        self.steps = int(steps)
        self.width = float(half_width)
        self.dt = float(dt)

        if self.steps < 1 or self.tread <= 0.0 or self.dt <= 0.0:
            raise ValueError("strict gait requires positive geometry and timestep")
        # G1 sole is ~0.18 m long. The strict contract needs 2.5 cm margins.
        if self.tread <= 0.18 + 2.0 * self.TREAD_MARGIN_M:
            raise ValueError("strict gait tread is too short for sole plus margins")

        self.count = torch.zeros(n, dtype=torch.long, device=device)
        self.last_foot = torch.full_like(self.count, -1)
        self.failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.reason = torch.zeros_like(self.count)
        self.unloaded = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.support_valid = torch.ones_like(self.unloaded)
        self.cleared = torch.zeros_like(self.unloaded)
        self.pending = torch.full_like(self.count, -1)
        self.dwell = torch.zeros(n, device=device)
        self.lane = torch.zeros(n, 2, device=device)
        self.lane_initialized = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids):
        for value in (
            self.count,
            self.failed,
            self.reason,
            self.unloaded,
            self.cleared,
            self.dwell,
            self.lane,
            self.lane_initialized,
        ):
            value[ids] = 0
        self.support_valid[ids] = True
        self.last_foot[ids] = -1
        self.pending[ids] = -1

    def update(self, bounds, force, speed, tilt, yaw, height):
        """Return ``(violation, complete)`` for each environment."""
        if bounds.ndim != 3 or bounds.shape[1:] != (2, 6):
            raise ValueError("bounds must have shape [N, 2, 6]")
        if force.shape[:2] != bounds.shape[:2] or force.shape[-1] != 3:
            raise ValueError("force must have shape [N, 2, 3]")
        for value, name in ((speed, "speed"), (tilt, "tilt"), (yaw, "yaw")):
            if value.shape != bounds.shape[:2]:
                raise ValueError(f"{name} must have shape [N, 2]")
        if height.shape != (bounds.shape[0],):
            raise ValueError("height must have shape [N]")

        active = (height > 0) & (self.count < self.steps) & ~self.failed
        rows = torch.arange(len(height), device=height.device)
        center_y = 0.5 * (bounds[..., 2] + bounds[..., 3])
        init = ~self.lane_initialized
        self.lane.copy_(torch.where(init[:, None], center_y, self.lane))
        self.lane_initialized |= active | (height <= 0)

        start = self.first + self.count.to(height.dtype) * self.tread
        end = start + self.tread  # third tread stays a tread; platform is later.
        top = (self.count + 1).to(height.dtype) * height

        fz = force[..., 2].clamp_min(0)
        loaded = fz >= self.SUPPORT_FORCE_N
        unloaded_now = fz <= self.UNLOAD_FORCE_N
        opposite_loaded = loaded.flip(1)

        beginning_unload = unloaded_now & ~self.unloaded & active[:, None]
        self.support_valid.copy_(torch.where(
            beginning_unload,
            opposite_loaded,
            self.support_valid,
        ))
        continuing_swing = self.unloaded & unloaded_now & active[:, None]
        self.support_valid.copy_(torch.where(
            continuing_swing,
            self.support_valid & opposite_loaded,
            self.support_valid,
        ))
        self.unloaded |= unloaded_now & active[:, None]

        # A valid swing must have whole-sole vertical clearance while the
        # opposite foot remains load-bearing.
        self.cleared |= (
            (bounds[..., 4] >= top[:, None] + self.CLEARANCE_M)
            & (bounds[..., 1] >= start[:, None] - 0.05)
            & unloaded_now
            & self.support_valid
            & active[:, None]
        )

        fresh = self.unloaded & loaded
        height_tolerance = torch.minimum(
            height * 0.20,
            torch.full_like(height, 0.01),
        )
        center_x = 0.5 * (bounds[..., 0] + bounds[..., 1])
        tread_center = 0.5 * (start + end)
        good = (
            (bounds[..., 0] >= start[:, None] + self.TREAD_MARGIN_M)
            & (bounds[..., 1] <= end[:, None] - self.TREAD_MARGIN_M)
            & (bounds[..., 2] >= -self.width + self.TREAD_MARGIN_M)
            & (bounds[..., 3] <= self.width - self.TREAD_MARGIN_M)
            & ((center_x - tread_center[:, None]).abs() <= 0.010)
            & ((center_y - self.lane).abs() <= self.LANE_TOLERANCE_M)
            & ((bounds[..., 4] - top[:, None]).abs() <= height_tolerance[:, None])
            & ((bounds[..., 5] - top[:, None]).abs() <= height_tolerance[:, None])
            & (speed <= self.TOUCHDOWN_SPEED_MPS)
            & (tilt <= self.MAX_TILT_RAD)
            & (yaw.abs() <= self.MAX_YAW_RAD)
            & loaded
        )

        # Loaded contact while the sole spans a height discontinuity is the
        # heel-on-old/toe-on-new bridge observed in C6.4. Horizontal force on
        # that span is tracked separately as a conservative riser strike.
        edge_x = self.first + torch.arange(
            self.steps, device=height.device, dtype=height.dtype
        ) * self.tread
        edge_top = height[:, None] * (
            torch.arange(self.steps, device=height.device, dtype=height.dtype) + 1
        )
        spans = (
            (bounds[..., 0, None] <= edge_x)
            & (bounds[..., 1, None] >= edge_x)
            & (bounds[..., 4, None] < edge_top[:, None] - 0.005)
            & (bounds[..., 5, None] > edge_top[:, None] + 0.005)
        )
        bridge = (spans.any(-1) & loaded & active[:, None]).any(-1)
        horizontal_force = force[..., :2].norm(dim=-1)
        strike = (
            spans.any(-1)
            & (horizontal_force >= self.SUPPORT_FORCE_N)
            & active[:, None]
        ).any(-1)

        # Ignore normal flat approach contacts. Once a foot reaches the first
        # stair region, every fresh landing must acquire exactly the expected
        # tread with the opposite foot from the previous acquisition.
        relevant = fresh & (
            (self.count[:, None] > 0) | (bounds[..., 1] >= self.first)
        ) & active[:, None]
        number = relevant.sum(-1)
        foot = relevant.long().argmax(-1)
        expected = (self.last_foot < 0) | (foot != self.last_foot)
        valid_landing = (
            (number == 1)
            & expected
            & good[rows, foot]
            & self.cleared[rows, foot]
            & self.support_valid[rows, foot]
        )
        bad = (number > 0) & ~valid_landing
        bad |= (number > 0) & (self.pending >= 0)

        begin = valid_landing & (self.pending < 0)
        self.pending = torch.where(begin, foot, self.pending)
        pending_idx = self.pending.clamp_min(0)
        pending_good = good[rows, pending_idx]
        lost = (self.pending >= 0) & ~pending_good

        failure = active & (strike | bridge | bad | lost)
        failure_reason = torch.where(
            strike,
            torch.full_like(self.reason, self.RISER_STRIKE),
            torch.where(
                bridge,
                torch.full_like(self.reason, self.EDGE_BRIDGE),
                torch.where(
                    bad,
                    torch.full_like(self.reason, self.BAD_LANDING),
                    torch.full_like(self.reason, self.LOST_SUPPORT),
                ),
            ),
        )
        self.reason = torch.where(failure, failure_reason, self.reason)
        self.failed |= failure

        self.dwell = torch.where(
            active & ~self.failed & (self.pending >= 0) & pending_good,
            self.dwell + self.dt,
            torch.zeros_like(self.dwell),
        )
        acquired = active & ~self.failed & (
            self.dwell >= self.SUPPORT_DWELL_S - 1e-6
        )
        self.last_foot = torch.where(acquired, self.pending, self.last_foot)
        self.count += acquired.long()
        self.pending = torch.where(acquired, -1, self.pending)
        self.dwell = torch.where(acquired, 0.0, self.dwell)
        self.cleared &= ~acquired[:, None]

        # A completed landing ends that foot's swing state. Failed rows are
        # latched and never recycled into a later "success".
        reset_swing = fresh | acquired[:, None]
        self.unloaded &= ~reset_swing
        self.support_valid.copy_(torch.where(
            reset_swing,
            torch.ones_like(self.support_valid),
            self.support_valid,
        ))

        complete = (height <= 0) | ((self.count == self.steps) & ~self.failed)
        return self.failed, complete
