"""Training-only supported alternating stair progression; not strict qualification."""
import torch


class SupportedAscent:
    """One credited, supported new tread per step; alternate feet across treads."""

    def __init__(self, num_envs, device, step_count=3, dwell_s=0.10):
        self.step_count = step_count
        self.dwell_s = dwell_s
        self.next_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.last_foot = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.pending_time = torch.zeros((num_envs, 2), device=device)
        self.first = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.second = torch.zeros_like(self.first)
        self.third = torch.zeros_like(self.first)
        self.alternating_complete = torch.zeros_like(self.first)

    def update(self, sole_bounds, forces_z, heights, first_riser, tread, half_width, dt, active):
        """Call once per simulation step, before auto-reset.

        Requires full-sole X/Y containment, both sole surfaces at the tread
        elevation, upward foot contact, and sustained support. Neither a
        riser-face strike nor unsupported base passage can earn credit.
        """
        n = sole_bounds.shape[0]
        target = self.next_step
        index = target.clamp(max=self.step_count - 1)
        start = first_riser + index.to(heights.dtype) * tread
        end = start + tread
        top = (index + 1).to(heights.dtype) * heights
        eligible = (
            (target < self.step_count)[:, None]
            & active[:, None]
            & (heights > 0)[:, None]
            & (sole_bounds[:, :, 0] >= start[:, None] + 0.005)
            & (sole_bounds[:, :, 1] <= end[:, None] - 0.005)
            & (sole_bounds[:, :, 2] >= -half_width + 0.005)
            & (sole_bounds[:, :, 3] <= half_width - 0.005)
            & ((sole_bounds[:, :, 4] - top[:, None]).abs() <= 0.025)
            & ((sole_bounds[:, :, 5] - top[:, None]).abs() <= 0.025)
            & (forces_z >= 30.0)
        )
        self.pending_time = torch.where(eligible, self.pending_time + dt, 0.0)
        ready = self.pending_time >= self.dwell_s
        # No credit for simultaneous acquisition or same-foot acquisition
        # on the next stair. Same-tread corrective loading has no credit.
        unique = ready.sum(dim=1) == 1
        foot = ready.to(torch.long).argmax(dim=1)
        allowed = (self.last_foot < 0) | (foot != self.last_foot)
        credit = unique & allowed & active & (target < self.step_count)
        old = target.clone()
        self.first |= credit & (old == 0)
        self.second |= credit & (old == 1)
        self.third |= credit & (old == 2)
        self.last_foot = torch.where(credit, foot, self.last_foot)
        self.next_step += credit.to(torch.long)
        self.alternating_complete |= credit & (old == self.step_count - 1)
        self.pending_time[credit] = 0.0
        return credit

    def reset(self, ids):
        self.next_step[ids] = 0
        self.last_foot[ids] = -1
        self.pending_time[ids] = 0.0
        self.first[ids] = False
        self.second[ids] = False
        self.third[ids] = False
        self.alternating_complete[ids] = False
