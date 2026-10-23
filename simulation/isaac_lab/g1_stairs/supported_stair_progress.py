"""Training-only supported alternating stair progression; not strict qualification."""
import torch


class SupportedAscent:
    """One credited, supported new tread per step; alternate feet across treads."""

    def __init__(
        self, num_envs, device, step_count=3, dwell_s=0.10,
        minimum_load_share=0.0, transfer_margin_m=0.0,
        maximum_planar_speed_mps=1000.0,
        maximum_vertical_speed_mps=1000.0,
        maximum_angular_speed_radps=1000.0,
        maximum_foot_speed_mps=1000.0,
        maximum_roll_pitch_rad=0.8,
    ):
        if dwell_s <= 0:
            raise ValueError("dwell_s must be positive")
        if not 0.0 <= minimum_load_share <= 1.0:
            raise ValueError("minimum_load_share must be in [0, 1]")
        if transfer_margin_m < 0:
            raise ValueError("transfer_margin_m must be nonnegative")
        self.step_count = step_count
        self.dwell_s = dwell_s
        self.minimum_load_share = minimum_load_share
        self.transfer_margin_m = transfer_margin_m
        self.maximum_planar_speed_mps = maximum_planar_speed_mps
        self.maximum_vertical_speed_mps = maximum_vertical_speed_mps
        self.maximum_angular_speed_radps = maximum_angular_speed_radps
        self.maximum_foot_speed_mps = maximum_foot_speed_mps
        self.maximum_roll_pitch_rad = maximum_roll_pitch_rad
        self.next_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.last_foot = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.pending_time = torch.zeros((num_envs, 2), device=device)
        self.first = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.second = torch.zeros_like(self.first)
        self.third = torch.zeros_like(self.first)
        self.alternating_complete = torch.zeros_like(self.first)
        self.last_credit = torch.zeros_like(self.first)
        self.sequence_failed = torch.zeros_like(self.first)
        self.last_invalid = torch.zeros_like(self.first)

    def update(
        self, sole_bounds, forces_z, heights, first_riser, tread,
        half_width, dt, active, invalid_contact=None, body_x=None,
        body_velocity=None, body_angular_velocity=None, roll_pitch=None,
        foot_speed=None,
    ):
        """Call once per simulation step, before auto-reset.

        Requires substantial tread overlap, sole-bottom near tread height,
        vertical loading and sustained support. Known riser strikes/bridges
        are explicitly excluded. This is a learning milestone, not a clean
        landing or strict qualification event.
        """
        self.last_credit.zero_()
        self.last_invalid.zero_()
        if sole_bounds is None:
            self.pending_time.zero_()
            return torch.zeros_like(active)
        if invalid_contact is None:
            invalid_contact = torch.zeros_like(active)
        target = self.next_step
        index = target.clamp(max=self.step_count - 1)
        start = first_riser + index.to(heights.dtype) * tread
        end = start + tread
        top = (index + 1).to(heights.dtype) * heights
        load_share = forces_z / forces_z.sum(dim=1, keepdim=True).clamp_min(1.0)
        if self.transfer_margin_m > 0.0:
            if body_x is None or body_x.shape != active.shape:
                raise ValueError(
                    "body_x with shape [num_envs] is required for transfer"
                )
            body_transferred = body_x >= start + self.transfer_margin_m
        else:
            body_transferred = torch.ones_like(active)
        controlled_inputs = (
            body_velocity, body_angular_velocity, roll_pitch, foot_speed
        )
        if any(value is None for value in controlled_inputs) and not all(
            value is None for value in controlled_inputs
        ):
            raise ValueError("controlled-transfer inputs must be supplied together")
        if any(value is None for value in controlled_inputs):
            controlled = torch.ones_like(active)
            supported_foot_slow = torch.ones_like(forces_z, dtype=torch.bool)
        else:
            controlled = (
                (torch.linalg.vector_norm(body_velocity[:, :2], dim=1)
                 <= self.maximum_planar_speed_mps)
                & (body_velocity[:, 2].abs()
                   <= self.maximum_vertical_speed_mps)
                & (torch.linalg.vector_norm(body_angular_velocity, dim=1)
                   <= self.maximum_angular_speed_radps)
                & (roll_pitch.abs() <= self.maximum_roll_pitch_rad).all(dim=1)
            )
            supported_foot_slow = foot_speed <= self.maximum_foot_speed_mps
        eligible = (
            (target < self.step_count)[:, None]
            & active[:, None]
            & ~self.sequence_failed[:, None]
            & ~invalid_contact[:, None]
            & (heights > 0)[:, None]
            & ((torch.minimum(sole_bounds[:, :, 1], end[:, None])
                - torch.maximum(sole_bounds[:, :, 0], start[:, None])) >= 0.10)
            & (sole_bounds[:, :, 2] <= half_width - 0.005)
            & (sole_bounds[:, :, 3] >= -half_width + 0.005)
            & ((sole_bounds[:, :, 4] - top[:, None]).abs() <= 0.025)
            & (sole_bounds[:, :, 5] <= top[:, None] + 0.05)
            & (forces_z >= 30.0)
            & (load_share >= self.minimum_load_share)
            & body_transferred[:, None]
            & controlled[:, None]
            & supported_foot_slow
        )
        self.pending_time = torch.where(eligible, self.pending_time + dt, 0.0)
        ready = self.pending_time >= self.dwell_s
        # No credit for simultaneous acquisition or same-foot acquisition
        # on the next stair. Same-tread corrective loading has no credit.
        unique = ready.sum(dim=1) == 1
        foot = ready.to(torch.long).argmax(dim=1)
        allowed = (self.last_foot < 0) | (foot != self.last_foot)
        attempted = ready.any(dim=1) & active & (target < self.step_count)
        invalid = attempted & (~unique | ~allowed)
        self.last_invalid.copy_(invalid & ~self.sequence_failed)
        self.sequence_failed |= invalid
        credit = (
            unique & allowed & active & ~self.sequence_failed
            & (target < self.step_count)
        )
        self.last_credit.copy_(credit)
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
        self.last_credit[ids] = False
        self.sequence_failed[ids] = False
        self.last_invalid[ids] = False
