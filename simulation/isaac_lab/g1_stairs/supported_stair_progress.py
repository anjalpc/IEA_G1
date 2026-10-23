"""Training-only supported alternating stair progression; not qualification."""
import torch


class SupportedAscent:
    """Track spatial swing, latched touchdown, and supported body transfer."""

    def __init__(
        self, num_envs, device, step_count=3, dwell_s=0.10,
        minimum_load_share=0.0, transfer_margin_m=0.0,
        touchdown_dwell_s=0.04, maximum_roll_pitch_rad=0.8,
        swing_activation_distance_m=0.30,
        swing_clearance_margin_m=0.015, swing_insertion_m=0.055,
    ):
        if dwell_s <= 0 or touchdown_dwell_s <= 0:
            raise ValueError("support and touchdown dwell must be positive")
        if not 0.0 <= minimum_load_share <= 1.0:
            raise ValueError("minimum_load_share must be in [0, 1]")
        if transfer_margin_m < 0:
            raise ValueError("transfer_margin_m must be nonnegative")
        if min(swing_activation_distance_m, swing_clearance_margin_m,
               swing_insertion_m) <= 0:
            raise ValueError("swing geometry parameters must be positive")
        self.step_count = step_count
        self.dwell_s = dwell_s
        self.minimum_load_share = minimum_load_share
        self.transfer_margin_m = transfer_margin_m
        self.touchdown_dwell_s = touchdown_dwell_s
        self.maximum_roll_pitch_rad = maximum_roll_pitch_rad
        self.swing_activation_distance_m = swing_activation_distance_m
        self.swing_clearance_margin_m = swing_clearance_margin_m
        self.swing_insertion_m = swing_insertion_m
        self.latching_enabled = minimum_load_share > 0 or transfer_margin_m > 0

        self.next_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.last_foot = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.pending_time = torch.zeros((num_envs, 2), device=device)
        self.candidate_foot = torch.full_like(self.last_foot, -1)
        self.transfer_time = torch.zeros(num_envs, device=device)
        self.transfer_blocked = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.swing_potential = torch.zeros(num_envs, device=device)
        self.transfer_start_potential = torch.zeros(num_envs, device=device)
        self.last_swing_progress = torch.zeros(num_envs, device=device)
        self.first = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.second = torch.zeros_like(self.first)
        self.third = torch.zeros_like(self.first)
        self.alternating_complete = torch.zeros_like(self.first)
        self.last_credit = torch.zeros_like(self.first)
        self.sequence_failed = torch.zeros_like(self.first)
        self.last_invalid = torch.zeros_like(self.first)

    def _complete(self, credit, foot, target):
        self.last_credit.copy_(credit)
        self.first |= credit & (target == 0)
        self.second |= credit & (target == 1)
        self.third |= credit & (target == 2)
        self.last_foot = torch.where(credit, foot, self.last_foot)
        self.next_step += credit.to(torch.long)
        self.alternating_complete |= credit & (target == self.step_count - 1)
        self.pending_time[credit] = 0.0
        self.candidate_foot[credit] = -1
        self.transfer_time[credit] = 0.0
        self.transfer_blocked[credit] = False
        self.transfer_start_potential[credit] = 0.0

    def update(
        self, sole_bounds, forces_z, heights, first_riser, tread,
        half_width, dt, active, invalid_contact=None, body_x=None,
        roll_pitch=None, **_unused,
    ):
        """Update once before reset; return newly confirmed acquisitions."""
        self.last_credit.zero_()
        self.last_invalid.zero_()
        self.last_swing_progress.zero_()
        if sole_bounds is None:
            self.pending_time.zero_()
            self.candidate_foot.fill_(-1)
            self.transfer_time.zero_()
            self.transfer_blocked.zero_()
            self.swing_potential.zero_()
            self.transfer_start_potential.zero_()
            return torch.zeros_like(active)
        if invalid_contact is None:
            invalid_contact = torch.zeros_like(active)

        target = self.next_step
        index = target.clamp(max=self.step_count - 1)
        start = first_riser + index.to(heights.dtype) * tread
        end = start + tread
        support_top = index.to(heights.dtype) * heights
        target_top = (index + 1).to(heights.dtype) * heights
        in_task = (
            active & ~self.sequence_failed & ~invalid_contact
            & (target < self.step_count) & (heights > 0)
        )
        overlap = (
            torch.minimum(sole_bounds[:, :, 1], end[:, None])
            - torch.maximum(sole_bounds[:, :, 0], start[:, None])
        )
        landing = (
            in_task[:, None] & (overlap >= 0.10)
            & (sole_bounds[:, :, 2] <= half_width - 0.005)
            & (sole_bounds[:, :, 3] >= -half_width + 0.005)
            & ((sole_bounds[:, :, 4] - target_top[:, None]).abs() <= 0.025)
            & (sole_bounds[:, :, 5] <= target_top[:, None] + 0.05)
            & (forces_z >= 30.0)
        )

        # Preserve historical configs exactly; the new state machine is opt-in.
        if not self.latching_enabled:
            self.pending_time = torch.where(landing, self.pending_time + dt, 0.0)
            ready = self.pending_time >= self.dwell_s
            unique = ready.sum(dim=1) == 1
            foot = ready.to(torch.long).argmax(dim=1)
            allowed = (self.last_foot < 0) | (foot != self.last_foot)
            attempted = ready.any(dim=1) & in_task
            invalid = attempted & (~unique | ~allowed)
            self.last_invalid.copy_(invalid & ~self.sequence_failed)
            self.sequence_failed |= invalid
            credit = unique & allowed & in_task & ~self.sequence_failed
            self._complete(credit, foot, target.clone())
            return credit

        if body_x is None or roll_pitch is None:
            raise ValueError("latched transfer requires body_x and roll_pitch")

        expected = torch.ones_like(landing)
        known = self.last_foot >= 0
        feet = torch.arange(2, device=forces_z.device)
        expected[known] = feet[None, :] != self.last_foot[known, None]
        unloaded = forces_z <= 20.0
        opposite_supported = torch.flip(forces_z, dims=(1,)) >= 30.0
        lift = (
            (sole_bounds[:, :, 4] - support_top[:, None])
            / (heights[:, None] + self.swing_clearance_margin_m)
        ).clamp(0.0, 1.0)
        reach = (
            (sole_bounds[:, :, 0]
             - (start[:, None] - self.swing_activation_distance_m))
            / (self.swing_activation_distance_m + self.swing_insertion_m)
        ).clamp(0.0, 1.0)
        swing = torch.sqrt(lift * reach) * expected * unloaded * opposite_supported
        swing = swing.amax(dim=1)
        swing = torch.where(in_task & (self.candidate_foot < 0), swing, 0.0)

        # Touchdown owns this target before body-transfer evaluation begins.
        no_candidate = self.candidate_foot < 0
        self.pending_time = torch.where(
            landing & no_candidate[:, None], self.pending_time + dt, 0.0
        )
        ready = self.pending_time >= self.touchdown_dwell_s
        unique = ready.sum(dim=1) == 1
        foot = ready.to(torch.long).argmax(dim=1)
        allowed = (self.last_foot < 0) | (foot != self.last_foot)
        attempted = ready.any(dim=1) & no_candidate & in_task
        invalid = attempted & (~unique | ~allowed)
        latch = attempted & unique & allowed
        self.candidate_foot = torch.where(latch, foot, self.candidate_foot)
        self.transfer_start_potential = torch.where(
            latch, swing, self.transfer_start_potential
        )
        self.pending_time[latch] = 0.0

        rows = torch.arange(active.shape[0], device=active.device)
        candidate = self.candidate_foot >= 0
        candidate_index = self.candidate_foot.clamp_min(0)
        candidate_supported = candidate & landing[rows, candidate_index]
        other = 1 - candidate_index
        trailing_on_target = candidate & landing[rows, other]
        invalid |= trailing_on_target & ~self.transfer_blocked
        self.transfer_blocked.copy_(trailing_on_target)
        self.last_invalid.copy_(invalid & ~self.sequence_failed)
        self.sequence_failed |= invalid & ~trailing_on_target

        lost = candidate & (~candidate_supported | ~in_task)
        self.candidate_foot[lost] = -1
        self.transfer_time[lost] = 0.0
        self.transfer_blocked[lost] = False
        self.transfer_start_potential[lost] = 0.0

        candidate = self.candidate_foot >= 0
        candidate_index = self.candidate_foot.clamp_min(0)
        total_force = forces_z.sum(dim=1).clamp_min(1.0)
        candidate_share = forces_z[rows, candidate_index] / total_force
        maximum_tilt = roll_pitch.abs().amax(dim=1)
        upright = maximum_tilt <= self.maximum_roll_pitch_rad
        load_progress = (
            (candidate_share - 0.15)
            / max(self.minimum_load_share - 0.15, 1.0e-6)
        ).clamp(0.0, 1.0)
        body_progress = (
            (body_x - (start - 0.10))
            / (self.transfer_margin_m + 0.10)
        ).clamp(0.0, 1.0)
        upright_progress = (
            1.0 - maximum_tilt / self.maximum_roll_pitch_rad
        ).clamp(0.0, 1.0)
        transfer_quality = torch.pow(
            load_progress * body_progress * upright_progress, 1.0 / 3.0
        )
        transfer_potential = self.transfer_start_potential + (
            1.0 - self.transfer_start_potential
        ) * transfer_quality
        potential = torch.where(candidate & candidate_supported, transfer_potential, swing)
        potential = torch.where(in_task & ~self.sequence_failed, potential, 0.0)
        self.last_swing_progress.copy_(potential - self.swing_potential)
        self.swing_potential.copy_(potential)
        transfer = (
            candidate & ~self.sequence_failed & ~self.transfer_blocked & in_task
            & (candidate_share >= self.minimum_load_share)
            & (body_x >= start + self.transfer_margin_m) & upright
        )
        self.transfer_time = torch.where(transfer, self.transfer_time + dt, 0.0)
        credit = transfer & (self.transfer_time + 1.0e-6 >= self.dwell_s)
        self._complete(credit, candidate_index, target.clone())
        return credit

    def reset(self, ids):
        self.next_step[ids] = 0
        self.last_foot[ids] = -1
        self.pending_time[ids] = 0.0
        self.candidate_foot[ids] = -1
        self.transfer_time[ids] = 0.0
        self.transfer_blocked[ids] = False
        self.swing_potential[ids] = 0.0
        self.transfer_start_potential[ids] = 0.0
        self.last_swing_progress[ids] = 0.0
        self.first[ids] = False
        self.second[ids] = False
        self.third[ids] = False
        self.alternating_complete[ids] = False
        self.last_credit[ids] = False
        self.sequence_failed[ids] = False
        self.last_invalid[ids] = False
