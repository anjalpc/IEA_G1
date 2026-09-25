"""Project-owned RSL-RL PPO extension for actor-only approach preservation."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from .approach_teacher import (
    control_aware_actor_mean,
    gated_teacher_loss,
    smooth_l1_per_environment,
    teacher_diagnostic_terms,
    teacher_gate,
    teacher_observation,
)


class ApproachTeacherPPO(PPO):
    """RSL-RL 3.0.1 PPO with one gated actor-mean auxiliary loss."""

    def configure_approach_teacher(self, teacher, config) -> None:
        self._approach_teacher = teacher
        self._approach_teacher_cfg = config

    def update(self):  # noqa: C901
        if not hasattr(self, "_approach_teacher"):
            return super().update()
        if self.policy.is_recurrent or self.rnd or self.symmetry or self.is_multi_gpu:
            raise RuntimeError(
                "approach teacher currently requires feed-forward, single-GPU "
                "PPO without RND or symmetry"
            )

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_teacher_loss = 0.0
        mean_teacher_error = 0.0
        mean_teacher_gate = 0.0

        teacher_phase_error_sum = {
            "far": 0.0,
            "release": 0.0,
            "near": 0.0,
        }

        teacher_phase_count = {
            "far": 0,
            "release": 0,
            "near": 0,
        }

        teacher_group_error_sum = {
            "legs": 0.0,
            "arms": 0.0,
            "waist": 0.0,
        }

        teacher_group_count = {
            "legs": 0,
            "arms": 0,
            "waist": 0,
        }

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        advantages_batch - advantages_batch.mean()
                    ) / (advantages_batch.std() + 1.0e-8)

            self.policy.act(
                obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.policy.evaluate(
                obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            student_mean = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - student_mean)
                        ) / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            1.0e-5, self.learning_rate / 1.5
                        )
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(
                            1.0e-2, self.learning_rate * 1.5
                        )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_values = torch.max(
                surrogate,
                surrogate_clipped,
            )

            stored_gate = teacher_gate(obs_batch)
            # The student action has no physical authority where the teacher
            # gate is one. 047 therefore removes task-policy and entropy
            # gradients from those samples. Historical modes keep the exact
            # unweighted PPO objective through the opt-in flag.
            surrogate_loss = control_aware_actor_mean(
                surrogate_values,
                stored_gate,
                enabled=(
                    self._approach_teacher_cfg
                    .gate_task_actor_loss
                ),
            )
            entropy_loss = control_aware_actor_mean(
                entropy_batch,
                stored_gate,
                enabled=(
                    self._approach_teacher_cfg
                    .gate_task_actor_loss
                ),
            )

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(
                    value_losses,
                    value_losses_clipped,
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            stored_teacher_obs = teacher_observation(obs_batch)
            with torch.no_grad():
                teacher_action = self._approach_teacher(stored_teacher_obs)
            teacher_loss = gated_teacher_loss(
                student_mean,
                teacher_action,
                stored_gate,
                self._approach_teacher_cfg.huber_delta,
            )
            teacher_error = smooth_l1_per_environment(
                student_mean,
                teacher_action,
                self._approach_teacher_cfg.huber_delta,
            ).mean()

            # Read-only diagnostics. Detach explicitly so these
            # measurements cannot affect the PPO/teacher gradient path.
            with torch.no_grad():
                teacher_diag = teacher_diagnostic_terms(
                    student_mean.detach(),
                    teacher_action,
                    stored_gate,
                    self._approach_teacher_cfg.huber_delta,
                )

                for phase in (
                    "far",
                    "release",
                    "near",
                ):
                    mask = teacher_diag[
                        f"{phase}_mask"
                    ]

                    count = int(
                        torch.count_nonzero(
                            mask
                        ).item()
                    )

                    teacher_phase_count[
                        phase
                    ] += count

                    if count:
                        teacher_phase_error_sum[
                            phase
                        ] += float(
                            teacher_diag[
                                "error"
                            ][mask].sum().item()
                        )

                for group in (
                    "legs",
                    "arms",
                    "waist",
                ):
                    values = teacher_diag[
                        f"{group}_error"
                    ]

                    teacher_group_error_sum[
                        group
                    ] += float(
                        values.sum().item()
                    )

                    teacher_group_count[
                        group
                    ] += int(
                        values.numel()
                    )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_loss
                + self._approach_teacher_cfg.coefficient * teacher_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_teacher_loss += teacher_loss.item()
            mean_teacher_error += teacher_error.item()
            mean_teacher_gate += stored_gate.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches

        result = {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "approach_teacher": mean_teacher_loss / num_updates,
            "approach_teacher_error": mean_teacher_error / num_updates,
            "approach_teacher_gate": mean_teacher_gate / num_updates,
        }

        total_phase_count = sum(
            teacher_phase_count.values()
        )

        if total_phase_count:
            for phase in (
                "far",
                "release",
                "near",
            ):
                count = teacher_phase_count[
                    phase
                ]

                result[
                    f"approach_teacher_{phase}_fraction"
                ] = (
                    count
                    / total_phase_count
                )

                if count:
                    result[
                        f"approach_teacher_error_{phase}"
                    ] = (
                        teacher_phase_error_sum[
                            phase
                        ]
                        / count
                    )

        for group in (
            "legs",
            "arms",
            "waist",
        ):
            count = teacher_group_count[
                group
            ]

            if count:
                result[
                    f"approach_teacher_error_{group}"
                ] = (
                    teacher_group_error_sum[
                        group
                    ]
                    / count
                )

        self.storage.clear()

        return result
