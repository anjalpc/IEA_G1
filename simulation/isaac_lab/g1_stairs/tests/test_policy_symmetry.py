import unittest

import torch

from simulation.isaac_lab.g1_stairs.footstep_symmetry import (
    POLICY_WIDTH, mirror_actions, mirror_policy_obs)
from simulation.isaac_lab.g1_stairs.policy_shaping import PolicyShaping
from simulation.isaac_lab.g1_stairs.policy_symmetry import (
    apply_from_config, make_actor_symmetric)


class _Policy(torch.nn.Module):
    """Minimal stand-in for rsl-rl 3.0.1 ActorCritic (actor + scalar std)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.actor = torch.nn.Sequential(torch.nn.Linear(POLICY_WIDTH, 64), torch.nn.ELU(), torch.nn.Linear(64, 29))
        with torch.no_grad():
            self.actor[2].bias += 3.0 * torch.randn(29)    # some outputs far beyond the clip, lopsided
        self.std = torch.nn.Parameter(0.3 * torch.ones(29))
        self.actor_obs_normalization = False
        self.distribution = None

    def update_distribution(self, obs):
        mean = self.actor(obs)
        self.distribution = torch.distributions.Normal(mean, self.std.expand_as(mean))

    def act_inference(self, obs):
        return self.actor(obs)


CONFIG = {"ppo": {"symmetric_actor": True}, "balance_residual_control": {"residual_clip": 2.0}}


class SymmetricActorTests(unittest.TestCase):
    def test_exactly_equivariant_and_inside_clip(self):
        p = _Policy(); apply_from_config(p, CONFIG)
        s = torch.randn(16, POLICY_WIDTH)
        with torch.no_grad():
            a, a_m = p.actor(s), p.actor(mirror_policy_obs(s))
        torch.testing.assert_close(a_m, mirror_actions(a), atol=1e-5, rtol=0)
        self.assertLessEqual(float(a.abs().max()), 2.0 + 1e-6)

    def test_matches_replay_average(self):
        # record_view.py --symmetry average: (clip(pi(s)) + M clip(pi(M s))) / 2
        p = _Policy(); s = torch.randn(8, POLICY_WIDTH)
        with torch.no_grad():
            raw, raw_m = p.actor(s).clamp(-2, 2), mirror_actions(p.actor(mirror_policy_obs(s)).clamp(-2, 2))
            make_actor_symmetric(p, 2.0)
            torch.testing.assert_close(p.actor(s), 0.5 * (raw + raw_m))

    def test_single_observation_and_idempotent(self):
        p = _Policy(); make_actor_symmetric(p, 2.0); make_actor_symmetric(p, 2.0)
        s = torch.randn(POLICY_WIDTH)
        with torch.no_grad():
            self.assertEqual(tuple(p.actor(s).shape), (29,))
            torch.testing.assert_close(p.actor(s), p.actor(s[None])[0])

    def test_checkpoint_layout_unchanged(self):
        p = _Policy(); keys = list(p.state_dict()); apply_from_config(p, CONFIG)
        self.assertEqual(list(p.state_dict()), keys)

    def test_off_unless_configured(self):
        p = _Policy()
        self.assertFalse(apply_from_config(p, {"ppo": {}}))
        self.assertFalse(apply_from_config(p, None))
        self.assertFalse(getattr(p.actor, "_g1_symmetric", False))

    def test_refuses_obs_normalisation(self):
        p = _Policy(); p.actor_obs_normalization = True
        with self.assertRaises(ValueError):
            make_actor_symmetric(p, 2.0)

    def test_gradient_reaches_weights(self):
        p = _Policy(); make_actor_symmetric(p, 2.0)
        p.actor(torch.randn(4, POLICY_WIDTH)).sum().backward()
        self.assertGreater(float(p.actor[0].weight.grad.abs().sum()), 0.0)


class SymmetricBoundTests(unittest.TestCase):
    def test_bound_pulls_raw_outputs_beyond_clip(self):
        p = _Policy()
        shaping = PolicyShaping(p, residual_clip=2.0, bound_coeff=0.5, symmetric=True)
        make_actor_symmetric(p, 2.0, raw_hook=shaping.raw_hook)
        s = torch.randn(32, POLICY_WIDTH)
        bias = p.actor[2].bias
        beyond = (bias.detach().abs() > 4.0)               # saturated for any state
        p.update_distribution(s)
        p.distribution.mean.sum().mul(0.0).backward()       # task gradient zero: only the bound remains
        g = bias.grad
        self.assertTrue(bool((g[beyond] * torch.sign(bias.detach()[beyond]) > 0).all()))  # descent moves them inward

    def test_hook_inactive_without_grad(self):
        p = _Policy(); calls = []
        make_actor_symmetric(p, 2.0, raw_hook=lambda r: calls.append(1) or r)
        with torch.inference_mode():
            p.actor(torch.randn(2, POLICY_WIDTH))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
