import unittest

import torch

from simulation.isaac_lab.g1_stairs.policy_shaping import PolicyShaping, bound_gradient, bound_loss

try:
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict
except Exception:  # simulator-free environments without rsl-rl
    ActorCritic = None


@unittest.skipIf(ActorCritic is None, "rsl-rl-lib not installed")
class PolicyShapingTests(unittest.TestCase):
    def make(self, std=0.02, bias=None):
        torch.manual_seed(0)
        obs = TensorDict({"policy": torch.randn(8, 6)}, batch_size=[8])
        policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 3,
                             actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=std)
        if bias is not None:
            with torch.no_grad():
                policy.actor[-1].bias.copy_(torch.tensor(bias))
        return policy, obs

    def test_forward_unchanged_and_std_floored(self):
        policy, obs = self.make(std=0.02)
        policy.act(obs)
        mean0 = policy.action_mean.detach().clone()
        PolicyShaping(policy, residual_clip=2.0, bound_coeff=0.01, std_floor=0.1)
        policy.act(obs)
        self.assertTrue(torch.equal(policy.action_mean.detach(), mean0))
        self.assertTrue(torch.allclose(policy.action_std, torch.full_like(policy.action_std, 0.1)))
        self.assertTrue(torch.equal(policy.act_inference(obs), mean0))   # deployment path untouched
        self.assertAlmostEqual(float(policy.std.detach()[0]), 0.02, places=6)       # parameter unchanged

    def test_bound_gradient_equals_explicit_loss(self):
        policy, obs = self.make(std=0.5, bias=[5.0, -4.0, 0.0])
        shaping = PolicyShaping(policy, residual_clip=2.0, bound_coeff=0.01)
        policy.act(obs)
        actions = policy.distribution.sample()
        loss = -policy.get_actions_log_prob(actions).mean()
        loss.backward()
        hooked = policy.actor[-1].bias.grad.clone()
        policy.zero_grad()
        # same loss with the bound written out explicitly, no shaping
        policy.update_distribution = shaping._original
        policy.act(obs)
        explicit = -policy.get_actions_log_prob(actions).mean() + bound_loss(policy.action_mean, 2.0, 0.01)
        explicit.backward()
        self.assertTrue(torch.allclose(hooked, policy.actor[-1].bias.grad, atol=1e-6))
        # the bound's own part pulls inward: + on the +5 output, - on the -4
        # one, nothing on the output inside the clip (gradient descent)
        policy.zero_grad()
        policy.act(obs)
        (-policy.get_actions_log_prob(actions).mean()).backward()
        extra = hooked - policy.actor[-1].bias.grad
        self.assertTrue(torch.allclose(extra, torch.tensor([0.01, -0.01, 0.0]), atol=1e-6))

    def test_no_pull_inside_the_clip(self):
        policy, obs = self.make(std=0.5, bias=[0.5, -0.5, 0.0])
        policy.act(obs)
        a = policy.distribution.sample()
        (-policy.get_actions_log_prob(a).mean()).backward()
        plain = policy.actor[-1].bias.grad.clone()
        policy.zero_grad()
        PolicyShaping(policy, residual_clip=2.0, bound_coeff=1.0)
        policy.act(obs)
        (-policy.get_actions_log_prob(a).mean()).backward()
        self.assertTrue(torch.allclose(plain, policy.actor[-1].bias.grad, atol=1e-7))

    def test_paired_bound_scales_the_smaller_side(self):
        # outputs 0/1 mirror onto each other, output 2 onto itself
        swap = lambda m: m[..., [1, 0, 2]] * torch.tensor([-1.0, -1.0, 1.0])
        policy, obs = self.make(std=0.5, bias=[5.0, -3.0, 6.0])
        with torch.no_grad():
            policy.actor[-1].weight.zero_()                # means = bias exactly
        policy.act(obs)
        a = policy.distribution.sample()
        (-policy.get_actions_log_prob(a).mean()).backward()
        plain = policy.actor[-1].bias.grad.clone()
        policy.zero_grad()
        PolicyShaping(policy, residual_clip=2.0, bound_coeff=0.01, partner=swap)
        policy.act(obs)
        (-policy.get_actions_log_prob(a).mean()).backward()
        extra = policy.actor[-1].bias.grad - plain
        # excess 3 (full pull), excess 1 (1/3 of it), self-mirrored waist-like joint: plain L1
        self.assertTrue(torch.allclose(extra, torch.tensor([0.01, -0.01 / 3, 0.01]), atol=1e-6))

    def test_paired_bound_brings_a_pair_to_the_clip_together(self):
        swap = lambda m: m[..., [1, 0]] * -1.0
        def crossings(partner):
            x = torch.tensor([[5.0, -3.0]])                 # left excess 3, right excess 1
            hit = [None, None]
            for step in range(1, 1000):
                x = x - 0.5 * bound_gradient(x, 2.0, 0.01, partner)
                for j in (0, 1):
                    if hit[j] is None and float(x[0, j].abs()) <= 2.0 + 1e-6:
                        hit[j] = step
            return hit
        plain, paired = crossings(None), crossings(swap)
        self.assertGreater(plain[0] - plain[1], 300)        # plain L1: right side free 400 steps early
        self.assertLessEqual(abs(paired[0] - paired[1]), 1)  # paired: together
        self.assertLessEqual(max(paired), max(plain) + 1)   # and no slower

    def test_paired_bound_without_partner_excess_is_plain(self):
        swap = lambda m: m[..., [1, 0]] * -1.0
        x = torch.tensor([[4.0, 0.5], [-1.0, -2.5]])
        self.assertTrue(torch.allclose(bound_gradient(x, 2.0, 0.1, swap), bound_gradient(x, 2.0, 0.1)))

    def test_rejects_bad_settings(self):
        policy, _ = self.make()
        with self.assertRaises(ValueError):
            PolicyShaping(policy, residual_clip=2.0, bound_coeff=-1.0)


if __name__ == "__main__":
    unittest.main()
