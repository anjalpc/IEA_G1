import unittest

import torch

from simulation.isaac_lab.g1_stairs.policy_shaping import PolicyShaping, bound_loss

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

    def test_rejects_bad_settings(self):
        policy, _ = self.make()
        with self.assertRaises(ValueError):
            PolicyShaping(policy, residual_clip=2.0, bound_coeff=-1.0)


if __name__ == "__main__":
    unittest.main()
