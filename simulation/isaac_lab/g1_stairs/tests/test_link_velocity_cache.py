"""LinkVelocity (training metrics) equals Isaac Lab's body_link_lin_vel_w."""
import types
import unittest

import torch

from simulation.isaac_lab.g1_stairs.evaluation import isaac_training_metrics as m


def isaac_link_velocity(com_vel, quat, com_b):
    """Isaac Lab 2.3 ArticulationData.body_link_vel_w (math_utils.quat_apply)."""
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, -com_b, dim=-1) * 2
    offset = -com_b + quat[..., :1] * t + torch.linalg.cross(xyz, t, dim=-1)
    return com_vel[..., :3] + torch.linalg.cross(com_vel[..., 3:], offset, dim=-1)


class FakeData:
    def __init__(self, n=5, b=4):
        g = torch.Generator().manual_seed(0)
        self.com = torch.randn(n, b, 3, generator=g) * 0.05
        self.com_reads = 0
        self.step(g)

    def step(self, g=None):
        g = g or torch.Generator().manual_seed(self.com_reads + 7)
        self.body_com_vel_w = torch.randn(*self.com.shape[:2], 6, generator=g)
        q = torch.randn(*self.com.shape[:2], 4, generator=g)
        self.body_link_quat_w = q / q.norm(dim=-1, keepdim=True)

    @property
    def body_com_pos_b(self):
        self.com_reads += 1
        return self.com

    @property
    def body_link_lin_vel_w(self):
        return isaac_link_velocity(self.body_com_vel_w, self.body_link_quat_w, self.com)


class LinkVelocityTests(unittest.TestCase):
    def test_matches_isaac_and_reads_the_coms_rarely(self):
        data, cache, ids = FakeData(), m.LinkVelocity(), [1, 3]
        for _ in range(2 * m.COM_RECHECK_STEPS + 1):
            got = cache(data, ids)
            self.assertTrue(torch.allclose(got, data.body_link_lin_vel_w[:, ids], atol=1e-6))
            data.step()
        self.assertEqual(data.com_reads, 3)          # first call, then every 1000
        self.assertFalse(cache.fallback)

    def test_changed_coms_fall_back_for_good(self):
        data, cache, ids = FakeData(), m.LinkVelocity(), [0, 2]
        cache(data, ids)
        data.com = data.com + 0.01
        for _ in range(m.COM_RECHECK_STEPS + 2):
            got = cache(data, ids)
            self.assertTrue(torch.allclose(got, data.body_link_lin_vel_w[:, ids], atol=1e-6)
                            or not cache.fallback)
        self.assertTrue(cache.fallback)
        self.assertTrue(torch.allclose(cache(data, ids), data.body_link_lin_vel_w[:, ids], atol=1e-6))

    def test_without_com_data_uses_the_isaac_property(self):
        data = types.SimpleNamespace(body_link_lin_vel_w=torch.ones(2, 3, 3))
        self.assertTrue(torch.equal(m.LinkVelocity()(data, [0, 2]), torch.ones(2, 2, 3)))


if __name__ == "__main__":
    unittest.main()
