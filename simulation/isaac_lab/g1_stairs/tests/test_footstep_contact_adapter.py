"""Focused tests for source-faithful Isaac contact extraction."""
import unittest
from types import SimpleNamespace as NS
import torch
from simulation.isaac_lab.g1_stairs.footstep_contact_adapter import measured_foot_state


class ContactAdapterTests(unittest.TestCase):
    def build(self):
        names = ["right_ankle_roll_link", "pelvis", "left_ankle_roll_link"]
        n = 2
        position = torch.tensor([[[2., 0., .5], [0., 0., 0.], [1., 0., .5]],
                                 [[12., 0., .5], [0., 0., 0.], [11., 0., .5]]])
        quat = torch.zeros((n, 3, 4))
        quat[..., 0] = 1
        zeros = torch.zeros((n, 3, 3))
        forces = torch.zeros((n, 3, 3))
        forces[:, 0, 2] = torch.tensor([110., 0.])
        forces[:, 2, 2] = torch.tensor([0., 110.])
        robot = NS(body_names=names, data=NS(body_pos_w=position, body_quat_w=quat,
                    body_lin_vel_w=zeros, body_ang_vel_w=zeros))
        sensor = NS(body_names=names, data=NS(net_forces_w=forces))
        class Scene(NS):
            def __getitem__(self, key):
                return self.__dict__[key]

        scene = Scene(sensors={"contact_forces": sensor},
                   env_origins=torch.tensor([[0., 0., 0.], [10., 0., 0.]]))
        scene.__dict__["robot"] = robot
        return NS(scene=scene, num_envs=n)

    def test_body_order_and_course_locality(self):
        sole, support = measured_foot_state(self.build(), support_force_n=100.)
        self.assertEqual(sole.shape, (2, 2, 3))
        self.assertTrue(torch.allclose(sole[:, 0, 0], torch.full((2,), 1.035)))
        self.assertTrue(torch.allclose(sole[:, 1, 0], torch.full((2,), 2.035)))
        self.assertEqual(support.tolist(), [[False, True], [True, False]])

    def test_invalid_force_threshold(self):
        with self.assertRaises(ValueError):
            measured_foot_state(self.build(), support_force_n=0.)

    def test_missing_foot_sensor_fails(self):
        env = self.build()
        env.scene.sensors["contact_forces"].body_names = ["pelvis"]
        with self.assertRaises(RuntimeError):
            measured_foot_state(env, support_force_n=100.)


if __name__ == "__main__":
    unittest.main()
