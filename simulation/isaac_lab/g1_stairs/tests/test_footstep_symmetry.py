import unittest

import torch

from simulation.isaac_lab.g1_stairs import footstep_symmetry as fs
from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS


def joint(name):
    return BODY_JOINTS.index(name)


class JointMapTests(unittest.TestCase):
    def test_pairs_and_signs(self):
        p, s = fs.JOINT_PERMUTATION, fs.JOINT_SIGN
        self.assertEqual(p[joint("left_knee_joint")], joint("right_knee_joint"))
        self.assertEqual(p[joint("right_hip_roll_joint")], joint("left_hip_roll_joint"))
        self.assertEqual(p[joint("waist_yaw_joint")], joint("waist_yaw_joint"))
        for name, flipped in (("left_hip_pitch_joint", False), ("left_hip_roll_joint", True),
                              ("left_hip_yaw_joint", True), ("left_knee_joint", False),
                              ("left_ankle_pitch_joint", False), ("left_ankle_roll_joint", True),
                              ("waist_yaw_joint", True), ("waist_roll_joint", True),
                              ("waist_pitch_joint", False), ("left_shoulder_roll_joint", True),
                              ("left_shoulder_yaw_joint", True), ("left_elbow_joint", False),
                              ("left_wrist_roll_joint", True), ("left_wrist_pitch_joint", False),
                              ("left_wrist_yaw_joint", True)):
            self.assertEqual(s[joint(name)], -1.0 if flipped else 1.0, name)

    def test_default_pose_is_mirror_symmetric(self):
        # unitree_rl_lab 4960b84 UNITREE_G1_29DOF_CFG init_state.joint_pos
        default = torch.zeros(len(BODY_JOINTS))
        for name, value in (("left_hip_pitch_joint", -0.1), ("right_hip_pitch_joint", -0.1),
                            ("left_knee_joint", 0.3), ("right_knee_joint", 0.3),
                            ("left_ankle_pitch_joint", -0.2), ("right_ankle_pitch_joint", -0.2),
                            ("left_shoulder_pitch_joint", 0.3), ("right_shoulder_pitch_joint", 0.3),
                            ("left_shoulder_roll_joint", 0.25), ("right_shoulder_roll_joint", -0.25),
                            ("left_elbow_joint", 0.97), ("right_elbow_joint", 0.97),
                            ("left_wrist_roll_joint", 0.15), ("right_wrist_roll_joint", -0.15)):
            default[joint(name)] = value
        self.assertTrue(torch.equal(fs.mirror_actions(default[None])[0], default))


class ObservationTests(unittest.TestCase):
    def test_widths(self):
        self.assertEqual(fs.POLICY_WIDTH, 688)
        self.assertEqual(fs.CRITIC_WIDTH, 703)
        with self.assertRaises(ValueError):
            fs.mirror_policy_obs(torch.zeros(1, 687))

    def test_involution(self):
        torch.manual_seed(0)
        for f, w in ((fs.mirror_policy_obs, 688), (fs.mirror_critic_obs, 703), (fs.mirror_actions, 29)):
            x = torch.randn(4, w)
            self.assertTrue(torch.equal(f(f(x)), x))

    def test_policy_layout(self):
        x = torch.zeros(1, 688)
        # newest frame (index 4) of base angular velocity: roll and yaw flip
        x[0, 12:15] = torch.tensor([1.0, 2.0, 3.0])
        # gravity frame 0: y flips; command frame 0: vy and wz flip
        x[0, 15:18] = torch.tensor([0.1, 0.2, -0.9])
        x[0, 30:33] = torch.tensor([0.4, 0.1, 0.2])
        # joint position frame 2: left knee 0.5, left hip roll 0.1
        q = 45 + 2 * 29
        x[0, q + joint("left_knee_joint")] = 0.5
        x[0, q + joint("left_hip_roll_joint")] = 0.1
        # height scan: row iy=0, ix=3 (right side, y = -0.5)
        scan = 480
        x[0, scan + 3] = 0.7
        # stair geometry: active, height, depth, yaw, distance
        geo = scan + 187
        x[0, geo:geo + 5] = torch.tensor([1.0, 0.1, 1.0, 0.05, 0.4])
        # footsteps: slot 0 left foot at y=+0.12, yaw 0.02; clock (sin, cos)
        fp = geo + 5
        x[0, fp:fp + 7] = torch.tensor([1.0, 1.0, 0.0, 0.2, 0.12, 0.02, 0.02])
        x[0, fp + 14:fp + 16] = torch.tensor([0.6, 0.8])
        m = fs.mirror_policy_obs(x)[0]
        self.assertTrue(torch.allclose(m[12:15], torch.tensor([-1.0, 2.0, -3.0])))
        self.assertTrue(torch.allclose(m[15:18], torch.tensor([0.1, -0.2, -0.9])))
        self.assertTrue(torch.allclose(m[30:33], torch.tensor([0.4, -0.1, -0.2])))
        self.assertAlmostEqual(m[q + joint("right_knee_joint")].item(), 0.5)
        self.assertAlmostEqual(m[q + joint("right_hip_roll_joint")].item(), -0.1)
        self.assertEqual(m[q + joint("left_knee_joint")].item(), 0.0)
        self.assertAlmostEqual(m[scan + 10 * 17 + 3].item(), 0.7)       # row iy=10 (y = +0.5)
        self.assertTrue(torch.allclose(m[geo:geo + 5], torch.tensor([1.0, 0.1, 1.0, -0.05, 0.4])))
        self.assertTrue(torch.allclose(m[fp:fp + 7], torch.tensor([1.0, 0.0, 1.0, 0.2, -0.12, 0.02, -0.02])))
        self.assertTrue(torch.allclose(m[fp + 14:fp + 16], torch.tensor([-0.6, -0.8])))

    def test_teacher_mirror_is_policy_mirror_prefix(self):
        torch.manual_seed(2)
        x = torch.randn(3, 688)
        self.assertEqual(fs.TEACHER_WIDTH, 480)
        self.assertTrue(torch.equal(fs.mirror_teacher_obs(x[:, :480]), fs.mirror_policy_obs(x)[:, :480]))
        self.assertTrue(torch.equal(fs.mirror_teacher_obs(fs.mirror_teacher_obs(x[:, :480])), x[:, :480]))

    def test_critic_has_linear_velocity_prefix(self):
        x = torch.zeros(1, 703)
        x[0, 0:3] = torch.tensor([0.5, 0.1, 0.0])
        x[0, 15 + 12:15 + 15] = torch.tensor([1.0, 2.0, 3.0])   # ang vel newest frame
        m = fs.mirror_critic_obs(x)[0]
        self.assertTrue(torch.allclose(m[0:3], torch.tensor([0.5, -0.1, 0.0])))
        self.assertTrue(torch.allclose(m[27:30], torch.tensor([-1.0, 2.0, -3.0])))


class AugmentationTests(unittest.TestCase):
    def test_tensordict_original_first_then_mirror(self):
        try:
            from tensordict import TensorDict
        except ImportError:
            self.skipTest("tensordict not installed")
        torch.manual_seed(1)
        obs = TensorDict({"policy": torch.randn(3, 688), "critic": torch.randn(3, 703)}, batch_size=[3])
        act = torch.randn(3, 29)
        o, a = fs.compute_symmetric_states(env=None, obs=obs, actions=act)
        self.assertEqual(o.batch_size[0], 6)
        self.assertTrue(torch.equal(o["policy"][:3], obs["policy"]))
        self.assertTrue(torch.equal(o["policy"][3:], fs.mirror_policy_obs(obs["policy"])))
        self.assertTrue(torch.equal(o["critic"][3:], fs.mirror_critic_obs(obs["critic"])))
        self.assertTrue(torch.equal(a[:3], act))
        self.assertTrue(torch.equal(a[3:], fs.mirror_actions(act)))
        with self.assertRaises(KeyError):
            fs.compute_symmetric_states(obs=TensorDict({"other": torch.zeros(1, 4)}, batch_size=[1]))


if __name__ == "__main__":
    unittest.main()
