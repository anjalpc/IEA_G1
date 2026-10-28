import unittest
import torch

from simulation.isaac_lab.g1_stairs.footstep_coordinator import FootstepCoordinator
from simulation.isaac_lab.g1_stairs.footstep_runtime import FootstepRuntime
from simulation.isaac_lab.g1_stairs.footstep_gait_clock import FootstepGaitClock


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.r = FootstepRuntime(num_envs=2,device="cpu",step_count=3,
                                 first_foot=torch.zeros(2,dtype=torch.long))
        self.c = FootstepGaitClock(num_envs=2,device="cpu",
                                   single_support_s=.8,double_support_s=.2)
        self.x = FootstepCoordinator(runtime=self.r,clock=self.c)
        self.targets = torch.zeros((2,2,4))
        self.targets[:,0,:3] = torch.tensor([1.125,.08,.2])
        self.feet = torch.tensor([[0,1],[0,1]])
        self.valid = torch.ones((2,2),dtype=torch.bool)
        self.sole = torch.zeros((2,2,3))
        self.sole[:,0] = self.targets[:,0,:3]
        self.support = torch.zeros((2,2),dtype=torch.bool)
        self.good = torch.zeros((2,2),dtype=torch.bool)

    def tick(self, step):
        return self.x.step(step_id=step,dt=.02,targets=self.targets,
                           feet=self.feet,valid=self.valid,sole_xyz=self.sole,
                           supported=self.support, qualified_tread=self.good,
                           xy_tolerance=.08,z_tolerance=.05)

    def test_reset_sets_lead_foot_for_selected_envs_only(self):
        self.x.reset(torch.tensor([1]), first_foot=torch.tensor([1]))
        self.assertEqual(self.r.first_foot.tolist(), [0, 1])
        self.x.reset(None, first_foot=torch.tensor([1, 0]))
        self.assertEqual(self.r.first_foot.tolist(), [1, 0])

    def test_initial_contact_never_advances(self):
        self.support[:,0] = True
        self.good[:,0] = True
        self.assertEqual(self.tick(1).tolist(),[False,False])
        self.assertEqual(self.r.target_index.tolist(),[0,0])

    def test_qualified_landing_advances_only_one_env_once(self):
        self.tick(1)
        self.support[0,0] = True
        self.good[0,0] = True
        self.assertEqual(self.tick(2).tolist(),[True,False])
        self.assertEqual(self.r.target_index.tolist(),[1,0])
        self.assertAlmostEqual(self.c.elapsed_s[0].item(),.04,places=6)
        self.assertEqual(self.tick(2).tolist(),[False,False])
        self.assertEqual(self.r.target_index.tolist(),[1,0])

    def test_unqualified_contact_is_not_progress(self):
        self.tick(1)
        self.support[0,0] = True
        self.good[0,0] = False
        self.assertEqual(self.tick(2).tolist(),[False,False])
        self.assertEqual(self.r.target_index.tolist(),[0,0])

    def test_contact_qualification_without_new_touchdown_cannot_advance(self):
        self.tick(1)
        self.support[0,0] = True
        self.tick(2)
        self.good[0,0] = True
        self.assertEqual(self.tick(3).tolist(),[False,False])

    def test_reset_primes_individual_env(self):
        self.tick(1)
        self.support[0,0] = True
        self.good[0,0] = True
        self.tick(2)
        self.x.reset(torch.tensor([0]))
        self.assertEqual(self.r.target_index.tolist(),[0,0])
        # Already-supported foot after reset must not count as landing.
        self.assertEqual(self.tick(3).tolist(),[False,False])

    def test_rejects_invalid_contract(self):
        with self.assertRaises(ValueError):
            self.x.step(step_id=1,dt=.02,targets=self.targets,feet=self.feet,
                        valid=self.valid,sole_xyz=self.sole,supported=self.support,
                        qualified_tread=torch.ones((2,2)),xy_tolerance=.08,z_tolerance=.05)


if __name__=="__main__":
    unittest.main()
