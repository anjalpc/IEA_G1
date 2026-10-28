import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_rewards import (
    footstep_target_score,clock_contact_scores)
from simulation.isaac_lab.g1_stairs.footstep_rewards import assert_paper_only_rewards

class PaperScoreTests(unittest.TestCase):
    def test_no_positive_without_acquisition(self):
        sole=torch.tensor([[[1.125,.08,.2],[0.,-.08,0.]]])
        targets=torch.tensor([[[1.125,.08,.2,0.],[1.375,-.08,.4,0.]]])
        feet=torch.tensor([[0,1]])
        valid=torch.tensor([[True,True]])
        x=footstep_target_score(sole,targets,feet,valid,torch.tensor([False]))
        self.assertEqual(x.item(),0.0)
        y=footstep_target_score(sole,targets,feet,valid,torch.tensor([True]))
        self.assertAlmostEqual(y.item(),1.0)
    def test_signed_gait_rewards_both_feet(self):
        force = torch.tensor([[100., 0.]])
        sole = torch.zeros((1, 2, 3))
        velocity = torch.tensor([[[0., 0., 0.], [.3, 0., 0.]]])
        targets = torch.zeros((1, 2, 4))
        feet = torch.tensor([[0, 1]])
        valid = torch.tensor([[True, True]])
        swing = torch.tensor([[0., 1.]])
        stance = torch.tensor([[1., 0.]])

        def scores():
            return clock_contact_scores(
                force, sole, velocity, targets, feet, valid, swing, stance
            )

        contact, speed = scores()
        self.assertAlmostEqual(contact.item(), .5)
        self.assertAlmostEqual(speed.item(), .5)

        # Right swing still receives credit while LEFT target is assigned.
        feet[0, 0] = 0
        self.assertAlmostEqual(scores()[1].item(), .5)

        # Contact on the scheduled swing foot is penalized.
        force[0, 1] = 100.
        self.assertAlmostEqual(scores()[0].item(), 0.)

        # Stance-foot motion reduces the foot-speed reward.
        velocity[0, 0, 0] = .3
        self.assertAlmostEqual(scores()[1].item(), 0.)

        # Mirror both feet: same physical scores.
        force = force.flip(1)
        velocity = velocity.flip(1)
        swing = swing.flip(1)
        stance = stance.flip(1)
        self.assertAlmostEqual(scores()[0].item(), 0.)
        self.assertAlmostEqual(scores()[1].item(), 0.)

    def test_plan_complete_stand_mode_rewards_standing(self):
        # Stand mode = double stance masks from the clock after the plan ends.
        force = torch.tensor([[100., 100.]])
        sole = torch.zeros((1, 2, 3))
        still = torch.zeros((1, 2, 3))
        moving = torch.tensor([[[.3, 0., 0.], [0., 0., 0.]]])
        targets = torch.zeros((1, 2, 4))
        feet = torch.tensor([[0, 1]])
        done = torch.tensor([[False, False]])
        swing = torch.zeros((1, 2))
        stance = torch.ones((1, 2))
        frc, vel = clock_contact_scores(
            force, sole, still, targets, feet, done, swing, stance)
        self.assertAlmostEqual(frc.item(), 1.)
        self.assertAlmostEqual(vel.item(), 0.)
        _, vel = clock_contact_scores(
            force, sole, moving, targets, feet, done, swing, stance)
        self.assertAlmostEqual(vel.item(), -.5)

    def test_double_support_penalizes_foot_motion(self):
        force = torch.tensor([[100., 100.]])
        sole = torch.zeros((1, 2, 3))
        velocity = torch.zeros((1, 2, 3))
        targets = torch.zeros((1, 2, 4))
        feet = torch.tensor([[0, 1]])
        valid = torch.tensor([[True, True]])
        swing = torch.zeros((1, 2))
        stance = torch.ones((1, 2))

        def scores():
            return clock_contact_scores(
                force, sole, velocity, targets, feet, valid, swing, stance
            )

        self.assertAlmostEqual(scores()[0].item(), 1.)
        self.assertAlmostEqual(scores()[1].item(), 0.)

        velocity[:, :, 0] = .3
        self.assertAlmostEqual(scores()[1].item(), -1.)

    def test_new_progress_no_recycling(self):
        from simulation.isaac_lab.g1_stairs.footstep_rewards import new_target_progress
        previous=torch.tensor([1.0])
        best=torch.tensor([1.0])
        valid=torch.tensor([True])
        total=0.0
        for current in (0.8, 0.95, 0.8, 0.75):
            score,best=new_target_progress(previous,torch.tensor([current]),best,valid)
            total+=score.item()
            previous=torch.tensor([current])
        self.assertAlmostEqual(total,2.5,places=5)
        self.assertAlmostEqual(best.item(),0.75,places=5)

    def test_new_progress_invalid_is_zero(self):
        from simulation.isaac_lab.g1_stairs.footstep_rewards import new_target_progress
        score,best=new_target_progress(torch.tensor([1.]),torch.tensor([.5]),
                                      torch.tensor([1.]),torch.tensor([False]))
        self.assertEqual(score.item(),0.)
        self.assertEqual(best.item(),1.)

    def test_event_terms_cancel_manager_dt(self):
        from types import SimpleNamespace
        from simulation.isaac_lab.g1_stairs import footstep_rewards as fr
        flag = torch.tensor([True, False])
        env = SimpleNamespace(
            num_envs=2, step_dt=0.02,
            _g1_footstep_last_advanced=flag,
            _g1_footstep_net_progress=torch.tensor([0.5, 0.]),
            _g1_footstep_clearance_progress=torch.tensor([0.25, 0.]),
            _g1_footstep_failure_event=flag.float(),
            _g1_footstep_stall_event=flag.float(),
            _g1_footstep_directed_swing=torch.tensor([0.5, 0.]),
            _g1_footstep_stance_support=torch.tensor([0.5, 0.]),
        )
        # Reward manager multiplies by step_dt: event terms become per-event.
        for fn, per_event in ((fr.cached_stall_event, 1.0),
                              (fr.cached_acquisition, 1.0),
                              (fr.cached_net_progress, 0.5),
                              (fr.cached_clearance_progress, 0.25),
                              (fr.cached_failure_event, 1.0)):
            self.assertAlmostEqual(fn(env)[0].item() * env.step_dt, per_event, places=6)
        # Per-step rate terms are NOT rescaled.
        self.assertAlmostEqual(fr.cached_directed_swing(env)[0].item(), 0.5)
        self.assertAlmostEqual(fr.cached_stance_support(env)[0].item(), 0.5)

    def test_reject_legacy_rewards(self):
        with self.assertRaises(RuntimeError):
            assert_paper_only_rewards(["track_lin_vel_xy"])
        self.assertTrue(assert_paper_only_rewards(["footstep_acquisition"]))
if __name__=="__main__":
    unittest.main()


class TreadLiftTests(unittest.TestCase):
    def lift(self, z, target_z, riser, index):
        from simulation.isaac_lab.g1_stairs.footstep_rewards import tread_lift
        f = lambda v: torch.tensor([float(v)])
        return tread_lift(f(z), f(target_z), f(riser), f(index), 3)

    def test_riser_height_alone_is_not_full_credit(self):
        lift, goal = self.lift(0.04, 0.04, 0.04, 0)
        self.assertAlmostEqual(goal.item(), 0.07, places=6)
        self.assertLess((lift / goal).item(), 0.6)
        lift, goal = self.lift(0.09, 0.04, 0.04, 0)
        self.assertAlmostEqual((lift / goal).item(), 1.0, places=6)

    def test_lift_is_measured_from_the_tread_below(self):
        # Foot resting on tread 1 (z = 0.04) while targeting tread 2.
        lift, _ = self.lift(0.04, 0.08, 0.04, 1)
        self.assertAlmostEqual(lift.item(), 0.0, places=6)

    def test_no_credit_without_a_riser(self):
        for index in (-2, -1, 3, 4):  # approach and platform targets
            self.assertEqual(self.lift(0.12, 0.12, 0.04, index)[0].item(), 0.0)
        self.assertEqual(self.lift(0.05, 0.0, 0.0, 0)[0].item(), 0.0)  # flat
