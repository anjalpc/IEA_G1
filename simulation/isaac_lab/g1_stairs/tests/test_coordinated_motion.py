"""Tensor lifecycle checks with an isolated ManagerTermBase stub; no Kit."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import torch
from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS

stub = types.ModuleType('isaaclab.managers')
class Base:
    def __init__(self, cfg, env):
        pass
stub.ManagerTermBase = Base
stub.ActionTerm = Base
stub.ActionTermCfg = type('ActionTermCfg', (), {})
stub.RewardTermCfg = types.SimpleNamespace
utils = types.ModuleType('isaaclab.utils')
utils.configclass = lambda cls: cls
spec = importlib.util.spec_from_file_location(
    'simulation.isaac_lab.g1_stairs._c4_test_adapter',
    Path(__file__).parents[1]/'coordinated_motion.py')
module = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'isaaclab.managers':stub, 'isaaclab.utils':utils}):
    spec.loader.exec_module(module)
    gait_spec = importlib.util.spec_from_file_location(
        'simulation.isaac_lab.g1_stairs._c4_gait_test_adapter',
        Path(__file__).parents[1]/'gait_runtime.py')
    gait_module = importlib.util.module_from_spec(gait_spec)
    gait_spec.loader.exec_module(gait_module)


def fixture():
    NS = types.SimpleNamespace
    n = 2
    zeros = torch.zeros(n, 29)
    data = NS(body_pos_w=torch.tensor([[[1.09,.12,.075],[.80,-.12,.035]]]*n),
              body_quat_w=torch.tensor([1.,0.,0.,0.]).expand(n,2,4).clone(),
              body_lin_vel_w=torch.zeros(n,2,3), body_ang_vel_w=torch.zeros(n,2,3),
              root_pos_w=torch.tensor([[1.,0.,.8]]*n),root_lin_vel_w=torch.zeros(n,3),
              root_ang_vel_w=torch.zeros(n,3),default_joint_pos=zeros.clone(),
              joint_pos=zeros.clone(),joint_pos_target=zeros.clone())
    robot=NS(data=data,joint_names=BODY_JOINTS)
    sensor=NS(data=NS(net_forces_w=torch.tensor([[[0.,0.,200.],[0.,0.,100.]]]*n)))
    cfg=NS(size=(6.,1.2),approach_length=2.,start_clearance=1.,tread_depth=.25,
           step_count=3,riser_heights=None,riser_height_range=(.04,.04))
    class Scene(dict): pass
    scene=Scene(robot=robot)
    scene.env_origins=torch.zeros(n,3)
    scene.terrain=NS(cfg=NS(terrain_generator=NS(sub_terrains={'three_step_ascent':cfg})))
    gait=NS(processed_actions=torch.ones(n,1),reward_phase=torch.zeros(n),phase=torch.zeros(n))
    runtime=NS(last_step=1,robot=robot,robot_foot_ids=[0,1],sensor=sensor,sensor_foot_ids=[0,1],
               target_step=torch.ones(n,2,dtype=torch.long),preferred_foot=torch.ones(n,dtype=torch.long),
               step_acquired_event=torch.ones(n,dtype=torch.bool),
               physical_touchdown_event=torch.tensor([[True,False]]*n),
               preferred_transition_event_code=torch.zeros(n,dtype=torch.long),
               clean_step_over_sequence_failed=torch.zeros(n,dtype=torch.bool))
    env=NS(num_envs=n,device='cpu',scene=scene,common_step_counter=1,step_dt=.02,
           cfg=NS(coordinated_motion_enabled=True,coordinated_landing_rear_margin_m=.055),
           command_manager=NS(get_command=lambda name: torch.tensor([[.35,0.,0.]]*n)),
           action_manager=NS(get_term=lambda name:gait),_g1_v04_acquisition=runtime)
    real = gait_module.GaitFrequencyAction.__new__(gait_module.GaitFrequencyAction)
    real._env, real.cfg = env, NS(
        learned=False, smoothing_tau_s=0.0, adaptive_transition_clock=False,
        liftoff_force_n=15.0, support_force_n=30.0,
        first_step_activation_distance_m=.35, first_step_unload_dwell_s=.06,
        liftoff_timeout_s=.30, touchdown_grace_s=.30)
    real._raw = torch.zeros(n,1)
    real._frequency = torch.ones(n,1)
    real.phase = torch.zeros(n)
    real.active = torch.zeros(n,dtype=torch.bool)
    real.aligned = torch.zeros(n,dtype=torch.bool)
    real.reward_phase = torch.zeros(n)
    real.reward_active = torch.zeros(n,dtype=torch.bool)
    real.contact_loss = torch.zeros(n)
    real.swing_started = torch.zeros(n,dtype=torch.bool)
    real.reference_expired = torch.zeros(n,dtype=torch.bool)
    real.endpoint_hold_s = torch.zeros(n)
    real.selection_wait_s = torch.zeros(n)
    real.approach_unload_dwell_s = torch.zeros(n,2)
    real.transition_motion_miss = torch.zeros(n,dtype=torch.bool)
    real.transition_motion_miss_event = torch.zeros(n,dtype=torch.bool)
    real.transition_motion_miss_reason = torch.zeros(n,dtype=torch.long)
    real.approach_simultaneous_unload = torch.zeros(n,dtype=torch.bool)
    real.approach_preferred_foot = torch.full((n,),-1,dtype=torch.long)
    real.last_finished_step = -1
    real._checked_contract = True
    runtime.dynamic_alternating_rewards = True
    runtime.global_stair_acquisition = True
    runtime.sample_preferred_foot = torch.full((n,),-1,dtype=torch.long)
    env.action_manager.get_term = lambda name: real
    return env,runtime,real


def run_step(env, runtime, gait, term, **term_kwargs):
    # Production ordering: action snapshot -> acquisition/clock -> reward -> obs.
    gait.process_actions(torch.zeros(env.num_envs,1))
    gait.finish_step(runtime,30.)
    next_phase = gait.phase.clone()
    result = term(env, **term_kwargs)
    torch.testing.assert_close(gait.phase,next_phase)
    return result



class TestCoordinatedMotion(unittest.TestCase):
    def test_same_step_phase_order_and_endpoint_hold(self):
        env,runtime,gait=fixture()
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term)
        self.assertFalse(bool(term.trace['reference_active'].any()))
        self.assertAlmostEqual(float(term.landing[0,0]),1.395,places=5)
        runtime.step_acquired_event.zero_()
        runtime.physical_touchdown_event.zero_()
        runtime.sample_preferred_foot.fill_(1)
        for i in range(2,35):
            env.common_step_counter=runtime.last_step=i
            before = gait.phase.clone()
            run_step(env,runtime,gait,term)
            torch.testing.assert_close(gait.reward_phase,before)
            torch.testing.assert_close(term.trace['swing_progress']/2,gait.phase)
            expected = torch.stack((torch.sin(2*torch.pi*gait.phase),
                                    torch.cos(2*torch.pi*gait.phase)),-1)
            torch.testing.assert_close(gait.observation(),expected)
        torch.testing.assert_close(term.progress,torch.ones(2))
        torch.testing.assert_close(term.trace['desired_sole_centre_local'],term.landing)

    def test_acquisition_scores_old_foot_then_observes_new_clock(self):
        env,runtime,gait=fixture()
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term)
        env.common_step_counter=runtime.last_step=2
        runtime.sample_preferred_foot.fill_(1)
        runtime.preferred_foot.fill_(0)
        runtime.target_step.fill_(2)
        runtime.physical_touchdown_event[:]=torch.tensor([False,True])
        run_step(env,runtime,gait,term)
        self.assertEqual(term.trace['reference_target'].tolist(),[1,1])
        self.assertEqual(term.trace['reference_foot'].tolist(),[1,1])
        self.assertEqual(term.target.tolist(),[2,2])
        torch.testing.assert_close(gait.phase,torch.full((2,),.5))

    def test_partial_reset_same_step_neutralizes_only_reset_cache(self):
        env,runtime,gait=fixture()
        runtime.robot.data.joint_pos[:,11:] = 1.
        term=module.CoordinatedMotion(None,env)
        first=run_step(env,runtime,gait,term).clone()
        self.assertGreater(float(first[0]),0.)
        term.reset(torch.tensor([0]))
        unchanged_target=term.target[1].clone()
        repeated=term(env)
        self.assertEqual(float(repeated[0]),0.)
        torch.testing.assert_close(repeated[1],first[1])
        torch.testing.assert_close(term.target[1],unchanged_target)
        self.assertEqual(term.target.tolist(),[-1,1])
        self.assertFalse(bool(term.trace['reference_active'][0]))
        term.reset()
        torch.testing.assert_close(term(env),torch.zeros(2))

    def test_ambiguous_acquisition_cannot_start_reference(self):
        env,runtime,gait=fixture()
        runtime.physical_touchdown_event.fill_(True)
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term)
        self.assertEqual(term.target.tolist(),[-1,-1])

    def test_c6_first_step_waits_for_real_liftoff_then_guides_target_zero(self):
        env,runtime,gait=fixture()
        gait.cfg.learned=True
        gait.cfg.smoothing_tau_s=.20
        gait.cfg.adaptive_transition_clock=True
        runtime.target_step.zero_()
        runtime.preferred_foot.fill_(-1)
        runtime.step_acquired_event.zero_()
        runtime.physical_touchdown_event.zero_()
        # left foot unloaded, right foot supports: select left for first tread.
        runtime.sensor.data.net_forces_w[:,0,2]=0.
        runtime.sensor.data.net_forces_w[:,1,2]=100.
        term=module.CoordinatedMotion(None,env)
        for i in range(1,4):
            env.common_step_counter=runtime.last_step=i
            run_step(env,runtime,gait,term, adaptive_timing=True,
                     direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertEqual(gait.approach_preferred_foot.tolist(),[0,0])
        self.assertTrue(bool(gait.swing_started.all()))
        self.assertEqual(term.target.tolist(),[0,0])
        self.assertEqual(term.preferred.tolist(),[0,0])
        self.assertTrue(bool(term.trace['reference_selected'].all()))
        # first step is guided but cannot earn canonical DIRECT escrow credit.
        torch.testing.assert_close(term.trace['direct_quality_credit'],torch.zeros(2))

    def test_c6_selected_next_foot_phase_holds_until_liftoff(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term, adaptive_timing=True,
                 direct_quality_weight=.5, ascent_only_auxiliary=True)
        runtime.step_acquired_event.zero_()
        runtime.physical_touchdown_event.zero_()
        runtime.sample_preferred_foot.fill_(1)
        # preferred right remains loaded; phase must hold at its aligned start.
        runtime.sensor.data.net_forces_w[:,0,2]=100.
        runtime.sensor.data.net_forces_w[:,1,2]=100.
        env.common_step_counter=runtime.last_step=2
        run_step(env,runtime,gait,term, adaptive_timing=True,
                 direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertFalse(bool(gait.swing_started.any()))
        torch.testing.assert_close(gait.phase,torch.zeros(2))
        self.assertFalse(bool(term.trace['reference_active'].any()))
        # unloading the preferred right foot starts the clock.
        runtime.sensor.data.net_forces_w[:,1,2]=0.
        env.common_step_counter=runtime.last_step=3
        run_step(env,runtime,gait,term, adaptive_timing=True,
                 direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertTrue(bool(gait.swing_started.all()))
        self.assertTrue(bool(gait.phase.gt(0).all()))

    def test_c6_endpoint_expires_instead_of_freezing_forever(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True
        gait.cfg.touchdown_grace_s=.04
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term, adaptive_timing=True,
                 direct_quality_weight=.5, ascent_only_auxiliary=True)
        runtime.step_acquired_event.zero_(); runtime.physical_touchdown_event.zero_()
        runtime.sample_preferred_foot.fill_(1)
        runtime.sensor.data.net_forces_w[:,1,2]=0.; runtime.sensor.data.net_forces_w[:,0,2]=100.
        gait.phase.fill_(.5)
        gait.swing_started.fill_(True)
        for i in range(2,6):
            env.common_step_counter=runtime.last_step=i
            run_step(env,runtime,gait,term, adaptive_timing=True,
                     direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertTrue(bool(gait.reference_expired.all()))
        self.assertFalse(bool(gait.active.any()))
        self.assertFalse(bool(term.trace['reference_active'].any()))

    def test_c6_old_tread_load_cannot_satisfy_target_support(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term, adaptive_timing=True,
                 direct_quality_weight=.5, ascent_only_auxiliary=True)
        runtime.step_acquired_event.zero_(); runtime.physical_touchdown_event.zero_()
        runtime.sample_preferred_foot.fill_(1)
        gait.swing_started.fill_(True)
        # preferred right foot is heavily loaded but remains on the old/ground surface.
        runtime.sensor.data.net_forces_w[:,1,2]=200.; runtime.sensor.data.net_forces_w[:,0,2]=100.
        env.common_step_counter=runtime.last_step=2
        run_step(env,runtime,gait,term, adaptive_timing=True,
                 direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertFalse(bool(term.trace['target_support_valid'].any()))
        self.assertTrue(bool((term.trace['cost_transfer'] >= 0).all()))

    def test_c61_first_step_requires_sustained_asymmetric_unloading(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True
        runtime.target_step.zero_(); runtime.preferred_foot.fill_(-1)
        runtime.step_acquired_event.zero_(); runtime.physical_touchdown_event.zero_()
        runtime.sensor.data.net_forces_w[:,0,2]=0.; runtime.sensor.data.net_forces_w[:,1,2]=100.
        term=module.CoordinatedMotion(None,env)
        for i in range(1,3):
            env.common_step_counter=runtime.last_step=i
            run_step(env,runtime,gait,term, adaptive_timing=True,
                     direct_quality_weight=.5, ascent_only_auxiliary=True)
            self.assertEqual(gait.approach_preferred_foot.tolist(),[-1,-1])
        env.common_step_counter=runtime.last_step=3
        run_step(env,runtime,gait,term, adaptive_timing=True,
                 direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertEqual(gait.approach_preferred_foot.tolist(),[0,0])

    def test_c61_simultaneous_unloading_resets_first_step_dwell(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True
        runtime.target_step.zero_(); runtime.preferred_foot.fill_(-1)
        runtime.step_acquired_event.zero_(); runtime.physical_touchdown_event.zero_()
        term=module.CoordinatedMotion(None,env)
        runtime.sensor.data.net_forces_w[:,0,2]=0.; runtime.sensor.data.net_forces_w[:,1,2]=100.
        env.common_step_counter=runtime.last_step=1
        run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
        runtime.sensor.data.net_forces_w[:,:,2]=0.
        env.common_step_counter=runtime.last_step=2
        run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertTrue(bool(gait.approach_simultaneous_unload.all()))
        torch.testing.assert_close(gait.approach_unload_dwell_s,torch.zeros_like(gait.approach_unload_dwell_s))
        self.assertEqual(gait.approach_preferred_foot.tolist(),[-1,-1])

    def test_c61_selected_foot_must_liftoff_before_deadline(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True
        gait.cfg.liftoff_timeout_s=.04
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
        runtime.step_acquired_event.zero_(); runtime.physical_touchdown_event.zero_(); runtime.sample_preferred_foot.fill_(1)
        runtime.sensor.data.net_forces_w[:,:,2]=100.
        seen=0
        for i in range(2,6):
            env.common_step_counter=runtime.last_step=i
            run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
            seen += int(gait.transition_motion_miss_event.sum())
        self.assertTrue(bool(gait.reference_expired.all()))
        self.assertTrue(bool(gait.transition_motion_miss.all()))
        self.assertEqual(gait.transition_motion_miss_reason.tolist(),[1,1])
        self.assertEqual(seen,2)

    def test_c61_endpoint_expiry_emits_one_shot_miss_and_stays_latched(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True; gait.cfg.touchdown_grace_s=.04
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
        runtime.step_acquired_event.zero_(); runtime.physical_touchdown_event.zero_(); runtime.sample_preferred_foot.fill_(1)
        runtime.sensor.data.net_forces_w[:,1,2]=0.; runtime.sensor.data.net_forces_w[:,0,2]=100.
        gait.phase.fill_(.5); gait.swing_started.fill_(True)
        seen=0
        for i in range(2,8):
            env.common_step_counter=runtime.last_step=i
            run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
            seen += int(gait.transition_motion_miss_event.sum())
        self.assertEqual(seen,2)
        self.assertEqual(gait.transition_motion_miss_reason.tolist(),[2,2])
        self.assertTrue(bool(gait.transition_motion_miss.all()))
        self.assertFalse(bool(term.trace['reference_active'].any()))
        torch.testing.assert_close(term.trace['direct_quality_credit'],torch.zeros(2))

    def test_c61_late_direct_after_motion_miss_cannot_release_escrow(self):
        env,runtime,gait=fixture()
        gait.cfg.adaptive_transition_clock=True; gait.cfg.liftoff_timeout_s=.02
        term=module.CoordinatedMotion(None,env)
        run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
        runtime.step_acquired_event.zero_(); runtime.physical_touchdown_event.zero_(); runtime.sample_preferred_foot.fill_(1)
        runtime.sensor.data.net_forces_w[:,:,2]=100.
        env.common_step_counter=runtime.last_step=2
        run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
        self.assertTrue(bool(gait.transition_motion_miss.all()))
        runtime.step_acquired_event.fill_(True)
        runtime.physical_touchdown_event[:]=torch.tensor([False,True])
        runtime.preferred_transition_event_code.fill_(1)
        runtime.target_step.fill_(2)
        runtime.preferred_foot.fill_(0)
        env.common_step_counter=runtime.last_step=3
        run_step(env,runtime,gait,term, adaptive_timing=True, direct_quality_weight=.5, ascent_only_auxiliary=True)
        torch.testing.assert_close(term.trace['direct_quality_credit'],torch.zeros(2))

    def test_c61_reset_clears_wait_miss_and_first_step_dwell(self):
        env,runtime,gait=fixture()
        gait.selection_wait_s.fill_(1); gait.approach_unload_dwell_s.fill_(1)
        gait.transition_motion_miss.fill_(True); gait.transition_motion_miss_event.fill_(True)
        gait.transition_motion_miss_reason.fill_(2); gait.approach_simultaneous_unload.fill_(True)
        gait.reset(torch.tensor([0]))
        self.assertEqual(float(gait.selection_wait_s[0]),0.)
        torch.testing.assert_close(gait.approach_unload_dwell_s[0],torch.zeros(2))
        self.assertFalse(bool(gait.transition_motion_miss[0]))
        self.assertFalse(bool(gait.transition_motion_miss_event[0]))
        self.assertEqual(int(gait.transition_motion_miss_reason[0]),0)
        self.assertFalse(bool(gait.approach_simultaneous_unload[0]))


if __name__ == '__main__':
    unittest.main()
