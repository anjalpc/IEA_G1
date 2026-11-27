from types import SimpleNamespace

import pytest
import torch

from simulation.isaac_lab.g1_stairs.domain_randomization import (
    reset_stair_belief, stair_belief_offset, validate_domain_cfg)


def test_validate():
    assert validate_domain_cfg(None) is None
    out = validate_domain_cfg({"friction": True, "stair_offset_m": [-0.03, 0.03]})
    assert out == {"friction": True, "mass": False, "push": False, "obs_noise": False,
                   "stair_offset_m": (-0.03, 0.03)}
    legacy_off = {"enabled": False, "rigid_body_material": False, "added_mass": False,
                  "pushes": False, "observation_noise": False, "action_latency": False}
    assert validate_domain_cfg(legacy_off) is None          # every resolved config carries this
    assert validate_domain_cfg({**legacy_off, "push": True})["push"] is True
    assert validate_domain_cfg({**legacy_off, "pushes": True})["push"] is True
    with pytest.raises(ValueError):
        validate_domain_cfg({**legacy_off, "action_latency": True})
    for bad in ({"wind": True}, {"stair_offset_m": [0.02, -0.02]}, {"stair_offset_m": [0.0, 0.08]}):
        with pytest.raises(ValueError):
            validate_domain_cfg(bad)


def test_offset_drawn_per_reset_and_zero_when_off():
    env = SimpleNamespace(num_envs=4, device="cpu",
                          cfg=SimpleNamespace(domain_randomization=None))
    reset_stair_belief(env, None)
    assert stair_belief_offset(env) == 0.0
    env.cfg.domain_randomization = validate_domain_cfg({"stair_offset_m": [-0.03, 0.03]})
    reset_stair_belief(env, None)
    off = stair_belief_offset(env)
    assert off.shape == (4,) and bool((off.abs() <= 0.03).all())
    before = off.clone()
    reset_stair_belief(env, torch.tensor([1]))
    assert stair_belief_offset(env)[0] == before[0] and stair_belief_offset(env)[2] == before[2]
    env.cfg.domain_randomization = validate_domain_cfg({"stair_offset_m": [0.02, 0.02]})
    reset_stair_belief(env, None)
    assert torch.allclose(stair_belief_offset(env), torch.full((4,), 0.02))


def test_real_stair_frame_off_is_identity():
    from simulation.isaac_lab.g1_stairs.domain_randomization import real_stair_frame
    course = {"first_riser_x": torch.ones(2), "tread_depth": torch.full((2,), 0.25)}
    targets = torch.zeros(2, 2, 4)
    real, out = real_stair_frame(course, targets, 0.0, torch.ones(2, dtype=torch.bool))
    assert real is course and out is targets


def test_real_stair_frame_moves_plan_rows_onto_real_treads():
    from simulation.isaac_lab.g1_stairs.domain_randomization import real_stair_frame
    from simulation.isaac_lab.g1_stairs.footstep_tensor import tread_target_x
    from simulation.isaac_lab.g1_stairs.footstep_tread import qualified_tread_mask
    offset = torch.tensor([0.03, 0.03])                 # believed 3 cm further
    believed = torch.full((2,), 1.0) + offset
    depth, height = torch.full((2,), 0.25), torch.full((2,), 0.10)
    course = {"first_riser_x": believed, "tread_depth": depth, "riser_height": height}
    targets = torch.zeros(2, 2, 4)
    idx = torch.zeros(2, dtype=torch.long)
    targets[:, :, 0] = tread_target_x(believed, depth, idx, 3)[:, None]
    targets[:, :, 2] = 0.10
    plan_rows = torch.tensor([True, False])             # row 1: rolling clock
    real, out = real_stair_frame(course, targets, offset, plan_rows)
    assert torch.allclose(real["first_riser_x"], torch.ones(2))
    assert torch.allclose(out[0, :, 0], tread_target_x(torch.ones(1), depth[:1], idx[:1], 3))
    assert torch.equal(out[1], targets[1])
    assert torch.allclose(targets[0, 0, 0], tread_target_x(believed, depth, idx, 3)[0])  # input untouched
    # A foot on the real tread 1 centre, where the believed plan target is 3 cm ahead.
    x = out[0, 0, 0].item()
    bounds = torch.tensor([[[x - 0.1, x + 0.1, 0.05, 0.15, 0.10, 0.10],
                            [0.7, 0.9, -0.15, -0.05, 0.0, 0.0]]] * 2)
    force = torch.full((2, 2), 200.0)
    feet = torch.tensor([[0, 1]] * 2)
    valid = torch.ones(2, 2, dtype=torch.bool)
    ok = qualified_tread_mask(sole_bounds=bounds, vertical_force=force, targets=out,
                              feet=feet, valid=valid, first_riser_x=real["first_riser_x"],
                              tread_depth=depth, riser_height=height, step_count=3,
                              half_width=1.0, approach_steps=2)
    assert bool(ok[0, 0])
