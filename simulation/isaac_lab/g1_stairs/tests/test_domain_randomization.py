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
