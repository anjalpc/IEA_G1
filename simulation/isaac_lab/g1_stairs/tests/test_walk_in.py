import pytest
import torch

from simulation.isaac_lab.g1_stairs.walk_in import (
    CLIMB, STOP, WALK, advance_phase, stop_trigger_x, validate_walk_in_cfg,
    walk_in_forward_command)
from simulation.isaac_lab.g1_stairs.footstep_clock import validate_clock_cfg

CFG = {"enabled": True, "start_distance_m": 1.7, "stop_distance_m": [0.4, 1.0],
       "walk_speed_mps": 0.35, "ramp_s": 1.0, "stop_ramp_s": 0.5, "settle_s": [0.5, 1.5]}


def test_validate_and_bounds():
    out = validate_walk_in_cfg(CFG)
    assert out["stop_distance_m"] == (0.4, 1.0) and out["start_distance_m"] == 1.7
    assert validate_walk_in_cfg({"enabled": False}) is None
    for bad in ({"start_distance_m": 1.8}, {"stop_distance_m": [0.4, 1.5]},
                {"walk_speed_mps": 0.8}, {"settle_s": [1.0, 0.5]}):
        with pytest.raises(ValueError):
            validate_walk_in_cfg({**CFG, **bad})
    with pytest.raises(ValueError):
        validate_walk_in_cfg({**CFG, "extra": 1})


def test_clock_cfg_carries_walk_in():
    clock = {"enabled": True, "period_slow_s": 1.0, "period_fast_s": 0.7, "period_stairs_s": 1.25,
             "period_jitter": 0.1, "max_stride_m": 0.6, "walk_in": CFG}
    assert validate_clock_cfg(clock)["walk_in"]["walk_speed_mps"] == 0.35
    assert "walk_in" not in validate_clock_cfg({k: v for k, v in clock.items() if k != "walk_in"})


def test_stop_trigger_leads_the_stop_point():
    x = stop_trigger_x(torch.tensor([1.0]), torch.tensor([0.7]), 0.35, 0.5)
    torch.testing.assert_close(x, torch.tensor([1.0 - 0.7 - 0.0875]))


def test_phase_machine_walk_stop_settle_climb():
    phase = torch.tensor([WALK, WALK, WALK])
    stairs = torch.tensor([True, True, False])
    t_stop = torch.zeros(3); settle = torch.tensor([1.0, 1.0, 1.0])
    trig = torch.tensor([0.2, 0.2, 0.2])
    # row 0 reaches the trigger at t=3, row 1 not yet, row 2 is flat (never moves)
    phase, t_stop, now = advance_phase(phase, torch.tensor(3.0), t_stop, settle,
                                       torch.tensor([0.25, 0.1, 0.9]), trig, stairs, 0.5)
    assert phase.tolist() == [STOP, WALK, WALK] and not now.any() and float(t_stop[0]) == 3.0
    phase, t_stop, now = advance_phase(phase, torch.tensor(4.4), t_stop, settle,
                                       torch.tensor([0.3, 0.1, 0.9]), trig, stairs, 0.5)
    assert phase.tolist() == [STOP, WALK, WALK] and not now.any()      # 1.4 s < 0.5 + 1.0
    phase, t_stop, now = advance_phase(phase, torch.tensor(4.5), t_stop, settle,
                                       torch.tensor([0.3, 0.1, 0.9]), trig, stairs, 0.5)
    assert phase.tolist() == [CLIMB, WALK, WALK] and now.tolist() == [True, False, False]


def test_forward_command_ramps_and_releases():
    phase = torch.tensor([WALK, WALK, STOP, STOP, CLIMB])
    t = torch.tensor([0.5, 2.0, 3.25, 4.0, 5.0]); t_stop = torch.tensor([0.0, 0.0, 3.0, 3.0, 0.0])
    out = walk_in_forward_command(phase, t, t_stop, 0.4, 1.0, 0.5)
    torch.testing.assert_close(out[:4], torch.tensor([0.2, 0.4, 0.2, 0.0]))
    assert torch.isnan(out[4])


def test_walk_back_limit_follows_the_walk_in_start_inside_the_tile():
    from simulation.isaac_lab.g1_stairs.walk_in import walk_back_limit
    assert walk_back_limit(1.0, 1.0, 2.0) == -0.5                         # usual start: unchanged
    assert abs(walk_back_limit(1.7, 1.0, 2.0) - (-0.95)) < 1e-9          # capped 5 cm inside the tile
    assert abs(walk_back_limit(1.3, 1.0, 2.0) - (-0.8)) < 1e-9


def test_first_foot_option_validated():
    assert validate_walk_in_cfg(CFG)["first_foot"] == "random"
    assert validate_walk_in_cfg({**CFG, "first_foot": "rear"})["first_foot"] == "rear"
    with pytest.raises(ValueError):
        validate_walk_in_cfg({**CFG, "first_foot": "left"})


def test_rear_first_foot_follows_heading_and_draws_ties():
    from simulation.isaac_lab.g1_stairs.walk_in import rear_first_foot
    import math
    # rows: left ahead (-> right first), right ahead (-> left first), tie (-> draw),
    # left ahead along a heading of 90 deg (y forward) -> right first
    feet = torch.tensor([[[0.10, 0.1], [0.03, -0.1]],
                         [[0.02, 0.1], [0.09, -0.1]],
                         [[0.05, 0.1], [0.046, -0.1]],
                         [[-0.1, 0.12], [0.1, 0.05]]])
    yaw = torch.tensor([0.0, 0.0, 0.0, math.pi / 2])
    out = rear_first_foot(feet, yaw, torch.tensor([0, 0, 1, 0]))
    assert out.tolist() == [1, 0, 1, 1]
