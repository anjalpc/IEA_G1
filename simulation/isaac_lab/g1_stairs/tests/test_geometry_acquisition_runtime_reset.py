"""Reset-contract check with isolated Isaac Lab manager stubs; no Kit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import types
from unittest.mock import patch

import torch


stub = types.ModuleType("isaaclab.managers")

class Base:
    def __init__(self, cfg, env):
        pass

class SceneEntityCfg:
    def __init__(self, name, *args, **kwargs):
        self.name = name

stub.ManagerTermBase = Base
stub.SceneEntityCfg = SceneEntityCfg

spec = importlib.util.spec_from_file_location(
    "simulation.isaac_lab.g1_stairs._geometry_acquisition_reset_test_adapter",
    Path(__file__).parents[1] / "geometry_acquisition_rewards.py",
)
module = importlib.util.module_from_spec(spec)
with patch.dict("sys.modules", {"isaaclab.managers": stub}):
    spec.loader.exec_module(module)

GeometryAcquisitionRuntime = module.GeometryAcquisitionRuntime


def _runtime_stub() -> SimpleNamespace:
    shape = (2, 2)
    return SimpleNamespace(
        target_step=torch.ones(shape, dtype=torch.long),
        preferred_foot=torch.ones(2, dtype=torch.long),
        support_elapsed=torch.ones(shape),
        touchdown_attempt_active=torch.ones(shape, dtype=torch.bool),
        touchdown_attempt_good=torch.ones(shape, dtype=torch.bool),
        touchdown_placement_scored=torch.ones(shape, dtype=torch.bool),
        first_touchdown_reward_seen=torch.ones(2, dtype=torch.bool),
        best_clearance=torch.ones(shape),
        best_insertion=torch.ones(shape),
        pending_clearance=torch.ones(shape),
        clearance_potential=torch.ones(shape),
        previous_foot_vx=torch.ones(shape),
        previous_foot_vx_initialized=torch.ones(2, dtype=torch.bool),
        clearance_reward=torch.tensor([-0.4, 0.7]),
        insertion_reward=torch.tensor([0.2, 0.8]),
        first_touchdown_placement_reward=torch.tensor([0.3, 0.9]),
        nonpreferred_acquisition_reward=torch.tensor([0.5, 1.0]),
        physical_touchdown_event=torch.ones(shape, dtype=torch.bool),
        step_acquired_event=torch.ones(2, dtype=torch.bool),
        acquired_target_index=torch.ones(2, dtype=torch.long),
        sample_target_index=torch.ones(2, dtype=torch.long),
        riser_face_strike_sample=torch.ones(shape, dtype=torch.bool),
        riser_strike_force_x=torch.ones(shape),
        crossing_attempt_invalid=torch.ones(shape, dtype=torch.bool),
        contact_current_duration=torch.zeros((2, 0)),
        contact_previous_over=torch.zeros((2, 0), dtype=torch.bool),
        last_step=37,
    )


def test_partial_reset_clears_cached_rewards_without_reopening_same_step():
    runtime = _runtime_stub()

    GeometryAcquisitionRuntime.reset(runtime, [0])

    torch.testing.assert_close(runtime.clearance_reward, torch.tensor([0.0, 0.7]))
    torch.testing.assert_close(runtime.insertion_reward, torch.tensor([0.0, 0.8]))
    torch.testing.assert_close(
        runtime.first_touchdown_placement_reward,
        torch.tensor([0.0, 0.9]),
    )
    torch.testing.assert_close(
        runtime.nonpreferred_acquisition_reward,
        torch.tensor([0.0, 1.0]),
    )

    assert runtime.last_step == 37
