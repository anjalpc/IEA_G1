"""MuJoCo binding tests for the planar carrier."""

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np
import pytest


SIMULATION_ROOT = (
    Path(__file__).parents[4]
    / "simulation"
    / "mujoco"
)
sys.path.insert(0, str(SIMULATION_ROOT))

from model_composition import (  # noqa: E402
    build_combined_model,
)
from navigation.mujoco_planar_carrier import (  # noqa: E402
    MuJoCoPlanarCarrier,
)


@pytest.fixture
def runtime():
    """Return a real composed MuJoCo model and carrier."""
    model = build_combined_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    harness = SimpleNamespace(
        anchor=np.array([0.0, 0.0, 3.0])
    )
    carrier = MuJoCoPlanarCarrier(
        mujoco=mujoco,
        model=model,
        data=data,
        harness=harness,
        command_timeout_seconds=0.15,
        max_forward_mps=0.15,
        max_lateral_mps=0.01,
        max_yaw_rate_rps=0.40,
        allow_lateral=False,
    )
    return model, data, harness, carrier


def free_joint_addresses(model):
    """Return qpos and qvel addresses for the named base."""
    identifier = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "floating_base_joint",
    )
    assert identifier >= 0
    return (
        int(model.jnt_qposadr[identifier]),
        int(model.jnt_dofadr[identifier]),
    )


def test_applies_planar_pose_velocity_and_harness_anchor(runtime):
    """One carrier step updates all related MuJoCo state."""
    model, data, harness, carrier = runtime
    qpos, qvel = free_joint_addresses(model)
    initial_z = float(data.qpos[qpos + 2])

    carrier.command(
        forward_mps=0.10,
        lateral_mps=0.0,
        yaw_rate_rps=0.20,
        simulation_time=0.0,
    )
    pose = carrier.after_step(simulation_time=0.10)

    assert data.qpos[qpos] == pytest.approx(pose.x)
    assert data.qpos[qpos + 1] == pytest.approx(pose.y)
    assert data.qpos[qpos + 2] == initial_z
    assert data.qpos[qpos + 3] == pytest.approx(
        math.cos(pose.yaw / 2.0)
    )
    assert data.qpos[qpos + 4] == 0.0
    assert data.qpos[qpos + 5] == 0.0
    assert data.qpos[qpos + 6] == pytest.approx(
        math.sin(pose.yaw / 2.0)
    )

    assert data.qvel[qvel] == pytest.approx(
        0.10 * math.cos(pose.yaw)
    )
    assert data.qvel[qvel + 1] == pytest.approx(
        0.10 * math.sin(pose.yaw)
    )
    assert data.qvel[qvel + 3] == 0.0
    assert data.qvel[qvel + 4] == 0.0
    assert data.qvel[qvel + 5] == pytest.approx(0.20)

    assert harness.anchor[0] == pytest.approx(pose.x)
    assert harness.anchor[1] == pytest.approx(pose.y)


def test_timeout_stops_reported_planar_velocity(runtime):
    """A stale carrier command becomes zero MuJoCo velocity."""
    model, data, _harness, carrier = runtime
    _qpos, qvel = free_joint_addresses(model)

    carrier.command(
        forward_mps=0.10,
        lateral_mps=0.0,
        yaw_rate_rps=0.0,
        simulation_time=0.0,
    )
    carrier.after_step(simulation_time=0.30)

    assert carrier.stopped is True
    assert data.qvel[qvel] == 0.0
    assert data.qvel[qvel + 1] == 0.0
    assert data.qvel[qvel + 5] == 0.0


def test_explicit_stop_zeroes_planar_velocity(runtime):
    """Stopping the fixture immediately updates MuJoCo state."""
    model, data, _harness, carrier = runtime
    _qpos, qvel = free_joint_addresses(model)

    carrier.command(
        forward_mps=0.10,
        lateral_mps=0.0,
        yaw_rate_rps=0.20,
        simulation_time=0.0,
    )
    carrier.after_step(simulation_time=0.05)
    carrier.stop()

    assert data.qvel[qvel] == 0.0
    assert data.qvel[qvel + 1] == 0.0
    assert data.qvel[qvel + 5] == 0.0
