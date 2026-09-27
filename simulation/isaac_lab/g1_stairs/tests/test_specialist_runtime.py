from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import simulation.isaac_lab.g1_stairs.specialist_runtime as specialist_runtime
from simulation.isaac_lab.g1_stairs.specialist_runtime import (
    apply_specialist_runtime,
    parse_specialist_staging_config,
)


def _config() -> dict:
    return {
        "specialist_stair": {
            "enabled": True,
            "episode_length_s": 8.0,
            "staging": {
                "distance_before_riser_m": [0.35, 0.50],
                "lateral_offset_m": [-0.04, 0.04],
                "yaw_offset_rad": [-0.05236, 0.05236],
            },
        },
        "policy_initialization": {"type": "random"},
    }


def test_specialist_staging_distribution_is_centered_on_0425_m() -> None:
    parsed = parse_specialist_staging_config(_config())

    assert parsed is not None
    assert parsed.nominal_distance_m == pytest.approx(0.425)
    assert parsed.reset_x_range_m == pytest.approx((-0.075, 0.075))
    assert parsed.lateral_min_m == pytest.approx(-0.04)
    assert parsed.lateral_max_m == pytest.approx(0.04)
    assert parsed.episode_length_s == pytest.approx(8.0)


def test_specialist_profile_accepts_checkpoint_actor_transfer() -> None:
    config = _config()
    config["policy_initialization"] = {
        "type": "rsl_rl_checkpoint_actor_transfer",
        "source": {"path": "/tmp/model_950.pt"},
        "critic_initialization": "fresh",
    }

    parsed = parse_specialist_staging_config(config)
    assert parsed is not None


def test_specialist_profile_rejects_other_initialization_modes() -> None:
    config = _config()
    config["policy_initialization"]["type"] = "unitree_velocity_onnx_transfer"

    with pytest.raises(
        ValueError,
        match="random, rsl_rl_checkpoint_actor_transfer, or zero_residual_actor",
    ):
        parse_specialist_staging_config(config)


def test_specialist_profile_rejects_legacy_runtime_mix() -> None:
    config = _config()
    config["structured_step"] = {"enabled": True}

    with pytest.raises(ValueError, match="archived pre-specialist runtime"):
        parse_specialist_staging_config(config)


def test_specialist_profile_is_opt_in() -> None:
    assert parse_specialist_staging_config({}) is None


class _FakeRobotCfg:
    def replace(self, **kwargs):
        return self


def test_apply_specialist_runtime_reports_checkpoint_initialization(monkeypatch) -> None:
    config = _config()
    config["policy_initialization"] = {
        "type": "rsl_rl_checkpoint_actor_transfer",
        "source": {"path": "/tmp/model_950.pt"},
        "critic_initialization": "fresh",
    }

    fake_robot_cfg_module = SimpleNamespace(
        G1_SPECIALIST_ROBOT_CFG=_FakeRobotCfg()
    )
    monkeypatch.setitem(
        sys.modules,
        "simulation.isaac_lab.g1_stairs.robot_cfg",
        fake_robot_cfg_module,
    )
    monkeypatch.setattr(
        specialist_runtime,
        "_configure_specialist_observations",
        lambda env_cfg, resolved: {
            "proprioception_history_steps": 1,
            "velocity_command_observation": False,
        },
    )

    stair_cfg = SimpleNamespace(start_clearance=None)
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(
            robot=None,
            terrain=SimpleNamespace(
                terrain_generator=SimpleNamespace(
                    sub_terrains={"three_step_ascent": stair_cfg}
                )
            ),
        ),
        events=SimpleNamespace(
            reset_base=SimpleNamespace(
                params={
                    "pose_range": {"x": None, "y": None, "yaw": None},
                    "velocity_range": {
                        key: None
                        for key in ("x", "y", "z", "roll", "pitch", "yaw")
                    },
                }
            ),
            reset_robot_joints=SimpleNamespace(
                params={"position_range": None, "velocity_range": None}
            ),
        ),
        episode_length_s=20.0,
    )

    summary = apply_specialist_runtime(env_cfg, config)

    assert summary is not None
    assert summary["policy_initialization"] == "rsl_rl_checkpoint_actor_transfer"


def test_specialist_profile_accepts_zero_residual_actor() -> None:
    config = _config()
    config["policy_initialization"] = {
        "type": "zero_residual_actor",
        "critic_initialization": "fresh",
        "optimizer_initialization": "fresh",
    }
    config["balance_residual_control"] = {
        "enabled": True,
        "teacher_checkpoint": "/tmp/model_1500.pt",
        "scale_by_joint": {"placeholder": 0.1},
    }

    parsed = parse_specialist_staging_config(config)
    assert parsed is not None


def test_zero_residual_actor_requires_balance_wrapper() -> None:
    config = _config()
    config["policy_initialization"] = {"type": "zero_residual_actor"}

    with pytest.raises(
        ValueError,
        match="requires enabled balance_residual_control",
    ):
        parse_specialist_staging_config(config)
