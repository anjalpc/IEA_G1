from __future__ import annotations

import pytest

from simulation.isaac_lab.g1_stairs.specialist_runtime import (
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


def test_specialist_profile_requires_random_initialization() -> None:
    config = _config()
    config["policy_initialization"]["type"] = "unitree_velocity_onnx_transfer"

    with pytest.raises(ValueError, match="requires policy_initialization.type: random"):
        parse_specialist_staging_config(config)


def test_specialist_profile_rejects_legacy_runtime_mix() -> None:
    config = _config()
    config["structured_step"] = {"enabled": True}

    with pytest.raises(ValueError, match="archived pre-specialist runtime"):
        parse_specialist_staging_config(config)


def test_specialist_profile_is_opt_in() -> None:
    assert parse_specialist_staging_config({}) is None
