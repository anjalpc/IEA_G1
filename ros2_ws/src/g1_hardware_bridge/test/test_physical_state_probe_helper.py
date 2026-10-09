"""CLI and binding contract for the physical-state probe."""

from types import SimpleNamespace

import pytest

from g1_hardware_bridge.state_probe.helper import (
    CANDIDATE_BINDINGS,
    parse_args,
    resolve_candidate_types,
)


def test_candidate_bindings_are_exact():
    assert CANDIDATE_BINDINGS == {
        "rt/lowstate": (
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            "LowState_",
        ),
        "rt/secondary_imu": (
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            "IMUState_",
        ),
        "rt/sportmodestate": (
            "unitree_sdk2py.idl.unitree_go.msg.dds_",
            "SportModeState_",
        ),
    }


def test_parses_explicit_safe_arguments():
    args = parse_args([
        "--interface",
        "test-interface",
        "--duration-seconds",
        "10",
        "--minimum-samples",
        "20",
    ])

    assert args.interface == "test-interface"
    assert args.duration_seconds == 10.0
    assert args.minimum_samples == 20


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--duration-seconds", "10"],
        ["--interface", ""],
        ["--interface", "test-interface", "--duration-seconds", "0"],
        ["--interface", "test-interface", "--duration-seconds", "-1"],
        ["--interface", "test-interface", "--duration-seconds", "nan"],
        ["--interface", "test-interface", "--duration-seconds", "inf"],
        ["--interface", "test-interface", "--minimum-samples", "0"],
        ["--interface", "test-interface", "--minimum-samples", "-1"],
    ],
)
def test_rejects_invalid_arguments(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_resolves_all_available_candidate_types():
    low_state = type("LowState", (), {})
    imu_state = type("ImuState", (), {})
    sport_state = type("SportState", (), {})

    modules = {
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": SimpleNamespace(
            LowState_=low_state,
            IMUState_=imu_state,
        ),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": SimpleNamespace(
            SportModeState_=sport_state,
        ),
    }

    result = resolve_candidate_types(
        import_module=lambda name: modules[name],
    )

    assert result == {
        "rt/lowstate": low_state,
        "rt/secondary_imu": imu_state,
        "rt/sportmodestate": sport_state,
    }


def test_contains_missing_optional_type_without_exception():
    module = SimpleNamespace(LowState_=type("LowState", (), {}))

    result = resolve_candidate_types(
        import_module=lambda _name: module,
    )

    assert result["rt/lowstate"] is module.LowState_
    assert result["rt/secondary_imu"] is None
    assert result["rt/sportmodestate"] is None


def test_contains_missing_module_without_exception():
    def unavailable(_name):
        raise ImportError("private binding detail")

    result = resolve_candidate_types(import_module=unavailable)

    assert result == {
        "rt/lowstate": None,
        "rt/secondary_imu": None,
        "rt/sportmodestate": None,
    }
