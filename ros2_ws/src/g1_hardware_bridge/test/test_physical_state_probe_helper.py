"""CLI and binding contract for the physical-state probe."""

from types import SimpleNamespace

import pytest

from g1_hardware_bridge.state_probe.helper import (
    CANDIDATE_BINDINGS,
    extract_candidate_observation,
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


def imu_state(value=0.1):
    return SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[value, 0.0, 0.0],
        accelerometer=[0.0, 0.0, 9.81],
    )


def test_extracts_sanitized_low_state_observation():
    message = SimpleNamespace(
        tick=42,
        imu_state=imu_state(),
    )

    result = extract_candidate_observation(
        "rt/lowstate",
        message,
        callback_sequence=1,
    )

    assert result == {
        "topic": "rt/lowstate",
        "progress": 42,
        "field_lengths": {
            "accelerometer": 3,
            "gyroscope": 3,
            "quaternion": 4,
        },
        "finite": True,
    }


def test_extracts_sanitized_secondary_imu_observation():
    result = extract_candidate_observation(
        "rt/secondary_imu",
        imu_state(),
        callback_sequence=7,
    )

    assert result["topic"] == "rt/secondary_imu"
    assert result["progress"] == 7
    assert result["field_lengths"] == {
        "accelerometer": 3,
        "gyroscope": 3,
        "quaternion": 4,
    }


def test_extracts_sanitized_sport_state_observation():
    message = SimpleNamespace(
        position=[1.0, 2.0, 3.0],
        velocity=[0.1, 0.2, 0.3],
    )

    result = extract_candidate_observation(
        "rt/sportmodestate",
        message,
        callback_sequence=9,
    )

    assert result == {
        "topic": "rt/sportmodestate",
        "progress": 9,
        "field_lengths": {
            "position": 3,
            "velocity": 3,
        },
        "finite": True,
    }


def test_candidate_extraction_does_not_retain_raw_values():
    message = SimpleNamespace(
        position=[123.456, 234.567, 345.678],
        velocity=[456.789, 567.891, 678.912],
    )

    result = extract_candidate_observation(
        "rt/sportmodestate",
        message,
        callback_sequence=1,
    )

    serialized = repr(result)
    assert "123.456" not in serialized
    assert "678.912" not in serialized


@pytest.mark.parametrize(
    "topic,message",
    [
        ("rt/lowstate", SimpleNamespace(tick=1)),
        ("rt/secondary_imu", SimpleNamespace()),
        (
            "rt/sportmodestate",
            SimpleNamespace(position=[0.0, 0.0, 0.0]),
        ),
    ],
)
def test_rejects_candidate_messages_with_missing_fields(
    topic,
    message,
):
    with pytest.raises(AttributeError):
        extract_candidate_observation(
            topic,
            message,
            callback_sequence=1,
        )
