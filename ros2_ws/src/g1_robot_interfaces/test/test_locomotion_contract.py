"""Contract tests for backend-neutral locomotion validation."""

import math

import pytest

from g1_locomotion_interfaces.msg import (
    LocomotionCommand,
    LocomotionState,
)
from g1_locomotion_interfaces.srv import (
    AcquireLocomotion,
    ReleaseLocomotion,
)

from g1_robot_interfaces.locomotion import (
    OWNER_MANUAL,
    OWNER_NAVIGATION,
    OWNER_NONE,
    OWNER_STAIR_POLICY,
    PlanarVelocityLimits,
    validate_lease_id,
    validate_locomotion_owner,
    validate_planar_velocity,
)


def limits():
    return PlanarVelocityLimits(
        forward_mps=0.5,
        lateral_mps=0.25,
        yaw_rate_rps=0.4,
    )


@pytest.mark.parametrize(
    "owner",
    [OWNER_MANUAL, OWNER_NAVIGATION, OWNER_STAIR_POLICY],
)
def test_accepts_command_owners(owner):
    assert validate_locomotion_owner(owner) == owner


def test_none_owner_requires_explicit_permission():
    with pytest.raises(ValueError, match="active locomotion owner"):
        validate_locomotion_owner(OWNER_NONE)

    assert validate_locomotion_owner(
        OWNER_NONE, allow_none=True
    ) == OWNER_NONE


@pytest.mark.parametrize("owner", [-1, 4, 255, "navigation"])
def test_rejects_unsupported_owners(owner):
    with pytest.raises(ValueError, match="locomotion owner"):
        validate_locomotion_owner(owner)


def test_accepts_nonempty_opaque_lease_without_normalizing_it():
    lease = " Lease-Token_01 "
    assert validate_lease_id(lease) == lease


@pytest.mark.parametrize("lease", ["", "   ", None])
def test_rejects_missing_lease(lease):
    with pytest.raises(ValueError, match="lease"):
        validate_lease_id(lease)


def test_accepts_planar_velocity_at_configured_limits():
    result = validate_planar_velocity(
        linear_x=0.5,
        linear_y=-0.25,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.4,
        limits=limits(),
    )

    assert result == (0.5, -0.25, 0.4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("linear_x", math.nan),
        ("linear_y", math.inf),
        ("angular_z", -math.inf),
    ],
)
def test_rejects_nonfinite_supported_velocity(field, value):
    command = {
        "linear_x": 0.0,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angular_x": 0.0,
        "angular_y": 0.0,
        "angular_z": 0.0,
    }
    command[field] = value

    with pytest.raises(ValueError, match="finite"):
        validate_planar_velocity(**command, limits=limits())


@pytest.mark.parametrize(
    "field",
    ["linear_z", "angular_x", "angular_y"],
)
def test_rejects_unsupported_velocity_axes(field):
    command = {
        "linear_x": 0.0,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angular_x": 0.0,
        "angular_y": 0.0,
        "angular_z": 0.0,
    }
    command[field] = 0.01

    with pytest.raises(ValueError, match="unsupported"):
        validate_planar_velocity(**command, limits=limits())


@pytest.mark.parametrize(
    "field,value",
    [
        ("linear_x", 0.5001),
        ("linear_x", -0.5001),
        ("linear_y", 0.2501),
        ("angular_z", -0.4001),
    ],
)
def test_rejects_velocity_outside_hard_limits(field, value):
    command = {
        "linear_x": 0.0,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angular_x": 0.0,
        "angular_y": 0.0,
        "angular_z": 0.0,
    }
    command[field] = value

    with pytest.raises(ValueError, match="limit"):
        validate_planar_velocity(**command, limits=limits())


@pytest.mark.parametrize(
    "field,value",
    [
        ("forward_mps", 0.0),
        ("lateral_mps", -0.1),
        ("yaw_rate_rps", math.inf),
    ],
)
def test_rejects_invalid_velocity_limits(field, value):
    values = {
        "forward_mps": 0.5,
        "lateral_mps": 0.25,
        "yaw_rate_rps": 0.4,
    }
    values[field] = value

    with pytest.raises(ValueError, match="finite and positive"):
        PlanarVelocityLimits(**values)


def test_generated_owner_constants_match_shared_contract():
    expected = {
        "OWNER_NONE": OWNER_NONE,
        "OWNER_MANUAL": OWNER_MANUAL,
        "OWNER_NAVIGATION": OWNER_NAVIGATION,
        "OWNER_STAIR_POLICY": OWNER_STAIR_POLICY,
    }

    for name, value in expected.items():
        assert getattr(LocomotionCommand, name) == value
        assert getattr(LocomotionState, name) == value
        assert getattr(AcquireLocomotion.Request, name) == value


def test_public_state_does_not_expose_lease_identifier():
    fields = LocomotionState.get_fields_and_field_types()
    assert "lease_id" not in fields


def test_release_contract_requires_owner_and_lease():
    request_fields = (
        ReleaseLocomotion.Request.get_fields_and_field_types()
    )
    assert request_fields == {
        "owner": "uint8",
        "lease_id": "string",
    }
