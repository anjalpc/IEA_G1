"""Contract tests for extracting navigation state from MuJoCo."""

import math

import pytest

from g1_simulation_bridge.simulation_state import (
    SimulationStateReader,
    diagonal_covariance,
)


class FakeMujoco:
    """Minimal named-object API used by the state reader."""

    class mjtObj:
        mjOBJ_JOINT = 1
        mjOBJ_SENSOR = 2

    @staticmethod
    def mj_name2id(model, object_type, name):
        return model.names.get((object_type, name), -1)


class FakeModel:
    """Model layout with nonzero addresses to reject fixed slicing."""

    def __init__(self):
        self.names = {
            (FakeMujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"): 0,
            (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_quat"): 0,
            (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"): 1,
            (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_acc"): 2,
        }
        self.jnt_qposadr = [2]
        self.jnt_dofadr = [1]
        self.sensor_adr = [0, 4, 7]
        self.sensor_dim = [4, 3, 3]


class FakeData:
    """State values matching FakeModel's resolved addresses."""

    def __init__(self):
        self.qpos = [
            99.0,
            99.0,
            1.0,
            2.0,
            3.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.qvel = [
            99.0,
            1.0,
            2.0,
            3.0,
            0.1,
            0.2,
            0.3,
        ]
        self.sensordata = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.4,
            0.5,
            0.6,
            4.0,
            5.0,
            6.0,
        ]


def reader(model=None):
    return SimulationStateReader(
        mujoco=FakeMujoco,
        model=model or FakeModel(),
    )


def test_resolves_named_sources_and_extracts_one_sample():
    sample = reader().sample(FakeData())

    assert sample.position_xyz == (1.0, 2.0, 3.0)
    assert sample.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert sample.linear_velocity_xyz == (1.0, 2.0, 3.0)
    assert sample.angular_velocity_xyz == (0.1, 0.2, 0.3)
    assert sample.imu_orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert sample.imu_angular_velocity_xyz == (0.4, 0.5, 0.6)
    assert sample.imu_linear_acceleration_xyz == (4.0, 5.0, 6.0)


def test_rotates_global_linear_velocity_into_pelvis_frame():
    data = FakeData()
    half_angle = math.sqrt(0.5)

    data.qpos[5:9] = [
        half_angle,
        0.0,
        0.0,
        half_angle,
    ]
    data.qvel[1:4] = [1.0, 0.0, 0.0]

    sample = reader().sample(data)

    assert sample.linear_velocity_xyz == pytest.approx(
        (0.0, -1.0, 0.0),
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "object_type,name",
    [
        (FakeMujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"),
        (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_quat"),
        (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"),
        (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_acc"),
    ],
)
def test_rejects_missing_required_named_source(object_type, name):
    model = FakeModel()
    del model.names[(object_type, name)]

    with pytest.raises(ValueError, match="required MuJoCo source"):
        reader(model)


def test_builds_explicit_diagonal_covariance():
    covariance = diagonal_covariance(
        width=3,
        variance=1e-6,
    )

    assert covariance == (
        1e-6, 0.0, 0.0,
        0.0, 1e-6, 0.0,
        0.0, 0.0, 1e-6,
    )


def test_rejects_aliased_required_sensors():
    model = FakeModel()
    gyro_id = model.names[
        (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    ]
    model.names[
        (FakeMujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
    ] = gyro_id

    with pytest.raises(ValueError, match="unique"):
        reader(model)


@pytest.mark.parametrize(
    "sensor_name,sensor_id,dimension",
    [
        ("imu_quat", 0, 3),
        ("imu_gyro", 1, 4),
        ("imu_acc", 2, 4),
    ],
)
def test_rejects_wrong_sensor_dimension(
    sensor_name,
    sensor_id,
    dimension,
):
    model = FakeModel()
    model.sensor_dim[sensor_id] = dimension

    with pytest.raises(
        ValueError,
        match=f"{sensor_name}.*dimension",
    ):
        reader(model)


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("qpos", 2, math.nan),
        ("qvel", 1, math.inf),
        ("sensordata", 4, -math.inf),
    ],
)
def test_rejects_nonfinite_simulation_state(
    field,
    index,
    value,
):
    data = FakeData()
    getattr(data, field)[index] = value

    with pytest.raises(ValueError, match="finite"):
        reader().sample(data)


@pytest.mark.parametrize(
    "source,components",
    [
        ("base", (0.0, 0.0, 0.0, 0.0)),
        ("base", (2.0, 0.0, 0.0, 0.0)),
        ("imu", (0.0, 0.0, 0.0, 0.0)),
        ("imu", (2.0, 0.0, 0.0, 0.0)),
    ],
)
def test_rejects_invalid_source_quaternion(
    source,
    components,
):
    data = FakeData()

    if source == "base":
        data.qpos[5:9] = components
    else:
        data.sensordata[0:4] = components

    with pytest.raises(ValueError, match="unit quaternion"):
        reader().sample(data)


@pytest.mark.parametrize(
    "field,length",
    [
        ("qpos", 8),
        ("qvel", 6),
        ("sensordata", 9),
    ],
)
def test_rejects_truncated_state_layout(field, length):
    data = FakeData()
    setattr(data, field, getattr(data, field)[:length])

    with pytest.raises(ValueError, match="state layout"):
        reader().sample(data)


@pytest.mark.parametrize(
    "width",
    [0, -1, 1.5],
)
def test_rejects_invalid_covariance_width(width):
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        diagonal_covariance(
            width=width,
            variance=1e-6,
        )


@pytest.mark.parametrize(
    "variance",
    [0.0, -1e-6, math.inf, math.nan],
)
def test_rejects_invalid_covariance_variance(variance):
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        diagonal_covariance(
            width=3,
            variance=variance,
        )
