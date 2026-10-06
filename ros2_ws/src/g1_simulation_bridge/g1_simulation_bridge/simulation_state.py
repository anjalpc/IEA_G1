"""Extract navigation state from named MuJoCo model sources."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimulationStateSample:
    """One hardware-independent simulation state sample."""

    position_xyz: tuple
    orientation_xyzw: tuple
    linear_velocity_xyz: tuple
    angular_velocity_xyz: tuple
    imu_orientation_xyzw: tuple
    imu_angular_velocity_xyz: tuple
    imu_linear_acceleration_xyz: tuple


def diagonal_covariance(*, width, variance):
    """Return a row-major square covariance with one diagonal value."""

    if not isinstance(width, int) or width <= 0:
        raise ValueError("covariance width must be a positive integer")
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "covariance variance must be finite and positive"
        )

    values = [0.0] * (width * width)
    for index in range(width):
        values[index * width + index] = variance
    return tuple(values)


def _wxyz_to_xyzw(quaternion):
    w, x, y, z = quaternion
    return (x, y, z, w)


def _rotate_global_to_local(vector, quaternion):
    """Rotate a global vector into the quaternion's local frame."""

    w, x, y, z = quaternion
    global_x, global_y, global_z = vector

    return (
        (
            (1.0 - 2.0 * (y * y + z * z)) * global_x
            + 2.0 * (x * y + w * z) * global_y
            + 2.0 * (x * z - w * y) * global_z
        ),
        (
            2.0 * (x * y - w * z) * global_x
            + (1.0 - 2.0 * (x * x + z * z)) * global_y
            + 2.0 * (y * z + w * x) * global_z
        ),
        (
            2.0 * (x * z + w * y) * global_x
            + 2.0 * (y * z - w * x) * global_y
            + (1.0 - 2.0 * (x * x + y * y)) * global_z
        ),
    )


class SimulationStateReader:
    """Resolve and read the approved MuJoCo state sources."""

    def __init__(self, *, mujoco, model):
        self._mujoco = mujoco
        self._model = model

        joint_type = mujoco.mjtObj.mjOBJ_JOINT
        sensor_type = mujoco.mjtObj.mjOBJ_SENSOR

        joint_id = self._required_id(
            joint_type,
            "floating_base_joint",
        )
        self._qpos_address = int(
            model.jnt_qposadr[joint_id]
        )
        self._qvel_address = int(
            model.jnt_dofadr[joint_id]
        )

        self._sensor_sources = {}
        resolved_sensor_ids = set()
        expected_dimensions = {
            "imu_quat": 4,
            "imu_gyro": 3,
            "imu_acc": 3,
        }
        for name, expected_dimension in expected_dimensions.items():
            sensor_id = self._required_id(sensor_type, name)
            if sensor_id in resolved_sensor_ids:
                raise ValueError(
                    "required MuJoCo sensors must resolve uniquely"
                )
            resolved_sensor_ids.add(sensor_id)

            dimension = int(model.sensor_dim[sensor_id])
            if dimension != expected_dimension:
                raise ValueError(
                    f"required MuJoCo source {name} has "
                    f"dimension {dimension}, expected "
                    f"{expected_dimension}"
                )
            self._sensor_sources[name] = (
                int(model.sensor_adr[sensor_id]),
                dimension,
            )

    def _required_id(self, object_type, name):
        object_id = self._mujoco.mj_name2id(
            self._model,
            object_type,
            name,
        )
        if object_id < 0:
            raise ValueError(
                f"required MuJoCo source is missing: {name}"
            )
        return int(object_id)

    def _sensor_values(self, data, name):
        address, dimension = self._sensor_sources[name]
        return tuple(
            float(value)
            for value in data.sensordata[
                address:address + dimension
            ]
        )

    def _validate_layout(self, data):
        required_qpos = self._qpos_address + 7
        required_qvel = self._qvel_address + 6
        required_sensor_data = max(
            address + dimension
            for address, dimension
            in self._sensor_sources.values()
        )

        if len(data.qpos) < required_qpos:
            raise ValueError(
                "MuJoCo state layout is truncated: qpos"
            )
        if len(data.qvel) < required_qvel:
            raise ValueError(
                "MuJoCo state layout is truncated: qvel"
            )
        if len(data.sensordata) < required_sensor_data:
            raise ValueError(
                "MuJoCo state layout is truncated: sensordata"
            )

    def _validate_finite(self, values):
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "MuJoCo state values must be finite"
            )

    def _validate_unit_quaternion(self, quaternion, name):
        norm = math.sqrt(
            sum(value * value for value in quaternion)
        )
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{name} must be a unit quaternion"
            )

    def sample(self, data):
        """Extract one finite sample from the resolved sources."""

        self._validate_layout(data)

        qpos = self._qpos_address
        qvel = self._qvel_address

        position = tuple(
            float(value)
            for value in data.qpos[qpos:qpos + 3]
        )
        base_wxyz = tuple(
            float(value)
            for value in data.qpos[qpos + 3:qpos + 7]
        )
        global_linear_velocity = tuple(
            float(value)
            for value in data.qvel[qvel:qvel + 3]
        )
        local_angular_velocity = tuple(
            float(value)
            for value in data.qvel[qvel + 3:qvel + 6]
        )
        imu_wxyz = self._sensor_values(
            data,
            "imu_quat",
        )
        imu_gyro = self._sensor_values(
            data,
            "imu_gyro",
        )
        imu_acceleration = self._sensor_values(
            data,
            "imu_acc",
        )

        values = (
            *position,
            *base_wxyz,
            *global_linear_velocity,
            *local_angular_velocity,
            *imu_wxyz,
            *imu_gyro,
            *imu_acceleration,
        )
        self._validate_finite(values)
        self._validate_unit_quaternion(
            base_wxyz,
            "floating-base orientation",
        )
        self._validate_unit_quaternion(
            imu_wxyz,
            "IMU orientation",
        )

        return SimulationStateSample(
            position_xyz=position,
            orientation_xyzw=_wxyz_to_xyzw(base_wxyz),
            linear_velocity_xyz=_rotate_global_to_local(
                global_linear_velocity,
                base_wxyz,
            ),
            angular_velocity_xyz=local_angular_velocity,
            imu_orientation_xyzw=_wxyz_to_xyzw(imu_wxyz),
            imu_angular_velocity_xyz=imu_gyro,
            imu_linear_acceleration_xyz=imu_acceleration,
        )
