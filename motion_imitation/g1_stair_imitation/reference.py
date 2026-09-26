"""Canonical loader for the G1 20 cm task-space stair imitation reference.

This module intentionally has no Isaac Lab dependency.  It validates and loads
``g1_stair_reference_v1``.  The reference is a task-space/contact guide, not an
exact motor-command trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


SCHEMA = "g1_stair_reference_v1"
EXPECTED_DOF = 29
EXPECTED_FPS = 50.0
EXPECTED_DT_S = 0.02
EXPECTED_SOURCE_FRAME_START = 151
EXPECTED_SOURCE_FRAME_END = 277
EXPECTED_RISER_M = 0.20
EXPECTED_TREAD_M = 0.25
EXPECTED_STEP_COUNT = 3

SOLE_POINT_NAMES = (
    "rear_left",
    "rear_right",
    "front_left",
    "front_right",
)

EXPECTED_WEAK_POSE_JOINTS = {
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
}

EXPECTED_DISABLED_LATERAL_JOINTS = {
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
}

EXPECTED_DEFAULT_ONLY_WRIST_JOINTS = {
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
}


@dataclass(frozen=True)
class G1StairImitationReference:
    path: Path
    fps: float
    dt: float
    source_frame_start: int
    source_frame_end: int
    joint_names: tuple[str, ...]
    q_default: np.ndarray
    pose_mask: np.ndarray
    q_ref: np.ndarray
    dq_ref: np.ndarray
    root_pos: np.ndarray
    root_quat_xyzw: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel_body: np.ndarray
    left_ankle_pos: np.ndarray
    right_ankle_pos: np.ndarray
    left_sole_points: np.ndarray
    right_sole_points: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    left_step: np.ndarray
    right_step: np.ndarray
    taskspace_valid: np.ndarray
    phase01: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.q_ref.shape[0])

    @property
    def source_frames(self) -> np.ndarray:
        return np.arange(
            self.source_frame_start,
            self.source_frame_end + 1,
            dtype=np.int64,
        )

    def reference_index_for_source_frame(self, source_frame: int) -> int:
        source_frame = int(source_frame)
        if not self.source_frame_start <= source_frame <= self.source_frame_end:
            raise IndexError(
                f"source frame {source_frame} outside "
                f"{self.source_frame_start}->{self.source_frame_end}"
            )
        return source_frame - self.source_frame_start


def _as_float_array(value, *, name: str, shape=None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def _sole_points(side_data: dict) -> np.ndarray:
    sole = side_data["sole_points_world_m"]
    missing = [name for name in SOLE_POINT_NAMES if name not in sole]
    if missing:
        raise ValueError(f"missing sole points: {missing}")
    return np.asarray([sole[name] for name in SOLE_POINT_NAMES], dtype=np.float64)


def load_g1_stair_imitation_reference(path: str | Path) -> G1StairImitationReference:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"expected schema {SCHEMA!r}, got {payload.get('schema')!r}"
        )

    robot = payload["robot"]
    timing = payload["timing"]
    terrain = payload["terrain"]
    tracking = payload["tracking_contract"]
    frames = payload["frames"]

    if int(robot["controlled_dof"]) != EXPECTED_DOF:
        raise ValueError("reference is not a 29-DOF G1 contract")

    joint_names = tuple(robot["joint_names"])
    if len(joint_names) != EXPECTED_DOF:
        raise ValueError(f"expected 29 joint names, got {len(joint_names)}")
    if len(set(joint_names)) != EXPECTED_DOF:
        raise ValueError("reference joint names are not unique")

    q_default = _as_float_array(
        robot["q_default_rad"], name="q_default_rad", shape=(EXPECTED_DOF,)
    )
    pose_mask = _as_float_array(
        tracking["joint_pose_mask_29"],
        name="joint_pose_mask_29",
        shape=(EXPECTED_DOF,),
    )

    enabled = {
        name for name, weight in zip(joint_names, pose_mask) if weight > 0.0
    }
    if enabled != EXPECTED_WEAK_POSE_JOINTS:
        raise ValueError(f"weak-pose mask mismatch: {sorted(enabled)}")

    for name in EXPECTED_DISABLED_LATERAL_JOINTS:
        index = joint_names.index(name)
        if pose_mask[index] != 0.0:
            raise ValueError(f"{name} must have zero direct imitation weight")

    declared_wrists = set(tracking["default_only_missing_source_joints"])
    if declared_wrists != EXPECTED_DEFAULT_ONLY_WRIST_JOINTS:
        raise ValueError(
            "default-only wrist contract mismatch: " f"{sorted(declared_wrists)}"
        )

    if int(terrain["step_count"]) != EXPECTED_STEP_COUNT:
        raise ValueError("reference must contain exactly 3 stairs")
    if not np.isclose(float(terrain["riser_height_m"]), EXPECTED_RISER_M):
        raise ValueError("reference riser height must be 0.20 m")
    if not np.isclose(float(terrain["tread_depth_m"]), EXPECTED_TREAD_M):
        raise ValueError("reference tread depth must be 0.25 m")

    fps = float(timing["reference_rate_hz"])
    dt = float(timing["control_dt_s"])
    if not np.isclose(fps, EXPECTED_FPS):
        raise ValueError(f"expected 50 Hz reference, got {fps}")
    if not np.isclose(dt, EXPECTED_DT_S):
        raise ValueError(f"expected 0.02 s control dt, got {dt}")

    start = int(timing["source_frame_start"])
    end = int(timing["source_frame_end"])
    if (start, end) != (
        EXPECTED_SOURCE_FRAME_START,
        EXPECTED_SOURCE_FRAME_END,
    ):
        raise ValueError(f"unexpected source frame range {start}->{end}")

    expected_count = end - start + 1
    if len(frames) != expected_count:
        raise ValueError(f"expected {expected_count} frames, got {len(frames)}")

    q_ref = []
    dq_ref = []
    root_pos = []
    root_quat = []
    root_lin_vel = []
    root_ang_vel = []
    left_ankle = []
    right_ankle = []
    left_sole = []
    right_sole = []
    left_contact = []
    right_contact = []
    left_step = []
    right_step = []
    taskspace_valid = []
    phase01 = []

    for reference_index, row in enumerate(frames):
        expected_source_frame = start + reference_index
        if int(row["reference_index"]) != reference_index:
            raise ValueError("reference_index sequence mismatch")
        if int(row["source_frame_index"]) != expected_source_frame:
            raise ValueError("source_frame_index sequence mismatch")

        q_ref.append(
            _as_float_array(
                row["q_ref_29_rad"],
                name=f"q_ref[{reference_index}]",
                shape=(EXPECTED_DOF,),
            )
        )
        dq_ref.append(
            _as_float_array(
                row["dq_ref_29_radps"],
                name=f"dq_ref[{reference_index}]",
                shape=(EXPECTED_DOF,),
            )
        )
        root_pos.append(
            _as_float_array(
                row["root_pos_stair_frame_m"],
                name=f"root_pos[{reference_index}]",
                shape=(3,),
            )
        )
        root_quat.append(
            _as_float_array(
                row["root_quat_xyzw"],
                name=f"root_quat[{reference_index}]",
                shape=(4,),
            )
        )
        root_lin_vel.append(
            _as_float_array(
                row["root_linear_velocity_stair_frame_mps"],
                name=f"root_lin_vel[{reference_index}]",
                shape=(3,),
            )
        )
        root_ang_vel.append(
            _as_float_array(
                row["root_angular_velocity_body_radps"],
                name=f"root_ang_vel[{reference_index}]",
                shape=(3,),
            )
        )

        valid = bool(row["taskspace_primary_valid"])
        taskspace_valid.append(valid)
        phase01.append(float(row["phase01"]))

        contact = row["contact"]
        left_contact.append(bool(contact["left"]["persistent_contact"]))
        right_contact.append(bool(contact["right"]["persistent_contact"]))

        left_step_value = contact["left"]["step"]
        right_step_value = contact["right"]["step"]
        left_step.append(-1 if left_step_value is None else int(left_step_value))
        right_step.append(-1 if right_step_value is None else int(right_step_value))

        if valid:
            feet = row["feet"]
            if feet is None:
                raise ValueError(
                    f"frame {expected_source_frame} marks task-space valid but feet=None"
                )
            left = feet["left"]
            right = feet["right"]
            left_ankle.append(
                _as_float_array(
                    left["ankle_pos_m"],
                    name=f"left_ankle[{reference_index}]",
                    shape=(3,),
                )
            )
            right_ankle.append(
                _as_float_array(
                    right["ankle_pos_m"],
                    name=f"right_ankle[{reference_index}]",
                    shape=(3,),
                )
            )
            left_sole.append(_sole_points(left))
            right_sole.append(_sole_points(right))
        else:
            if row["feet"] is not None:
                raise ValueError(
                    f"frame {expected_source_frame} is initialization-only but feet present"
                )
            left_ankle.append(np.full(3, np.nan))
            right_ankle.append(np.full(3, np.nan))
            left_sole.append(np.full((4, 3), np.nan))
            right_sole.append(np.full((4, 3), np.nan))

    q_ref = np.asarray(q_ref)
    dq_ref = np.asarray(dq_ref)
    root_pos = np.asarray(root_pos)
    root_quat = np.asarray(root_quat)
    root_lin_vel = np.asarray(root_lin_vel)
    root_ang_vel = np.asarray(root_ang_vel)
    left_ankle = np.asarray(left_ankle)
    right_ankle = np.asarray(right_ankle)
    left_sole = np.asarray(left_sole)
    right_sole = np.asarray(right_sole)
    left_contact = np.asarray(left_contact, dtype=bool)
    right_contact = np.asarray(right_contact, dtype=bool)
    left_step = np.asarray(left_step, dtype=np.int64)
    right_step = np.asarray(right_step, dtype=np.int64)
    taskspace_valid = np.asarray(taskspace_valid, dtype=bool)
    phase01 = np.asarray(phase01, dtype=np.float64)

    if taskspace_valid[0]:
        raise ValueError("frame 151 must be initialization-only")
    if not np.all(taskspace_valid[1:]):
        raise ValueError("frames 152..277 must all have task-space targets")
    if q_ref.shape != (expected_count, EXPECTED_DOF):
        raise ValueError(f"bad q_ref shape {q_ref.shape}")
    if dq_ref.shape != (expected_count, EXPECTED_DOF):
        raise ValueError(f"bad dq_ref shape {dq_ref.shape}")
    if left_sole.shape != (expected_count, 4, 3):
        raise ValueError(f"bad left sole shape {left_sole.shape}")
    if right_sole.shape != (expected_count, 4, 3):
        raise ValueError(f"bad right sole shape {right_sole.shape}")

    return G1StairImitationReference(
        path=path,
        fps=fps,
        dt=dt,
        source_frame_start=start,
        source_frame_end=end,
        joint_names=joint_names,
        q_default=q_default,
        pose_mask=pose_mask,
        q_ref=q_ref,
        dq_ref=dq_ref,
        root_pos=root_pos,
        root_quat_xyzw=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel_body=root_ang_vel,
        left_ankle_pos=left_ankle,
        right_ankle_pos=right_ankle,
        left_sole_points=left_sole,
        right_sole_points=right_sole,
        left_contact=left_contact,
        right_contact=right_contact,
        left_step=left_step,
        right_step=right_step,
        taskspace_valid=taskspace_valid,
        phase01=phase01,
    )
