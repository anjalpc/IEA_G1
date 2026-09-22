"""Load and validate the frozen stair-ascent reference motion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class ReferenceMotion:
    frequency_hz: float
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.dof_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.frame_count - 1) / self.frequency_hz

    def body_index(self, name: str) -> int:
        try:
            return TRACKED_BODIES.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown tracked body: {name}") from exc

    def body_position(self, name: str) -> np.ndarray:
        return self.body_pos_w[:, self.body_index(name), :]

    @property
    def pelvis(self) -> np.ndarray:
        return self.body_position("pelvis")

    @property
    def left_ankle(self) -> np.ndarray:
        return self.body_position("left_ankle_roll_link")

    @property
    def right_ankle(self) -> np.ndarray:
        return self.body_position("right_ankle_roll_link")

    def positions_in_task_frame(self) -> np.ndarray:
        """Rotate source -Y-forward coordinates into task +X-forward coordinates.

        Mapping:
            task X = -source Y
            task Y =  source X
            task Z =  source Z

        The first pelvis frame is translated to X=0, Y=0.
        Z is preserved.
        """
        src = self.body_pos_w

        dst = np.empty_like(src)
        dst[..., 0] = -src[..., 1]
        dst[..., 1] = src[..., 0]
        dst[..., 2] = src[..., 2]

        pelvis_idx = self.body_index("pelvis")

        dst[..., 0] -= dst[0, pelvis_idx, 0]
        dst[..., 1] -= dst[0, pelvis_idx, 1]

        return dst


def load_reference_motion(
    metadata_path: str | Path,
) -> ReferenceMotion:
    metadata_path = Path(metadata_path)

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    data_path = metadata_path.parent / metadata["data"]["file"]

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    with np.load(data_path, allow_pickle=False) as data:
        required = {
            "body_pos_w",
            "body_quat_w",
            "dof_pos",
            "dof_vel",
        }

        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"reference data missing arrays: {sorted(missing)}"
            )

        body_pos_w = np.asarray(data["body_pos_w"])
        body_quat_w = np.asarray(data["body_quat_w"])
        dof_pos = np.asarray(data["dof_pos"])
        dof_vel = np.asarray(data["dof_vel"])

    expected_frames = int(metadata["reference"]["frame_count"])
    expected_hz = float(metadata["reference"]["frequency_hz"])
    expected_dofs = int(metadata["robot"]["policy_dofs"])
    expected_bodies = int(metadata["tracked_body_count"])

    if body_pos_w.shape != (expected_frames, expected_bodies, 3):
        raise ValueError(
            "unexpected body_pos_w shape: "
            f"{body_pos_w.shape}, expected "
            f"({expected_frames}, {expected_bodies}, 3)"
        )

    if body_quat_w.shape != (expected_frames, expected_bodies, 4):
        raise ValueError(
            "unexpected body_quat_w shape: "
            f"{body_quat_w.shape}, expected "
            f"({expected_frames}, {expected_bodies}, 4)"
        )

    if dof_pos.shape != (expected_frames, expected_dofs):
        raise ValueError(
            f"unexpected dof_pos shape: {dof_pos.shape}"
        )

    if dof_vel.shape != (expected_frames, expected_dofs):
        raise ValueError(
            f"unexpected dof_vel shape: {dof_vel.shape}"
        )

    arrays = (
        body_pos_w,
        body_quat_w,
        dof_pos,
        dof_vel,
    )

    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("reference data contains NaN or infinity")

    return ReferenceMotion(
        frequency_hz=expected_hz,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
    )
