from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class LocalizationStatus(str, Enum):
    OFFLINE = "offline"
    LOCALIZING = "localizing"
    VALID = "valid"
    DEGRADED = "degraded"


class MissionStatus(str, Enum):
    IDLE = "idle"
    LOCALIZING = "localizing"
    NAVIGATING = "navigating"
    TERRAIN_APPROACH = "terrain_approach"
    TERRAIN_TRAVERSAL = "terrain_traversal"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TerrainType(str, Enum):
    UNKNOWN = "unknown"
    FLAT = "flat"
    STAIR_APPROACH = "stair_approach"
    STAIR_ASCENT = "stair_ascent"
    UPPER_PLATFORM = "upper_platform"


class ControlOwner(str, Enum):
    NONE = "none"
    MANUAL = "manual"
    MISSION = "mission"
    NAVIGATION = "navigation"
    STAIR_APPROACH = "stair_approach"
    STAIR_RL = "stair_rl"
    RECOVERY = "recovery"
    EMERGENCY = "emergency"


class Pose2D(BaseModel):
    x: float
    y: float
    yaw: float


class RobotPose(BaseModel):
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    frame: str = "map"
    valid: bool = False

    covariance: list[float] = Field(
        default_factory=list
    )


class NavigationState(BaseModel):
    localization: LocalizationStatus = (
        LocalizationStatus.OFFLINE
    )

    pose: RobotPose = Field(
        default_factory=RobotPose
    )

    goal: Pose2D | None = None

    global_path: list[Pose2D] = Field(
        default_factory=list
    )

    local_path: list[Pose2D] = Field(
        default_factory=list
    )

    distance_remaining_m: float | None = None
    eta_seconds: float | None = None


class StairGeometry(BaseModel):
    rise_m: float | None = None
    tread_m: float | None = None
    steps: int | None = None

    alignment_error_deg: float | None = None
    distance_m: float | None = None


class TerrainState(BaseModel):
    current: TerrainType = TerrainType.UNKNOWN

    detected: bool = False
    confidence: float | None = None

    stair: StairGeometry | None = None


class MissionStage(BaseModel):
    id: str
    label: str

    status: Literal[
        "waiting",
        "active",
        "complete",
        "failed",
    ] = "waiting"


class MissionState(BaseModel):
    id: str | None = None
    name: str | None = None

    status: MissionStatus = MissionStatus.IDLE

    stage_index: int = 0
    progress: float = 0.0

    stages: list[MissionStage] = Field(
        default_factory=list
    )


class SafetyState(BaseModel):
    ready: bool = False

    stable: bool | None = None
    fall_detected: bool | None = None
    command_timeout: bool | None = None

    reason: str | None = None


class RobotState(BaseModel):
    connected: bool = False

    robot_name: str = "G1 EDU U7"

    mode: str | None = None

    battery_percent: float | None = None

    control_owner: ControlOwner = (
        ControlOwner.NONE
    )


class AutonomyEvent(BaseModel):
    sequence: int

    kind: str
    message: str

    source: str = "dashboard_backend"


class DashboardState(BaseModel):
    source: Literal[
        "preview",
        "ros",
    ] = "preview"

    robot: RobotState = Field(
        default_factory=RobotState
    )

    navigation: NavigationState = Field(
        default_factory=NavigationState
    )

    mission: MissionState = Field(
        default_factory=MissionState
    )

    terrain: TerrainState = Field(
        default_factory=TerrainState
    )

    safety: SafetyState = Field(
        default_factory=SafetyState
    )

    events: list[AutonomyEvent] = Field(
        default_factory=list
    )


class BackendHealth(BaseModel):
    ok: bool = True
    service: str = "g1-dashboard-backend"
    source: str = "preview"
