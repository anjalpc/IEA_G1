from __future__ import annotations

import asyncio
from copy import deepcopy

from .models import (
    AutonomyEvent,
    ControlOwner,
    DashboardState,
    LocalizationStatus,
    MissionStage,
    MissionState,
    MissionStatus,
    NavigationState,
    Pose2D,
    RobotPose,
    RobotState,
    SafetyState,
    StairGeometry,
    TerrainState,
    TerrainType,
)


PREVIEW_STAGES = [
    ("localize", "Localize"),
    ("navigate", "Navigate"),
    ("terrain", "Terrain"),
    ("traverse", "Traverse"),
    ("resume", "Resume"),
    ("goal", "Goal"),
]


def create_preview_stages() -> list[MissionStage]:
    return [
        MissionStage(
            id=stage_id,
            label=label,
        )
        for stage_id, label
        in PREVIEW_STAGES
    ]


def create_initial_state() -> DashboardState:
    return DashboardState(
        source="preview",
        robot=RobotState(
            connected=False,
            robot_name="G1 EDU U7",
            mode=None,
            battery_percent=None,
            control_owner=ControlOwner.NONE,
        ),
        navigation=NavigationState(
            localization=LocalizationStatus.OFFLINE,
            pose=RobotPose(
                x=0.0,
                y=0.0,
                yaw=0.0,
                frame="map",
                valid=False,
            ),
        ),
        mission=MissionState(
            id=None,
            name=None,
            status=MissionStatus.IDLE,
            stage_index=0,
            progress=0.0,
            stages=create_preview_stages(),
        ),
        terrain=TerrainState(
            current=TerrainType.UNKNOWN,
            detected=False,
        ),
        safety=SafetyState(
            ready=False,
        ),
    )


class StateStore:
    def __init__(self) -> None:
        self._state = create_initial_state()

        self._lock = asyncio.Lock()

        self._version = 0

        self._event_sequence = 0

        self._changed = asyncio.Condition()


    async def snapshot(self) -> DashboardState:
        async with self._lock:
            return deepcopy(self._state)


    async def version(self) -> int:
        async with self._lock:
            return self._version


    async def update(
        self,
        state: DashboardState,
    ) -> None:
        async with self._lock:
            self._state = deepcopy(state)
            self._version += 1

        async with self._changed:
            self._changed.notify_all()


    async def reset(self) -> None:
        await self.update(
            create_initial_state()
        )


    async def wait_for_change(
        self,
        last_version: int,
    ) -> tuple[int, DashboardState]:
        while True:
            current_version = (
                await self.version()
            )

            if current_version != last_version:
                return (
                    current_version,
                    await self.snapshot(),
                )

            async with self._changed:
                await self._changed.wait()


    async def add_event(
        self,
        kind: str,
        message: str,
        *,
        source: str = "preview",
    ) -> None:
        async with self._lock:
            self._event_sequence += 1

            event = AutonomyEvent(
                sequence=self._event_sequence,
                kind=kind,
                message=message,
                source=source,
            )

            self._state.events.append(
                event
            )

            self._state.events = (
                self._state.events[-100:]
            )

            self._version += 1

        async with self._changed:
            self._changed.notify_all()


    async def set_preview_step(
        self,
        step: int,
    ) -> None:
        async with self._lock:
            state = self._state

            step = max(
                0,
                min(
                    step,
                    len(PREVIEW_STAGES) - 1,
                ),
            )

            state.source = "preview"

            state.mission.id = (
                "preview-upper-platform"
            )

            state.mission.name = (
                "Navigate to Upper Platform"
            )

            state.mission.stage_index = step

            state.mission.progress = (
                (step + 1)
                / len(PREVIEW_STAGES)
            ) * 100.0

            for index, stage in enumerate(
                state.mission.stages
            ):
                if index < step:
                    stage.status = "complete"
                elif index == step:
                    stage.status = "active"
                else:
                    stage.status = "waiting"


            # -----------------------------------------------
            # Stage 0: localize
            # -----------------------------------------------

            if step == 0:
                state.mission.status = (
                    MissionStatus.LOCALIZING
                )

                state.robot.control_owner = (
                    ControlOwner.MISSION
                )

                state.navigation.localization = (
                    LocalizationStatus.LOCALIZING
                )

                state.navigation.pose = RobotPose(
                    x=0.0,
                    y=0.0,
                    yaw=0.0,
                    frame="map",
                    valid=False,
                )

                state.terrain = TerrainState(
                    current=TerrainType.FLAT,
                    detected=False,
                )


            # -----------------------------------------------
            # Stage 1: normal navigation
            # -----------------------------------------------

            elif step == 1:
                state.mission.status = (
                    MissionStatus.NAVIGATING
                )

                state.robot.control_owner = (
                    ControlOwner.NAVIGATION
                )

                state.navigation.localization = (
                    LocalizationStatus.VALID
                )

                state.navigation.pose = RobotPose(
                    x=1.2,
                    y=0.5,
                    yaw=0.25,
                    frame="map",
                    valid=True,
                )

                state.terrain = TerrainState(
                    current=TerrainType.FLAT,
                    detected=False,
                )


            # -----------------------------------------------
            # Stage 2: stair detection + approach
            # -----------------------------------------------

            elif step == 2:
                state.mission.status = (
                    MissionStatus.TERRAIN_APPROACH
                )

                state.robot.control_owner = (
                    ControlOwner.STAIR_APPROACH
                )

                state.navigation.pose = RobotPose(
                    x=2.2,
                    y=1.1,
                    yaw=0.30,
                    frame="map",
                    valid=True,
                )

                state.terrain = TerrainState(
                    current=(
                        TerrainType.STAIR_APPROACH
                    ),
                    detected=True,
                    confidence=0.94,
                    stair=StairGeometry(
                        rise_m=0.20,
                        tread_m=0.25,
                        steps=3,
                        alignment_error_deg=2.1,
                        distance_m=0.74,
                    ),
                )


            # -----------------------------------------------
            # Stage 3: stair RL owns locomotion
            # -----------------------------------------------

            elif step == 3:
                state.mission.status = (
                    MissionStatus.TERRAIN_TRAVERSAL
                )

                state.robot.control_owner = (
                    ControlOwner.STAIR_RL
                )

                state.navigation.pose = RobotPose(
                    x=2.8,
                    y=1.6,
                    yaw=0.31,
                    frame="map",
                    valid=True,
                )

                state.terrain = TerrainState(
                    current=(
                        TerrainType.STAIR_ASCENT
                    ),
                    detected=True,
                    confidence=0.97,
                    stair=StairGeometry(
                        rise_m=0.20,
                        tread_m=0.25,
                        steps=3,
                        alignment_error_deg=0.8,
                        distance_m=0.0,
                    ),
                )


            # -----------------------------------------------
            # Stage 4: platform verified, nav restored
            # -----------------------------------------------

            elif step == 4:
                state.mission.status = (
                    MissionStatus.RESUMING
                )

                state.robot.control_owner = (
                    ControlOwner.NAVIGATION
                )

                state.navigation.pose = RobotPose(
                    x=3.6,
                    y=2.3,
                    yaw=0.32,
                    frame="map",
                    valid=True,
                )

                state.terrain = TerrainState(
                    current=(
                        TerrainType.UPPER_PLATFORM
                    ),
                    detected=False,
                )


            # -----------------------------------------------
            # Stage 5: mission goal
            # -----------------------------------------------

            else:
                state.mission.status = (
                    MissionStatus.COMPLETED
                )

                state.robot.control_owner = (
                    ControlOwner.NAVIGATION
                )

                state.navigation.pose = RobotPose(
                    x=4.7,
                    y=3.1,
                    yaw=0.32,
                    frame="map",
                    valid=True,
                )

                state.terrain = TerrainState(
                    current=(
                        TerrainType.UPPER_PLATFORM
                    ),
                    detected=False,
                )

            state.navigation.goal = Pose2D(
                x=4.7,
                y=3.1,
                yaw=0.32,
            )

            state.navigation.global_path = [
                Pose2D(
                    x=0.0,
                    y=0.0,
                    yaw=0.0,
                ),
                Pose2D(
                    x=1.2,
                    y=0.5,
                    yaw=0.25,
                ),
                Pose2D(
                    x=2.2,
                    y=1.1,
                    yaw=0.30,
                ),
                Pose2D(
                    x=2.8,
                    y=1.6,
                    yaw=0.31,
                ),
                Pose2D(
                    x=3.6,
                    y=2.3,
                    yaw=0.32,
                ),
                Pose2D(
                    x=4.7,
                    y=3.1,
                    yaw=0.32,
                ),
            ]

            remaining = [
                5.63,
                4.33,
                3.17,
                2.39,
                1.36,
                0.0,
            ][step]

            state.navigation.distance_remaining_m = (
                remaining
            )

            state.navigation.eta_seconds = (
                None
                if step == 5
                else remaining / 0.45
            )

            self._version += 1

        async with self._changed:
            self._changed.notify_all()


state_store = StateStore()
