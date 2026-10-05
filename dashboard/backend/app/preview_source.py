from __future__ import annotations

import asyncio

from .models import (
    ControlOwner,
    MissionStatus,
    TerrainState,
    TerrainType,
)
from .source import DashboardSource
from .state_store import (
    PREVIEW_STAGES,
    state_store,
)


class PreviewMission(DashboardSource):
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None


    @property
    def name(self) -> str:
        return "preview"


    @property
    def running(self) -> bool:
        return (
            self._task is not None
            and not self._task.done()
        )


    async def start(self) -> bool:
        if self.running:
            return False

        await self.stop(
            reset=False
        )

        await state_store.reset()

        await state_store.add_event(
            "mission",
            "Preview mission started",
        )

        self._task = asyncio.create_task(
            self._run()
        )

        return True


    async def stop(
        self,
        *,
        reset: bool = True,
    ) -> None:
        task = self._task

        self._task = None

        if task is not None:
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        if reset:
            state = await state_store.snapshot()

            state.mission.status = (
                MissionStatus.STOPPED
            )

            state.robot.control_owner = (
                ControlOwner.NONE
            )

            state.terrain = TerrainState(
                current=TerrainType.UNKNOWN,
                detected=False,
            )

            await state_store.update(
                state
            )

            await state_store.add_event(
                "mission",
                "Preview mission stopped",
            )


    async def _run(self) -> None:
        try:
            for step in range(
                len(PREVIEW_STAGES)
            ):
                await state_store.set_preview_step(
                    step
                )

                state = (
                    await state_store.snapshot()
                )

                await state_store.add_event(
                    "mission_stage",
                    (
                        "Preview stage: "
                        f"{state.mission.status.value}"
                    ),
                )

                if (
                    step
                    < len(PREVIEW_STAGES) - 1
                ):
                    await asyncio.sleep(
                        2.2
                    )

        except asyncio.CancelledError:
            raise
