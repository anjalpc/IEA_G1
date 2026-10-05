from __future__ import annotations

from .source import DashboardSource


class RosSource(DashboardSource):
    def __init__(self) -> None:
        self._running = False


    @property
    def name(self) -> str:
        return "ros"


    @property
    def running(self) -> bool:
        return self._running


    async def start(self) -> bool:
        if self._running:
            return False

        self._running = True

        # ROS 2 subscriptions will be added here later.
        #
        # Planned inputs:
        # - robot connection / mode
        # - localization pose
        # - global path
        # - local trajectory
        # - navigation goal
        # - terrain state
        # - control owner
        # - safety state
        # - mission events

        return True


    async def stop(self) -> None:
        self._running = False
