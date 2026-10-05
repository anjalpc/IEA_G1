from __future__ import annotations

from .preview_source import PreviewMission
from .ros_source import RosSource
from .source import DashboardSource


class SourceManager:
    def __init__(self) -> None:
        self.preview = PreviewMission()
        self.ros = RosSource()

        self._active: DashboardSource = (
            self.preview
        )


    @property
    def active(self) -> DashboardSource:
        return self._active


    @property
    def active_name(self) -> str:
        return self._active.name


    async def use_preview(self) -> bool:
        if self._active is not self.preview:
            await self._active.stop()
            self._active = self.preview

        return True


    async def use_ros(self) -> bool:
        if self._active is not self.ros:
            await self._active.stop()
            self._active = self.ros

        return True


source_manager = SourceManager()
