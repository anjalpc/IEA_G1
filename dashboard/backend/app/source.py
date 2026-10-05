from __future__ import annotations

from abc import ABC, abstractmethod


class DashboardSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


    @property
    @abstractmethod
    def running(self) -> bool:
        raise NotImplementedError


    @abstractmethod
    async def start(self) -> bool:
        raise NotImplementedError


    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError
