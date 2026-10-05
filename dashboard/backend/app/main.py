from __future__ import annotations

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    BackendHealth,
    DashboardState,
)
from .source_manager import source_manager
from .state_store import state_store
from .websocket import stream_dashboard_state


app = FastAPI(
    title="G1 Dashboard Backend",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/api/health",
    response_model=BackendHealth,
)
async def health() -> BackendHealth:
    return BackendHealth()


@app.get(
    "/api/state",
    response_model=DashboardState,
)
async def state() -> DashboardState:
    return await state_store.snapshot()


@app.get(
    "/api/source",
)
async def source() -> dict[str, object]:
    return {
        "active": source_manager.active_name,
        "preview_running": (
            source_manager.preview.running
        ),
        "ros_running": (
            source_manager.ros.running
        ),
    }


@app.post(
    "/api/mission/preview/start",
)
async def start_preview() -> dict[str, bool]:
    await source_manager.use_preview()

    started = (
        await source_manager.preview.start()
    )

    return {
        "ok": True,
        "started": started,
    }


@app.post(
    "/api/mission/preview/stop",
)
async def stop_preview() -> dict[str, bool]:
    await source_manager.preview.stop()

    return {
        "ok": True,
    }


@app.websocket(
    "/ws/state",
)
async def state_websocket(
    websocket: WebSocket,
) -> None:
    await stream_dashboard_state(
        websocket
    )
