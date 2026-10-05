from __future__ import annotations

import asyncio

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .state_store import state_store


async def stream_dashboard_state(
    websocket: WebSocket,
) -> None:
    await websocket.accept()

    version = -1

    try:
        while True:
            try:
                version, state = (
                    await asyncio.wait_for(
                        state_store.wait_for_change(
                            version
                        ),
                        timeout=1.0,
                    )
                )

                await websocket.send_json(
                    {
                        "type": "state",
                        "version": version,
                        "data": state.model_dump(
                            mode="json"
                        ),
                    }
                )

            except asyncio.TimeoutError:
                # Periodic heartbeat serves two purposes:
                #
                # 1. Detect disconnected browser clients even
                #    when dashboard state is not changing.
                #
                # 2. Prevent a WebSocket handler from remaining
                #    blocked forever during server shutdown /
                #    Uvicorn reload.
                await websocket.send_json(
                    {
                        "type": "ping",
                    }
                )

    except (
        WebSocketDisconnect,
        RuntimeError,
        OSError,
    ):
        return
