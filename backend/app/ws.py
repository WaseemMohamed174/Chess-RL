"""WebSocket channels:  /ws/training   /ws/game/{id}   /ws/match/{id}   /ws/system"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket

from .state import S

router = APIRouter()


async def _stream(ws: WebSocket, channel: str, hello: dict | None = None) -> None:
    await ws.accept()
    async with S.bus.subscribe(channel) as q:
        try:
            if hello is not None:
                await ws.send_json(hello)
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 15)
                except asyncio.TimeoutError:
                    ev = {"type": "ping"}            # also detects a closed browser tab
                await ws.send_json(ev)
        except Exception:
            pass


@router.websocket("/ws/training")
async def ws_training(ws: WebSocket):
    await _stream(ws, "training", {"type": "hello", "runs": S.db.active_runs(),
                                   "watch": S.watch_run})


@router.websocket("/ws/population")
async def ws_population(ws: WebSocket):
    await _stream(ws, "population", {"type": "hello",
                                     "populations": [c.info() for c in S.coordinators.values()]})


@router.websocket("/ws/game/{gid}")
async def ws_game(ws: WebSocket, gid: int):
    g = S.games.get(gid)
    await _stream(ws, f"game:{gid}", {"type": "state", **g.state()} if g else None)


@router.websocket("/ws/match/{mid}")
async def ws_match(ws: WebSocket, mid: int):
    await _stream(ws, f"match:{mid}", {"type": "hello", "match": S.db.get_match(mid),
                                      "running": mid in S.matches})


@router.websocket("/ws/system")
async def ws_system(ws: WebSocket):
    await _stream(ws, "system", {"type": "workers", "workers": S.sup.status()})
