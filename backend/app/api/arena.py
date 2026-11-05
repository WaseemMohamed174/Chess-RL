from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import services
from ..state import S

router = APIRouter(prefix="/api/arena", tags=["arena"])


class NewMatch(BaseModel):
    white_id: int
    black_id: int
    games: int = Field(default=10, ge=1, le=200)
    simulations: int = Field(default=100, ge=10, le=800)
    delay_ms: int = Field(default=400, ge=0, le=5000)


class Speed(BaseModel):
    delay_ms: int = Field(ge=0, le=5000)


@router.post("/match", status_code=201)
async def start(body: NewMatch):
    if body.white_id == body.black_id:
        raise HTTPException(400, "choose two different models")
    a, b = services.need_model(body.white_id), services.need_model(body.black_id)
    row = S.db.create_match(a["id"], b["id"], body.games)
    S.match_delay[row["id"]] = body.delay_ms / 1000
    S.matches[row["id"]] = asyncio.create_task(
        services.run_match(row["id"], a, b, body.games, body.simulations))
    return row


@router.get("/match/{mid}")
def get_match(mid: int):
    row = S.db.get_match(mid)
    if not row:
        raise HTTPException(404, "match not found")
    return {**row, "running": mid in S.matches}


@router.put("/match/{mid}/speed")
def speed(mid: int, body: Speed):
    if mid not in S.matches:
        raise HTTPException(404, "match is not running")
    S.match_delay[mid] = body.delay_ms / 1000
    return {"delay_ms": body.delay_ms}


@router.post("/match/{mid}/cancel")
def cancel(mid: int):
    t = S.matches.get(mid)
    if not t:
        raise HTTPException(404, "match is not running")
    t.cancel()
    return {"ok": True}
