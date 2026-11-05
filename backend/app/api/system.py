from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from ... import hardware
from .. import services
from ..state import S

router = APIRouter(prefix="/api/system", tags=["system"])
_hw: dict | None = None


@router.get("/status")
async def status():
    global _hw
    if _hw is None:
        _hw = await asyncio.to_thread(lambda: hardware.detect_hardware().to_dict())
    info = None
    w = S.sup.get("play")
    if w and w.is_ready():
        try:
            info = await S.play.info()
        except Exception:
            pass
    return {"workers": S.sup.status(), "selection": S.selection, "hardware": _hw,
            "engine": info}


@router.get("/capacity")
async def capacity_check(runs: int = Query(1, ge=1, le=1000), model_id: int | None = None):
    """What `runs` training runs at the same moment would need (RAM / VRAM) against what is
    free, with a warning level. Only measures; the website decides what to show."""
    model = S.db.get_model(model_id) if model_id else None
    return await services.capacity_estimate(runs, model)


@router.post("/restart/{name}")
async def restart(name: str):
    w = S.sup.get(name)
    if not w:
        return {"ok": False, "detail": f"no worker named {name}"}
    await asyncio.to_thread(w.restart)
    return {"ok": True}
