"""Populations API: create one, run it generation after generation, stop it, read it."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import coordinator, services
from ..state import S

router = APIRouter(prefix="/api/populations", tags=["populations"])


class Create(BaseModel):
    name: str
    base_model_id: int
    overrides: dict = {}                  # values of the "population" settings for this one


class Start(BaseModel):
    generations: int = Field(default=0, ge=0)      # generations to finish now; 0 = until stopped
    parallel: int = Field(default=0, ge=0)         # agents training at the same moment; 0 = all
    delete_replaced: bool = False                  # remove the models of replaced agents
    force: bool = False                            # start even if the machine is over its capacity


def _need(pid: int) -> dict:
    pop = S.db.get_population(pid)
    if not pop:
        raise HTTPException(404, "population not found")
    return pop


def _view(pop: dict) -> dict:
    c = coordinator.get(pop["id"])
    return {**pop, "live": c.info() if c else None}


@router.get("")
def list_populations():
    return [_view(p) for p in S.db.list_populations()]


@router.post("", status_code=201)
async def create(body: Create):
    base = services.need_model(body.base_model_id)
    out = await services.create_population(body.name, base, body.overrides)
    return {**_view(out["population"]), "members": out["members"]}


@router.get("/{pid}")
def get_population(pid: int):
    pop = _need(pid)
    return {**_view(pop), "members": S.db.list_members(pid),
            "summary": S.db.generation_summary(pid)}


@router.post("/{pid}/start", status_code=202)
async def start(pid: int, body: Start | None = None):
    _need(pid)
    body = body or Start()
    if not S.selection.get("train_enabled", True):
        raise HTTPException(400, "training was disabled in the run.py selection")
    pop = _need(pid)
    live = [m for m in S.db.list_members(pid, pop["generation"]) if m["model_id"] is not None]
    at_once = min(body.parallel, len(live)) if body.parallel else len(live)
    first = S.db.get_model(live[0]["model_id"]) if live else None
    await services.need_capacity(len(S.db.active_runs()) + max(at_once, 1), first, body.force)
    try:
        c = coordinator.begin(pid, max_generations=body.generations, parallel=body.parallel,
                              delete_replaced=body.delete_replaced)
    except coordinator.CoordinatorError as e:
        raise HTTPException(409, str(e))
    return c.info()


@router.post("/{pid}/stop")
async def stop(pid: int):
    _need(pid)
    c = coordinator.get(pid)
    if c is None or not c.alive:
        raise HTTPException(409, "this population is not running")
    await c.stop()
    return c.info()


@router.delete("/{pid}", status_code=204)
def delete(pid: int):
    """Removes the population and its history. The models stay."""
    _need(pid)
    c = coordinator.get(pid)
    if c is not None and c.alive:
        raise HTTPException(409, "this population is running; stop it first")
    S.coordinators.pop(pid, None)
    S.db.delete_population(pid)
