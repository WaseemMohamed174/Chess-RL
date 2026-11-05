"""Training API. Several models can train at the same moment; each run has its own process."""
from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ... import capacity, paths
from ... import settings as cfg
from .. import services
from ..state import S

router = APIRouter(prefix="/api/training", tags=["training"])


class Start(BaseModel):
    model_id: int
    overrides: dict = {}                  # same shape as /api/settings, for this run only
    max_iterations: int = Field(default=0, ge=0)   # stop by itself after this many iterations; 0 = never
    force: bool = False                   # start even if the machine is over its capacity


class Watch(BaseModel):
    run_id: int | None = None             # None = nobody is watching, no live board is sent


def max_parallel() -> int:
    """The suggested number of runs for this machine. It is advice for the website, not a limit."""
    return max(1, int(S.selection.get("max_parallel_runs", 4)))


async def _inference_for_run() -> dict | None:
    """Start (or reuse) the shared inference server and describe it for a train worker.
    None when it is switched off or will not start: the run then plays self-play inside its
    own process, exactly as before."""
    if not capacity.inference_enabled(S.selection):
        return None
    cfg_ = S.inference_cfg()
    w = S.sup.ensure_inference(cfg_)
    if not await asyncio.to_thread(w.wait_ready, 60):
        return None
    active = len(S.db.active_runs())                         # the run being started is in the DB already
    return {"port_file": cfg_["port_file"], "max_n": cfg_["max_n"],
            "workers": capacity.workers_for_run(S.selection.get("selfplay_workers", 1), active)}


async def _push_watch() -> None:
    """Tell every training process whether it is the one being watched."""
    for run in S.db.active_runs():
        try:
            await S.train_client(run["model_id"]).watch(run["id"] == S.watch_run)
        except Exception:
            pass


@router.post("/start", status_code=201)
async def start(body: Start):
    if not S.selection.get("train_enabled", True):
        raise HTTPException(400, "training was disabled in the run.py selection")
    model = services.need_model(body.model_id)
    if S.db.active_run_for_model(model["id"]):
        raise HTTPException(409, "this model is already being trained")
    await services.need_capacity(len(S.db.active_runs()) + 1, model, body.force)
    return await begin_run(model, body.overrides, body.max_iterations)


class StartAll(BaseModel):
    model_ids: list[int] | None = None    # None = every model that is not training yet
    force: bool = False                   # start even if the machine is over its capacity
    max_iterations: int = Field(default=0, ge=0)


@router.post("/start-all", status_code=201)
async def start_all(body: StartAll):
    """Start training for many single models at once (all idle ones by default; the agents of a
    population are left to the Population tab). The capacity is
    checked once for the whole group; a model that fails to start does not stop the others."""
    if not S.selection.get("train_enabled", True):
        raise HTTPException(400, "training was disabled in the run.py selection")
    models = S.db.list_models()
    if body.model_ids is not None:
        wanted = set(body.model_ids)
        models = [m for m in models if m["id"] in wanted]
    started, skipped, failed = [], [], []
    todo = []
    pops = services.population_model_map()
    for m in models:
        if m["id"] in pops:                          # trained from the Population tab
            skipped.append({"id": m["id"], "name": m["name"], "reason": "population agent"})
        elif S.db.active_run_for_model(m["id"]):
            skipped.append({"id": m["id"], "name": m["name"], "reason": "already training"})
        elif not os.path.exists(m["path"]):
            skipped.append({"id": m["id"], "name": m["name"], "reason": "file missing"})
        else:
            todo.append(m)
    if not todo:
        raise HTTPException(409, "there is no model to start")
    await services.need_capacity(len(S.db.active_runs()) + len(todo), todo[0], body.force)
    for m in todo:
        try:
            run = await begin_run(m, None, body.max_iterations)
            started.append({"id": m["id"], "name": m["name"], "run_id": run["id"]})
        except HTTPException as e:
            failed.append({"id": m["id"], "name": m["name"], "reason": str(e.detail)})
        except Exception as e:                       # one bad model must not stop the rest
            failed.append({"id": m["id"], "name": m["name"], "reason": str(e)})
    return {"started": started, "skipped": skipped, "failed": failed}


async def begin_run(model: dict, overrides: dict | None = None, max_iterations: int = 0) -> dict:
    """Start one training run (used by the Start button and by the population coordinator)."""
    overrides = overrides or {}
    try:
        cfg.validate_settings(overrides)
    except ValueError as e:
        raise HTTPException(422, str(e))
    merged = cfg.merged_settings(S.db.get_settings())
    for section, values in overrides.items():
        merged[section].update(values)
    name = f"train-{model['id']}"
    w = S.sup.ensure_train(model["id"])
    if not await asyncio.to_thread(w.wait_ready, 90):
        S.sup.release(name)
        await asyncio.to_thread(w.stop)
        raise HTTPException(503, f"the training worker did not start (see data/runtime/logs/{name}.log)")
    run = S.db.create_run(model["id"], merged)
    if S.watch_run is None:                          # first run: show its board right away
        S.watch_run = run["id"]
    try:
        inference = await _inference_for_run()
        await S.train_client(model["id"]).start(
            {"run_id": run["id"], "model_path": model["path"], "elo": model["elo"],
             "iterations": model["iterations"], "max_iterations": max_iterations,
             "mcts": merged["mcts"],
             "train": merged["train"], "watch": S.watch_run == run["id"],
             "inference": inference,
             "learner_gate": {"path": str(paths.RUNTIME_DIR / "learner.lock"),
                              "slots": int(S.selection.get("max_learners", 1))}})
    except Exception:
        S.db.set_run_status(run["id"], "failed")
        S.sup.release(name)
        await asyncio.to_thread(w.stop)
        raise
    await push_workers()                             # the runs that were already training share with it
    return run


async def push_workers() -> None:
    """Share the self-play workers evenly between the runs training now. Called when a run
    starts or ends; each run applies it at the start of its next iteration."""
    if not capacity.inference_enabled(S.selection):
        return
    runs = S.db.active_runs()
    if not runs:
        return
    n = capacity.workers_for_run(S.selection.get("selfplay_workers", 1), len(runs))
    for run in runs:
        try:
            await S.train_client(run["model_id"]).set_workers(n)
        except Exception:
            pass


@router.get("/server")
async def server_status():
    """The shared inference server for the Training page: state, capacity and raw counters
    (the page turns them into live rates from two samples)."""
    import json
    import time
    enabled = capacity.inference_enabled(S.selection)
    w = S.sup.get("inference") if S.sup else None
    out = {"enabled": enabled, "state": w.state.value if w else None,
           "restarts": w.restarts if w else 0,
           "workers": int(S.selection.get("selfplay_workers", 1)),
           "learners": int(S.selection.get("max_learners", 1)), "stats": None, "runs": []}
    if not enabled:
        return out
    for run in S.db.active_runs():                           # how many workers each model has now
        try:
            st = await S.train_client(run["model_id"]).status()
            out["runs"].append({"run_id": run["id"], "workers": int(st.get("workers", 0))})
        except Exception:
            out["runs"].append({"run_id": run["id"], "workers": 0})
    if w is not None:
        f = paths.RUNTIME_DIR / "inference_stats.json"
        try:
            if time.time() - f.stat().st_mtime < 10:         # an old file belongs to a dead server
                out["stats"] = json.loads(f.read_text(encoding="utf-8"))
                out["stats"].pop("models", None)
        except (OSError, ValueError):
            pass
    return out


async def _command(run: dict, name: str):
    try:
        return await S.train_client(run["model_id"])._call("POST", f"/{name}")
    except Exception:
        if name == "stop":                  # worker is gone: just close the run
            S.db.set_run_status(run["id"], "stopped")
            return {"state": "stopped"}
        raise


def _need_active(rid: int) -> dict:
    run = S.db.get_run(rid)
    if not run or run["status"] not in ("running", "paused"):
        raise HTTPException(409, "this training run is not active")
    return run


@router.post("/watch")
async def watch(body: Watch):
    """Choose the run whose live games are streamed; the others train without a board."""
    if body.run_id is not None:
        _need_active(body.run_id)
    S.watch_run = body.run_id
    await _push_watch()
    return {"watch": S.watch_run}


@router.post("/runs/{rid}/{name}")
async def run_command(rid: int, name: str):
    if name not in ("pause", "resume", "stop"):
        raise HTTPException(404, "unknown command")
    return await _command(_need_active(rid), name)


@router.post("/{name}")
async def all_command(name: str):
    """pause / resume / stop for every active run at once."""
    if name not in ("pause", "resume", "stop"):
        raise HTTPException(404, "unknown command")
    runs = S.db.active_runs()
    if not runs:
        raise HTTPException(409, "no active training run")
    return [await _command(r, name) for r in runs]


def _population_scope() -> tuple[dict, dict]:
    """Which runs belong to a population that is running now: (run id -> population id,
    model id -> population id). The second map covers the moment between a run being created
    and the coordinator noting it, so a population's run never shows up as a single model."""
    by_run: dict[int, int] = {}
    by_model: dict[int, int] = {}
    for pid, c in list(S.coordinators.items()):
        if not c.alive:
            continue
        for r in list(c.active.values()):
            by_run[r["id"]] = pid
        pop = S.db.get_population(pid)
        for m in S.db.list_members(pid, pop["generation"] if pop else None):
            if m["model_id"] is not None:
                by_model[m["model_id"]] = pid
    return by_run, by_model


def _population_of(run: dict, scope: tuple[dict, dict]) -> int | None:
    by_run, by_model = scope
    return by_run.get(run["id"], by_model.get(run["model_id"]))


def _view(run: dict, live: dict | None, population_id: int | None = None) -> dict:
    m = S.db.get_model(run["model_id"])
    return {"run": run, "live": live, "population_id": population_id,
            "model": {"id": run["model_id"], "name": m["name"] if m else f"Model {run['model_id']}",
                      "elo": m["elo"] if m else None},
            "metrics": S.db.list_metrics(run["id"])}


@router.get("/status")
async def status(population_id: str | None = None):
    """The runs training now. `population_id` narrows it: "single" = runs of single models,
    a number = the runs of that population, left out = every run. `busy_models` always lists
    every model that is training, whoever started it."""
    if population_id not in (None, "single") and not population_id.isdigit():
        raise HTTPException(400, 'population_id must be "single" or a number')
    scope = _population_scope()
    want = None if population_id is None else None if population_id == "single" else int(population_id)

    def mine(run: dict) -> bool:
        if population_id is None:
            return True
        return _population_of(run, scope) == want

    active = S.db.active_runs()
    runs = []
    for run in active:
        if not mine(run):
            continue
        live = None
        try:
            live = await S.train_client(run["model_id"]).status()
        except Exception:
            pass
        runs.append(_view(run, live, _population_of(run, scope)))
    last = None
    if not runs:
        for r in S.db.list_runs(limit=50):
            if population_id is None or _was_in_scope(r, population_id, want):
                last = _view(r, None, (r.get("hyperparams") or {}).get("meta", {}).get("population_id"))
                break
    return {"runs": runs, "last": last, "watch": S.watch_run, "max_parallel": max_parallel(),
            "busy_models": [r["model_id"] for r in active]}


def _was_in_scope(run: dict, population_id: str, want: int | None) -> bool:
    """For the 'last finished run' of a view: did it belong to that population (or to none)?
    Finished runs keep the population in their saved settings (see begin_run)."""
    pid = (run.get("hyperparams") or {}).get("meta", {}).get("population_id")
    return pid is None if population_id == "single" else pid == want


@router.get("/runs")
def runs(model_id: int | None = None):
    return S.db.list_runs(model_id)


@router.get("/runs/{rid}/metrics")
def metrics(rid: int):
    if not S.db.get_run(rid):
        raise HTTPException(404, "run not found")
    return S.db.list_metrics(rid)
