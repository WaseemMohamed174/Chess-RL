"""Gateway: REST + WebSocket + the website. No PyTorch in this process."""
from __future__ import annotations

import asyncio
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import paths
from ..clients import WorkerError, WorkerUnavailable
from . import services, ws
from .api import arena, games, models, populations, settings as settings_api, system, training
from .state import S

log = logging.getLogger("chessrl")


# ------------------------------------------------- training events -> DB + website
def _handle_train_event(ev: dict) -> None:
    kind, run_id = ev.get("type"), ev.get("run_id")
    run = S.db.get_run(run_id) if run_id else None
    if kind == "selfplay_move" and run_id != S.watch_run:
        return                                       # nobody is looking at this run's board
    if kind == "metrics" and run:
        S.db.add_metric(run_id, ev)
        S.db.update_model(run["model_id"], elo=ev["elo"], iterations=ev["iteration"])
    elif kind == "status" and run:
        S.db.set_run_status(run_id, ev["state"])
    elif kind == "game" and run and ev.get("pgn"):
        g = S.db.create_game(run["model_id"], run["model_id"], "selfplay", run_id=run_id)
        S.db.update_game(g["id"], ev["pgn"], {1: "1-0", -1: "0-1"}.get(ev["result"], "1/2-1/2"))
    S.bus.publish("training", ev)


def _close_run(run: dict, state: str, error: str | None = None) -> None:
    """The run ended without telling us (crash): record it and tell the website."""
    S.db.set_run_status(run["id"], state)
    S.bus.publish("training", {"type": "status", "run_id": run["id"], "state": state,
                               "error": error})


async def _release(model_id: int) -> None:
    w = S.sup.release(f"train-{model_id}")            # removed at once; stopped in the background
    if w:
        await asyncio.to_thread(w.stop)


async def training_poller() -> None:
    """Every training process keeps a numbered event log; we fetch what is new twice a second."""
    seen: dict[int, tuple] = {}                       # run_id -> (boot_id, last seq)
    while True:
        await asyncio.sleep(0.5)
        for run in S.db.active_runs():
            rid, mid = run["id"], run["model_id"]
            w = S.sup.get(f"train-{mid}")
            if w is None or w.state.value == "FAILED":
                _close_run(run, "failed", "the training process stopped unexpectedly")
                seen.pop(rid, None)
                await _release(mid)
                continue
            if not w.is_ready():
                continue
            boot, after = seen.get(rid, (None, 0))
            try:
                data = await S.train_client(mid)._call("GET", f"/events?after={after}")
            except Exception:
                continue
            if boot is not None and data["boot_id"] != boot:     # process restarted = run lost
                _close_run(run, "failed", "the training process crashed and was restarted")
                seen.pop(rid, None)
                await _release(mid)
                continue
            ended = False
            for ev in data["events"]:
                try:
                    _handle_train_event(ev)
                except Exception:
                    log.exception("bad training event")
                if ev.get("type") == "status" and ev.get("state") in ("stopped", "failed", "done"):
                    ended = True
            seen[rid] = (data["boot_id"], data["last"])
            if ended:
                seen.pop(rid, None)
                if S.watch_run == rid:
                    S.watch_run = None
                await _release(mid)
                await training.push_workers()          # its workers go to the runs that are left


async def ensure_selected_model() -> None:
    """Make sure the model size picked in run.py exists as a ready file.

    - no model of that size (blocks x filters)  -> create "Starter <size>"
    - a model whose file is gone and that was never trained -> write a fresh file at the same path
    - a trained model whose file is gone -> only a warning (re-creating it would hide lost weights)
    """
    net = S.selection["network"]
    models = S.db.list_models()
    for m in models:
        if os.path.exists(m["path"]):
            continue
        if m["iterations"] == 0:
            await S.play.init_model(m["path"], m["config"]["blocks"], m["config"]["filters"])
            log.info("model '%s': file was missing, created it again", m["name"])
        else:
            log.warning("model '%s' has %s iterations but its file is missing: %s",
                        m["name"], m["iterations"], m["path"])
    same = [m for m in models
            if m["config"]["blocks"] == net["blocks"] and m["config"]["filters"] == net["filters"]]
    if same:
        return
    names = {m["name"] for m in models}
    name = f"Starter {net['name']}"
    if name in names:
        name = f"{name} ({net['blocks']}x{net['filters']})"
    await services.create_model(name, net["blocks"], net["filters"])
    log.info("created model '%s'", name)


async def bootstrap() -> None:
    """Every start: wait for the play worker, then make sure the selected model exists."""
    w = S.sup.get("play")
    for attempt in range(1, 6):                    # torch can take a while to import on a first run
        if await asyncio.to_thread(w.wait_ready, 120):
            break
        log.warning("play worker not ready yet (try %d/5)", attempt)
    else:
        log.warning("play worker is not ready; the selected model was not created")
        return
    try:
        await ensure_selected_model()
    except Exception:
        log.exception("could not create the selected model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.start(asyncio.get_running_loop())
    S.tasks = [asyncio.create_task(training_poller()), asyncio.create_task(bootstrap())]
    try:
        yield
    finally:
        S.stop()


app = FastAPI(title="Chess-RL", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(WorkerUnavailable)
async def _unavailable(_: Request, e: WorkerUnavailable):
    return JSONResponse({"detail": str(e)}, status_code=503)


@app.exception_handler(WorkerError)
async def _worker_error(_: Request, e: WorkerError):
    return JSONResponse({"detail": f"engine: {e.detail}"}, status_code=502)


@app.get("/health")
def health():
    return {"status": "ok"}


for r in (system.router, models.router, games.router, arena.router, training.router,
          populations.router, settings_api.router, ws.router):
    app.include_router(r)

# ------------------------------------------------------------- the website
_dist = paths.FRONTEND_DIST
if (_dist / "index.html").exists():
    if (_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "ws/")):
            return JSONResponse({"detail": "not found"}, status_code=404)
        f = (_dist / path).resolve()
        if path and f.is_file() and _dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(_dist / "index.html")
