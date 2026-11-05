"""train_worker: owns the training loop. Started by the Supervisor when the user presses Start.

    POST /start {run_id, model_path, elo, iterations, mcts, train}
    POST /pause | /resume | /stop
    POST /watch {on}          only the run open on the website sends the full live board
    GET  /status
    GET  /events?after=N     the gateway polls this and forwards events to the website
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..rl.config import TrainConfig
from ..rl.orchestrator import EventLog, FakeBackend, Orchestrator, TorchBackend
from .common import fake_engine, require_token, torch_device

app = FastAPI(title="chess-rl train worker", dependencies=[Depends(require_token)])
EVENTS = EventLog()
RUN: Orchestrator | None = None


class StartReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    run_id: int
    model_path: str
    elo: float = 1000.0
    iterations: int = 0
    max_iterations: int = Field(default=0, ge=0)   # stop by itself after this many iterations; 0 = never
    mcts: dict = {}
    train: dict = {}
    watch: bool = False
    learner_gate: dict | None = None   # {"path", "slots"}: share the learner slots of all runs
    inference: dict | None = None      # {"port_file", "max_n", "workers"} -> self-play through the server


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/start")
def start(r: StartReq):
    global RUN
    if RUN is not None and RUN.alive():
        raise HTTPException(409, "a training run is already active")
    tc = TrainConfig.from_dict(r.train)
    backend_cls = FakeBackend if fake_engine() else TorchBackend
    backend = backend_cls(r.model_path, torch_device(), tc)
    remote = None
    if r.inference:
        remote = {"workers": int(r.inference["workers"]),
                  "spec": {"kind": "sock", "token": os.environ["CHESSRL_TOKEN"],
                           "max_n": int(r.inference["max_n"]), "port_file": r.inference["port_file"]}}
    gate = None
    if r.learner_gate:
        from ..rl.gate import LearnerGate
        gate = LearnerGate(r.learner_gate["path"], int(r.learner_gate["slots"]))
    RUN = Orchestrator(backend, r.model_dump(), EVENTS, remote, gate)
    RUN.set_watch(r.watch)
    RUN.start()
    return RUN.status()


def _need_run() -> Orchestrator:
    if RUN is None or not RUN.alive():
        raise HTTPException(409, "no active training run")
    return RUN


@app.post("/pause")
def pause():
    _need_run().pause()
    return RUN.status()


@app.post("/resume")
def resume():
    _need_run().resume()
    return RUN.status()


@app.post("/stop")
def stop():
    _need_run().stop()
    return RUN.status()


class WorkersReq(BaseModel):
    n: int


@app.post("/workers")
def workers(r: WorkersReq):
    """The gateway rebalances the self-play workers when runs start or end."""
    if RUN is not None and RUN.alive():
        RUN.set_workers(r.n)
    return {"workers": r.n}


class WatchReq(BaseModel):
    on: bool = False


@app.post("/watch")
def watch(r: WatchReq):
    if RUN is not None:
        RUN.set_watch(r.on)
    return {"watched": r.on}


@app.get("/status")
def status():
    return RUN.status() if RUN else {"state": "idle"}


@app.get("/events")
def events(after: int = 0):
    return EVENTS.since(after)
