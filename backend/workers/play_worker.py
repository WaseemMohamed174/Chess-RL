"""play_worker: answers "what would the model play here?" (inference + MCTS).

    POST /search      {fen, model_path, mcts, ply, seed, inference} -> {move, value, top, sims, ms, via}

With "inference" (the shared inference server, sent by the gateway while it runs) every
position is evaluated by the server: no PyTorch here, no global lock, so Play, hint and Arena
moves are answered while the training runs use the same server. Without it, or when the
server cannot answer, the search runs locally exactly as before (via = "local").
    POST /init_model  {path, blocks, filters}            -> writes a fresh random checkpoint
    GET  /info, /health
"""
from __future__ import annotations

import os
import threading
import time

import chess
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from ..rl.config import MCTSConfig
from ..rl.evaluator import UniformEvaluator
from ..rl.mcts import search
from .common import fake_engine, require_token, torch_device

app = FastAPI(title="chess-rl play worker", dependencies=[Depends(require_token)])
_lock = threading.Lock()                   # LOCAL searches only: one at a time, the net is shared
_tls = threading.local()                   # one server connection per request thread
PLAY_MAX_N = 64                            # rows per server request: one game never needs more
_cache: dict[str, tuple[float, object]] = {}
_MAX_CACHED = 4


def _evaluator(path: str | None):
    if fake_engine() or not path:
        return UniformEvaluator()
    mtime = os.path.getmtime(path)
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    from ..rl.evaluator import TorchEvaluator
    from ..rl.trainer import load_network
    dev = torch_device()
    ev = TorchEvaluator(load_network(path, dev), dev)
    _cache.pop(path, None)
    while len(_cache) >= _MAX_CACHED:
        _cache.pop(next(iter(_cache)))
    _cache[path] = (mtime, ev)
    return ev


def _remote(inf: dict, path: str):
    """Evaluator of this model on the shared server (connection reused by this thread)."""
    from ..inference.client import RemoteEvaluator
    cl = getattr(_tls, "client", None)
    if cl is None or getattr(_tls, "key", None) != inf["port_file"]:
        spec = {"kind": "sock", "token": os.environ["CHESSRL_TOKEN"],
                "max_n": min(int(inf.get("max_n", PLAY_MAX_N)), PLAY_MAX_N),
                "port_file": inf["port_file"], "retry_s": 5.0}
        cl = _tls.client = RemoteEvaluator(spec, "", timeout=60.0)
        _tls.key = inf["port_file"]
    return cl.for_model(os.path.abspath(path))


class SearchReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    fen: str
    model_path: str | None = None
    mcts: dict = {}
    ply: int = 0
    seed: int | None = None
    inference: dict | None = None          # {"port_file", "max_n"} of the shared server


class InitReq(BaseModel):
    path: str
    blocks: int = 3
    filters: int = 64


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    out = {"fake": fake_engine(), "device": torch_device(), "cached": list(_cache)}
    if not fake_engine():
        try:
            import torch
            out["torch"] = torch.__version__
            out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
    return out


@app.post("/search")
def do_search(r: SearchReq):
    try:
        board = chess.Board(r.fen)
    except ValueError:
        raise HTTPException(400, "invalid FEN")
    if not any(board.legal_moves):
        raise HTTPException(409, "no legal moves: the game is over")
    if r.model_path and not fake_engine() and not os.path.exists(r.model_path):
        raise HTTPException(404, "model file not found")
    cfg = MCTSConfig.from_dict(r.mcts)
    t0 = time.perf_counter()
    job, via = None, "local"
    if r.inference and r.model_path:
        from ..inference.client import InferenceError
        try:
            job = search(board, _remote(r.inference, r.model_path), cfg, seed=r.seed, noise=False)
            via = "server"
        except InferenceError:             # server down / model refused: play on locally
            job = None
    if job is None:
        with _lock:
            job = search(board, _evaluator(r.model_path), cfg, seed=r.seed, noise=False)
    temp = cfg.temperature if r.ply < cfg.temperature_moves else 0.0
    rng = np.random.default_rng(r.seed)
    probs = job.policy(temp)
    move = job.root.moves[int(rng.choice(len(probs), p=probs))]
    return {"move": move.uci(), "value": job.root_value(), "top": job.top_moves(5),
            "sims": int(job.root.N.sum()), "ms": round((time.perf_counter() - t0) * 1000, 1), "via": via}


@app.post("/init_model")
def init_model(r: InitReq):
    os.makedirs(os.path.dirname(r.path) or ".", exist_ok=True)
    tmp = r.path + ".tmp"
    if fake_engine():
        with open(tmp, "w") as f:
            f.write("fake model")
    else:
        import torch
        from ..rl.network import build_network
        net = build_network(r.blocks, r.filters)
        torch.save({"config": net.config(), "model": net.state_dict(), "opt": None,
                    "steps": 0}, tmp)
    os.replace(tmp, r.path)
    return {"path": r.path}
