"""The inference server loop.

    requests (from any worker, any model)
        -> one pending list per model
        -> flush a model when:  its batch is full (max_batch positions)
                           or   its oldest request waited max_wait_ms
                           or   every client is already waiting (nobody else can arrive)
        -> ONE evaluate() for the whole batch, answers go back row by row

Batches are per model: two different networks can never share a forward pass.
The device is whatever the source builds its evaluators on ("cpu" or "cuda"); this
file does not know or care, which is what keeps one code path for both."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np

from ..rl.evaluator import EvalStats, check_response
from .transport import open_server_port


@dataclass
class ServerConfig:
    max_batch: int = 256            # positions per forward pass
    max_wait_ms: float = 2.0        # how long the oldest request may wait for company
    max_models: int = 4             # networks kept loaded (least recently used leave first)
    reload_check_s: float = 1.0     # how often to look for changed weights
    check: bool = False             # verify the evaluator contract on every batch
    stats_path: str | None = None   # JSON snapshot written about once a second


class ModelRegistry:
    """LRU cache of loaded models, reloaded when source.version() changes."""

    def __init__(self, source, max_models: int, reload_check_s: float):
        self.source, self.max_models, self.every = source, max(1, max_models), reload_check_s
        self._items: OrderedDict = OrderedDict()      # id -> [evaluator, version, last_check]
        self.loads = 0

    def get(self, model_id: str):
        now = time.monotonic()
        it = self._items.get(model_id)
        if it is not None:
            self._items.move_to_end(model_id)
            if now - it[2] < self.every:
                return it[0]
            version = self.source.version(model_id)
            it[2] = now
            if version == it[1]:
                return it[0]
        ev = self.source.load(model_id)
        self.loads += 1
        self._items[model_id] = [ev, self.source.version(model_id), now]
        self._items.move_to_end(model_id)
        while len(self._items) > self.max_models:
            self._items.popitem(last=False)
        return ev

    def invalidate(self, model_id: str) -> None:
        """Check this model's version on its next use, whatever reload_check_s says."""
        it = self._items.get(model_id)
        if it is not None:
            it[2] = float("-inf")

    def loaded(self) -> list:
        return list(self._items)


class InferenceServer:
    def __init__(self, port, source, cfg: ServerConfig | None = None):
        self.port, self.cfg = port, cfg or ServerConfig()
        self.models = ModelRegistry(source, self.cfg.max_models, self.cfg.reload_check_s)
        self.pending: dict = {}                      # model_id -> [Request]
        self._stats: dict = {}                       # model_id -> EvalStats
        self._wait: dict = {}                        # model_id -> [requests, queue-wait seconds]
        self._t0 = time.time()
        self._last_dump = 0.0

    # ------------------------------------------------------------------ loop
    def serve(self, stop) -> None:
        cfg, wait_s = self.cfg, self.cfg.max_wait_ms / 1000.0
        while not stop.is_set():
            now = time.perf_counter()
            if self.pending:
                due = min(r[0].t_arrive for r in self.pending.values()) + wait_s
                timeout = max(0.0, due - now)
            else:
                timeout = 0.05
            req = self.port.recv(timeout)
            while req is not None:                   # take everything that already arrived
                if req.op == "refresh":
                    self.models.invalidate(req.model_id)
                    self.port.reply(req, None, None)
                else:
                    self.pending.setdefault(req.model_id, []).append(req)
                req = self.port.recv(0)
            self._flush()
            self._dump()

    def _outstanding(self) -> int:
        return sum(len(v) for v in self.pending.values())

    def _flush(self) -> None:
        wait_s = self.cfg.max_wait_ms / 1000.0
        while self.pending:
            now = time.perf_counter()
            everyone_waits = self._outstanding() >= getattr(self.port, "n_clients", 1 << 30)
            ready = [m for m, rs in self.pending.items()
                     if everyone_waits or now - rs[0].t_arrive >= wait_s
                     or sum(r.n for r in rs) >= self.cfg.max_batch
                     or any(r.urgent for r in rs)]   # pipelined caller: its other half is busy
            if not ready:
                return
            # the model whose oldest request has waited longest goes first
            m = min(ready, key=lambda k: self.pending[k][0].t_arrive)
            rs, take, size = self.pending[m], [], 0
            while rs and (not take or size + rs[0].n <= self.cfg.max_batch):
                r = rs.pop(0)
                take.append(r)
                size += r.n
            if not rs:
                del self.pending[m]
            self._run(m, take)

    def _run(self, model_id: str, reqs: list) -> None:
        t0 = time.perf_counter()
        try:
            ev = self.models.get(model_id)
            x = reqs[0].x if len(reqs) == 1 else np.concatenate([r.x for r in reqs])
            logits, values = ev.evaluate(x)
            logits = np.asarray(logits, np.float32)
            values = np.asarray(values, np.float32)
            if self.cfg.check:
                check_response(logits, values, len(x))
        except Exception as e:                       # one bad model must not stop the server
            msg = f"{type(e).__name__}: {e}"
            if os.environ.get("CHESSRL_DEBUG"):
                msg += "\n" + traceback.format_exc(limit=4)
            for r in reqs:
                self.port.fail(r, msg)
            return
        secs = time.perf_counter() - t0
        st = self._stats.setdefault(model_id, EvalStats())
        st.add(len(x), secs)
        w = self._wait.setdefault(model_id, [0, 0.0])
        w[0] += len(reqs)
        w[1] += sum(t0 - r.t_arrive for r in reqs)
        at = 0
        for r in reqs:
            self.port.reply(r, logits[at:at + r.n], values[at:at + r.n])
            at += r.n

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        up = max(time.time() - self._t0, 1e-9)
        per = {}
        for m, st in self._stats.items():
            reqs, waited = self._wait.get(m, [0, 0.0])
            per[m] = {**st.to_dict(), "requests": reqs,
                      "avg_wait_ms": round(1000 * waited / reqs, 2) if reqs else 0.0}
        pos = sum(s.positions for s in self._stats.values())
        calls = sum(s.calls for s in self._stats.values())
        busy = sum(s.secs for s in self._stats.values())
        reqs_all = sum(w[0] for w in self._wait.values())
        wait_all = sum(w[1] for w in self._wait.values())
        return {"uptime_s": round(up, 1), "positions": pos, "calls": calls,
                "busy_s": round(busy, 3), "requests": reqs_all, "wait_s": round(wait_all, 3),
                "avg_batch": round(pos / calls, 1) if calls else 0.0,
                "pos_per_s": round(pos / up), "busy_pct": round(100 * busy / up, 1),
                "loaded": self.models.loaded(), "loads": self.models.loads, "models": per}

    def _dump(self) -> None:
        if not self.cfg.stats_path or time.time() - self._last_dump < 1.0:
            return
        self._last_dump = time.time()
        tmp = self.cfg.stats_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.stats(), f)
            os.replace(tmp, self.cfg.stats_path)
        except OSError:
            pass


# ---------------------------------------------------------------------- launchers
class _Handle:
    def __init__(self, worker, stop, server=None):
        self.worker, self.stop_event, self.server = worker, stop, server

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.worker.join(timeout)
        if self.server is not None:
            self.server.port.close()
        if hasattr(self.worker, "terminate") and self.worker.is_alive():
            self.worker.terminate()

    def alive(self) -> bool:
        return self.worker.is_alive()


def start_thread(hub, source, cfg: ServerConfig | None = None) -> _Handle:
    """Server inside this process (tests, or a quick single-process setup)."""
    srv = InferenceServer(open_server_port(hub.server_spec()), source, cfg)
    stop = threading.Event()
    t = threading.Thread(target=srv.serve, args=(stop,), name="inference-server", daemon=True)
    t.start()
    return _Handle(t, stop, srv)


def _process_main(spec, source_factory, source_kwargs, cfg_dict, stop) -> None:
    srv = InferenceServer(open_server_port(spec), source_factory(**source_kwargs),
                          ServerConfig(**cfg_dict))
    try:
        srv.serve(stop)
    finally:
        srv.port.close()


def start_process(hub, source_factory, source_kwargs: dict | None = None,
                  cfg: ServerConfig | None = None, ctx=None) -> _Handle:
    """Server in its own process. source_factory must be a module-level callable (it is
    pickled), e.g. TorchSource with {"device": "cuda", "template": "..."}."""
    ctx = ctx or mp.get_context("spawn")
    stop = ctx.Event()
    p = ctx.Process(target=_process_main, name="inference-server", daemon=True,
                    args=(hub.server_spec(), source_factory, source_kwargs or {},
                          asdict(cfg or ServerConfig()), stop))
    p.start()
    return _Handle(p, stop)
