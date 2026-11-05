"""Self-play in separate worker processes that never import PyTorch.

    coordinator (train process, one thread)             worker 0..N-1 (CPU only)
    ----------------------------------------            ------------------------------
    run_games(model, n_games) --task-->                 GamePool + RemoteEvaluator
                              <--game / move / done--   (the network is in the inference server)

The same workers also play the arena match (candidate vs best) through the server.
A worker costs ~0.2 GB instead of the ~1 GB a PyTorch process needs, and uses its own core,
so one model can now keep several cores busy. Only worker 0 reports live moves: the card
and the live board of a model show one game, and one reporter is enough."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import time
import traceback

INDEX_STRIDE = 1_000_000          # game index = worker * stride + local index (unique per run)


def _worker_main(wid: int, spec: dict, task_q, out_q, pause_ev, stop_ev, emit_moves: bool) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ..inference.client import RemoteEvaluator, pipeline_connections
    from .config import MCTSConfig
    from .arena import play_match
    from .selfplay import GamePool, GameRecord

    parent = mp.parent_process()
    # two connections let GamePool search one half of its games while the server answers the
    # other half (only the sock transport can open a second connection from the same spec)
    conns = pipeline_connections() if spec.get("kind") == "sock" else 1
    client = RemoteEvaluator(spec, "", timeout=120.0, connections=conns)
    last_check = 0.0

    def should_stop() -> bool:
        nonlocal last_check
        while pause_ev.is_set() and not stop_ev.is_set():
            time.sleep(0.1)
        now = time.time()
        if now - last_check > 1.0:                       # the train process died: do not linger
            last_check = now
            if parent is not None and not parent.is_alive():
                return True
        return stop_ev.is_set()

    while True:
        try:
            task = task_q.get(timeout=2.0)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                return
            continue
        if task is None:
            return
        tid, kind, a = task
        cfg = MCTSConfig.from_dict(a["mcts"])
        n_games, concurrent = a["n_games"], max(1, min(a["concurrent"], a["n_games"]))
        try:
            if kind == "match":
                ev_a, ev_b = client.for_model(a["a"]), client.for_model(a["b"])

                def on_mgame(rec):
                    a_white = rec.white_eval is ev_a
                    if rec.result == 0:
                        o = "d"
                    else:
                        o = "a" if (rec.result == 1) == a_white else "b"
                    out_q.put(("mgame", wid, tid, o))

                play_match(ev_a, ev_b, n_games, cfg, concurrent, a["seed"], should_stop,
                           on_game=on_mgame, offset=a["offset"])
            else:
                ev = client.for_model(a["model"])
                pool = GamePool(cfg, seed=a["seed"])
                state = {"track": None, "last": 0.0}

                def on_move(rec, board, move, value, top):
                    if not emit_moves:
                        return
                    if state["track"] is None:
                        state["track"] = rec.index
                    now = time.time()
                    if rec.index != state["track"] or now - state["last"] < 0.12:
                        return
                    state["last"] = now
                    out_q.put(("move", wid, tid, {
                        "index": rec.index, "fen": board.fen(), "uci": move.uci(),
                        "ply": len(rec.moves), "value": float(value), "turn": board.turn,
                        "top": top[:3]}))

                def on_game(rec):
                    if rec.index == state["track"]:
                        state["track"] = None
                    slim = GameRecord(rec.index, rec.moves, rec.result, rec.samples, None, None,
                                      rec.start_fen)
                    out_q.put(("game", wid, tid, slim))

                pool.run([(ev, ev)] * n_games, concurrent=concurrent, should_stop=should_stop,
                         on_move=on_move, on_game=on_game)
            out_q.put(("done", wid, tid, None))
        except Exception:                                # report, stay alive for the next task
            out_q.put(("error", wid, tid, traceback.format_exc(limit=6)))


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextlib.contextmanager
def _single_thread_env():
    """A worker owns ONE core. Its numpy must not start a thread pool of its own (N workers
    x all cores would fight each other). BLAS reads these variables when it is imported, so
    they are set around process start instead of inside the child."""
    old = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class SelfplayWorkers:
    """A resizable set of long-lived workers, used by one training run."""

    def __init__(self, n: int, spec: dict, ctx=None, max_respawns: int = 3):
        self.n, self.spec, self.ctx = max(1, n), spec, ctx or mp.get_context("spawn")
        self.max_respawns = max_respawns
        self.out_q = self.ctx.Queue()
        self.pause_ev, self.stop_ev = self.ctx.Event(), self.ctx.Event()
        self.task_qs = [self.ctx.Queue() for _ in range(self.n)]
        self.procs: list = [None] * self.n
        self._tid = 0
        for i in range(self.n):
            self._spawn(i)

    # ---------------------------------------------------------------- life cycle
    def _spawn(self, i: int) -> None:
        self.task_qs[i] = self.ctx.Queue()
        p = self.ctx.Process(target=_worker_main, name=f"selfplay-{i}", daemon=True,
                             args=(i, self.spec, self.task_qs[i], self.out_q,
                                   self.pause_ev, self.stop_ev, i == 0))
        with _single_thread_env():
            p.start()
        self.procs[i] = p

    def resize(self, n: int) -> None:
        """Grow or shrink the pool. Call it between tasks (never inside run_games / run_match).
        Worker 0 always stays: it is the one that reports the live game."""
        n = max(1, int(n))
        while self.n < n:
            self.task_qs.append(self.ctx.Queue())
            self.procs.append(None)
            self.n += 1
            self._spawn(self.n - 1)
        while self.n > n:
            i = self.n - 1
            try:
                self.task_qs[i].put(None)
            except Exception:
                pass
            p = self.procs[i]
            p.join(3)
            if p.is_alive():
                p.terminate()
            self.task_qs.pop()
            self.procs.pop()
            self.n -= 1

    def pids(self) -> list:
        return [p.pid for p in self.procs]

    def set_paused(self, on: bool) -> None:
        (self.pause_ev.set if on else self.pause_ev.clear)()

    def stop(self) -> None:
        """Make every worker abandon what it is doing (run_games returns soon after)."""
        self.stop_ev.set()

    def close(self) -> None:
        self.stop_ev.set()
        for q in self.task_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(2)
            if p.is_alive():
                p.terminate()

    # ---------------------------------------------------------------- work
    def _send(self, i: int, tid: int, kind: str, base: dict, k: int, seed: int, offset: int) -> None:
        args = {**base, "n_games": k, "seed": seed + i * 7919 + tid, "offset": offset}
        self.task_qs[i].put((tid, kind, args))

    def run_games(self, model_id: str, n_games: int, mcts: dict, concurrent: int, seed: int,
                  on_game=None, on_move=None, poll_stop=lambda: False) -> int:
        """Self-play: n_games split over the workers. Blocks until all are finished (or
        stopped). Returns how many games were delivered. A worker that dies is replaced and
        its unfinished games are handed to the new one."""
        base = {"model": model_id, "mcts": mcts, "concurrent": concurrent}
        return self._dispatch("selfplay", base, n_games, seed, on_game, on_move, poll_stop)

    def run_match(self, a_id: str, b_id: str, n_games: int, mcts: dict, concurrent: int,
                  seed: int, on_progress=None, poll_stop=lambda: False) -> dict:
        """Arena: model A against model B over the workers, colours alternating (also across
        workers). Returns {"a": wins of A, "b": wins of B, "d": draws}.
        on_progress(finished_games) after every game."""
        tally = {"a": 0, "b": 0, "d": 0}

        def got(outcome):
            tally[outcome] += 1
            if on_progress:
                on_progress(sum(tally.values()))

        base = {"a": a_id, "b": b_id, "mcts": mcts, "concurrent": concurrent}
        self._dispatch("match", base, n_games, seed, got, None, poll_stop)
        return tally

    def _dispatch(self, kind: str, base: dict, n_games: int, seed: int, on_game, on_move,
                  poll_stop) -> int:
        self._tid += 1
        tid, n = self._tid, self.n
        assigned = {i: n_games // n + (1 if i < n_games % n else 0) for i in range(n)}
        assigned = {i: k for i, k in assigned.items() if k > 0}
        starts, at = {}, 0
        for i, k in assigned.items():                     # first game number of each worker
            starts[i], at = at, at + k
        finished = {i: 0 for i in assigned}
        respawns = {i: 0 for i in assigned}
        for i, k in assigned.items():
            self._send(i, tid, kind, base, k, seed, starts[i])
        pending, delivered = set(assigned), 0

        while pending:
            try:
                msg, wid, mtid, payload = self.out_q.get(timeout=0.2)
            except queue.Empty:
                if poll_stop():
                    self.stop()
                    break
                for i in list(pending):
                    if not self.procs[i].is_alive():
                        if respawns[i] >= self.max_respawns:
                            raise RuntimeError(f"self-play worker {i} keeps dying")
                        respawns[i] += 1
                        left = assigned[i] - finished[i]
                        self._spawn(i)
                        if left > 0:
                            self._send(i, tid, kind, base, left, seed + 1000 * respawns[i],
                                       starts[i] + finished[i])
                        else:
                            pending.discard(i)
                continue
            if mtid != tid:
                continue                                  # leftovers of an earlier task
            if msg in ("game", "mgame"):
                finished[wid] = finished.get(wid, 0) + 1
                delivered += 1
                if msg == "game":
                    payload.index += wid * INDEX_STRIDE
                if on_game:
                    on_game(payload)
            elif msg == "move":
                payload["index"] += wid * INDEX_STRIDE
                if on_move:
                    on_move(payload)
            elif msg == "done":
                pending.discard(wid)
            elif msg == "error":
                raise RuntimeError(f"self-play worker {wid} failed:\n{payload}")
            if poll_stop():
                self.stop()
                break
        return delivered
