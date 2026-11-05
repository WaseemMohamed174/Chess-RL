"""The training loop of ONE run (runs in a thread inside train_worker).

    iteration:  self-play (best net)  ->  learner (candidate net)  ->  arena (candidate vs best)
                -> promote when the candidate scores >= arena_threshold

Everything that needs PyTorch sits behind a "backend" object, so the same loop also
runs with FakeBackend (CHESSRL_FAKE_ENGINE=1): real chess games and real arena
matches, but no network and synthetic losses. That is how the pipeline is tested
on machines without PyTorch."""
from __future__ import annotations

import os
import threading
import time
import traceback
from collections import deque

import chess
import numpy as np

from .arena import EloRating, MatchResult, play_match
from .config import MCTSConfig, TrainConfig
from .evaluator import CountingEvaluator, UniformEvaluator
from .replay_buffer import ReplayBuffer
from .selfplay import GamePool


class EventLog:
    """Numbered events the gateway polls with GET /events?after=N."""

    def __init__(self, keep: int = 3000):
        self.boot_id = os.urandom(4).hex()
        self._lock = threading.Lock()
        self._items: deque = deque(maxlen=keep)
        self._seq = 0

    def add(self, type_: str, **data) -> int:
        with self._lock:
            self._seq += 1
            self._items.append({"seq": self._seq, "type": type_, **data})
            return self._seq

    def since(self, after: int) -> dict:
        with self._lock:
            return {"boot_id": self.boot_id, "last": self._seq,
                    "events": [e for e in self._items if e["seq"] > after]}


def _atomic(path: str, write) -> None:
    tmp = f"{path}.tmp"
    write(tmp)
    os.replace(tmp, path)


class TorchBackend:
    def __init__(self, model_path: str, device: str, tc: TrainConfig):
        self.path, self.cand_path, self.device, self.tc = model_path, model_path + ".cand", device, tc

    def setup(self) -> None:
        from .evaluator import TorchEvaluator
        from .trainer import Trainer, load_network
        best = load_network(self.path, self.device)
        cand = load_network(self.cand_path if os.path.exists(self.cand_path) else self.path,
                            self.device)
        self.trainer = Trainer(cand, self.tc, self.device)
        if os.path.exists(self.cand_path):
            try:
                self.trainer.load_optimizer(self.cand_path)
            except Exception:
                pass
        self.cand_net = cand
        self.best_ev = TorchEvaluator(best, self.device)
        self.cand_ev = TorchEvaluator(cand, self.device)

    def train(self, buffer, steps, rng, should_stop) -> dict:
        try:
            return self.trainer.train(buffer, steps, rng, should_stop)
        finally:
            if self.device == "cuda":              # let the next learner use the activation memory
                import torch
                torch.cuda.empty_cache()

    def steps_for(self, n: int) -> int:
        return self.trainer.steps_for(n)

    def save_candidate(self) -> None:
        _atomic(self.cand_path, self.trainer.save)

    def promote(self) -> None:
        import torch
        self.best_ev.set_weights(self.cand_net.state_dict())

        def write(p):
            torch.save({"config": self.cand_net.config(), "model": self.cand_net.state_dict(),
                        "opt": None, "steps": self.trainer.step_count}, p)
        _atomic(self.path, write)


class FakeBackend:
    def __init__(self, model_path: str, device: str, tc: TrainConfig):
        self.path, self.cand_path = model_path, model_path + ".cand"
        self._k = 0

    def setup(self) -> None:
        self.best_ev, self.cand_ev = UniformEvaluator(), UniformEvaluator()

    def steps_for(self, n: int) -> int:
        return 5

    def train(self, buffer, steps, rng, should_stop) -> dict:
        time.sleep(0.2)
        self._k += 1
        return {"policy_loss": 6.0 * 0.95 ** self._k + 0.5, "value_loss": 0.9 * 0.97 ** self._k,
                "steps": steps}

    def save_candidate(self) -> None:
        _atomic(self.cand_path, lambda p: open(p, "w").write("fake candidate"))

    def promote(self) -> None:
        _atomic(self.path, lambda p: open(p, "w").write("fake best"))


class Orchestrator:
    def __init__(self, backend, payload: dict, events: EventLog, remote: dict | None = None,
                 gate=None):
        """remote = {"spec": client spec of the inference server, "workers": n} plays self-play
        in n separate worker processes through the server; None = play inside this process."""
        p = payload
        self.gate = gate                           # LearnerGate shared by all runs, or None
        self.remote = remote
        self.workers = None                        # SelfplayWorkers, created in _run()
        self.want_workers: int | None = None       # asked by the gateway when other runs start / end
        self.refresher = None                      # tells the server a model file changed
        self.be, self.ev = backend, events
        self.run_id = p["run_id"]
        self.mcts = MCTSConfig.from_dict(p.get("mcts"))
        self.tc = TrainConfig.from_dict(p.get("train"))
        self.path = p["model_path"]
        self.elo = float(p.get("elo", 1000.0))
        self.iteration = int(p.get("iterations", 0))     # all iterations this model has ever done
        self.max_iterations = max(0, int(p.get("max_iterations") or 0))   # 0 = until stopped
        self.iterations_run = 0                          # iterations done by THIS run
        self.total_games = 0
        self.state = "starting"
        self.last_metrics: dict | None = None
        self.error: str | None = None
        self.buffer = ReplayBuffer(self.tc.buffer_size)
        self.rng = np.random.default_rng()
        self._stop, self._resume = threading.Event(), threading.Event()
        self._resume.set()
        self._paused_s = 0.0                       # total time spent paused (not counted as work)
        self._pause_t0: float | None = None
        self._thread = threading.Thread(target=self._run, name="train-loop", daemon=True)
        self.watched = False                       # True only for the run open on the website
        self._track: int | None = None             # index of the game shown on this run's card
        self._last_live = 0.0                      # last light "live" event (every run)
        self._last_full = 0.0                      # last full "selfplay_move" event (watched only)
        self.live: dict | None = None              # newest summary of the tracked game
        self.phase = {"name": "starting", "done": 0, "total": 0}   # what the loop does now

    # ----------------------------------------------------------- controls
    def start(self) -> None:
        self._thread.start()

    def pause(self) -> None:
        if self.state == "running":
            self._pause_t0 = time.time()
            self._resume.clear()
            if self.workers:
                self.workers.set_paused(True)
            self._set("paused")

    def resume(self) -> None:
        if self.state == "paused":
            self._end_pause()
            self._set("running")
            if self.workers:
                self.workers.set_paused(False)
            self._resume.set()

    def _end_pause(self) -> None:
        if self._pause_t0 is not None:
            self._paused_s += time.time() - self._pause_t0
            self._pause_t0 = None

    def _paused_total(self) -> float:
        extra = time.time() - self._pause_t0 if self._pause_t0 is not None else 0.0
        return self._paused_s + extra

    def stop(self) -> None:
        self._end_pause()
        self._stop.set()
        self._resume.set()
        if self.workers:
            self.workers.set_paused(False)
            self.workers.stop()

    def alive(self) -> bool:
        return self._thread.is_alive()

    def set_workers(self, n: int) -> None:
        """Ask for n self-play workers; applied at the start of the next iteration."""
        self.want_workers = max(1, int(n))

    def set_watch(self, on: bool) -> None:
        self.watched = bool(on)

    def status(self) -> dict:
        return {"run_id": self.run_id, "state": self.state, "iteration": self.iteration,
                "elo": round(self.elo, 1), "games": self.total_games,
                "buffer": len(self.buffer), "error": self.error, "last": self.last_metrics,
                "iterations_run": self.iterations_run, "max_iterations": self.max_iterations,
                "watched": self.watched, "live": self.live, "phase": self.phase,
                "workers": self.workers.n if self.workers else 0}

    # ------------------------------------------------------------- loop
    def _set(self, state: str) -> None:
        self.state = state
        self.ev.add("status", run_id=self.run_id, state=state, error=self.error)

    def _set_phase(self, name: str, done: int = 0, total: int = 0) -> None:
        self.phase = {"name": name, "done": done, "total": total}
        self.ev.add("phase", run_id=self.run_id, iteration=self.iteration, **self.phase)

    def _should_stop(self) -> bool:
        while not self._resume.is_set() and not self._stop.is_set():
            self._resume.wait(0.2)                 # paused: block the loop right here
        return self._stop.is_set()

    def _run(self) -> None:
        try:
            self._set("running")
            self.be.setup()
            # one wrapper per network for the whole run: GamePool groups games by evaluator
            # identity, and the wrappers count what the network side sees.
            self.ev_best = CountingEvaluator(self.be.best_ev)
            self.ev_cand = CountingEvaluator(self.be.cand_ev)
            if self.remote:
                from ..inference.client import RemoteEvaluator
                from .selfplay_workers import SelfplayWorkers
                self.refresher = RemoteEvaluator(self.remote["spec"], os.path.abspath(self.path))
                self.workers = SelfplayWorkers(self.remote["workers"], self.remote["spec"])
            buf = self.path + ".buffer.npz"
            if os.path.exists(buf):
                try:
                    self.buffer.load(buf)
                except Exception:
                    pass
            finished = False                       # True: reached max_iterations by itself
            while not self._should_stop():
                self.iteration += 1
                m = self._iteration()
                if m is None:                      # stopped in the middle
                    self.iteration -= 1
                    break
                self.iterations_run += 1
                self.last_metrics = m
                self.ev.add("metrics", run_id=self.run_id, **m)
                if self.max_iterations and self.iterations_run >= self.max_iterations:
                    finished = True
                    break
            if len(self.buffer):
                try:
                    self.buffer.save(buf)
                except Exception:
                    pass
            self._set("done" if finished else "stopped")
        except Exception:
            self.error = traceback.format_exc(limit=6)
            self._set("failed")
        finally:
            if self.workers:
                self.workers.close()
            if self.refresher:
                self.refresher.close()

    def _iteration(self) -> dict | None:
        tc, be = self.tc, self.be
        if self.workers and self.want_workers and self.want_workers != self.workers.n:
            self.workers.resize(self.want_workers)   # other runs started or ended: share fairly
        # 1) self-play with the best network
        t0, done, p0 = time.time(), 0, self._paused_total()
        self._set_phase("selfplay", 0, tc.games_per_iteration)

        def on_game(rec):
            nonlocal done
            done += 1
            self.total_games += 1
            self.buffer.add_game(rec)
            self._set_phase("selfplay", done, tc.games_per_iteration)
            if rec.index == self._track:
                self._track = None
                self._publish_live(_replay(rec), len(rec.moves), None, ended=rec.result)
            keep_pgn = self.total_games % 10 == 0
            self.ev.add("game", run_id=self.run_id, plies=len(rec.moves), result=rec.result,
                        pgn=rec.pgn() if keep_pgn else None)

        seed = int(self.rng.integers(1 << 31))
        self.ev_best.stats(reset=True)
        self.ev_cand.stats(reset=True)
        if self.workers:
            per_worker = -(-tc.games_per_iteration // self.workers.n)
            self.workers.run_games(
                os.path.abspath(self.path), tc.games_per_iteration, self.mcts.to_dict(),
                concurrent=min(tc.concurrent_games, per_worker), seed=seed,
                on_game=on_game, on_move=self._on_remote_move, poll_stop=self._stop.is_set)
        else:
            pool = GamePool(self.mcts, seed=seed)
            pool.run([(self.ev_best, self.ev_best)] * tc.games_per_iteration,
                     concurrent=min(tc.concurrent_games, tc.games_per_iteration),
                     should_stop=self._should_stop, on_move=self._on_move, on_game=on_game)
        if self._stop.is_set():
            return None
        t_play = max(time.time() - t0 - (self._paused_total() - p0), 1e-6)   # pauses are not work
        infer_play = self.ev_best.stats(reset=True)

        # 2) learner
        stats = {"policy_loss": None, "value_loss": None, "steps": 0}
        t1, p1 = time.time(), self._paused_total()
        if len(self.buffer) >= max(tc.min_buffer, 1):
            self._set_phase("learning")
            if self.gate and not self.gate.acquire(self._should_stop):
                return None                        # stopped while waiting for a free learner slot
            try:
                stats = be.train(self.buffer, be.steps_for(len(self.buffer)), self.rng,
                                 self._should_stop)
            finally:
                if self.gate:
                    self.gate.release()
            if self._stop.is_set():
                return None
            be.save_candidate()
            if self.refresher:                     # the server must see the new candidate weights
                self.refresher.refresh(self._cand_id())
        t_train = max(time.time() - t1 - (self._paused_total() - p1), 0.0)

        # 3) arena gate
        arena, promoted = None, False
        if stats["steps"] > 0:
            self._set_phase("arena", 0, tc.arena_games)
            cfg = MCTSConfig.from_dict({**self.mcts.to_dict(),
                                        "simulations": tc.arena_simulations})
            seed = int(self.rng.integers(1 << 31))
            if self.workers:                       # the workers play it, the server thinks
                per_worker = -(-tc.arena_games // self.workers.n)
                tally = self.workers.run_match(
                    self._cand_id(), os.path.abspath(self.path), tc.arena_games, cfg.to_dict(),
                    concurrent=max(1, min(tc.concurrent_games, per_worker)), seed=seed,
                    on_progress=lambda k: self._set_phase("arena", k, tc.arena_games),
                    poll_stop=self._stop.is_set)
                res = MatchResult(tally["a"], tally["b"], tally["d"], [])
            else:
                res = play_match(self.ev_cand, self.ev_best, tc.arena_games, cfg,
                                 concurrent=max(1, min(tc.concurrent_games, tc.arena_games)),
                                 seed=seed, should_stop=self._should_stop)
            if self._stop.is_set():
                return None
            arena = {"wins": res.a_wins, "losses": res.b_wins, "draws": res.draws,
                     "score": round(res.score_a, 3)}
            if res.score_a >= tc.arena_threshold:
                promoted = True
                self.elo += max(EloRating.diff_from_score(res.score_a, res.n), 0.0)
                be.promote()
                if self.refresher:                 # the server must not keep the old weights
                    self.refresher.refresh(os.path.abspath(self.path))
        infer_arena = self.ev_cand.stats(reset=True).merge(self.ev_best.stats(reset=True))
        return {"iteration": self.iteration,
                "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"],
                "elo": round(self.elo, 1), "games": self.total_games,
                "games_per_min": round(done / max(t_play / 60, 1e-6), 1),
                "promoted": promoted, "arena": arena, "buffer": len(self.buffer),
                "steps": stats["steps"],
                "seconds": {"selfplay": round(t_play, 1), "train": round(t_train, 1)},
                "infer": {"selfplay": {"remote": True} if self.workers else infer_play.to_dict(),
                          "arena": {"remote": True} if self.workers else infer_arena.to_dict()}}

    def _on_move(self, rec, board: chess.Board, move, value: float, top) -> None:
        """Follow one game at a time.

        Every run sends a light "live" summary (piece counts, move number) about once a
        second, so its card stays up to date. Only the run that is open on the website
        (self.watched) also sends the full board position ~8 times a second."""
        if self._track is None:
            self._track = rec.index
        if rec.index != self._track:
            return
        white = value if board.turn == chess.BLACK else -value     # value was for the mover
        self._emit(white, move.uci(), len(rec.moves), top, board=board)

    def _cand_id(self) -> str:
        return os.path.abspath(self.path) + ".cand"        # same naming as the backends use

    def _on_remote_move(self, m: dict) -> None:
        """Same as _on_move, for a move reported by a self-play worker (plain dict)."""
        if self._track is None:
            self._track = m["index"]
        if m["index"] != self._track:
            return
        white = m["value"] if m["turn"] == chess.BLACK else -m["value"]
        self._emit(white, m["uci"], m["ply"], m["top"], fen=m["fen"])

    def _emit(self, white: float, uci: str, ply: int, top, board=None, fen: str | None = None) -> None:
        now = time.time()
        if self.watched and now - self._last_full >= 0.12:
            self._last_full = now
            self.ev.add("selfplay_move", run_id=self.run_id, fen=fen or board.fen(), uci=uci,
                        ply=ply, eval=round(white, 3), top=top[:3])
        if now - self._last_live >= 1.0:
            self._publish_live(board or chess.Board(fen), ply, round(white, 3))

    def _publish_live(self, board: chess.Board, ply: int, ev, ended: int | None = None) -> None:
        self._last_live = time.time()
        self.live = {"ply": ply, "eval": ev,
                     "white_pieces": chess.popcount(board.occupied_co[chess.WHITE]),
                     "black_pieces": chess.popcount(board.occupied_co[chess.BLACK]),
                     "ended": ended is not None,
                     "result": None if ended is None else {1: "white", -1: "black", 0: "draw"}[ended]}
        extra = {"fen": board.fen()} if ended is not None else {}   # final position
        self.ev.add("live", run_id=self.run_id, **self.live, **extra)


def _replay(rec) -> chess.Board:
    """Final position of a finished game record."""
    b = chess.Board(rec.start_fen)
    for u in rec.moves:
        b.push(chess.Move.from_uci(u))
    return b
