"""Plays many games at once and feeds every leaf to the network in big batches.

One GamePool = one process. All its running games are advanced together:

    loop:  every game collects leaves  ->  ONE evaluate() call per network
           ->  results go back to their games  ->  finished moves are played

A game that ends is replaced at once, so the GPU batch stays full.
Self-play and arena matches use the same pool; only the (white, black)
evaluators of each game differ."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.pgn
import numpy as np

from .config import MCTSConfig
from .encoding import board_to_tensor
from .env import game_result
from .mcts import SearchJob


@dataclass
class GameRecord:
    index: int
    moves: list = field(default_factory=list)      # UCI strings
    result: int = 0                                # White's view: 1 / 0 / -1
    samples: list = field(default_factory=list)    # (state, idxs, probs, z)
    white_eval: object = None
    black_eval: object = None
    start_fen: str = chess.STARTING_FEN

    def pgn(self) -> str:
        b = chess.Board(self.start_fen)
        game = chess.pgn.Game()
        game.setup(b)
        node = game
        for u in self.moves:
            node = node.add_variation(chess.Move.from_uci(u))
        game.headers["Result"] = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}[self.result]
        return str(game)


class _Slot:
    def __init__(self, rec: GameRecord, cfg: MCTSConfig):
        self.rec, self.cfg = rec, cfg
        self.board = chess.Board(rec.start_fen)
        self.job = None
        self.raw: list = []                        # (state, idxs, probs, white_to_move)
        self.reuse = cfg.reuse_tree and rec.white_eval is rec.black_eval
        self.noise = cfg.dirichlet_eps > 0 and rec.white_eval is rec.black_eval

    def evaluator(self):
        return self.rec.white_eval if self.board.turn == chess.WHITE else self.rec.black_eval


class GamePool:
    def __init__(self, cfg: MCTSConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def run(self, specs: list, concurrent: int = 32, should_stop=None, on_move=None,
            on_game=None, collect_samples: bool = True, pipeline: bool | None = None) -> list[GameRecord]:
        """specs: list of (white_evaluator, black_evaluator). Returns finished records.

        on_move(record, board_after_move, move, root_value, top_moves)  - live view
        on_game(record)                                                 - after each game
        should_stop() -> bool                                            - pause / stop
        pipeline: None = automatic. The games are split in two halves that take turns: while
        the network answers one half, this process searches the other. It needs evaluators with
        evaluate_async (a RemoteEvaluator with 2 connections) and at least 4 games at once;
        otherwise the plain loop runs, exactly as before."""
        if pipeline is None:
            pipeline = self._can_pipeline(specs, concurrent)
        if pipeline:
            return self._run_pipelined(specs, concurrent, should_stop, on_move, on_game, collect_samples)
        return self._run_serial(specs, concurrent, should_stop, on_move, on_game, collect_samples)

    @staticmethod
    def _can_pipeline(specs: list, concurrent: int) -> bool:
        if concurrent < 4 or not specs:
            return False
        seen = {}
        for w, b in specs:
            seen[id(w)] = w
            seen[id(b)] = b
        return all(getattr(e, "pipelined", False) for e in seen.values())

    def _run_serial(self, specs, concurrent, should_stop, on_move, on_game, collect_samples):
        queue = [GameRecord(i, white_eval=w, black_eval=b) for i, (w, b) in enumerate(specs)]
        queue.reverse()
        active: list[_Slot] = []
        finished: list[GameRecord] = []
        cfg = self.cfg

        while (queue or active) and not (should_stop and should_stop()):
            while queue and len(active) < concurrent:
                active.append(_Slot(queue.pop(), cfg))

            # 1) gather the leaves of every game, grouped by network
            groups = self._gather(active)

            # 2) one network call per network, then hand the answers back
            for ev, items in groups.values():
                logits, values = ev.evaluate(np.concatenate([x for _, x in items]))
                self._hand_back(items, logits, values)

            # 3) play the moves whose search is complete
            self._play_ready(active, finished, collect_samples, on_move, on_game)
        return finished

    def _run_pipelined(self, specs, concurrent, should_stop, on_move, on_game, collect_samples):
        """Two halves of the games take turns. Each turn of a half is: take the answers of its
        last request, play the finished moves, collect new leaves, send them WITHOUT waiting.
        While that request is at the server, the other half takes its turn."""
        queue = [GameRecord(i, white_eval=w, black_eval=b) for i, (w, b) in enumerate(specs)]
        queue.reverse()
        finished: list[GameRecord] = []
        cfg = self.cfg
        caps = [(concurrent + 1) // 2, concurrent // 2]
        active: list[list[_Slot]] = [[], []]
        sent: list = [None, None]                  # per half: [(future, items), ...] in flight
        try:
            while (queue or active[0] or active[1] or sent[0] or sent[1]) \
                    and not (should_stop and should_stop()):
                for h in (0, 1):
                    if sent[h]:
                        for fut, items in sent[h]:
                            logits, values = fut.result()
                            self._hand_back(items, logits, values)
                        sent[h] = None
                    self._play_ready(active[h], finished, collect_samples, on_move, on_game)
                    while queue and len(active[h]) < caps[h]:
                        active[h].append(_Slot(queue.pop(), cfg))
                    groups = self._gather(active[h])
                    if groups:
                        sent[h] = [(ev.evaluate_async(np.concatenate([x for _, x in items])), items)
                                   for ev, items in groups.values()]
        finally:
            for h in (0, 1):                       # never leave a request on a connection
                for fut, _ in sent[h] or []:       # that the next caller is going to use
                    try:
                        fut.result()
                    except Exception:              # noqa: BLE001 - the original error wins
                        pass
        return finished

    # ---------------------------------------------------------------- shared steps
    def _gather(self, active: list) -> dict:
        """Every game collects leaves; returns {id(network): (network, [(slot, leaves)])}."""
        groups: dict = {}
        for s in active:
            if s.job is None:
                s.job = SearchJob(s.board, self.cfg, self.rng, noise=s.noise)
            x = s.job.collect(self.cfg.leaves_per_batch)
            if len(x):
                groups.setdefault(id(s.evaluator()), (s.evaluator(), []))[1].append((s, x))
        return groups

    @staticmethod
    def _hand_back(items: list, logits, values) -> None:
        at = 0
        for s, x in items:
            s.job.apply(logits[at:at + len(x)], values[at:at + len(x)])
            at += len(x)

    def _play_ready(self, active: list, finished: list, collect_samples, on_move, on_game) -> None:
        """Play the moves whose search is complete; a game that ends leaves `active`."""
        cfg = self.cfg
        for s in list(active):
            if s.job is not None and s.job.done and s.job.root.moves:
                self._play(s, collect_samples, on_move)
                res = game_result(s.board, len(s.rec.moves), cfg.max_plies)
                if res is not None:
                    self._finish(s, res)
                    active.remove(s)
                    finished.append(s.rec)
                    if on_game:
                        on_game(s.rec)
            elif s.job is not None and s.job.root.expanded and not s.job.root.moves:
                self._finish(s, game_result(s.board, 10**9, 0) or 0)   # no legal move
                active.remove(s)
                finished.append(s.rec)

    # ------------------------------------------------------------------
    def _play(self, s: _Slot, collect: bool, on_move) -> None:
        job, cfg, board = s.job, self.cfg, s.board
        if collect:
            pi = job.policy(1.0)
            keep = pi > 0
            idxs = np.array([_idx(board, m) for m, k in zip(job.root.moves, keep) if k])
            s.raw.append((board_to_tensor(board), idxs, pi[keep], board.turn == chess.WHITE))
        t = cfg.temperature if len(s.rec.moves) < cfg.temperature_moves else 0.0
        probs = job.policy(t)
        a = int(self.rng.choice(len(probs), p=probs))
        move = job.root.moves[a]
        value, top = job.root_value(), job.top_moves(5)
        child = job.root.kids[a] if s.reuse else None
        board.push(move)
        s.rec.moves.append(move.uci())
        s.job = (SearchJob(board, cfg, self.rng, root=child, noise=s.noise)
                 if child is not None and child.expanded else None)
        if on_move:
            on_move(s.rec, board, move, value, top)

    def _finish(self, s: _Slot, result: int) -> None:
        s.rec.result = result
        s.rec.samples = [(st, i, p, float(result if white else -result))
                         for st, i, p, white in s.raw]


def _idx(board, move):
    from .encoding import move_to_index
    return move_to_index(move, board)
