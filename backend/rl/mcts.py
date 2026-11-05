"""Monte-Carlo tree search (PUCT) built for BATCHED evaluation.

A SearchJob never calls the network itself. It hands out the positions it wants
evaluated (collect), and takes the answers back (apply). The caller can therefore
mix the leaves of many games into one big GPU batch (see selfplay.py), and a
single game can also send several leaves at once thanks to virtual loss.

Value convention: every value is from the point of view of the side to move in
that position. Edge statistics (N, W) are from the point of view of the side
that chooses the edge, so a leaf value v is stored as -v on the edge into it.
"""
from __future__ import annotations

import math

import chess
import numpy as np

from .config import MCTSConfig
from .encoding import board_to_tensor, move_to_index
from .env import leaf_value


class Node:
    __slots__ = ("expanded", "terminal_v", "pending", "moves", "P0", "P", "N", "W", "kids")

    def __init__(self):
        self.expanded = False
        self.terminal_v = None
        self.pending = False
        self.moves = self.P0 = self.P = self.N = self.W = self.kids = None

    def expand(self, legal, logits):
        idxs = np.fromiter((i for i, _ in legal), dtype=np.int64, count=len(legal))
        z = logits[idxs].astype(np.float64)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        self.moves = [m for _, m in legal]
        self.P0 = p.astype(np.float32)
        self.P = self.P0.copy()
        self.N = np.zeros(len(legal), np.float32)
        self.W = np.zeros(len(legal), np.float32)
        self.kids = [None] * len(legal)
        self.expanded = True


class SearchJob:
    """The search for one move of one game."""

    def __init__(self, board: chess.Board, cfg: MCTSConfig, rng: np.random.Generator,
                 root: Node | None = None, noise: bool = False):
        self.board = board.copy(stack=False)       # working copy, always returned to the root
        self.cfg, self.rng, self.noise = cfg, rng, noise
        self.root = root or Node()
        self.remaining = cfg.simulations
        self._inflight: list = []
        if self.root.expanded:
            self._add_noise()

    # ------------------------------------------------------------ public API
    @property
    def done(self) -> bool:
        return self.root.expanded and self.remaining <= 0 and not self._inflight

    def collect(self, k: int) -> np.ndarray:
        """Positions (uint8 [n,18,8,8]) this search wants evaluated now; may be empty."""
        if self._inflight:
            raise RuntimeError("apply() must be called before collect()")
        if not self.root.expanded:
            legal = [(move_to_index(m, self.board), m) for m in self.board.legal_moves]
            if legal:
                self._inflight.append(([], self.root, legal, board_to_tensor(self.board)))
        else:
            for _ in range(k):
                if self.remaining <= 0 or not self._descend():
                    break
        if not self._inflight:
            return np.zeros((0, 18, 8, 8), np.uint8)
        return np.stack([e[3] for e in self._inflight])

    def apply(self, logits: np.ndarray, values: np.ndarray) -> None:
        for (path, node, legal, _), lg, v in zip(self._inflight, logits, values):
            if not node.expanded:
                node.expand(legal, lg)
            node.pending = False
            if path:
                self._backup(path, float(v))
        was_root = bool(self._inflight) and not self._inflight[0][0]
        self._inflight = []
        if was_root:
            self._add_noise()

    def advance(self, move: chess.Move):
        """Subtree reuse: the chosen child becomes the next root."""
        child = None
        if self.root.expanded:
            for i, m in enumerate(self.root.moves):
                if m == move:
                    child = self.root.kids[i]
                    break
        return child

    # ------------------------------------------------------------ results
    def visits(self) -> np.ndarray:
        return self.root.N

    def root_value(self) -> float:
        n = float(self.root.N.sum())
        return float(self.root.W.sum() / n) if n else 0.0

    def policy(self, temperature: float) -> np.ndarray:
        n = self.root.N.astype(np.float64)
        if temperature < 1e-3:
            best = np.flatnonzero(n == n.max())
            out = np.zeros_like(n)
            out[self.rng.choice(best)] = 1.0
            return out
        n = n ** (1.0 / temperature)
        return n / n.sum()

    def top_moves(self, k: int = 5) -> list[dict]:
        r = self.root
        order = np.argsort(-r.N)[:k]
        return [{"uci": r.moves[i].uci(), "prior": float(r.P0[i]), "visits": int(r.N[i]),
                 "q": float(r.W[i] / r.N[i]) if r.N[i] else 0.0} for i in order]

    # ------------------------------------------------------------ internals
    def _add_noise(self):
        r, c = self.root, self.cfg
        if self.noise and c.dirichlet_eps > 0 and r.moves:
            d = self.rng.dirichlet([c.dirichlet_alpha] * len(r.moves)).astype(np.float32)
            r.P = (1 - c.dirichlet_eps) * r.P0 + c.dirichlet_eps * d

    def _select(self, node: Node) -> int:
        n = node.N
        q = node.W / np.maximum(n, 1.0)
        u = self.cfg.c_puct * node.P * math.sqrt(float(n.sum()) + 1.0) / (1.0 + n)
        return int(np.argmax(q + u))

    def _descend(self) -> bool:
        """One simulation: walk down, expand or finish at a leaf. False = nothing queued."""
        vl = self.cfg.virtual_loss
        board, node, path = self.board, self.root, []
        while node.expanded and node.terminal_v is None:
            a = self._select(node)
            node.N[a] += vl
            node.W[a] -= vl
            path.append((node, a))
            board.push(node.moves[a])
            child = node.kids[a]
            if child is None:
                child = node.kids[a] = Node()
            node = child
        queued = True
        if node.terminal_v is not None:
            self._backup(path, node.terminal_v)
        elif node.pending:                                  # another leaf of this batch is there
            self._undo_virtual_loss(path)
            queued = False
        else:
            legal = [(move_to_index(m, board), m) for m in board.legal_moves]
            tv = leaf_value(board, len(legal))
            if tv is not None:
                node.terminal_v, node.expanded = tv, True
                self._backup(path, tv)
            else:
                node.pending = True
                self._inflight.append((path, node, legal, board_to_tensor(board)))
        for _ in path:
            board.pop()
        if queued:
            self.remaining -= 1
        return queued

    def _backup(self, path, v: float):
        vl = self.cfg.virtual_loss
        for node, a in reversed(path):
            v = -v
            node.N[a] += 1.0 - vl
            node.W[a] += v + vl

    def _undo_virtual_loss(self, path):
        vl = self.cfg.virtual_loss
        for node, a in path:
            node.N[a] -= vl
            node.W[a] += vl


def search(board: chess.Board, evaluator, cfg: MCTSConfig, seed: int | None = None,
           noise: bool = False) -> SearchJob:
    """Run one complete search of a single game (used by the Play page)."""
    job = SearchJob(board, cfg, np.random.default_rng(seed), noise=noise)
    while not job.done:
        x = job.collect(cfg.leaves_per_batch)
        if len(x) == 0:
            if job.remaining <= 0 or not job.root.expanded:
                break
            continue
        job.apply(*evaluator.evaluate(x))
    return job
