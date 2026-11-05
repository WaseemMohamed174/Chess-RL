"""Ring buffer of training positions.

State planes are binary, so they are stored as uint8 (1.1 KB each instead of 4.6 KB),
and the MCTS policy target is stored sparse (legal moves only), then expanded to
a dense 4672-vector only for the batch being trained on."""
from __future__ import annotations

import numpy as np

from .encoding import INPUT_PLANES, POLICY_SIZE


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.states = np.zeros((capacity, INPUT_PLANES, 8, 8), np.uint8)
        self.z = np.zeros(capacity, np.float32)
        self.pi: list = [None] * capacity
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state: np.ndarray, idxs: np.ndarray, probs: np.ndarray, z: float) -> None:
        self.states[self.ptr] = state
        self.pi[self.ptr] = (idxs.astype(np.int16), probs.astype(np.float16))
        self.z[self.ptr] = z
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def add_game(self, record) -> None:
        for s in record.samples:
            self.add(*s)

    def sample(self, batch: int, rng: np.random.Generator):
        ids = rng.integers(0, self.size, size=min(batch, self.size))
        pi = np.zeros((len(ids), POLICY_SIZE), np.float32)
        for row, i in enumerate(ids):
            idx, p = self.pi[i]
            pi[row, idx.astype(np.int64)] = p.astype(np.float32)
        return self.states[ids], pi, self.z[ids]

    def save(self, path) -> None:
        n = self.size
        order = np.arange(n) if n < self.cap else (np.arange(n) + self.ptr) % self.cap
        idx = [self.pi[i][0] for i in order]
        np.savez_compressed(
            path, states=self.states[order], z=self.z[order],
            pi_idx=np.concatenate(idx) if idx else np.zeros(0, np.int16),
            pi_prob=np.concatenate([self.pi[i][1] for i in order]) if idx else np.zeros(0, np.float16),
            lens=np.array([len(a) for a in idx], np.int32))

    def load(self, path) -> None:
        d = np.load(path)
        cuts = np.concatenate([[0], np.cumsum(d["lens"])])
        for i in range(len(d["z"])):
            self.add(d["states"][i], d["pi_idx"][cuts[i]:cuts[i + 1]],
                     d["pi_prob"][cuts[i]:cuts[i + 1]], float(d["z"][i]))
