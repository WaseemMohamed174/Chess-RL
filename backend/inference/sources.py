"""Where the server gets its models. A source has two methods:

    load(model_id)    -> an evaluator (anything with evaluate(x))
    version(model_id) -> any comparable value that changes when the weights change

The server calls version() now and then and reloads when it changed, so a promoted
model is picked up without restarting anything."""
from __future__ import annotations

import os

import numpy as np

from ..rl.encoding import POLICY_SIZE
from ..rl.evaluator import UniformEvaluator


def _stamp(path: str):
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


class UniformSource:
    """Every model is the flat dummy network. For tests."""

    def load(self, model_id):
        return UniformEvaluator()

    def version(self, model_id):
        return 0


class SlowUniformSource:
    """The flat dummy network that takes `latency_ms + per_pos_us * n` to answer. It sleeps
    instead of computing, which is what waiting for a GPU looks like to the CPU. For benchmarks."""

    def __init__(self, latency_ms: float = 5.0, per_pos_us: float = 8.0):
        self.latency_ms, self.per_pos_us = latency_ms, per_pos_us

    def load(self, model_id):
        import time
        lat, per, inner = self.latency_ms / 1000.0, self.per_pos_us / 1e6, UniformEvaluator()

        class _Slow:
            def evaluate(self, x):
                time.sleep(lat + per * len(x))
                return inner.evaluate(x)
        return _Slow()

    def version(self, model_id):
        return 0


class DebugEvaluator:
    """Deterministic fake network: the answer depends on the input row AND on an offset,
    so tests can check that every row went back to the right caller and that a reload
    really changed the weights."""

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)

    def evaluate(self, x: np.ndarray):
        s = x.reshape(len(x), -1).sum(1).astype(np.float32) + np.float32(self.offset)
        logits = np.repeat(s[:, None], POLICY_SIZE, axis=1).astype(np.float32)
        return logits, ((s % 13) / 13).astype(np.float32)


class DebugSource:
    """Model <id> is the text file <root>/<id>.txt holding one number (its offset)."""

    def __init__(self, root: str):
        self.root = root
        self.loads = 0

    def _path(self, model_id):
        return os.path.join(self.root, f"{model_id}.txt")

    def load(self, model_id):
        self.loads += 1
        return DebugEvaluator(float(open(self._path(model_id)).read()))

    def version(self, model_id):
        return _stamp(self._path(model_id))


class TorchSource:
    """Real networks: model ids map to checkpoint files written by Trainer.save() / promote().
    `device` is the ONE place that decides cpu or cuda for the whole system.

        TorchSource("cuda", paths={"run7:best": "/data/m7.pt", ...})
        TorchSource("cpu",  template="/data/models/{id}.pt")"""

    def __init__(self, device: str = "cpu", paths: dict | None = None,
                 template: str | None = None, amp: bool = True, root: str | None = None):
        self.device, self.paths, self.template, self.amp = device, paths or {}, template, amp
        self.root = os.path.realpath(root) if root else None

    def _path(self, model_id):
        if model_id in self.paths:
            path = self.paths[model_id]
        elif self.template:
            path = self.template.format(id=model_id)
        else:
            raise KeyError(f"unknown model {model_id!r}")
        if self.root and os.path.commonpath([self.root, os.path.realpath(path)]) != self.root:
            raise PermissionError(f"model {model_id!r} is outside the allowed folder")
        return path

    def load(self, model_id):
        from ..rl.evaluator import TorchEvaluator
        from ..rl.trainer import load_network
        return TorchEvaluator(load_network(self._path(model_id), self.device), self.device, self.amp)

    def version(self, model_id):
        return _stamp(self._path(model_id))
