"""Anything that turns a batch of positions into (policy logits, values).

MCTS only knows this interface, so the same search runs on the GPU, on the CPU,
with ONNX Runtime later, or with a dummy in tests."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .encoding import INPUT_PLANES, POLICY_SIZE

# ----------------------------------------------------------------------------
# THE CONTRACT. Every evaluator, local or remote, honours exactly this:
#
#   evaluate(x)   x       uint8   [n, 18, 8, 8]   positions, seen from the side to move
#              -> logits  float32 [n, 4672]       raw policy scores (no softmax)
#                 values  float32 [n]             in [-1, 1], for the side to move
#
# n >= 1. Row i of the answer belongs to row i of the request. Nothing else may
# cross this boundary, so an inference server can later sit behind the same call
# without MCTS noticing.
# ----------------------------------------------------------------------------
INPUT_SHAPE = (INPUT_PLANES, 8, 8)
REQUEST_DTYPE = np.uint8
LOGITS_DTYPE = np.float32
VALUES_DTYPE = np.float32


class Evaluator(Protocol):
    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def check_request(x: np.ndarray) -> None:
    """Raise ValueError if x is not a valid request batch."""
    if not isinstance(x, np.ndarray) or x.dtype != REQUEST_DTYPE:
        raise ValueError(f"request must be a numpy uint8 array, got {getattr(x, 'dtype', type(x))}")
    if x.ndim != 4 or x.shape[1:] != INPUT_SHAPE or x.shape[0] < 1:
        raise ValueError(f"request shape must be [n>=1, {INPUT_PLANES}, 8, 8], got {x.shape}")


def check_response(logits: np.ndarray, values: np.ndarray, n: int) -> None:
    """Raise ValueError if (logits, values) is not a valid answer to a request of n rows."""
    if logits.dtype != LOGITS_DTYPE or logits.shape != (n, POLICY_SIZE):
        raise ValueError(f"logits must be float32 [{n}, {POLICY_SIZE}], got "
                         f"{logits.dtype} {logits.shape}")
    if values.dtype != VALUES_DTYPE or values.shape != (n,):
        raise ValueError(f"values must be float32 [{n}], got {values.dtype} {values.shape}")


@dataclass
class EvalStats:
    """What the network side saw: calls, positions, time spent inside evaluate()."""
    calls: int = 0
    positions: int = 0
    secs: float = 0.0
    max_batch: int = 0

    def add(self, n: int, secs: float) -> None:
        self.calls += 1
        self.positions += n
        self.secs += secs
        self.max_batch = max(self.max_batch, n)

    def merge(self, o: "EvalStats") -> "EvalStats":
        return EvalStats(self.calls + o.calls, self.positions + o.positions,
                         self.secs + o.secs, max(self.max_batch, o.max_batch))

    def to_dict(self) -> dict:
        return {"calls": self.calls, "positions": self.positions,
                "avg_batch": round(self.positions / self.calls, 1) if self.calls else 0.0,
                "max_batch": self.max_batch, "eval_secs": round(self.secs, 3)}


class CountingEvaluator:
    """Wraps any evaluator and records EvalStats. With check=True (or CHESSRL_CHECK=1)
    it also verifies the contract on every call; off by default because it costs time."""

    def __init__(self, inner, check: bool | None = None):
        self.inner = inner
        self.check = os.environ.get("CHESSRL_CHECK") == "1" if check is None else check
        self._stats = EvalStats()

    def evaluate(self, x: np.ndarray):
        if self.check:
            check_request(x)
        t0 = time.perf_counter()
        logits, values = self.inner.evaluate(x)
        self._stats.add(len(x), time.perf_counter() - t0)
        if self.check:
            check_response(logits, values, len(x))
        return logits, values

    def stats(self, reset: bool = False) -> EvalStats:
        out = self._stats
        if reset:
            self._stats = EvalStats()
        return out

    def __getattr__(self, name):                   # set_weights() etc. go to the real one
        if name in ("inner", "_stats", "check"):   # not built yet (copy / pickle): no recursion
            raise AttributeError(name)
        return getattr(self.inner, name)


class UniformEvaluator:
    """No network: flat policy, value 0. For tests and for a baseline opponent."""

    def evaluate(self, x: np.ndarray):
        return (np.zeros((len(x), POLICY_SIZE), np.float32), np.zeros(len(x), np.float32))


class TorchEvaluator:
    def __init__(self, net, device: str = "cpu", amp: bool = True, compile: bool = False):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.net = net.to(self.device).eval()
        self.amp = amp and self.cuda
        self.amp_dtype = (torch.bfloat16 if self.cuda and torch.cuda.is_bf16_supported()
                          else torch.float16)
        if self.cuda:
            torch.backends.cudnn.benchmark = True
            self.net = self.net.to(memory_format=torch.channels_last)
        self._fn = torch.compile(self.net) if compile else self.net

    def set_weights(self, state_dict) -> None:
        self.net.load_state_dict(state_dict)

    def evaluate(self, x: np.ndarray):
        torch = self.torch
        t = torch.from_numpy(x)
        if self.cuda:
            t = t.pin_memory().to(self.device, non_blocking=True)
            t = t.float().contiguous(memory_format=torch.channels_last)
        else:
            t = t.float()
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.amp_dtype,
                                                    enabled=self.amp):
            logits, value = self._fn(t)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()
