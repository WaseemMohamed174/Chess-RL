"""RemoteEvaluator: looks exactly like any other evaluator (evaluate(x) -> logits, values),
so MCTS and GamePool cannot tell the network lives in another process.

With more than one connection it can also answer evaluate_async(x): the request is SENT at once
(a memory copy and a small note) and the call returns a handle; .result() waits for the answer.
Everything happens in the calling thread, so there is no thread to fight over the GIL while the
caller is busy searching. GamePool uses this to overlap its own CPU work (python-chess, tree
search) with the time the server spends on the network."""
from __future__ import annotations

import os
from collections import deque

import numpy as np

from ..rl.evaluator import check_request
from .transport import InferenceError, open_client_port

__all__ = ["RemoteEvaluator", "InferenceError", "pipeline_connections"]


def pipeline_connections() -> int:
    """Connections a self-play worker opens: 2 lets it work on one half of its games while the
    server answers the other half. CHESSRL_PIPELINE=0 goes back to one blocking connection."""
    return 1 if os.environ.get("CHESSRL_PIPELINE", "1") == "0" else 2


class _Pool:
    """Several connections of one caller. A connection carries one request at a time, so a
    request that finds none free first waits for the oldest one in flight (which frees its)."""

    def __init__(self, ports: list):
        self.ports = ports
        self.free = list(ports)
        self.inflight: deque = deque()

    def acquire(self):
        if not self.free:
            self.inflight[0].wait()               # moves its port back to `free`
        return self.free.pop()

    def release(self, port, pending=None) -> None:
        if pending is not None:
            self.inflight.remove(pending)
        self.free.append(port)

    def close(self) -> None:
        for p in self.ports:
            p.close()


class _Pending:
    """A request that is on its way. result() returns (logits, values) or raises its error."""

    def __init__(self, pool: _Pool, model_id: str, x: np.ndarray, timeout: float):
        self._pool, self._done, self._out, self._err = pool, False, None, None
        self._port = pool.acquire()
        try:
            self._handle = self._port.begin(model_id, x, timeout, urgent=True)   # pipelined: no waiting for company
        except BaseException:
            pool.release(self._port)
            raise
        pool.inflight.append(self)

    def wait(self) -> None:
        if self._done:
            return
        try:
            self._out = self._port.finish(self._handle)
        except BaseException as e:               # noqa: BLE001 - re-raised by result()
            self._err = e
        finally:
            self._done = True
            self._pool.release(self._port, self)

    def result(self, timeout=None):               # timeout: kept for Future look-alikes, unused
        self.wait()
        if self._err is not None:
            raise self._err
        return self._out


def _call(port, model_id: str, x: np.ndarray, timeout: float):
    n, cap = len(x), port.max_n
    if n <= cap:
        return port.call(model_id, x, timeout)
    parts = [port.call(model_id, x[i:i + cap], timeout) for i in range(0, n, cap)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


class _Done:
    """The same handle for an answer that is already there (a request too big to pipeline)."""

    def __init__(self, out=None, err=None):
        self._out, self._err = out, err

    def result(self, timeout=None):
        if self._err is not None:
            raise self._err
        return self._out


class RemoteEvaluator:
    def __init__(self, spec_or_port, model_id: str, timeout: float = 60.0, connections: int = 1):
        """spec_or_port: one spec dict, one open port, or a list of them (one per connection).
        connections > 1 with a single "sock" spec opens that many independent connections."""
        if isinstance(spec_or_port, _Pool):               # for_model(): share the connections
            self._pool, self.port = spec_or_port, spec_or_port.ports[0]
        else:
            items = list(spec_or_port) if isinstance(spec_or_port, (list, tuple)) else [spec_or_port]
            if len(items) == 1 and connections > 1:
                if not (isinstance(items[0], dict) and items[0].get("kind") == "sock"):
                    raise ValueError("several connections from one spec need the sock transport")
                items = items * connections
            ports = [open_client_port(i) if isinstance(i, dict) else i for i in items]
            self.port = ports[0]
            self._pool = _Pool(ports) if len(ports) > 1 else None
        self.model_id, self.timeout = model_id, timeout

    @property
    def pipelined(self) -> bool:
        """True when evaluate_async can really overlap requests."""
        return self._pool is not None

    def for_model(self, model_id: str) -> "RemoteEvaluator":
        """Another model on the same connection(s) (e.g. candidate vs best in the arena).
        One connection serves one caller at a time; do not use both from two threads."""
        return RemoteEvaluator(self._pool if self._pool is not None else self.port,
                               model_id, self.timeout)

    def evaluate(self, x: np.ndarray):
        check_request(x)
        if self._pool is None:
            return _call(self.port, self.model_id, x, self.timeout)
        return self.evaluate_async(x).result()

    def evaluate_async(self, x: np.ndarray):
        """Send the request now, get the answer later with .result(). Needs connections > 1."""
        if self._pool is None:
            raise RuntimeError("evaluate_async needs more than one connection")
        check_request(x)
        if len(x) > self.port.max_n:                       # split in pieces: do it the plain way
            port = self._pool.acquire()
            try:
                return _Done(out=_call(port, self.model_id, x, self.timeout))
            except Exception as e:                         # noqa: BLE001 - raised by result()
                return _Done(err=e)
            finally:
                self._pool.release(port)
        return _Pending(self._pool, self.model_id, x, self.timeout)

    def refresh(self, model_id: str | None = None) -> None:
        """Tell the server this model's weights file changed, so it re-checks now instead of
        at its next periodic check. Only the sock transport supports it."""
        port = self._pool.acquire() if self._pool is not None else self.port
        try:
            if hasattr(port, "refresh"):
                port.refresh(model_id or self.model_id)
        finally:
            if self._pool is not None:
                self._pool.release(port)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
        else:
            self.port.close()
