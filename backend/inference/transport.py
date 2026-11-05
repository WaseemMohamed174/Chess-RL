"""How requests and answers travel between workers (clients) and the server.

Two interchangeable transports behind the same tiny interface:

    hub = make_hub("shm" | "queue", n_clients, max_n)      # created once, in the parent
    hub.client_spec(i)   -> picklable dict, give it to worker i
    hub.server_spec()    -> picklable dict, give it to the server
    open_client_port(spec).call(model_id, x) -> (logits, values)
    open_server_port(spec).recv(timeout) -> Request | None ; port.reply(req, l, v) ; port.fail(req, msg)

"shm"   one shared-memory block per client (request rows, logits, values) + one
        semaphore per client. Only a 3-number note goes through a queue, so the big
        arrays are never pickled. This is the default.
"queue" everything is pickled through multiprocessing queues. Slower, but needs no
        shared memory; kept as a fallback and as a reference for the tests.
"sock"  for processes NOT started by one parent (the supervisor starts the server and the
        workers independently). A client connects to 127.0.0.1 with a secret token; the
        server makes a named shared-memory block for it and only small notes cross the
        socket. Clients join and leave at any time and reconnect when the server restarts."""
from __future__ import annotations

import hmac
import multiprocessing as mp
import os
import queue
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import numpy as np

from ..rl.encoding import INPUT_PLANES, POLICY_SIZE

_ROW = INPUT_PLANES * 64                 # bytes of one request row (uint8)
_HDR = 512                               # status, n, error length, error text


class InferenceError(RuntimeError):
    """The server could not answer (unknown model, crash in the network, timeout...)."""


@dataclass
class Request:
    cid: int
    model_id: str
    n: int
    x: np.ndarray                        # [n,18,8,8] uint8; valid until the request is answered
    t_arrive: float = field(default_factory=time.perf_counter)
    op: str = "eval"                     # "eval", or "refresh" (re-check this model's weights)
    urgent: bool = False                 # pipelined caller: run now, do not wait for company


def _layout(max_n: int) -> dict:
    off_x = _HDR
    off_l = (off_x + max_n * _ROW + 63) // 64 * 64
    off_v = (off_l + max_n * POLICY_SIZE * 4 + 63) // 64 * 64
    return {"x": off_x, "l": off_l, "v": off_v, "size": off_v + max_n * 4}


class _Views:
    """numpy views onto one client's shared-memory block."""

    def __init__(self, shm, max_n: int):
        lay = _layout(max_n)
        b = shm.buf
        self.hdr = np.ndarray((3,), np.int32, b, 0)                  # status, n, len(err)
        self.err = np.ndarray((_HDR - 12,), np.uint8, b, 12)
        self.x = np.ndarray((max_n, INPUT_PLANES, 8, 8), np.uint8, b, lay["x"])
        self.logits = np.ndarray((max_n, POLICY_SIZE), np.float32, b, lay["l"])
        self.values = np.ndarray((max_n,), np.float32, b, lay["v"])


# ------------------------------------------------------------------ shared memory
class ShmHub:
    kind = "shm"

    def __init__(self, n_clients: int, max_n: int = 256, ctx=None):
        ctx = ctx or mp.get_context("spawn")
        self.n_clients, self.max_n = n_clients, max_n
        self.req_q = ctx.Queue()
        self.sems = [ctx.Semaphore(0) for _ in range(n_clients)]
        size = _layout(max_n)["size"]
        self.shms = [shared_memory.SharedMemory(create=True, size=size) for _ in range(n_clients)]

    def client_spec(self, i: int) -> dict:
        return {"kind": "shm", "cid": i, "name": self.shms[i].name, "max_n": self.max_n,
                "req_q": self.req_q, "sem": self.sems[i]}

    def server_spec(self) -> dict:
        return {"kind": "shm", "names": [s.name for s in self.shms], "max_n": self.max_n,
                "req_q": self.req_q, "sems": self.sems, "n_clients": self.n_clients}

    def close(self) -> None:
        for s in self.shms:
            try:
                s.close()
                s.unlink()
            except FileNotFoundError:
                pass


class ShmClientPort:
    def __init__(self, spec: dict):
        self.max_n, self.cid = spec["max_n"], spec["cid"]
        self._shm = shared_memory.SharedMemory(name=spec["name"])
        self._v = _Views(self._shm, self.max_n)
        self._q, self._sem = spec["req_q"], spec["sem"]
        self.broken = False

    def begin(self, model_id: str, x: np.ndarray, timeout: float, urgent: bool = False):
        """Send the request and return at once; finish(handle) waits for the answer.
        One request per connection at a time. urgent: the server runs it without waiting
        for other requests to join (what a pipelined caller wants, see GamePool)."""
        if self.broken:
            raise InferenceError("this connection timed out earlier; open a new one")
        n = len(x)
        self._v.x[:n] = x
        self._q.put((self.cid, model_id, n, urgent))
        return (n, timeout)

    def finish(self, handle):
        n, timeout = handle
        v = self._v
        if not self._sem.acquire(timeout=timeout):
            self.broken = True               # a late answer would desync the semaphore
            raise InferenceError(f"no answer from the inference server in {timeout:.0f}s")
        if v.hdr[0] != 0:
            raise InferenceError(bytes(v.err[:int(v.hdr[2])]).decode("utf-8", "replace"))
        return v.logits[:n].copy(), v.values[:n].copy()

    def call(self, model_id: str, x: np.ndarray, timeout: float):
        return self.finish(self.begin(model_id, x, timeout))

    def close(self) -> None:
        self._shm.close()


class ShmServerPort:
    def __init__(self, spec: dict):
        self.max_n, self.n_clients = spec["max_n"], spec["n_clients"]
        self._shms = [shared_memory.SharedMemory(name=n) for n in spec["names"]]
        self._v = [_Views(s, self.max_n) for s in self._shms]
        self._q, self._sems = spec["req_q"], spec["sems"]

    def recv(self, timeout: float):
        try:
            cid, model_id, n, urgent = self._q.get(timeout=timeout) if timeout > 0 else self._q.get_nowait()
        except queue.Empty:
            return None
        return Request(cid, model_id, n, self._v[cid].x[:n], urgent=urgent)

    def reply(self, req: Request, logits: np.ndarray, values: np.ndarray) -> None:
        v = self._v[req.cid]
        v.logits[:req.n] = logits
        v.values[:req.n] = values
        v.hdr[0] = 0
        self._sems[req.cid].release()

    def fail(self, req: Request, msg: str) -> None:
        v = self._v[req.cid]
        raw = msg.encode("utf-8")[:v.err.size]
        v.err[:len(raw)] = np.frombuffer(raw, np.uint8)
        v.hdr[2], v.hdr[0] = len(raw), 1
        self._sems[req.cid].release()

    def close(self) -> None:
        for s in self._shms:
            s.close()


# ------------------------------------------------------------------ plain queues
class QueueHub:
    kind = "queue"

    def __init__(self, n_clients: int, max_n: int = 256, ctx=None):
        ctx = ctx or mp.get_context("spawn")
        self.n_clients, self.max_n = n_clients, max_n
        self.req_q = ctx.Queue()
        self.resp_qs = [ctx.Queue() for _ in range(n_clients)]

    def client_spec(self, i: int) -> dict:
        return {"kind": "queue", "cid": i, "max_n": self.max_n,
                "req_q": self.req_q, "resp_q": self.resp_qs[i]}

    def server_spec(self) -> dict:
        return {"kind": "queue", "max_n": self.max_n, "req_q": self.req_q,
                "resp_qs": self.resp_qs, "n_clients": self.n_clients}

    def close(self) -> None:
        pass


class QueueClientPort:
    def __init__(self, spec: dict):
        self.max_n, self.cid = spec["max_n"], spec["cid"]
        self._q, self._rq = spec["req_q"], spec["resp_q"]
        self.broken = False

    def begin(self, model_id: str, x: np.ndarray, timeout: float, urgent: bool = False):
        if self.broken:
            raise InferenceError("this connection timed out earlier; open a new one")
        self._q.put((self.cid, model_id, np.ascontiguousarray(x), urgent))
        return timeout

    def call(self, model_id: str, x: np.ndarray, timeout: float):
        return self.finish(self.begin(model_id, x, timeout))

    def finish(self, timeout):
        try:
            err, logits, values = self._rq.get(timeout=timeout)
        except queue.Empty:
            self.broken = True
            raise InferenceError(f"no answer from the inference server in {timeout:.0f}s") from None
        if err:
            raise InferenceError(err)
        return logits, values

    def close(self) -> None:
        pass


class QueueServerPort:
    def __init__(self, spec: dict):
        self.max_n, self.n_clients = spec["max_n"], spec["n_clients"]
        self._q, self._rqs = spec["req_q"], spec["resp_qs"]

    def recv(self, timeout: float):
        try:
            cid, model_id, x, urgent = self._q.get(timeout=timeout) if timeout > 0 else self._q.get_nowait()
        except queue.Empty:
            return None
        return Request(cid, model_id, len(x), x, urgent=urgent)

    def reply(self, req: Request, logits, values) -> None:
        self._rqs[req.cid].put((None, logits, values))

    def fail(self, req: Request, msg: str) -> None:
        self._rqs[req.cid].put((msg, None, None))

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ sockets + shared memory
_MAGIC = b"CRL1"
_OP_EVAL, _OP_REFRESH, _OP_EVAL_NOW = 1, 2, 3     # _NOW: eval that must not wait for company
_REQ = struct.Struct("!BIH")             # op, n, len(model id)
_RESP = struct.Struct("!BH")             # status (0 ok / 1 error), len(error text)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def _close_shm(shm, unlink: bool) -> None:
    if unlink:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
    try:
        shm.close()
    except BufferError:                  # numpy views still alive: freed when they are
        pass


def _attach_untracked(name: str):
    """Attach to a block someone else owns, without the resource tracker (Python < 3.13)
    deleting it when this process exits."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")   # noqa: SLF001
    except Exception:
        pass
    return shm


class SockHub:
    """Spec factory for the sock transport (a port file tells clients where the server is)."""
    kind = "sock"

    def __init__(self, n_clients: int = 0, max_n: int = 256, token: str | None = None,
                 port_file: str | None = None, ctx=None):
        import tempfile
        self.max_n = max_n
        self.token = token or uuid.uuid4().hex
        self.port_file = port_file or os.path.join(tempfile.mkdtemp(prefix="crl-hub-"), "port")

    def server_spec(self) -> dict:
        return {"kind": "sock", "host": "127.0.0.1", "port": 0, "token": self.token,
                "max_n": self.max_n, "port_file": self.port_file}

    def client_spec(self, i: int = 0) -> dict:
        return {"kind": "sock", "token": self.token, "max_n": self.max_n,
                "port_file": self.port_file}

    def close(self) -> None:
        pass


class _Conn:
    def __init__(self, cid, sock, shm, views):
        self.cid, self.sock, self.shm, self.views = cid, sock, shm, views


class SockServerPort:
    def __init__(self, spec: dict):
        self.max_n = spec["max_n"]
        self._token = spec["token"].encode()
        self._prefix = spec.get("shm_prefix", "crl-")
        self._q: queue.Queue = queue.Queue()
        self._conns: dict = {}
        self._lock = threading.Lock()
        self._next = 0
        self._closed = False
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((spec.get("host", "127.0.0.1"), spec.get("port", 0)))
        self._srv.listen(64)
        self.port = self._srv.getsockname()[1]
        if spec.get("port_file"):                       # clients read the port from here
            tmp = spec["port_file"] + ".tmp"
            with open(tmp, "w") as f:
                f.write(str(self.port))
            os.replace(tmp, spec["port_file"])
        threading.Thread(target=self._accept, name="inference-accept", daemon=True).start()

    @property
    def n_clients(self) -> int:
        return len(self._conns)

    # -- connections
    def _accept(self) -> None:
        while not self._closed:
            try:
                sock, _ = self._srv.accept()
            except OSError:
                return
            if self._closed:                            # woken up by close(): nobody serves any more
                sock.close()
                return
            threading.Thread(target=self._serve_conn, args=(sock,), daemon=True).start()

    def _serve_conn(self, sock) -> None:
        conn = None
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(5)
            if _recv_exact(sock, 4) != _MAGIC:
                return
            (tlen,) = struct.unpack("!H", _recv_exact(sock, 2))
            token = _recv_exact(sock, tlen)
            (want_n,) = struct.unpack("!I", _recv_exact(sock, 4))
            if not hmac.compare_digest(token, self._token):
                return
            max_n = max(1, min(want_n, self.max_n))
            shm = shared_memory.SharedMemory(create=True, size=_layout(max_n)["size"],
                                             name=f"{self._prefix}{uuid.uuid4().hex[:16]}")
            with self._lock:
                cid, self._next = self._next, self._next + 1
                conn = _Conn(cid, sock, shm, _Views(shm, max_n))
                self._conns[cid] = conn
            name = shm.name.encode()
            sock.sendall(struct.pack("!IH", max_n, len(name)) + name)
            sock.settimeout(None)
            while True:
                op, n, mlen = _REQ.unpack(_recv_exact(sock, _REQ.size))
                model = _recv_exact(sock, mlen).decode("utf-8", "replace")
                if op == _OP_REFRESH:
                    self._q.put(Request(cid, model, 0, conn.views.x[:0], op="refresh"))
                elif 1 <= n <= max_n:
                    self._q.put(Request(cid, model, n, conn.views.x[:n], urgent=op == _OP_EVAL_NOW))
                else:
                    self._send(conn, 1, f"bad request size {n} (max {max_n})")
        except (OSError, ConnectionError, struct.error):
            pass
        finally:
            with self._lock:
                if conn is not None:
                    self._conns.pop(conn.cid, None)
            if conn is not None:
                _close_shm(conn.shm, unlink=True)
            try:
                sock.close()
            except OSError:
                pass

    def _send(self, conn, status: int, msg: str = "") -> None:
        raw = msg.encode("utf-8")[:4000]
        try:
            conn.sock.sendall(_RESP.pack(status, len(raw)) + raw)
        except OSError:
            pass                                        # the client is gone; its thread cleans up

    # -- server-port interface
    def recv(self, timeout: float):
        try:
            return self._q.get(timeout=timeout) if timeout > 0 else self._q.get_nowait()
        except queue.Empty:
            return None

    def reply(self, req: Request, logits, values) -> None:
        conn = self._conns.get(req.cid)
        if conn is None:
            return
        if req.n:
            conn.views.logits[:req.n] = logits
            conn.views.values[:req.n] = values
        self._send(conn, 0)

    def fail(self, req: Request, msg: str) -> None:
        conn = self._conns.get(req.cid)
        if conn is not None:
            self._send(conn, 1, msg)

    def close(self) -> None:
        self._closed = True
        try:                                            # wake the thread blocked in accept()
            socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
        except OSError:
            pass
        try:
            self._srv.close()
        except OSError:
            pass
        for c in list(self._conns.values()):
            try:
                c.sock.shutdown(socket.SHUT_RDWR)       # really cut the line: close() alone does not
            except OSError:                             # wake the thread blocked in recv(), so the
                pass                                    # client would wait for its whole timeout
            try:
                c.sock.close()
            except OSError:
                pass


class SockClientPort:
    """Connects lazily, reconnects after any failure, and retries a call for up to
    `retry_s` seconds, so a restarting server only causes a pause, not an error."""

    def __init__(self, spec: dict):
        self.spec, self.max_n = spec, spec["max_n"]
        self.retry_s = float(spec.get("retry_s", 30.0))
        self._sock = self._shm = self._v = None
        self.reconnects = 0

    def _address(self) -> int:
        if spec_port := self.spec.get("port"):
            return int(spec_port)
        with open(self.spec["port_file"]) as f:
            return int(f.read().strip())

    def _drop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._shm is not None:
            self._v = None
            _close_shm(self._shm, unlink=False)
        self._sock = self._shm = self._v = None

    def _connect(self) -> None:
        sock = socket.create_connection(("127.0.0.1", self._address()), timeout=5)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            tok = self.spec["token"].encode()
            sock.sendall(_MAGIC + struct.pack("!H", len(tok)) + tok + struct.pack("!I", self.max_n))
            granted, nlen = struct.unpack("!IH", _recv_exact(sock, 6))
            name = _recv_exact(sock, nlen).decode()
        except Exception:
            sock.close()
            raise
        self._shm = _attach_untracked(name)
        self._v, self.max_n, self._sock = _Views(self._shm, granted), granted, sock
        self.reconnects += 1

    def _send(self, op: int, model_id: str, n: int, timeout: float) -> None:
        raw = model_id.encode("utf-8")
        self._sock.settimeout(timeout)
        self._sock.sendall(_REQ.pack(op, n, len(raw)) + raw)

    def _recv_status(self, timeout: float) -> None:
        self._sock.settimeout(timeout)
        status, elen = _RESP.unpack(_recv_exact(self._sock, _RESP.size))
        if status != 0:
            raise InferenceError(_recv_exact(self._sock, elen).decode("utf-8", "replace"))

    def _roundtrip(self, op: int, model_id: str, n: int, timeout: float) -> None:
        self._send(op, model_id, n, timeout)
        self._recv_status(timeout)

    def _with_retry(self, fn):
        end = time.monotonic() + self.retry_s
        while True:
            try:
                if self._sock is None:
                    self._connect()
                return fn()
            except InferenceError:
                raise                                   # the server answered: do not retry
            except (OSError, ConnectionError, struct.error, ValueError) as e:
                self._drop()
                if time.monotonic() >= end:
                    raise InferenceError(f"inference server unreachable: {e}") from e
                time.sleep(0.25)

    def call(self, model_id: str, x: np.ndarray, timeout: float):
        def once():
            n = len(x)
            self._v.x[:n] = x
            self._roundtrip(_OP_EVAL, model_id, n, timeout)
            return self._v.logits[:n].copy(), self._v.values[:n].copy()
        return self._with_retry(once)

    def begin(self, model_id: str, x: np.ndarray, timeout: float, urgent: bool = False):
        """Send the request and return at once; finish(handle) waits for the answer. The handle
        keeps x, so a server that restarts in between only causes the request to be sent again."""
        n = len(x)
        op = _OP_EVAL_NOW if urgent else _OP_EVAL

        def send():
            self._v.x[:n] = x
            self._send(op, model_id, n, timeout)
        self._with_retry(send)
        return (model_id, x, n, timeout, self.reconnects, op)

    def finish(self, handle):
        model_id, x, n, timeout, gen, op = handle
        resend = [False]

        def once():
            if resend[0] or self.reconnects != gen:      # the connection the request left on is gone
                self._v.x[:n] = x
                self._roundtrip(op, model_id, n, timeout)
            else:
                resend[0] = True                         # if waiting fails, the retry sends it again
                self._recv_status(timeout)
            return self._v.logits[:n].copy(), self._v.values[:n].copy()
        return self._with_retry(once)

    def refresh(self, model_id: str, timeout: float = 10.0) -> None:
        self._with_retry(lambda: self._roundtrip(_OP_REFRESH, model_id, 0, timeout))

    def close(self) -> None:
        self._drop()


# ------------------------------------------------------------------ entry points
def make_hub(kind: str, n_clients: int, max_n: int = 256, ctx=None):
    if kind == "shm":
        return ShmHub(n_clients, max_n, ctx)
    if kind == "queue":
        return QueueHub(n_clients, max_n, ctx)
    if kind == "sock":
        return SockHub(n_clients, max_n)
    raise ValueError(f"unknown transport {kind!r} (use 'shm', 'queue' or 'sock')")


def open_client_port(spec: dict):
    return {"shm": ShmClientPort, "queue": QueueClientPort, "sock": SockClientPort}[spec["kind"]](spec)


def open_server_port(spec: dict):
    return {"shm": ShmServerPort, "queue": QueueServerPort, "sock": SockServerPort}[spec["kind"]](spec)
