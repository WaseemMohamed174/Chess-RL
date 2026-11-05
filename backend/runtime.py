"""Starts, watches and restarts the engine workers (separate processes).

    STARTING -> READY -> CRASHED -> BACKOFF -> STARTING ...   (max 5 restarts, then FAILED)

Each worker listens on a free 127.0.0.1 port and only answers requests that carry
the secret token the supervisor generated for this run."""
from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from enum import Enum

from . import paths

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class State(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    CRASHED = "CRASHED"
    BACKOFF = "BACKOFF"
    FAILED = "FAILED"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _can_connect(port_file: str, token: str) -> bool:
    """True when something answers the inference handshake on the port in port_file."""
    from .inference.transport import open_client_port
    p = open_client_port({"kind": "sock", "token": token, "max_n": 1,
                          "port_file": port_file, "retry_s": 0})
    try:
        p._connect()                                   # noqa: SLF001
        return True
    except Exception:
        return False
    finally:
        p.close()


def engine_python() -> str:
    """Interpreter of the engine venv (PyTorch lives there)."""
    override = os.environ.get("CHESSRL_ENGINE_PYTHON")
    if override:
        return override
    py = paths.venv_python("engine")
    return str(py) if py.exists() else sys.executable


class ManagedProcess:
    def __init__(self, name: str, module: str, token: str, env: dict | None = None,
                 max_restarts: int = 5, on_change=None):
        self.name, self.module, self.token = name, module, token
        self.extra_env, self.max_restarts, self.on_change = env or {}, max_restarts, on_change
        self.state = State.STOPPED
        self.port = 0
        self.restarts = 0
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._spawn()
        self._thread = threading.Thread(target=self._watch, name=f"watch-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kill()
        self._set(State.STOPPED)

    def restart(self) -> None:
        self.stop()
        if self._thread:
            self._thread.join(timeout=3)
        self.restarts = 0
        self.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_ready(self) -> bool:
        return self.state == State.READY

    def wait_ready(self, timeout: float = 60.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.state == State.READY:
                return True
            if self.state == State.FAILED:
                return False
            time.sleep(0.1)
        return False

    def status(self) -> dict:
        return {"name": self.name, "state": self.state.value, "port": self.port,
                "pid": self.proc.pid if self.proc else None, "restarts": self.restarts}

    def _set(self, state: State) -> None:
        if state != self.state:
            self.state = state
            if self.on_change:
                try:
                    self.on_change(self)
                except Exception:
                    pass

    def _spawn(self) -> None:
        self.port = _free_port()
        env = {**os.environ, **self.extra_env,
               "PYTHONPATH": str(paths.BASE_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "CHESSRL_TOKEN": self.token, "PYTHONUNBUFFERED": "1"}
        paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log = open(paths.LOGS_DIR / f"{self.name}.log", "a", encoding="utf-8")
        self._set(State.STARTING)
        self.proc = subprocess.Popen(self._argv(), cwd=str(paths.BASE_DIR), env=env,
                                     stdout=log, stderr=subprocess.STDOUT)

    def _argv(self) -> list[str]:
        return [engine_python(), "-m", "uvicorn", f"{self.module}:app", "--host", "127.0.0.1",
                "--port", str(self.port), "--log-level", "warning"]

    def _kill(self) -> None:
        p = self.proc
        if not p or p.poll() is not None:
            return
        kids = []
        try:
            import psutil
            kids = psutil.Process(p.pid).children(recursive=True)
        except Exception:
            pass
        p.terminate()
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.kill()
        for k in kids:
            try:
                k.kill()
            except Exception:
                pass

    def _probe(self) -> bool:
        try:
            req = urllib.request.Request(self.url + "/health",
                                         headers={"X-Worker-Token": self.token})
            with _NO_PROXY.open(req, timeout=2) as r:
                return r.status == 200 and json.loads(r.read()).get("status") == "ok"
        except Exception:
            return False

    def _watch(self) -> None:
        while not self._stop.is_set():
            if self.proc.poll() is not None:                       # process died
                if self._stop.is_set():
                    return
                self._set(State.CRASHED)
                if self.restarts >= self.max_restarts:
                    self._set(State.FAILED)
                    return
                self.restarts += 1
                self._set(State.BACKOFF)
                if self._stop.wait(min(2 ** self.restarts, 30)):
                    return
                self._spawn()
            elif self.state != State.READY and self._probe():
                self._set(State.READY)
            self._stop.wait(0.5)


class ManagedCommand(ManagedProcess):
    """A watched process that is not a web server: any `python -m module` that signals
    readiness itself. Same life cycle (STARTING -> READY -> CRASHED -> BACKOFF ...), so the
    inference server is restarted exactly like the workers are.
    ready(): called every 0.5 s while STARTING; return True when the process is usable."""

    def __init__(self, name: str, module: str, token: str, ready, env: dict | None = None,
                 max_restarts: int = 5, on_change=None):
        super().__init__(name, module, token, env, max_restarts, on_change)
        self._ready = ready

    def _argv(self) -> list[str]:
        return [engine_python(), "-m", self.module]

    def _probe(self) -> bool:
        try:
            return bool(self._ready())
        except Exception:
            return False


class Supervisor:
    def __init__(self, on_change=None):
        self.token = secrets.token_urlsafe(24)
        self.on_change = on_change
        self.workers: dict[str, ManagedProcess] = {}

    def _make(self, name: str, module: str, env: dict | None = None) -> ManagedProcess:
        w = self.workers.get(name)
        if w is None:
            w = self.workers[name] = ManagedProcess(name, module, self.token, env,
                                                    on_change=self.on_change)
        return w

    def start_play(self, env: dict | None = None) -> ManagedProcess:
        w = self._make("play", "backend.workers.play_worker", env)
        w.start()
        return w

    def ensure_train(self, model_id: int, env: dict | None = None) -> ManagedProcess:
        """One training process per model: runs of different models use different CPU
        cores and never wait for each other."""
        w = self._make(f"train-{model_id}", "backend.workers.train_worker", env)
        w.start()
        return w

    def ensure_inference(self, cfg: dict, env: dict | None = None) -> ManagedCommand:
        """The ONE inference server for every model. cfg is written to a JSON file the server
        reads at start (and again after each restart); it must hold port_file."""
        w = self.workers.get("inference")
        if w is None:
            cfg_path = paths.RUNTIME_DIR / "inference.json"
            paths.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            port_file = cfg["port_file"]
            w = self.workers["inference"] = ManagedCommand(
                "inference", "backend.inference.serve", self.token,
                ready=lambda: _can_connect(port_file, self.token),
                env={**(env or {}), "CHESSRL_INFERENCE_CONFIG": str(cfg_path)},
                on_change=self.on_change)
        w.start()
        return w

    def release(self, name: str) -> ManagedProcess | None:
        """Forget a worker right now (so nobody can reuse it); the caller stops it."""
        return self.workers.pop(name, None)

    def get(self, name: str) -> ManagedProcess | None:
        return self.workers.get(name)

    def status(self) -> dict:
        return {n: w.status() for n, w in self.workers.items()}

    def stop_all(self) -> None:
        for w in self.workers.values():
            w.stop()
