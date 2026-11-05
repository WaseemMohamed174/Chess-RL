"""How the gateway talks to the workers (HTTP on 127.0.0.1 + secret token)."""
from __future__ import annotations

import httpx

from .runtime import ManagedProcess


class WorkerUnavailable(RuntimeError):
    pass


class WorkerError(RuntimeError):
    def __init__(self, status: int, detail):
        super().__init__(f"worker answered {status}: {detail}")
        self.status, self.detail = status, detail


class WorkerClient:
    def __init__(self, worker: ManagedProcess | None, timeout: float = 300.0):
        self.worker, self.timeout = worker, timeout

    async def _call(self, method: str, path: str, json=None):
        w = self.worker
        if w is None or not w.is_ready():
            raise WorkerUnavailable(f"engine worker is {w.state.value if w else 'NOT STARTED'}")
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as c:
            r = await c.request(method, w.url + path, json=json,
                                headers={"X-Worker-Token": w.token})
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise WorkerError(r.status_code, detail)
        return r.json()


class PlayClient(WorkerClient):
    async def search(self, fen: str, model_path: str | None, mcts: dict, ply: int = 0,
                     seed=None, inference: dict | None = None) -> dict:
        return await self._call("POST", "/search", {"fen": fen, "model_path": model_path,
                                                    "mcts": mcts, "ply": ply, "seed": seed,
                                                    "inference": inference})

    async def init_model(self, path: str, blocks: int, filters: int) -> dict:
        return await self._call("POST", "/init_model",
                                {"path": path, "blocks": blocks, "filters": filters})

    async def info(self) -> dict:
        return await self._call("GET", "/info")


class TrainClient(WorkerClient):
    async def set_workers(self, n: int) -> dict:
        return await self._call("POST", "/workers", {"n": n})

    async def start(self, payload: dict) -> dict:
        return await self._call("POST", "/start", payload)

    async def command(self, name: str) -> dict:           # pause | resume | stop
        return await self._call("POST", f"/{name}")

    async def status(self) -> dict:
        return await self._call("GET", "/status")

    async def watch(self, on: bool) -> dict:
        return await self._call("POST", "/watch", {"on": on})
