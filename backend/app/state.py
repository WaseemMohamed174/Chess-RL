"""Everything the gateway shares, in one object (S)."""
from __future__ import annotations

import asyncio

from .. import paths, settings
from ..clients import PlayClient, TrainClient
from ..db import Database
from ..events import EventBus
from ..runtime import Supervisor


class AppState:
    def __init__(self):
        self.bus = EventBus()
        self.db: Database | None = None
        self.sup: Supervisor | None = None
        self.selection: dict = {}
        self.play = PlayClient(None)
        self.watch_run: int | None = None      # the training run whose live board is streamed
        self.games: dict = {}                  # game_id -> LiveGame (services.py)
        self.matches: dict[int, asyncio.Task] = {}
        self.match_delay: dict[int, float] = {}
        self.tasks: list[asyncio.Task] = []
        self.coordinators: dict = {}           # population_id -> Coordinator (app/coordinator.py)

    def start(self, loop) -> None:
        paths.ensure_runtime_dirs()
        self.selection = settings.load_selection()
        self.bus.bind(loop)
        self.db = Database(paths.DB_PATH)
        for run in self.db.list_runs():                      # left over from a crash
            if run["status"] in ("running", "paused"):
                self.db.set_run_status(run["id"], "stopped")
        for pop in self.db.list_populations():               # the coordinator died with the server;
            if pop["status"] in ("running", "paused"):       # the phase is kept, so Start goes on
                self.db.update_population(pop["id"], status="stopped")
        self.sup = Supervisor(on_change=lambda w: self.bus.publish(
            "system", {"type": "workers", "workers": self.sup.status()}))
        self.play.worker = self.sup.start_play()

    def stop(self) -> None:
        for t in self.tasks + list(self.matches.values()):
            t.cancel()
        for c in self.coordinators.values():
            if c.task is not None:
                c.task.cancel()
        if self.sup:
            self.sup.stop_all()
        if self.db:
            self.db.close()

    def inference_cfg(self) -> dict:
        """Settings of the one shared inference server (written to its JSON file)."""
        import hashlib
        sel = self.selection
        tag = hashlib.sha1(str(paths.DATA_DIR).encode()).hexdigest()[:6]
        return {"root": str(paths.CHECKPOINTS_DIR),
                "port_file": str(paths.RUNTIME_DIR / "inference.port"),
                "stats_path": str(paths.RUNTIME_DIR / "inference_stats.json"),
                "max_n": 512, "cpu_threads": int(sel.get("cpu_threads", 1)), "max_batch": int(sel.get("max_batch", 256)),
                "max_wait_ms": float(sel.get("max_wait_ms", 5.0)),
                "max_models": self._max_models(),
                "shm_prefix": f"crl{tag}-"}

    def _max_models(self) -> int:
        """Networks the inference server keeps loaded: two per run (best + candidate) for the
        suggested number of runs or for the biggest population, whichever is more. Not a limit:
        when more are needed the least recently used one leaves and is loaded again later."""
        runs = int(self.selection.get("max_parallel_runs", 4))
        try:
            runs = max([runs] + [int(p["config"].get("size", 0)) for p in self.db.list_populations()])
        except Exception:
            pass
        return min(max(4, 2 * runs), 256)

    def inference_info(self) -> dict | None:
        """How the play worker reaches the server, or None when it is off or not ready.
        Never starts anything: the server starts with the first training run."""
        from .. import capacity
        if self.sup is None or not capacity.inference_enabled(self.selection):
            return None
        w = self.sup.get("inference")
        if w is None or not w.is_ready():
            return None
        return {"port_file": self.inference_cfg()["port_file"], "max_n": 64}

    def train_client(self, model_id: int) -> TrainClient:
        """Client of the training process that belongs to this model."""
        return TrainClient(self.sup.get(f"train-{model_id}"))

    def mcts_settings(self, **override) -> dict:
        merged = settings.merged_settings(self.db.get_settings())
        return {**merged["mcts"], **override}


S = AppState()
