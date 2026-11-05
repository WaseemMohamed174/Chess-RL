"""Population coordinator (PBT): runs a population generation after generation.

    training   every agent trains K more iterations (as its own run, stops by itself)
    evaluating every agent plays the frozen champion; score + rank are saved
    selecting  the champion is frozen, the next generation is made (exploit + explore)

The population row keeps `generation` and `phase`, so a stopped coordinator (or a server that
was closed in the middle) goes on from the same point: agents that already trained enough are
skipped, a scored generation is not scored again, an existing next generation is not remade."""
from __future__ import annotations

import asyncio
import os
import shutil

from fastapi import HTTPException

from .. import paths
from ..rl.config import PopulationConfig
from ..rl.population import select
from . import services
from .state import S


class CoordinatorError(RuntimeError):
    pass


class Ops:
    """Everything the coordinator does to the outside world (replaced by a fake in tests)."""

    async def begin_run(self, model, overrides, max_iterations):
        from .api import training
        return await training.begin_run(model, overrides, max_iterations)

    async def stop_run(self, run):
        try:
            await S.train_client(run["model_id"])._call("POST", "/stop")
        except Exception:
            S.db.set_run_status(run["id"], "stopped")
        await self.wait_run(run["id"], timeout=30)

    async def wait_run(self, run_id, timeout=None):
        """Wait until the run is over; returns its status (done / stopped / failed)."""
        waited = 0.0
        while True:
            r = S.db.get_run(run_id)
            if not r or r["status"] not in ("running", "paused"):
                break
            await asyncio.sleep(0.5)
            waited += 0.5
            if timeout and waited >= timeout:
                return r["status"]
        for _ in range(40):                      # let the gateway release the train process
            if S.sup is None or S.sup.get(f"train-{r['model_id']}") is None:
                break
            await asyncio.sleep(0.25)
        return r["status"] if r else "failed"

    async def score(self, pid, gen, champion):
        return await services.score_generation(pid, gen, champion)

    async def next_generation(self, pid, gen):
        return await services.make_next_generation(pid, gen)

    async def freeze(self, src, dst):
        def copy():
            tmp = dst + ".tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        await asyncio.to_thread(copy)


class Coordinator:
    def __init__(self, pid, max_generations=0, parallel=0, delete_replaced=False, ops=None):
        self.pid, self.max_generations = pid, max_generations
        self.parallel, self.delete_replaced = parallel, delete_replaced
        self.ops = ops or Ops()
        self.task: asyncio.Task | None = None
        self.active: dict[int, dict] = {}        # member id -> its running run
        self.finished = 0                        # generations finished since Start
        self.error: str | None = None
        self.failed_agents: list[str] = []

    # --------------------------------------------------------------- control
    @property
    def db(self):
        return S.db

    @property
    def alive(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self) -> None:
        self._set(status="running")
        self.task = asyncio.ensure_future(self._main())

    async def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    def info(self) -> dict:
        pop = self.db.get_population(self.pid) if self.db else None
        return {"population_id": self.pid, "running": self.alive,
                "status": pop["status"] if pop else None, "generation": pop["generation"] if pop else None,
                "phase": pop["phase"] if pop else None, "finished": self.finished,
                "max_generations": self.max_generations,
                "active_runs": [r["id"] for r in self.active.values()],
                "error": self.error, "failed_agents": list(self.failed_agents)}

    def _set(self, **fields) -> None:
        if self.db.get_population(self.pid):
            self.db.update_population(self.pid, **fields)
        S.bus.publish("population", {"type": "population", **self.info()})

    # ------------------------------------------------------------------ loop
    async def _main(self) -> None:
        try:
            await self._loop()
        except asyncio.CancelledError:
            await self._cleanup()
            self._set(status="stopped")
            raise
        except Exception as e:                    # noqa: BLE001 - shown on the website
            self.error = e.detail if isinstance(e, HTTPException) else str(e)
            await self._cleanup()
            self._set(status="failed")

    async def _cleanup(self) -> None:
        runs = {r["id"]: r for r in self.active.values()}
        pop = self.db.get_population(self.pid)
        for mem in (self.db.list_members(self.pid, pop["generation"]) if pop else []):
            if mem["model_id"] is None:
                continue
            run = self.db.active_run_for_model(mem["model_id"])   # one that was just starting
            if run:
                runs[run["id"]] = run
        for run in runs.values():
            try:
                await self.ops.stop_run(run)
            except BaseException:                 # noqa: BLE001
                pass
        self.active.clear()

    def _members(self, gen):
        return [m for m in self.db.list_members(self.pid, gen) if m["model_id"] is not None]

    def _scored(self, gen) -> bool:
        live = self._members(gen)
        return bool(live) and all(m["rank"] is not None for m in live)

    def _config(self) -> PopulationConfig:
        return PopulationConfig.from_dict(self.db.get_population(self.pid)["config"])

    async def _loop(self) -> None:
        while True:
            pop = self.db.get_population(self.pid)
            if not pop:
                return
            gen, phase = pop["generation"], pop["phase"]
            if phase == "idle":                    # a finished generation goes on to selecting
                phase = "selecting" if self._scored(gen) else "training"
            if phase == "training":
                await self._ensure_champion(gen)
                self._set(phase="training")
                await self._train(gen)
                phase = "evaluating"
            if phase == "evaluating":
                self._set(phase="evaluating")
                if not self._scored(gen):
                    await self._evaluate(gen)
                self.finished += 1                 # a generation counts when it is scored
                if self.max_generations and self.finished >= self.max_generations:
                    self._set(status="done", phase="idle")
                    return
            self._set(phase="selecting")
            await self._select(gen)

    # -------------------------------------------------------------- champion
    def _champ_path(self) -> str:
        return str(paths.CHECKPOINTS_DIR / f"pop_{self.pid}_champion.pt")

    def _champion(self) -> dict:
        return {"id": -1, "name": "champion", "path": self._champ_path()}

    async def _ensure_champion(self, gen: int) -> None:
        """Generation 0 plays the model the population was made from. An existing frozen file
        is kept: a restart must not move the champion."""
        if os.path.exists(self._champ_path()):
            return
        pop = self.db.get_population(self.pid)
        base = self.db.get_model(pop["base_model_id"]) if pop["base_model_id"] else None
        if base is None or not os.path.exists(base["path"]):
            first = next(iter(self._members(gen)), None)
            base = self.db.get_model(first["model_id"]) if first else None
        if base is None:
            raise CoordinatorError("no model to use as the first champion")
        await self.ops.freeze(base["path"], self._champ_path())

    # -------------------------------------------------------------- training
    async def _train(self, gen: int) -> None:
        k = self._config().iterations_per_generation
        sem = asyncio.Semaphore(self.parallel if self.parallel > 0 else 10**6)
        jobs = []
        for mem in self._members(gen):
            model = self.db.get_model(mem["model_id"])
            if model is None:
                continue
            if mem["start_iterations"] is None:
                self.db.set_member_start(mem["id"], model["iterations"])
                mem["start_iterations"] = model["iterations"]
            jobs.append(self._train_agent(mem, mem["start_iterations"] + k, sem))
        await asyncio.gather(*jobs)

    async def _train_agent(self, mem, target, sem) -> None:
        async with sem:
            for _ in range(2):                    # one retry; then the agent is scored as it is
                model = self.db.get_model(mem["model_id"])
                if model is None or target - model["iterations"] <= 0:
                    return
                try:
                    run = await self.ops.begin_run(model, mem["hyperparams"],
                                                   target - model["iterations"])
                except Exception:                 # noqa: BLE001
                    await asyncio.sleep(1)
                    continue
                self.active[mem["id"]] = run
                self.db.tag_run(run["id"], {"population_id": self.pid})
                try:
                    status = await self.ops.wait_run(run["id"])
                finally:
                    self.active.pop(mem["id"], None)
                if status == "done":
                    return
            self.failed_agents.append(mem["model_name"])

    # ------------------------------------------------------------ evaluating
    async def _evaluate(self, gen: int) -> None:
        if not self._members(gen):
            raise CoordinatorError(f"generation {gen} has no agents left")
        await self.ops.score(self.pid, gen, self._champion())

    # ------------------------------------------------------------- selecting
    async def _select(self, gen: int) -> None:
        pc = self._config()
        nxt = gen + 1
        ranked = [m for m in self._members(gen) if m["rank"] is not None]
        picked = select(ranked, pc.survivors)
        best = picked["champion"]
        if best is None:
            raise CoordinatorError(f"generation {gen} has no ranked agents")
        best_model = self.db.get_model(best["model_id"])
        if best_model is None or not os.path.exists(best_model["path"]):
            raise CoordinatorError(f"the file of the best agent '{best['model_name']}' is missing")
        await self.ops.freeze(best_model["path"], self._champ_path())    # before anyone trains on
        if not self.db.list_members(self.pid, nxt):
            self._drop_orphans(nxt)
            await self.ops.next_generation(self.pid, gen)
        self._set(generation=nxt, phase="training", champion_id=best_model["id"])
        if self.delete_replaced:
            keep = {m["model_id"] for m in self._members(nxt)}
            for mem in picked["replaced"]:
                m = self.db.get_model(mem["model_id"]) if mem["model_id"] else None
                if m and m["id"] not in keep:
                    try:
                        services.delete_model(m)          # the history rows stay
                    except HTTPException:
                        pass

    def _drop_orphans(self, nxt: int) -> None:
        """A server that closed while the next generation was being made leaves children that
        have no member row. Remove them so the names are free again."""
        pop = self.db.get_population(self.pid)
        prefix = f"{pop['name']} G{nxt:02d} C"
        used = {m["model_id"] for m in self.db.list_members(self.pid)}
        for m in self.db.list_models():
            if m["name"].startswith(prefix) and m["id"] not in used:
                try:
                    services.delete_model(m)
                except HTTPException:
                    pass


# ---------------------------------------------------------------- module API
def get(pid: int) -> Coordinator | None:
    return S.coordinators.get(pid)


def begin(pid: int, max_generations: int = 0, parallel: int = 0,
          delete_replaced: bool = False, ops=None) -> Coordinator:
    pop = S.db.get_population(pid)
    if not pop:
        raise CoordinatorError("population not found")
    c = get(pid)
    if c is not None and c.alive:
        raise CoordinatorError("this population is already running")
    if not S.db.list_members(pid, pop["generation"]):
        raise CoordinatorError("this population has no agents")
    c = Coordinator(pid, max_generations, parallel, delete_replaced, ops)
    S.coordinators[pid] = c
    c.start()
    return c
