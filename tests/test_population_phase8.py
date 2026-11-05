"""Population Based Training, phase 8: the coordinator (train -> score -> select -> next)."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chess")

from backend import paths
from backend.app import coordinator as co
from backend.app import services
from backend.db import Database


class Bus:
    def publish(self, *a, **k):
        pass


class FakeOps(co.Ops):
    def __init__(self, db):
        self.db, self.runs = db, []

    async def begin_run(self, model, overrides, max_iterations):
        self.runs.append((model["id"], max_iterations))
        self.db.update_model(model["id"], iterations=model["iterations"] + max_iterations)
        r = self.db.create_run(model["id"], {})
        self.db.set_run_status(r["id"], "done")
        return r

    async def wait_run(self, run_id, timeout=None):
        return "done"

    async def score(self, pid, gen, champion):
        mems = [m for m in self.db.list_members(pid, gen) if m["model_id"] is not None]
        for rank, m in enumerate(sorted(mems, key=lambda m: -m["model_id"]), start=1):
            self.db.set_member_result(m["id"], 1 - rank / 100, 1.0, rank)


@pytest.fixture
def env(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", ckpt)
    db = Database(tmp_path / "t.db")
    S = SimpleNamespace(db=db, play=None, bus=Bus(), coordinators={}, sup=None)
    for mod in (services, co):
        monkeypatch.setattr(mod, "S", S)
    base_path = ckpt / "base.pt"
    base_path.write_bytes(b"base")
    base = db.create_model("Base", {"blocks": 3, "filters": 64}, str(base_path))
    out = asyncio.run(services.create_population("P", base, {"size": 4, "iterations_per_generation": 2}))
    yield SimpleNamespace(db=db, pid=out["population"]["id"], ops=FakeOps(db), S=S)
    db.close()


def go(env, **kw):
    async def run():
        c = co.begin(env.pid, ops=env.ops, **kw)
        await c.task
        return c
    return asyncio.run(run())


def test_one_generation_trains_scores_and_makes_the_next(env):
    c = go(env, max_generations=1)
    pop = env.db.get_population(env.pid)
    assert pop["status"] == "done" and c.error is None
    assert len(env.ops.runs) == 4 and all(k == 2 for _, k in env.ops.runs)
    assert all(m["rank"] is not None for m in env.db.list_members(env.pid, 0))
    assert not env.db.list_members(env.pid, 1)         # stops after scoring; next is made on go-on


def test_second_start_goes_on_with_the_next_generation(env):
    go(env, max_generations=1)
    go(env, max_generations=1)
    pop = env.db.get_population(env.pid)
    assert pop["generation"] == 1 and len(env.db.list_members(env.pid, 1)) == 4
    assert pop["champion_id"] is not None
    # generation 1: survivors keep their model, children inherit iterations; all get +2 more
    its = [env.db.get_model(m["model_id"])["iterations"] for m in env.db.list_members(env.pid, 1)]
    assert max(its) == 4


def test_a_restart_in_the_middle_does_not_train_again(env):
    # the server closed after two agents of generation 0 had trained
    mems = env.db.list_members(env.pid, 0)
    for m in mems[:2]:
        env.db.set_member_start(m["id"], 0)
        env.db.update_model(m["model_id"], iterations=2)
    env.db.update_population(env.pid, phase="training", status="stopped")
    go(env, max_generations=1)
    assert len(env.ops.runs) == 2                      # only the two that were missing
    assert all(m["rank"] is not None for m in env.db.list_members(env.pid, 0))


def test_stop_cancels_and_keeps_the_phase(env):
    async def run():
        c = co.begin(env.pid, ops=env.ops)
        await asyncio.sleep(0)
        await c.stop()
        return c
    asyncio.run(run())
    assert env.db.get_population(env.pid)["status"] == "stopped"
