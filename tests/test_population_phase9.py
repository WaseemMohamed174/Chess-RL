"""Population Based Training, phase 9: over capacity is a warning with numbers, not a block."""
import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from backend import capacity

CPU = {"device": "cpu", "selfplay_workers": 2}
HW = {"safe_ram": 8.0, "safe_vram": 0.0}


def test_few_runs_fit_and_many_do_not():
    assert capacity.estimate(1, CPU, HW)["level"] == "ok"
    big = capacity.estimate(100, CPU, HW)
    assert big["level"] == "severe" and big["need_ram"] > 8 and big["runs"] == 100


def test_fits_is_the_batch_size_and_at_least_one():
    e = capacity.estimate(100, CPU, HW)
    assert 1 <= e["fits"] < 100 and e["slower_by"] >= 1
    assert capacity.estimate(e["fits"], CPU, HW)["level"] == "ok"
    assert capacity.estimate(5, CPU, {"safe_ram": 0.5})["fits"] >= 1


def test_levels_grow_with_the_number_of_runs():
    order = ["ok", "warn", "severe"]
    levels = [order.index(capacity.estimate(n, CPU, HW)["level"]) for n in range(1, 60)]
    assert levels == sorted(levels) and levels[0] == 0 and levels[-1] == 2
    assert 1 in levels                                        # "warn" exists between them


def test_no_hard_limit_on_the_number_of_runs():
    assert capacity.estimate(500, CPU, HW)["runs"] == 500 > capacity.MAX_RUNS


def test_gpu_is_limited_by_vram_too():
    gpu = {"device": "gpu", "selfplay_workers": 2}
    e = capacity.estimate(10, gpu, {"safe_ram": 64.0, "safe_vram": 1.0})
    assert e["level"] in ("warn", "severe") and e["avail_vram"] == 1.0
    assert capacity.estimate(10, gpu, {"safe_ram": 64.0, "safe_vram": 0})["level"] == "severe"


def test_unknown_machine_never_warns():
    assert capacity.estimate(50, CPU, None)["level"] == "unknown"


def test_guard_answers_409_with_the_numbers_and_force_skips_it(monkeypatch):
    from fastapi import HTTPException
    from backend.app import services

    async def est(runs, model=None):
        return capacity.estimate(runs, CPU, HW)
    monkeypatch.setattr(services, "capacity_estimate", est)
    with pytest.raises(HTTPException) as e:
        asyncio.run(services.need_capacity(100, None, False))
    assert e.value.status_code == 409 and e.value.detail["code"] == "capacity"
    assert e.value.detail["need_ram"] > 8 and "fits" in e.value.detail
    asyncio.run(services.need_capacity(100, None, True))          # force: no error
    asyncio.run(services.need_capacity(1, None, False))           # fits: no error


def test_max_models_follows_the_biggest_population():
    from backend.app.state import AppState
    s = AppState()
    s.selection = {"max_parallel_runs": 4}
    s.db = SimpleNamespace(list_populations=lambda: [{"config": {"size": 100}}])
    assert s._max_models() == 200
    s.db = SimpleNamespace(list_populations=lambda: [])
    assert s._max_models() == 8
    s.db = SimpleNamespace(list_populations=lambda: [{"config": {"size": 5000}}])
    assert s._max_models() == 256                                 # sane ceiling for the cache


# ----------------------------------------------------------- through the gateway
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as c:
        t0 = time.time()
        while time.time() - t0 < 60:
            if c.get("/api/system/status").json()["workers"].get("play", {}).get("state") == "READY":
                break
            time.sleep(0.3)
        else:
            pytest.fail("play worker did not become READY")
        while time.time() - t0 < 60 and not c.get("/api/models").json():
            time.sleep(0.3)
        yield c


@pytest.fixture
def tight(monkeypatch):
    """A machine that can hold one run only, whatever this computer really is."""
    from backend.app import services

    async def est(runs, model=None):
        return capacity.estimate(runs, CPU, HW if runs < 3 else {"safe_ram": 3.0, "safe_vram": 0.0})
    monkeypatch.setattr(services, "capacity_estimate", est)


def test_capacity_endpoint_reports_the_numbers(client):
    r = client.get("/api/system/capacity", params={"runs": 40})
    assert r.status_code == 200
    d = r.json()
    assert d["runs"] == 40 and d["need_ram"] > 0 and d["level"] in ("ok", "warn", "severe", "unknown")
    assert client.get("/api/system/capacity", params={"runs": 0}).status_code == 422


def test_training_start_warns_then_goes_on_with_force(client, tight, monkeypatch):
    from backend.app.api import training
    started = []

    async def fake_begin(model, overrides=None, max_iterations=0):
        started.append(model["id"])
        return {"id": 1}
    monkeypatch.setattr(training, "begin_run", fake_begin)
    mid = client.get("/api/models").json()[0]["id"]
    monkeypatch.setattr("backend.app.state.S.db.active_runs", lambda: [{}] * 49)   # 50th run
    r = client.post("/api/training/start", json={"model_id": mid})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "capacity" and not started
    r = client.post("/api/training/start", json={"model_id": mid, "force": True})
    assert r.status_code == 201 and started == [mid]


def test_population_start_warns_then_goes_on_with_force(client, tight, monkeypatch):
    from backend.app import coordinator
    from backend.app.state import S
    pop = S.db.create_population(f"cap-{time.time_ns()}", {"size": 100}, None)
    begun = []

    class C:
        def info(self):
            return {"state": "running"}

    def fake_begin(pid, **kw):
        begun.append(kw)
        return C()
    monkeypatch.setattr(coordinator, "begin", fake_begin)
    monkeypatch.setattr(S.db, "active_runs", lambda: [{}] * 49)
    r = client.post(f"/api/populations/{pop['id']}/start", json={"parallel": 0})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "capacity" and not begun
    r = client.post(f"/api/populations/{pop['id']}/start", json={"force": True})
    assert r.status_code == 202 and len(begun) == 1
