"""Phase 9: the server panel (raw counters + /api/training/server) and fair worker sharing."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("chess")
pytest.importorskip("fastapi")

from backend.inference import RemoteEvaluator, make_hub, start_thread
from backend.inference.sources import UniformSource

MCTS = {"simulations": 8, "max_plies": 30}


def test_stats_hold_the_raw_counters_the_page_needs():
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    try:
        ev = RemoteEvaluator(hub.client_spec(), "m", timeout=5)
        x = np.zeros((4, 18, 8, 8), np.uint8)
        for _ in range(3):
            ev.evaluate(x)
        s = srv.server.stats()
        for k in ("uptime_s", "positions", "calls", "busy_s", "requests", "wait_s"):
            assert k in s, k
        assert s["positions"] == 12 and s["requests"] >= 3 and s["wait_s"] >= 0 and s["busy_s"] >= 0
    finally:
        srv.stop()


def test_capacity_gives_the_server_cpu_threads():
    from backend.capacity import plan
    cpu, gpu = plan(8, 16, "cpu"), plan(8, 16, "gpu", safe_vram=8)
    assert cpu["cpu_threads"] == 8 - cpu["selfplay_workers"] >= 1
    assert gpu["cpu_threads"] == 1
    assert plan(1, 4, "cpu")["cpu_threads"] == 1


def test_inference_cfg_carries_cpu_threads():
    from backend.app.state import S
    S.selection["cpu_threads"] = 3
    assert S.inference_cfg()["cpu_threads"] == 3


def test_workers_use_one_thread_each_and_the_parent_env_is_restored():
    from backend.rl.selfplay_workers import SelfplayWorkers, _single_thread_env
    before = os.environ.get("OMP_NUM_THREADS")
    with _single_thread_env():
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_pool_grows_and_shrinks_between_tasks():
    from backend.rl.selfplay_workers import SelfplayWorkers
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    workers = SelfplayWorkers(1, hub.client_spec())
    try:
        workers.resize(3)
        assert workers.n == 3 and len(workers.procs) == 3 and all(p.is_alive() for p in workers.procs)
        got = []
        assert workers.run_games("m", 6, MCTS, concurrent=2, seed=5, on_game=got.append) == 6
        assert len({g.index for g in got}) == 6
        gone = workers.procs[1:]
        workers.resize(1)
        assert workers.n == 1 and len(workers.task_qs) == 1 and workers.procs[0].is_alive()
        assert all(not p.is_alive() for p in gone)
        got.clear()
        assert workers.run_games("m", 2, MCTS, concurrent=2, seed=6, on_game=got.append) == 2
        workers.resize(0)                                  # never below one: worker 0 reports the live game
        assert workers.n == 1
    finally:
        workers.close()
        srv.stop()


def test_orchestrator_applies_a_new_worker_count_at_the_next_iteration(tmp_path):
    from backend.rl.config import TrainConfig
    from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    p = {"run_id": 9, "model_path": str(tmp_path / "m.pt"), "elo": 1000, "iterations": 0, "mcts": MCTS,
         "train": {"games_per_iteration": 4, "min_buffer": 1, "arena_games": 2, "arena_simulations": 8,
                   "arena_threshold": 0.5, "concurrent_games": 2, "buffer_size": 1000}}
    ev = EventLog()
    o = Orchestrator(FakeBackend(p["model_path"], "cpu", TrainConfig()), p, ev,
                     remote={"spec": hub.client_spec(), "workers": 2})
    o.start()
    try:
        t0 = time.time()
        while time.time() - t0 < 60 and (o.workers is None or o.status()["workers"] != 2):
            time.sleep(0.1)
        assert o.status()["workers"] == 2
        o.set_workers(1)
        t0 = time.time()
        while time.time() - t0 < 120 and o.status()["workers"] != 1:
            time.sleep(0.2)
        assert o.status()["workers"] == 1, o.error
    finally:
        o.stop()
        o._thread.join(timeout=40)
        srv.stop()


def test_gateway_server_panel_endpoint_and_rebalancing(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.state import S
    fast = {"mcts": {"simulations": 50, "max_plies": 40},
            "train": {"games_per_iteration": 6, "min_buffer": 1, "arena_games": 2,
                      "arena_simulations": 10, "concurrent_games": 2}}
    monkeypatch.setenv("CHESSRL_INFERENCE", "1")
    with TestClient(app) as c:
        t0 = time.time()
        while time.time() - t0 < 60:
            if c.get("/api/system/status").json()["workers"].get("play", {}).get("state") == "READY":
                break
            time.sleep(0.3)
        while time.time() - t0 < 60 and not c.get("/api/models").json():
            time.sleep(0.3)
        a = c.get("/api/models").json()[0]
        r = c.post("/api/models", json={"name": "Two", "size": "tiny"})
        assert r.status_code in (200, 201), r.text
        b = r.json()
        S.selection["selfplay_workers"] = 4
        try:
            idle = c.get("/api/training/server").json()
            assert idle["enabled"] and idle["stats"] is None and idle["runs"] == []
            assert c.post("/api/training/start", json={"model_id": a["id"], "overrides": fast, "force": True}).status_code == 201
            assert c.post("/api/training/start", json={"model_id": b["id"], "overrides": fast, "force": True}).status_code == 201
            t0, info = time.time(), None
            while time.time() - t0 < 120:
                time.sleep(0.7)
                info = c.get("/api/training/server").json()
                if info["stats"] and len(info["runs"]) == 2 and info["stats"]["positions"] > 0:
                    break
            assert info and info["state"] == "READY" and info["stats"]["positions"] > 0, info
            for k in ("busy_s", "requests", "wait_s", "calls", "uptime_s"):
                assert k in info["stats"]
            assert "models" not in info["stats"]
            # two runs share 4 workers: 2 each once the first run applied the rebalance
            t0 = time.time()
            while time.time() - t0 < 120:
                per = sorted(r_["workers"] for r_ in c.get("/api/training/server").json()["runs"])
                if per == [2, 2]:
                    break
                time.sleep(0.7)
            assert per == [2, 2], per
        finally:
            c.post("/api/training/stop")
            time.sleep(2)


def test_server_panel_is_hidden_when_the_server_is_off(monkeypatch):
    monkeypatch.setenv("CHESSRL_INFERENCE", "0")
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as c:
        d = c.get("/api/training/server").json()
        assert d["enabled"] is False and d["stats"] is None and d["runs"] == []
