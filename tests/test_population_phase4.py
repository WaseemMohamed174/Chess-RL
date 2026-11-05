"""Population Based Training, phase 4: a training run can stop by itself after K iterations."""
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("chess")

from backend.rl.config import TrainConfig
from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator

FAST = {"mcts": {"simulations": 8, "max_plies": 30, "leaves_per_batch": 4},
        "train": {"games_per_iteration": 2, "min_buffer": 1, "arena_games": 2,
                  "arena_simulations": 8, "arena_threshold": 0.5,
                  "concurrent_games": 2, "buffer_size": 1000}}


def make(tmp_path, name="m.pt", **extra):
    payload = {"run_id": 1, "model_path": str(tmp_path / name), "elo": 1000, **FAST, **extra}
    ev = EventLog()
    return Orchestrator(FakeBackend(payload["model_path"], "cpu", TrainConfig()), payload, ev), ev


def events(ev, kind):
    return [e for e in ev.since(0)["events"] if e["type"] == kind]


def wait(cond, secs=90):
    t0 = time.time()
    while time.time() - t0 < secs and not cond():
        time.sleep(0.1)
    return cond()


# ------------------------------------------------------ the orchestrator on its own
def test_stops_by_itself_after_k_iterations(tmp_path):
    o, ev = make(tmp_path, max_iterations=3)
    o.start()
    assert wait(lambda: not o.alive()), "the loop should end without anyone pressing stop"
    assert o.state == "done"
    assert [m["iteration"] for m in events(ev, "metrics")] == [1, 2, 3]
    st = o.status()
    assert (st["iterations_run"], st["max_iterations"]) == (3, 3)
    states = events(ev, "status")
    assert states[-1]["state"] == "done"
    assert states[-1]["seq"] > events(ev, "metrics")[-1]["seq"]      # the last metrics come first


def test_k_counts_this_run_not_the_lifetime_of_the_model(tmp_path):
    o, ev = make(tmp_path, iterations=10, max_iterations=2)           # the model did 10 before
    o.start()
    assert wait(lambda: not o.alive())
    assert [m["iteration"] for m in events(ev, "metrics")] == [11, 12]
    assert o.state == "done" and o.status()["iterations_run"] == 2


def test_zero_means_no_limit_and_a_stop_is_a_stop(tmp_path):
    o, ev = make(tmp_path)                                            # max_iterations not given
    assert o.max_iterations == 0
    o.start()
    assert wait(lambda: events(ev, "metrics"))
    o.stop()
    assert wait(lambda: not o.alive())
    assert o.state == "stopped" and events(ev, "status")[-1]["state"] == "stopped"


def test_stopping_before_the_limit_is_stopped_not_done(tmp_path):
    o, ev = make(tmp_path, max_iterations=500)
    o.start()
    assert wait(lambda: events(ev, "metrics"))
    o.stop()
    assert wait(lambda: not o.alive())
    assert o.state == "stopped" and 1 <= o.status()["iterations_run"] < 500
    assert "done" not in [e["state"] for e in events(ev, "status")]


@pytest.mark.parametrize("given,expected", [(-3, 0), (None, 0), (0, 0), (7, 7)])
def test_max_iterations_is_never_negative(tmp_path, given, expected):
    o, _ = make(tmp_path, max_iterations=given)
    assert o.max_iterations == expected


def test_a_finished_agent_keeps_its_buffer_for_the_next_generation(tmp_path):
    o, ev = make(tmp_path, max_iterations=2)
    o.start()
    assert wait(lambda: not o.alive())
    assert os.path.exists(str(tmp_path / "m.pt") + ".buffer.npz")
    before = events(ev, "metrics")[-1]["buffer"]
    o2, ev2 = make(tmp_path, iterations=2, max_iterations=1)          # the next generation
    o2.start()
    assert wait(lambda: not o2.alive())
    m = events(ev2, "metrics")[0]
    assert m["iteration"] == 3 and m["buffer"] > before               # it started from the old buffer


# ------------------------------------------------------------ the whole gateway
@pytest.fixture(scope="module")
def client():
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


def _run_until_done(client, mid, k):
    fast = {"mcts": {"simulations": 50, "max_plies": 40},
            "train": {"games_per_iteration": 2, "min_buffer": 1, "arena_games": 2,
                      "arena_simulations": 10, "concurrent_games": 2}}
    r = client.post("/api/training/start", json={"model_id": mid, "overrides": fast, "force": True, "max_iterations": k})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    t0 = time.time()
    while time.time() - t0 < 150:
        run = next(x for x in client.get(f"/api/training/runs?model_id={mid}").json() if x["id"] == rid)
        if run["status"] not in ("running", "paused"):
            return run
        time.sleep(0.5)
    pytest.fail("the run did not finish by itself")


def test_gateway_run_ends_as_done_and_the_next_run_continues(client):
    from backend.app.state import S
    m = client.post("/api/models", json={"name": "PBT-phase4", "size": "tiny"}).json()
    mid = m["id"]

    run = _run_until_done(client, mid, 2)
    assert run["status"] == "done" and run["ended_at"]
    assert [x["iteration"] for x in client.get(f"/api/training/runs/{run['id']}/metrics").json()] == [1, 2]
    assert client.get(f"/api/models/{mid}").json()["iterations"] == 2
    t0 = time.time()
    while time.time() - t0 < 15 and S.sup.get(f"train-{mid}") is not None:
        time.sleep(0.2)
    assert S.sup.get(f"train-{mid}") is None, "the finished training process must be released"
    assert all(x["run"]["id"] != run["id"] for x in client.get("/api/training/status").json()["runs"])

    run2 = _run_until_done(client, mid, 1)                    # the next generation, same agent
    assert run2["id"] != run["id"] and run2["status"] == "done"
    assert client.get(f"/api/models/{mid}").json()["iterations"] == 3     # continued, not restarted


def test_the_start_request_refuses_a_negative_limit(client):
    mid = client.get("/api/models").json()[0]["id"]
    r = client.post("/api/training/start", json={"model_id": mid, "max_iterations": -1})
    assert r.status_code == 422
