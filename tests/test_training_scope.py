"""The live training window of a population is kept apart from the single models' window."""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("chess")


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


def _wait(fn, secs=90):
    t0 = time.time()
    while time.time() - t0 < secs:
        v = fn()
        if v:
            return v
        time.sleep(0.4)
    return None


def test_population_runs_stay_out_of_the_single_models_window(client):
    client.put("/api/settings", json={
        "mcts": {"simulations": 50, "max_plies": 40},
        "train": {"games_per_iteration": 2, "min_buffer": 1, "arena_games": 2,
                  "arena_simulations": 10, "concurrent_games": 2},
        "population": {"size": 2, "iterations_per_generation": 50}})
    single = client.post("/api/models", json={"name": "Solo", "size": "tiny"}).json()
    base = client.post("/api/models", json={"name": "Base", "size": "tiny"}).json()
    pop = client.post("/api/populations", json={"name": "Pop", "base_model_id": base["id"]}).json()
    pid = pop["population"]["id"] if "population" in pop else pop["id"]

    # agents are marked on the Models list, the others are not
    models = {m["name"]: m for m in client.get("/api/models").json()}
    assert models["Solo"]["population_id"] is None and models["Base"]["population_id"] is None
    agents = [m for m in models.values() if m["population_id"] == pid]
    assert len(agents) == 2

    # one single model and one population train at the same time
    assert client.post("/api/training/start", json={"model_id": single["id"], "force": True}).status_code == 201
    assert client.post(f"/api/populations/{pid}/start", json={"force": True}).status_code == 202

    pop_runs = _wait(lambda: client.get(f"/api/training/status?population_id={pid}").json()["runs"])
    assert pop_runs, "the population's agents should be training"
    agent_ids = {m["id"] for m in agents}
    assert {r["model"]["id"] for r in pop_runs} <= agent_ids
    assert all(r["population_id"] == pid for r in pop_runs)

    solo = client.get("/api/training/status?population_id=single").json()
    assert [r["model"]["id"] for r in solo["runs"]] == [single["id"]]       # no agent in here
    assert all(r["population_id"] is None for r in solo["runs"])
    everyone = client.get("/api/training/status").json()
    assert len(everyone["runs"]) == len(pop_runs) + 1
    assert set(solo["busy_models"]) >= {single["id"]} | {r["model"]["id"] for r in pop_runs}
    assert client.get("/api/training/status?population_id=oops").status_code == 400

    # "Start all" leaves the agents to the Population tab
    extra = client.post("/api/models", json={"name": "Solo 2", "size": "tiny"}).json()
    out = client.post("/api/training/start-all", json={"force": True}).json()
    started = {s["id"] for s in out["started"]}
    assert extra["id"] in started and base["id"] in started          # plain models are started
    assert not (started & agent_ids)                                  # the agents are not
    skipped = {s["id"]: s["reason"] for s in out["skipped"]}
    assert skipped[single["id"]] == "already training"
    assert all(skipped[a] in ("population agent",) for a in agent_ids)

    # stop everything; the last finished run is remembered per window
    assert client.post(f"/api/populations/{pid}/stop").status_code == 200
    assert client.post("/api/training/stop").status_code == 200
    assert _wait(lambda: not client.get("/api/training/status").json()["runs"], 60)
    last_pop = client.get(f"/api/training/status?population_id={pid}").json()["last"]
    last_solo = client.get("/api/training/status?population_id=single").json()["last"]
    assert last_pop["population_id"] == pid and last_pop["model"]["id"] in agent_ids
    assert last_solo["population_id"] is None and last_solo["model"]["id"] not in agent_ids
