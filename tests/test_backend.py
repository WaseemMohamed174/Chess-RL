"""Backend tests. Need: fastapi, httpx, uvicorn, chess, numpy, pytest (no PyTorch)."""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("chess")


# --------------------------------------------------------------- pure logic
def test_orchestrator_runs_a_fake_training_loop(tmp_path):
    from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator
    from backend.rl.config import TrainConfig
    payload = {"run_id": 1, "model_path": str(tmp_path / "m.pt"), "elo": 1000, "iterations": 0,
               "mcts": {"simulations": 8, "max_plies": 30, "leaves_per_batch": 4},
               "train": {"games_per_iteration": 2, "min_buffer": 1, "arena_games": 2,
                         "arena_simulations": 8, "arena_threshold": 0.5,
                         "concurrent_games": 2, "buffer_size": 1000}}
    ev = EventLog()
    o = Orchestrator(FakeBackend(payload["model_path"], "cpu", TrainConfig()), payload, ev)
    o.set_watch(True)
    o.start()
    t0 = time.time()
    while time.time() - t0 < 60 and not any(e["type"] == "metrics" for e in ev.since(0)["events"]):
        time.sleep(0.2)
    o.pause()
    time.sleep(0.5)
    assert o.state == "paused"
    o.resume()
    o.stop()
    o._thread.join(timeout=30)
    events = ev.since(0)["events"]
    m = [e for e in events if e["type"] == "metrics"]
    assert m and m[0]["iteration"] == 1 and m[0]["games"] == 2 and m[0]["policy_loss"] is not None
    states = [e["state"] for e in events if e["type"] == "status"]
    assert states[0] == "running" and "paused" in states and states[-1] == "stopped"
    assert any(e["type"] == "selfplay_move" for e in events)       # watched: full board
    live = [e for e in events if e["type"] == "live"]               # every run: light summary
    assert live and 0 < live[0]["white_pieces"] <= 16 and 0 < live[0]["black_pieces"] <= 16
    assert any(e["ended"] and e["result"] in ("white", "black", "draw") for e in live)
    phases = {e["name"] for e in events if e["type"] == "phase"}
    assert {"selfplay", "learning", "arena"} <= phases


def test_unwatched_run_sends_no_full_board(tmp_path):
    from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator
    from backend.rl.config import TrainConfig
    payload = {"run_id": 2, "model_path": str(tmp_path / "m.pt"), "elo": 1000, "iterations": 0,
               "mcts": {"simulations": 8, "max_plies": 30, "leaves_per_batch": 4},
               "train": {"games_per_iteration": 2, "min_buffer": 1, "arena_games": 2,
                         "arena_simulations": 8, "concurrent_games": 2, "buffer_size": 1000}}
    ev = EventLog()
    o = Orchestrator(FakeBackend(payload["model_path"], "cpu", TrainConfig()), payload, ev)
    o.start()
    t0 = time.time()
    while time.time() - t0 < 60 and not any(e["type"] == "metrics" for e in ev.since(0)["events"]):
        time.sleep(0.2)
    o.stop()
    o._thread.join(timeout=30)
    kinds = {e["type"] for e in ev.since(0)["events"]}
    assert "selfplay_move" not in kinds and "live" in kinds


def test_settings_validation():
    from backend.settings import validate_settings
    validate_settings({"mcts": {"simulations": 200}})
    for bad in ({"mcts": {"simulations": 5}}, {"nope": {}}, {"mcts": {"c_puct": "x"}},
                {"train": {"lr": True}}, {"mcts": {"unknown": 1}}):
        with pytest.raises(ValueError):
            validate_settings(bad)


# ------------------------------------------------------------ whole gateway
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


def test_starter_model_and_architecture(client):
    models = client.get("/api/models").json()
    assert len(models) == 1 and models[0]["params"] > 0
    a = client.get(f"/api/models/{models[0]['id']}/architecture").json()
    assert a["params"] == models[0]["params"] and a["layers"]


def test_model_crud(client):
    m = client.post("/api/models", json={"name": "Alpha", "size": "small"}).json()
    assert (m["blocks"], m["filters"]) == (6, 128)
    assert client.post("/api/models", json={"name": "Alpha"}).status_code == 409
    c = client.post(f"/api/models/{m['id']}/clone", json={"name": "Alpha 2"}).json()
    assert c["parent_id"] == m["id"]
    assert client.put(f"/api/models/{c['id']}", json={"name": "Beta"}).json()["name"] == "Beta"
    assert client.get(f"/api/models/{c['id']}/download").status_code == 200
    assert client.delete(f"/api/models/{c['id']}").status_code == 204
    assert client.delete(f"/api/models/{m['id']}").status_code == 204


def test_play_flow(client):
    mid = client.get("/api/models").json()[0]["id"]
    g = client.post("/api/games/new", json={"model_id": mid, "color": "white"}).json()
    assert g["human_to_move"] and "e2e4" in g["legal_moves"]
    s = client.post("/api/games/play", json={"game_id": g["game_id"], "move_uci": "e2e4"}).json()
    assert len(s["moves"]) == 2 and s["human_to_move"] and s["eval"] is not None
    assert client.post("/api/games/play", json={"game_id": g["game_id"],
                                                "move_uci": "e2e5"}).status_code == 400
    assert client.post(f"/api/games/{g['game_id']}/hint").json()["move"]
    assert len(client.post(f"/api/games/{g['game_id']}/undo").json()["moves"]) == 0
    r = client.post(f"/api/games/{g['game_id']}/resign").json()
    assert r["over"] and r["result"] == "0-1"
    assert "[Result \"0-1\"]" in client.get(f"/api/games/{g['game_id']}/pgn").text
    b = client.post("/api/games/new", json={"model_id": mid, "color": "black"}).json()
    assert len(b["moves"]) == 1                      # the model, playing White, already moved


def test_settings_roundtrip(client):
    assert client.put("/api/settings", json={"mcts": {"simulations": 5}}).status_code == 422
    v = client.put("/api/settings", json={"mcts": {"simulations": 50, "max_plies": 40}}).json()
    assert v["values"]["mcts"]["simulations"] == 50
    assert client.get("/api/settings").json()["values"]["mcts"]["max_plies"] == 40


def test_arena_match_and_training(client):
    a = client.get("/api/models").json()[0]
    b = client.post("/api/models", json={"name": "Rival", "size": "tiny"}).json()
    m = client.post("/api/arena/match", json={"white_id": a["id"], "black_id": b["id"],
                                              "games": 1, "simulations": 50,
                                              "delay_ms": 0}).json()
    t0 = time.time()
    while time.time() - t0 < 120 and client.get(f"/api/arena/match/{m['id']}").json()["running"]:
        time.sleep(0.5)
    done = client.get(f"/api/arena/match/{m['id']}").json()
    assert done["white_wins"] + done["black_wins"] + done["draws"] == 1

    fast = {"mcts": {"simulations": 50, "max_plies": 40},
            "train": {"games_per_iteration": 2, "min_buffer": 1, "arena_games": 2,
                      "arena_simulations": 10, "concurrent_games": 2}}
    r = client.post("/api/training/start", json={"model_id": a["id"], "overrides": fast, "force": True})
    assert r.status_code == 201, r.text
    ra = r.json()
    assert client.post("/api/training/start", json={"model_id": a["id"]}).status_code == 409

    # a second model trains at the same time, in its own process
    r = client.post("/api/training/start", json={"model_id": b["id"], "overrides": fast, "force": True})
    assert r.status_code == 201, r.text
    rb = r.json()
    st = client.get("/api/training/status").json()
    assert {x["run"]["id"] for x in st["runs"]} == {ra["id"], rb["id"]}
    assert st["watch"] == ra["id"]                       # the first run is watched by default
    assert all(x["model"]["name"] for x in st["runs"])

    # the board is streamed for one run only
    assert client.post("/api/training/watch", json={"run_id": rb["id"]}).json()["watch"] == rb["id"]
    assert client.get("/api/training/status").json()["watch"] == rb["id"]
    assert client.post("/api/training/watch", json={"run_id": 9999}).status_code == 409

    t0 = time.time()
    done = False
    while time.time() - t0 < 120 and not done:
        time.sleep(0.5)
        runs = client.get("/api/training/status").json()["runs"]
        done = len(runs) == 2 and all(x["metrics"] for x in runs)
    assert done, "both runs should report iteration 1"

    # control one run without touching the other
    assert client.post(f"/api/training/runs/{rb['id']}/pause").status_code == 200
    assert client.post(f"/api/training/runs/{rb['id']}/resume").status_code == 200
    assert client.post(f"/api/training/runs/{ra['id']}/stop").status_code == 200
    t0 = time.time()
    by_id = {}
    while time.time() - t0 < 60:
        by_id = {r["id"]: r["status"] for r in client.get("/api/training/runs").json()}
        if by_id.get(ra["id"]) == "stopped":
            break
        time.sleep(0.5)
    assert by_id[ra["id"]] == "stopped" and by_id[rb["id"]] in ("running", "paused")

    # stop everything that is left
    assert client.post("/api/training/stop").status_code == 200
    t0 = time.time()
    while time.time() - t0 < 60 and client.get("/api/training/status").json()["runs"]:
        time.sleep(0.5)
    st = client.get("/api/training/status").json()
    assert st["runs"] == [] and st["last"]["run"]["status"] == "stopped"
    assert client.post("/api/training/stop").status_code == 409


def test_selected_model_is_created_again_when_missing(client):
    """ensure_selected_model(): the size picked in run.py exists, and a never-trained model with a
    deleted file gets a fresh file. It must also be safe to run on every start."""
    import asyncio, os
    from backend.app.main import ensure_selected_model
    from backend.app.state import S
    loop = S.loop if hasattr(S, "loop") else None
    net = S.selection["network"]

    async def run():
        # 1) a different size is selected on a later run -> a model of that size appears
        old = dict(S.selection["network"])
        S.selection["network"] = {"key": "custom", "name": "Custom", "blocks": 2, "filters": 32}
        try:
            await ensure_selected_model()
            await ensure_selected_model()               # second call changes nothing
        finally:
            S.selection["network"] = old
    asyncio.run(run()) if loop is None else asyncio.run_coroutine_threadsafe(run(), loop).result(60)
    models = client.get("/api/models").json()
    assert sum(1 for m in models if (m["blocks"], m["filters"]) == (2, 32)) == 1

    # 2) never-trained model whose file was deleted -> file comes back
    m = next(m for m in models if (m["blocks"], m["filters"]) == (2, 32))
    path = S.db.get_model(m["id"])["path"]
    os.remove(path)
    assert not [x for x in client.get("/api/models").json() if x["id"] == m["id"]][0]["file_ok"]
    if loop is None:
        asyncio.run(ensure_selected_model())
    else:
        asyncio.run_coroutine_threadsafe(ensure_selected_model(), loop).result(60)
    assert os.path.exists(path)
    assert [x for x in client.get("/api/models").json() if x["id"] == m["id"]][0]["file_ok"]
