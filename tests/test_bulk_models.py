"""Bulk buttons on the Models page: delete all, download all (.zip), start all."""
import io
import time
import zipfile

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


def test_download_all_is_one_zip_with_every_file(client):
    names = ["Zip One", "Zip Two"]
    made = [client.post("/api/models", json={"name": n, "size": "tiny"}).json() for n in names]
    r = client.get("/api/models/download-all")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert {"Zip One.pt", "Zip Two.pt"} <= set(z.namelist())
    only = client.get(f"/api/models/download-all?ids={made[0]['id']}")
    assert zipfile.ZipFile(io.BytesIO(only.content)).namelist() == ["Zip One.pt"]
    assert client.get("/api/models/download-all?ids=abc").status_code == 400
    assert client.get("/api/models/download-all?ids=999999").status_code == 404


def test_bulk_delete_removes_only_the_listed_models(client):
    a = client.post("/api/models", json={"name": "Bulk A", "size": "tiny"}).json()
    b = client.post("/api/models", json={"name": "Bulk B", "size": "tiny"}).json()
    out = client.post("/api/models/bulk-delete", json={"ids": [a["id"]]}).json()
    assert [d["id"] for d in out["deleted"]] == [a["id"]] and out["skipped"] == []
    ids = {m["id"] for m in client.get("/api/models").json()}
    assert a["id"] not in ids and b["id"] in ids


def test_bulk_delete_all_and_start_all_with_nothing_to_start(client):
    out = client.post("/api/models/bulk-delete", json={}).json()
    assert out["deleted"] and out["skipped"] == []
    assert client.get("/api/models").json() == []
    assert client.post("/api/training/start-all", json={}).status_code == 409   # no model to start


def test_start_all_then_delete_all_skips_the_training_models(client):
    client.put("/api/settings", json={
        "mcts": {"simulations": 50, "max_plies": 40},
        "train": {"games_per_iteration": 2, "min_buffer": 1, "arena_games": 2,
                  "arena_simulations": 10, "concurrent_games": 2}})
    a = client.post("/api/models", json={"name": "Run A", "size": "tiny"}).json()
    b = client.post("/api/models", json={"name": "Run B", "size": "tiny"}).json()
    r = client.post("/api/training/start-all", json={"force": True})
    assert r.status_code == 201, r.text
    out = r.json()
    assert {s["id"] for s in out["started"]} == {a["id"], b["id"]} and out["failed"] == []
    assert len(client.get("/api/training/status").json()["runs"]) == 2
    assert client.post("/api/training/start-all", json={"force": True}).status_code == 409  # all busy

    kept = client.post("/api/models/bulk-delete", json={}).json()          # training: not deleted
    assert kept["deleted"] == [] and {s["id"] for s in kept["skipped"]} == {a["id"], b["id"]}
    assert len(client.get("/api/models").json()) >= 2

    assert client.post("/api/training/stop").status_code == 200
    t0 = time.time()
    while time.time() - t0 < 60 and client.get("/api/training/status").json()["runs"]:
        time.sleep(0.5)
    gone = client.post("/api/models/bulk-delete", json={}).json()
    assert {d["id"] for d in gone["deleted"]} >= {a["id"], b["id"]} and gone["skipped"] == []
