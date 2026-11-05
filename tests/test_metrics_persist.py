"""A finished run must show the same numbers as a running one: the iteration's memory size,
arena result and 'promoted' flag are saved with its metrics (and old databases get the columns)."""
import sqlite3

from backend.db import Database


def _run(db):
    m = db.create_model("a", {"blocks": 2, "filters": 32}, "/tmp/a.pt")
    return db.create_run(m["id"], {})


def test_full_metric_roundtrip(tmp_path):
    db = Database(tmp_path / "t.db")
    run = _run(db)
    db.add_metric(run["id"], {"iteration": 1, "policy_loss": 2.1, "value_loss": 0.9, "elo": 1012.0,
                              "games": 20, "games_per_min": 14.3, "buffer": 1840, "promoted": True,
                              "arena": {"wins": 6, "losses": 3, "draws": 1, "score": 0.65}})
    r = db.list_metrics(run["id"])[0]
    assert r["buffer"] == 1840 and r["promoted"] is True
    assert r["arena"] == {"wins": 6, "losses": 3, "draws": 1, "score": 0.65}


def test_metric_without_arena(tmp_path):
    db = Database(tmp_path / "t.db")
    run = _run(db)
    db.add_metric(run["id"], {"iteration": 1, "policy_loss": 1.0, "promoted": False, "buffer": 5})
    r = db.list_metrics(run["id"])[0]
    assert r["arena"] is None and r["promoted"] is False and r["buffer"] == 5


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript("""CREATE TABLE metrics(id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL,
        iteration INTEGER NOT NULL, policy_loss REAL, value_loss REAL, elo REAL, games INTEGER,
        games_per_min REAL, created_at TEXT NOT NULL);
        INSERT INTO metrics(run_id,iteration,policy_loss,created_at) VALUES(1,1,1.5,'x');""")
    c.commit(); c.close()
    r = Database(path).list_metrics(1)[0]
    assert r["policy_loss"] == 1.5 and r["buffer"] is None and r["arena"] is None
