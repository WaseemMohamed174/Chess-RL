"""Population Based Training, phase 2: the populations / population_members tables."""
import sqlite3

import pytest

from backend.db import Database
from backend.rl.config import PopulationConfig


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    yield d
    d.close()


def make_models(db, n):
    return [db.create_model(f"agent-{i}", {"blocks": 3, "filters": 64}, f"/tmp/a{i}.pt") for i in range(n)]


def make_population(db, n=3):
    ms = make_models(db, n)
    pop = db.create_population("P1", PopulationConfig().to_dict(), base_model_id=ms[0]["id"])
    return pop, ms


# ---------------------------------------------------------------- the schema
def test_new_tables_exist(db):
    names = {r["name"] for r in db._all("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"populations", "population_members"} <= names


def test_an_old_database_is_upgraded_without_losing_data(tmp_path):
    path = tmp_path / "old.db"
    d = Database(path)
    m = d.create_model("keep-me", {}, "/tmp/k.pt")
    d._run("DROP TABLE population_members")              # what a database from before phase 2 looks like
    d._run("DROP TABLE populations")
    d.close()
    d2 = Database(path)                                  # opening it again must add the tables
    try:
        assert d2.get_model(m["id"])["name"] == "keep-me"
        assert d2.list_populations() == []
        d2.create_population("after-upgrade", {})
    finally:
        d2.close()


def test_foreign_keys_are_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.add_member(999, 0, None, "ghost")             # no such population


# ---------------------------------------------------------------- populations
def test_create_get_list(db):
    pop, ms = make_population(db)
    assert pop["name"] == "P1" and pop["generation"] == 0
    assert (pop["status"], pop["phase"], pop["champion_id"]) == ("idle", "idle", None)
    assert pop["base_model_id"] == ms[0]["id"]
    assert pop["config"] == PopulationConfig().to_dict()  # the settings it was started with
    assert db.get_population(pop["id"]) == pop
    assert [p["id"] for p in db.list_populations()] == [pop["id"]]
    assert db.get_population(12345) is None


def test_population_names_are_unique(db):
    db.create_population("same", {})
    with pytest.raises(sqlite3.IntegrityError):
        db.create_population("same", {})


def test_update_population(db):
    pop, ms = make_population(db)
    u = db.update_population(pop["id"], status="running", phase="training", generation=2,
                            champion_id=ms[1]["id"], config={"size": 4})
    assert (u["status"], u["phase"], u["generation"]) == ("running", "training", 2)
    assert u["champion_id"] == ms[1]["id"] and u["config"] == {"size": 4}
    u2 = db.update_population(pop["id"], created_at="x", id=99, bogus=1)   # unknown fields are ignored
    assert u2["id"] == pop["id"] and u2["created_at"] == pop["created_at"]
    assert u2["updated_at"] >= pop["updated_at"]


def test_update_population_refuses_unknown_status_and_phase(db):
    pop, _ = make_population(db)
    with pytest.raises(ValueError):
        db.update_population(pop["id"], status="exploding")
    with pytest.raises(ValueError):
        db.update_population(pop["id"], phase="dancing")
    assert db.get_population(pop["id"])["status"] == "idle"


# -------------------------------------------------------------------- members
def test_add_member_and_json_roundtrip(db):
    pop, ms = make_population(db)
    hp = {"mcts": {"c_puct": 1.2, "simulations": 120}, "train": {"lr": 0.0008}}
    m = db.add_member(pop["id"], 1, ms[1]["id"], ms[1]["name"], role="child",
                      parent_model_id=ms[0]["id"], hyperparams=hp)
    assert m["hyperparams"] == hp and m["role"] == "child" and m["parent_model_id"] == ms[0]["id"]
    assert (m["score"], m["loss"], m["rank"]) == (None, None, None)
    assert db.add_member(pop["id"], 0, ms[0]["id"], "a")["hyperparams"] == {}   # default


def test_member_rules(db):
    pop, ms = make_population(db)
    with pytest.raises(ValueError):
        db.add_member(pop["id"], 0, ms[0]["id"], "a", role="king")
    db.add_member(pop["id"], 0, ms[0]["id"], "a")
    with pytest.raises(sqlite3.IntegrityError):          # one row per agent per generation
        db.add_member(pop["id"], 0, ms[0]["id"], "a")
    db.add_member(pop["id"], 1, ms[0]["id"], "a", role="survivor")           # next generation: fine


def test_results_and_ranking_order(db):
    pop, ms = make_population(db, 4)
    rows = [db.add_member(pop["id"], 0, m["id"], m["name"]) for m in ms]
    db.set_member_result(rows[2]["id"], 0.70, loss=1.1, rank=1)
    db.set_member_result(rows[0]["id"], 0.40, loss=2.0, rank=2)
    # rows[1] and rows[3] are not ranked yet
    got = db.list_members(pop["id"], generation=0)
    assert [r["model_id"] for r in got] == [ms[2]["id"], ms[0]["id"], ms[1]["id"], ms[3]["id"]]
    assert (got[0]["score"], got[0]["loss"], got[0]["rank"]) == (0.70, 1.1, 1)


def test_list_members_by_generation(db):
    pop, ms = make_population(db, 2)
    db.add_member(pop["id"], 0, ms[0]["id"], "a")
    db.add_member(pop["id"], 1, ms[0]["id"], "a", role="survivor")
    db.add_member(pop["id"], 1, ms[1]["id"], "b", role="child", parent_model_id=ms[0]["id"])
    assert len(db.list_members(pop["id"])) == 3
    assert len(db.list_members(pop["id"], generation=1)) == 2
    assert db.list_members(pop["id"], generation=7) == []


def test_generation_summary_for_the_chart(db):
    pop, ms = make_population(db, 3)
    for g, scores in enumerate([[0.5, 0.3, None], [0.8, 0.6, 0.4]]):
        for m, sc in zip(ms, scores):
            row = db.add_member(pop["id"], g, m["id"], m["name"])
            if sc is not None:
                db.set_member_result(row["id"], sc)
    s = db.generation_summary(pop["id"])
    assert [r["generation"] for r in s] == [0, 1]
    assert (s[0]["agents"], s[0]["best_score"]) == (3, 0.5)
    assert s[0]["mean_score"] == pytest.approx(0.4)      # the unscored agent does not count as 0
    assert (s[1]["best_score"], s[1]["mean_score"]) == (0.8, pytest.approx(0.6))
    assert db.generation_summary(999) == []


def test_population_of_model(db):
    pop, ms = make_population(db, 2)
    assert db.population_of_model(ms[0]["id"]) is None
    db.add_member(pop["id"], 0, ms[0]["id"], "a")
    assert db.population_of_model(ms[0]["id"])["id"] == pop["id"]
    assert db.population_of_model(ms[1]["id"]) is None


# ------------------------------------------------- deleting things (the history)
def test_deleting_a_model_keeps_its_history(db):
    pop, ms = make_population(db, 3)
    db.add_member(pop["id"], 0, ms[0]["id"], "agent-0")
    db.add_member(pop["id"], 1, ms[1]["id"], "agent-1", role="child", parent_model_id=ms[0]["id"])
    db.update_population(pop["id"], champion_id=ms[0]["id"])
    db.delete_model(ms[0]["id"])                          # the parent, the champion and the base model
    rows = db.list_members(pop["id"])
    assert len(rows) == 2
    gone = next(r for r in rows if r["generation"] == 0)
    assert gone["model_id"] is None and gone["model_name"] == "agent-0"     # the name survives
    child = next(r for r in rows if r["generation"] == 1)
    assert child["parent_model_id"] is None and child["model_id"] == ms[1]["id"]
    p = db.get_population(pop["id"])
    assert p["champion_id"] is None and p["base_model_id"] is None


def test_deleting_a_population_keeps_the_models(db):
    pop, ms = make_population(db, 2)
    for m in ms:
        db.add_member(pop["id"], 0, m["id"], m["name"])
    db.delete_population(pop["id"])
    assert db.get_population(pop["id"]) is None and db.list_members(pop["id"]) == []
    assert len(db.list_models()) == 2
    assert db.population_of_model(ms[0]["id"]) is None


def test_the_existing_database_api_still_works(db):
    m = db.create_model("x", {}, "/tmp/x.pt")
    run = db.create_run(m["id"], {"a": 1})
    db.add_metric(run["id"], {"iteration": 1, "policy_loss": 1.0})
    assert db.list_metrics(run["id"])[0]["iteration"] == 1
    assert db.active_runs()[0]["id"] == run["id"]
    db.delete_model(m["id"])
    assert db.get_run(run["id"]) is None
