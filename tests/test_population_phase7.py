"""Phase 7: explore — children of the survivors, with nudged settings."""
import asyncio
import os
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException

from backend import paths, settings
from backend.db import Database
from backend.rl.config import explorable_values
from backend.rl.population import mutate_hyperparams

LIM = settings.LIMITS
START = explorable_values(settings.default_settings())


class Fixed:                       # a "random" that always answers the same
    def __init__(self, v): self.v = v
    def random(self): return self.v


# ------------------------------------------------------------------ mutation (pure)
def test_values_are_multiplied_by_08_or_12():
    up = mutate_hyperparams({"train": {"lr": 0.001}}, 0.2, LIM, Fixed(0.1))
    down = mutate_hyperparams({"train": {"lr": 0.001}}, 0.2, LIM, Fixed(0.9))
    assert up["train"]["lr"] == pytest.approx(0.0012) and down["train"]["lr"] == pytest.approx(0.0008)


def test_strength_zero_is_an_exact_copy():
    assert mutate_hyperparams(START, 0, LIM, random.Random(1)) == START


def test_input_is_not_changed_and_shape_is_kept():
    before = {s: dict(v) for s, v in START.items()}
    out = mutate_hyperparams(START, 0.2, LIM, random.Random(1))
    assert START == before and out is not START
    assert {s: set(v) for s, v in out.items()} == {s: set(v) for s, v in START.items()}


def test_always_inside_the_limits_even_after_many_generations():
    rng = random.Random(7)
    hp = START
    for _ in range(300):
        hp = mutate_hyperparams(hp, 0.5, LIM, rng)
        for sec, vals in hp.items():
            for k, v in vals.items():
                lo, hi = LIM[sec][k]
                assert lo <= v <= hi, (sec, k, v)


def test_clamped_at_both_ends():
    hi = mutate_hyperparams({"mcts": {"simulations": 800, "c_puct": 10}}, 0.5, LIM, Fixed(0.1))
    lo = mutate_hyperparams({"mcts": {"simulations": 50, "c_puct": 0.1}}, 0.5, LIM, Fixed(0.9))
    assert hi["mcts"] == {"simulations": 800, "c_puct": 10}
    assert lo["mcts"] == {"simulations": 50, "c_puct": 0.1}


def test_simulations_stays_a_whole_number():
    out = mutate_hyperparams({"mcts": {"simulations": 101}}, 0.2, LIM, Fixed(0.1))
    assert out["mcts"]["simulations"] == 121 and type(out["mcts"]["simulations"]) is int


def test_both_directions_happen():
    rng = random.Random(3)
    seen = {round(mutate_hyperparams({"train": {"lr": 0.01}}, 0.2, LIM, rng)["train"]["lr"], 6)
            for _ in range(40)}
    assert seen == {0.008, 0.012}


def test_a_value_saved_as_a_whole_number_does_not_collapse_to_zero():
    out = mutate_hyperparams({"mcts": {"c_puct": 1}}, 0.2, LIM, Fixed(0.9))
    assert out["mcts"]["c_puct"] == pytest.approx(0.8)


# ------------------------------------------------------------------ next generation
@pytest.fixture
def env(tmp_path, monkeypatch):
    from backend.app import services
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", ckpt)
    db = Database(tmp_path / "t.db")
    monkeypatch.setattr(services, "S", SimpleNamespace(db=db, play=None))
    path = ckpt / "base.pt"
    path.write_bytes(b"base" * 100)
    base = db.create_model("Base", {"blocks": 3, "filters": 64}, str(path), elo=1234.0)
    yield SimpleNamespace(services=services, db=db, base=base, ckpt=ckpt)
    db.close()


def make(env, **ov):
    out = asyncio.run(env.services.create_population("Pop", env.base, ov or None))
    pid = out["population"]["id"]
    # every agent gets its own weights, its own buffer and its own iteration count
    for i, mem in enumerate(out["members"]):
        m = env.db.get_model(mem["model_id"])
        open(m["path"], "wb").write(f"weights-{i}".encode())
        open(m["path"] + ".buffer.npz", "wb").write(f"buffer-{i}".encode())
        env.db.update_model(m["id"], iterations=10 + i, elo=1000.0 + i)
    return pid, out["members"]


def score(env, pid, order=None):
    """Rank the generation-0 agents: member i gets rank i+1 (best first) unless `order`."""
    mems = env.db.list_members(pid, 0)
    for rank, mem in enumerate(order or mems, start=1):
        env.db.set_member_result(mem["id"], 1 - rank / 100, 1.0, rank)


def run(env, pid, gen=0, seed=1):
    return asyncio.run(env.services.make_next_generation(pid, gen, rng=random.Random(seed)))


def test_ten_agents_make_two_survivors_and_eight_children(env):
    pid, _ = make(env)
    score(env, pid)
    out = run(env, pid)
    assert (out["generation"], out["survivors"], out["children"]) == (1, 2, 8)
    g1 = env.db.list_members(pid, 1)
    assert len(g1) == 10
    assert [m["role"] for m in g1].count("survivor") == 2 and [m["role"] for m in g1].count("child") == 8


def test_survivors_are_the_best_and_unchanged(env):
    pid, g0 = make(env)
    score(env, pid)
    run(env, pid)
    surv = [m for m in env.db.list_members(pid, 1) if m["role"] == "survivor"]
    assert {m["model_id"] for m in surv} == {g0[0]["model_id"], g0[1]["model_id"]}
    for m in surv:
        old = next(o for o in g0 if o["model_id"] == m["model_id"])
        assert m["hyperparams"] == old["hyperparams"]


def test_child_copies_weights_buffer_iterations_and_elo_of_its_parent(env):
    pid, g0 = make(env)
    score(env, pid)
    run(env, pid)
    for c in (m for m in env.db.list_members(pid, 1) if m["role"] == "child"):
        child, par = env.db.get_model(c["model_id"]), env.db.get_model(c["parent_model_id"])
        assert c["parent_model_id"] in (g0[0]["model_id"], g0[1]["model_id"])
        assert child["parent_id"] == par["id"] and child["path"] != par["path"]
        assert open(child["path"], "rb").read() == open(par["path"], "rb").read()
        assert open(child["path"] + ".buffer.npz", "rb").read() == open(par["path"] + ".buffer.npz", "rb").read()
        assert (child["iterations"], child["elo"]) == (par["iterations"], par["elo"])


def test_children_are_shared_four_and_four(env):
    pid, g0 = make(env)
    score(env, pid)
    run(env, pid)
    parents = [m["parent_model_id"] for m in env.db.list_members(pid, 1) if m["role"] == "child"]
    assert sorted((parents.count(g0[0]["model_id"]), parents.count(g0[1]["model_id"]))) == [4, 4]


def test_no_buffer_when_copy_buffer_is_off(env):
    pid, _ = make(env, copy_buffer=False)
    score(env, pid)
    run(env, pid)
    for c in (m for m in env.db.list_members(pid, 1) if m["role"] == "child"):
        assert not os.path.exists(env.db.get_model(c["model_id"])["path"] + ".buffer.npz")


def test_children_settings_are_nudged_inside_limits(env):
    pid, g0 = make(env)
    score(env, pid)
    run(env, pid)
    kids = [m for m in env.db.list_members(pid, 1) if m["role"] == "child"]
    assert any(k["hyperparams"] != g0[0]["hyperparams"] for k in kids)
    for k in kids:
        for sec, vals in k["hyperparams"].items():
            for name, v in vals.items():
                assert LIM[sec][name][0] <= v <= LIM[sec][name][1]


def test_strength_zero_gives_identical_settings(env):
    pid, g0 = make(env, mutation_strength=0)
    score(env, pid)
    run(env, pid)
    for k in (m for m in env.db.list_members(pid, 1) if m["role"] == "child"):
        assert k["hyperparams"] == g0[0]["hyperparams"]


def test_old_generation_is_untouched(env):
    pid, _ = make(env)
    score(env, pid)
    before = env.db.list_members(pid, 0)
    models = len(env.db.list_models())
    run(env, pid)
    assert env.db.list_members(pid, 0) == before
    assert len(env.db.list_models()) == models + 8            # nothing deleted, 8 added
    assert env.db.get_population(pid)["generation"] == 0       # the coordinator moves it


def test_same_seed_same_result(env):
    pid, _ = make(env)
    score(env, pid)
    a = run(env, pid, seed=5)
    hp_a = [m["hyperparams"] for m in a["members"]]
    env.db.delete_members_of_generation(pid, 1)
    for m in a["members"]:
        if m["role"] == "child":
            env.services.delete_model(env.db.get_model(m["model_id"]))
    b = run(env, pid, seed=5)
    assert [m["hyperparams"] for m in b["members"]] == hp_a


def test_generation_must_be_scored_first(env):
    pid, _ = make(env)
    with pytest.raises(HTTPException) as e:
        run(env, pid)
    assert e.value.status_code == 409 and len(env.db.list_models()) == 11


def test_cannot_make_the_same_generation_twice(env):
    pid, _ = make(env)
    score(env, pid)
    run(env, pid)
    n = len(env.db.list_models())
    with pytest.raises(HTTPException) as e:
        run(env, pid)
    assert e.value.status_code == 409 and len(env.db.list_models()) == n


def test_a_deleted_agent_is_replaced_so_the_size_stays(env):
    pid, g0 = make(env)
    score(env, pid)
    env.services.delete_model(env.db.get_model(g0[9]["model_id"]))     # the weakest is deleted
    out = run(env, pid)
    assert len(env.db.list_members(pid, 1)) == 10 and out["children"] == 8


def test_missing_survivor_file_stops_before_anything_is_made(env):
    pid, g0 = make(env)
    score(env, pid)
    os.remove(env.db.get_model(g0[0]["model_id"])["path"])
    models = len(env.db.list_models())
    with pytest.raises(HTTPException) as e:
        run(env, pid)
    assert e.value.status_code == 409
    assert len(env.db.list_models()) == models and env.db.list_members(pid, 1) == []


def test_failure_in_the_middle_removes_the_children_made_so_far(env, monkeypatch):
    pid, _ = make(env)
    score(env, pid)
    models, files = len(env.db.list_models()), sorted(os.listdir(env.ckpt))
    real, calls = env.services.create_model, {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 4:
            raise HTTPException(500, "boom")
        return await real(*a, **k)

    monkeypatch.setattr(env.services, "create_model", flaky)
    with pytest.raises(HTTPException):
        run(env, pid)
    assert len(env.db.list_models()) == models and sorted(os.listdir(env.ckpt)) == files
    assert env.db.list_members(pid, 1) == []


def test_works_for_the_next_round_too(env):
    pid, _ = make(env)
    score(env, pid)
    run(env, pid)
    for rank, mem in enumerate(env.db.list_members(pid, 1), start=1):
        env.db.set_member_result(mem["id"], 1 - rank / 100, 1.0, rank)
    out = run(env, pid, gen=1)
    assert out["generation"] == 2 and len(env.db.list_members(pid, 2)) == 10
    assert len({m["model_name"] for m in env.db.list_members(pid, 2)}) == 10   # no name twice
