"""Population Based Training, phase 3: create_population makes N agents from one model."""
import asyncio
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException

from backend import paths, settings
from backend.db import Database
from backend.rl.config import EXPLORABLE, PopulationConfig, explorable_values

BASE_BYTES = b"weights-of-the-base-model" * 50


@pytest.fixture
def env(tmp_path, monkeypatch):
    from backend.app import services
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", ckpt)
    db = Database(tmp_path / "t.db")
    monkeypatch.setattr(services, "S", SimpleNamespace(db=db, play=None))
    path = ckpt / "base.pt"
    path.write_bytes(BASE_BYTES)
    base = db.create_model("Base", {"blocks": 3, "filters": 64}, str(path), elo=1234.0)
    yield SimpleNamespace(services=services, db=db, base=base, ckpt=ckpt)
    db.close()


def make(env, name="Alpha", **overrides):
    return asyncio.run(env.services.create_population(name, env.base, overrides or None))


def files(env):
    return sorted(p.name for p in env.ckpt.iterdir())


# --------------------------------------------------------------------- the agents
def test_creates_the_default_ten_agents(env):
    out = make(env)
    pop, members = out["population"], out["members"]
    assert pop["name"] == "Alpha" and pop["generation"] == 0 and pop["base_model_id"] == env.base["id"]
    assert pop["config"] == PopulationConfig().to_dict()
    assert len(members) == 10 and len(env.db.list_models()) == 11          # the base + 10 agents
    names = [m["model_name"] for m in members]
    assert names == [f"Alpha A{i:02d}" for i in range(1, 11)]
    for mem in members:
        m = env.db.get_model(mem["model_id"])
        assert m["parent_id"] == env.base["id"] and m["elo"] == 1234.0
        assert m["config"] == env.base["config"]
        assert (mem["generation"], mem["role"]) == (0, "seed")
        assert mem["parent_model_id"] == env.base["id"]


def test_every_agent_has_its_own_weights_file(env):
    members = make(env, size=5)["members"]
    paths_ = [env.db.get_model(m["model_id"])["path"] for m in members]
    assert len(set(paths_)) == 5 and env.base["path"] not in paths_
    for p in paths_:
        assert open(p, "rb").read() == BASE_BYTES
    open(paths_[0], "wb").write(b"changed")                 # training one agent must not touch the rest
    assert open(paths_[1], "rb").read() == BASE_BYTES
    assert open(env.base["path"], "rb").read() == BASE_BYTES


def test_seeds_record_the_explorable_settings(env):
    members = make(env, size=3)["members"]
    want = explorable_values(settings.default_settings())
    assert all(m["hyperparams"] == want for m in members)
    assert want["train"] == {"lr": 1e-3} and set(want["mcts"]) == set(EXPLORABLE["mcts"])


def test_saved_user_settings_are_the_starting_point(env):
    env.db.set_settings({"population": {"size": 6}, "mcts": {"c_puct": 2.5}, "train": {"lr": 0.005}})
    out = make(env)
    assert len(out["members"]) == 6                          # saved size
    assert out["members"][0]["hyperparams"]["mcts"]["c_puct"] == 2.5
    assert out["members"][0]["hyperparams"]["train"]["lr"] == 0.005
    assert len(make(env, "Beta", size=3)["members"]) == 3    # an override beats the saved value


def test_a_hundred_agents_get_three_digit_names(env):
    names = [m["model_name"] for m in make(env, size=100)["members"]]
    assert names[0] == "Alpha A001" and names[-1] == "Alpha A100" and len(set(names)) == 100


# ----------------------------------------------------------------- replay buffers
def test_buffers_are_copied_when_the_setting_is_on(env):
    open(env.base["path"] + ".buffer.npz", "wb").write(b"positions" * 20)
    members = make(env, size=3)["members"]                   # copy_buffer is on by default
    for mem in members:
        p = env.db.get_model(mem["model_id"])["path"] + ".buffer.npz"
        assert open(p, "rb").read() == b"positions" * 20
    assert not [f for f in files(env) if f.endswith(".tmp")]


def test_buffers_are_left_out_when_the_setting_is_off(env):
    open(env.base["path"] + ".buffer.npz", "wb").write(b"positions")
    make(env, size=3, copy_buffer=False)
    assert sum(f.endswith(".buffer.npz") for f in files(env)) == 1          # only the base's own


def test_a_base_without_a_buffer_is_fine(env):
    make(env, size=2)
    assert not any(f.endswith(".buffer.npz") for f in files(env))


# ------------------------------------------------------------------- refusals
def status(env, name="Alpha", **overrides):
    with pytest.raises(HTTPException) as e:
        make(env, name, **overrides)
    return e.value.status_code


def test_bad_size_and_bad_names_are_refused(env):
    assert status(env, size=101) == 422 and status(env, size=1) == 422
    assert status(env, size="ten") == 422 and status(env, bogus=1) == 422
    assert status(env, name="   ") == 400 and status(env, name="x" * 51) == 400
    assert env.db.list_populations() == [] and len(env.db.list_models()) == 1
    assert files(env) == ["base.pt"]


def test_a_missing_base_file_is_refused(env):
    os.remove(env.base["path"])
    assert status(env) == 409 and env.db.list_populations() == []


def test_a_taken_population_name_is_refused(env):
    make(env, size=2)
    assert status(env, size=2) == 409
    assert len(env.db.list_populations()) == 1 and len(env.db.list_models()) == 3


def test_a_clashing_agent_name_is_refused_before_anything_is_made(env, monkeypatch):
    env.db.create_model("Alpha A03", {}, "/tmp/x.pt")        # someone already owns that name
    calls = []

    async def spy(*a, **kw):
        calls.append(1)
        raise AssertionError("an agent was made although a name was already taken")

    monkeypatch.setattr(env.services, "create_model", spy)   # no file may even be copied
    assert status(env, size=5) == 409 and calls == []
    assert env.db.list_populations() == [] and len(env.db.list_models()) == 2
    assert files(env) == ["base.pt"]


# ------------------------------------------------------------ all or nothing
def test_a_failure_halfway_removes_everything(env, monkeypatch):
    open(env.base["path"] + ".buffer.npz", "wb").write(b"positions")
    real, calls = env.services.create_model, []

    async def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("disk full")
        return await real(*a, **kw)

    monkeypatch.setattr(env.services, "create_model", flaky)
    with pytest.raises(RuntimeError):
        make(env, size=6)
    assert len(calls) == 4
    assert env.db.list_populations() == [] and len(env.db.list_models()) == 1
    assert sorted(files(env)) == ["base.pt", "base.pt.buffer.npz"]          # no orphan files
    monkeypatch.setattr(env.services, "create_model", real)
    assert len(make(env, size=2)["members"]) == 2                           # and it still works after


# ------------------------------------------------------------- what explore may touch
def test_explorable_settings_are_real_and_have_limits():
    d = settings.default_settings()
    for section, keys in EXPLORABLE.items():
        for k in keys:
            assert k in d[section] and k in settings.LIMITS[section]
