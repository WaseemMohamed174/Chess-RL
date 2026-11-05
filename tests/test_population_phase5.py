"""Population Based Training, phase 5: every agent plays a fixed champion, gets a score,
and the agents are ranked (ties go to the lower loss)."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chess")
from fastapi import HTTPException

from backend import paths
from backend.db import Database
from backend.rl.population import rank_agents, total_loss


# ------------------------------------------------------------- the ranking on its own
def R(score, loss, key=None):
    return {"key": key, "score": score, "loss": loss}


def keys(ranked):
    return [r["key"] for r in ranked]


def test_higher_score_ranks_first():
    out = rank_agents([R(0.3, 1.0, "a"), R(0.9, 5.0, "b"), R(0.6, 0.1, "c")])
    assert keys(out) == ["b", "c", "a"]
    assert [r["rank"] for r in out] == [1, 2, 3]


def test_same_score_lower_loss_wins():
    out = rank_agents([R(0.5, 3.0, "a"), R(0.5, 1.0, "b"), R(0.5, 2.0, "c")])
    assert keys(out) == ["b", "c", "a"]


def test_score_beats_loss():
    out = rank_agents([R(0.5, 0.1, "low-loss"), R(0.55, 9.0, "high-score")])
    assert keys(out) == ["high-score", "low-loss"]


def test_no_loss_comes_after_any_loss_among_equals():
    out = rank_agents([R(0.5, None, "none"), R(0.5, 99.0, "big")])
    assert keys(out) == ["big", "none"]
    out = rank_agents([R(0.7, None, "none"), R(0.5, 0.1, "x")])        # but score still wins
    assert keys(out) == ["none", "x"]


def test_full_tie_keeps_input_order_and_ranks_have_no_gaps():
    out = rank_agents([R(0.5, 1.0, "a"), R(0.5, 1.0, "b"), R(0.5, 1.0, "c")])
    assert keys(out) == ["a", "b", "c"] and [r["rank"] for r in out] == [1, 2, 3]
    assert rank_agents([]) == []


def test_total_loss():
    assert total_loss(2.0, 0.5) == 2.5
    assert total_loss(None, 0.5) == 0.5 and total_loss(2.0, None) == 2.0
    assert total_loss(None, None) is None


# ----------------------------------------------------------------- with the database
@pytest.fixture
def env(tmp_path, monkeypatch):
    from backend.app import services
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", ckpt)
    db = Database(tmp_path / "t.db")
    monkeypatch.setattr(services, "S", SimpleNamespace(db=db, play=None))

    def model(name):
        p = ckpt / f"{name}.pt"
        p.write_bytes(b"w")
        return db.create_model(name, {"blocks": 3, "filters": 64}, str(p))

    champ = model("Champ")
    pop = db.create_population("P", {}, base_model_id=champ["id"])
    yield SimpleNamespace(services=services, db=db, model=model, champ=champ, pop=pop, ckpt=ckpt)
    db.close()


def add_agents(env, names, generation=0):
    out = []
    for n in names:
        m = env.model(n)
        out.append((m, env.db.add_member(env.pop["id"], generation, m["id"], m["name"])))
    return out


def train_with_loss(env, model, policy, value):
    run = env.db.create_run(model["id"], {})
    env.db.add_metric(run["id"], {"iteration": 1, "policy_loss": 99.0, "value_loss": 99.0})
    env.db.add_metric(run["id"], {"iteration": 2, "policy_loss": policy, "value_loss": value})


def fake_play(points, log=None):
    """points: {path-name: score of that agent in every game}."""
    async def play(agent_path, champ_path, agent_white, cfg):
        await asyncio.sleep(0)
        name = agent_path.rsplit("/", 1)[-1][:-3]
        if log is not None:
            log.append((name, agent_white))
        return points[name]
    return play


def score(env, **kw):
    kw.setdefault("games", 4)
    return asyncio.run(env.services.score_generation(env.pop["id"], 0, env.champ, **kw))


def test_scores_ranks_and_saves_every_agent(env):
    add_agents(env, ["A1", "A2", "A3"])
    out = score(env, play=fake_play({"A1": 0.0, "A2": 1.0, "A3": 0.5}))
    assert [m["model_name"] for m in out] == ["A2", "A3", "A1"]        # best first
    assert [m["rank"] for m in out] == [1, 2, 3]
    assert [m["score"] for m in out] == [1.0, 0.5, 0.0]
    saved = env.db.list_members(env.pop["id"], 0)                       # really in the database
    assert [(m["model_name"], m["rank"]) for m in saved] == [("A2", 1), ("A3", 2), ("A1", 3)]


def test_every_agent_plays_all_games_and_colours_alternate(env):
    add_agents(env, ["A1", "A2"])
    log = []
    score(env, games=6, play=fake_play({"A1": 0.5, "A2": 0.5}, log))
    for name in ("A1", "A2"):
        assert sorted(w for n, w in log if n == name) == [False] * 3 + [True] * 3
    assert len(log) == 12


def test_score_is_the_share_of_points(env):
    add_agents(env, ["A1"])
    out = score(env, games=5, play=fake_play({"A1": 0.7}))
    assert out[0]["score"] == pytest.approx(0.7)


def test_tie_is_broken_by_the_lower_loss(env):
    (m1, _), (m2, _), (m3, _) = add_agents(env, ["A1", "A2", "A3"])
    train_with_loss(env, m1, 3.0, 1.0)                                 # 4.0
    train_with_loss(env, m2, 1.0, 0.5)                                 # 1.5  -> wins the tie
    out = score(env, play=fake_play({"A1": 0.5, "A2": 0.5, "A3": 0.5}))
    assert [m["model_name"] for m in out] == ["A2", "A1", "A3"]        # A3 never trained: last
    assert [m["loss"] for m in out] == [1.5, 4.0, None]


def test_latest_loss_uses_the_newest_metric_of_the_newest_run(env):
    m = env.model("X")
    assert env.services.latest_loss(m["id"]) is None
    train_with_loss(env, m, 5.0, 1.0)
    assert env.services.latest_loss(m["id"]) == 6.0
    train_with_loss(env, m, 2.0, 0.25)                                 # a later run
    assert env.services.latest_loss(m["id"]) == 2.25


def test_only_the_asked_generation_is_scored(env):
    add_agents(env, ["G0"], generation=0)
    add_agents(env, ["G1"], generation=1)
    score(env, play=fake_play({"G0": 1.0}))
    g1 = env.db.list_members(env.pop["id"], 1)[0]
    assert g1["score"] is None and g1["rank"] is None


def test_agent_that_is_the_champion_gets_half_without_playing(env):
    env.db.add_member(env.pop["id"], 0, env.champ["id"], "Champ")
    log = []
    out = score(env, play=fake_play({}, log))
    assert log == [] and out[0]["score"] == 0.5 and out[0]["rank"] == 1


def test_deleted_agent_is_skipped_and_the_rest_are_ranked(env):
    (m1, _), (m2, _) = add_agents(env, ["A1", "A2"])
    env.db.delete_model(m1["id"])                                      # row stays, model_id None
    out = score(env, play=fake_play({"A2": 0.5}))
    ranked = [m for m in out if m["rank"] is not None]
    assert [m["model_name"] for m in ranked] == ["A2"] and ranked[0]["rank"] == 1
    gone = next(m for m in out if m["model_name"] == "A1")
    assert gone["score"] is None and gone["rank"] is None


def test_missing_weights_file_stops_before_any_game(env):
    (m1, _), _ = add_agents(env, ["A1", "A2"])
    (env.ckpt / "A1.pt").unlink()
    log = []
    with pytest.raises(HTTPException) as e:
        score(env, play=fake_play({"A1": 1.0, "A2": 1.0}, log))
    assert e.value.status_code == 409 and log == []
    (env.ckpt / "Champ.pt").unlink()
    with pytest.raises(HTTPException) as e:
        score(env, play=fake_play({}, log))
    assert e.value.status_code == 409


def test_a_failing_game_leaves_the_rows_untouched(env):
    add_agents(env, ["A1", "A2"])

    async def broken(agent_path, champ_path, white, cfg):
        if "A2" in agent_path:
            raise RuntimeError("engine died")
        await asyncio.sleep(0)
        return 1.0
    with pytest.raises(RuntimeError):
        score(env, play=broken)
    assert all(m["score"] is None and m["rank"] is None
               for m in env.db.list_members(env.pop["id"], 0))


def test_no_game_goes_on_in_the_background_after_a_failure(env):
    add_agents(env, ["A1", "A2"])
    finished = []

    async def play(agent_path, champ_path, white, cfg):
        if "A1" in agent_path:
            raise RuntimeError("engine died")
        await asyncio.sleep(0.2)
        finished.append(agent_path)                  # only reached if nobody cancelled us
        return 1.0

    async def run_and_wait():
        with pytest.raises(RuntimeError):
            await env.services.score_generation(env.pop["id"], 0, env.champ, games=2, play=play)
        await asyncio.sleep(0.5)                     # long enough for a stray game to end

    asyncio.run(run_and_wait())
    assert finished == []


def test_empty_generation_and_bad_input(env):
    assert score(env, play=fake_play({})) == []
    with pytest.raises(HTTPException) as e:
        score(env, games=0, play=fake_play({}))
    assert e.value.status_code == 422
    with pytest.raises(HTTPException) as e:
        asyncio.run(env.services.score_generation(9999, 0, env.champ, games=2))
    assert e.value.status_code == 404


def test_progress_and_concurrency_limit(env):
    add_agents(env, ["A1", "A2", "A3"])
    seen, running, peak = [], 0, 0

    async def play(agent_path, champ_path, white, cfg):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return 0.5
    score(env, games=4, concurrent=2, play=play, on_progress=lambda d, t: seen.append((d, t)))
    assert peak == 2 and seen[-1] == (12, 12) and [d for d, _ in seen] == list(range(1, 13))


def test_settings_defaults_are_used_when_games_not_given(env):
    add_agents(env, ["A1"])
    env.db.set_settings({"train": {"arena_games": 3, "arena_simulations": 7}})
    log, cfgs = [], []

    async def play(agent_path, champ_path, white, cfg):
        log.append(white)
        cfgs.append(cfg)
        return 0.5
    asyncio.run(env.services.score_generation(env.pop["id"], 0, env.champ, play=play))
    assert log == [True, False, True]                # 3 games from the saved settings
    assert cfgs[0]["simulations"] == 7
    assert cfgs[0]["dirichlet_eps"] == 0.0           # no root noise, like every arena


# ------------------------------------------------------------- one real game of chess
WIN_WHITE = ["e2e4", "f1c4", "d1h5", "h5f7"]        # scholar's mate
LOSE_BLACK = ["a7a6", "b7b6", "c7c6"]
LOSE_WHITE = ["f2f3", "g2g4"]                        # fool's mate
WIN_BLACK = ["e7e5", "d8h4"]


async def scripted(fen, path, cfg, ply=0):
    white_to_move = ply % 2 == 0
    k = ply // 2
    winner = "win" in path
    script = ((WIN_WHITE if white_to_move else WIN_BLACK) if winner
              else (LOSE_WHITE if white_to_move else LOSE_BLACK))
    return {"move": script[k]}


def one_game(agent, champ, agent_white):
    from backend.app import services
    return asyncio.run(services._play_vs_champion(agent, champ, agent_white,
                                                  {"max_plies": 400}, search=scripted))


def test_real_game_scores_from_the_agents_point_of_view(env):
    assert one_game("win", "lose", True) == 1.0          # agent mates as White
    assert one_game("win", "lose", False) == 1.0         # agent mates as Black
    assert one_game("lose", "win", True) == 0.0
    assert one_game("lose", "win", False) == 0.0


def test_real_game_draw_by_max_plies_and_illegal_move(env):
    from backend.app import services

    async def shuffle(fen, path, cfg, ply=0):
        return {"move": ["g1f3", "g8f6", "f3g1", "f6g8"][ply % 4]}
    # threefold repetition ends it long before 400 plies
    assert asyncio.run(services._play_vs_champion("a", "b", True, {"max_plies": 400},
                                                  search=shuffle)) == 0.5

    async def cheat(fen, path, cfg, ply=0):
        return {"move": "e2e5"}
    with pytest.raises(HTTPException) as e:
        asyncio.run(services._play_vs_champion("a", "b", True, {"max_plies": 400}, search=cheat))
    assert e.value.status_code == 502
