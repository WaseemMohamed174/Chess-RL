"""Population Based Training, phase 1: the `population` settings section and its limits."""
import time

import pytest

from backend import settings
from backend.rl.config import PopulationConfig


# ------------------------------------------------------------------ defaults
def test_defaults_match_the_plan():
    c = PopulationConfig()
    assert (c.size, c.survivor_fraction, c.iterations_per_generation) == (10, 0.25, 3)
    assert c.mutation_strength == 0.2 and c.copy_buffer is True
    assert (c.survivors, c.replaced) == (2, 8)          # 10 agents: the best 2 stay, 8 are new


@pytest.mark.parametrize("size,frac,survivors", [
    (10, 0.25, 2), (100, 0.25, 25), (100, 0.29, 29),    # 0.29 * 100 is 28.999... in floating point
    (2, 0.5, 1), (3, 0.25, 1), (10, 0.05, 1),           # never fewer than one survivor
    (4, 0.5, 2), (2, 0.05, 1),
])
def test_survivor_count(size, frac, survivors):
    c = PopulationConfig(size=size, survivor_fraction=frac)
    assert c.survivors == survivors
    assert c.survivors + c.replaced == size
    assert 1 <= c.survivors < size                      # and always someone to replace


def test_whole_numbers_are_whole():
    c = PopulationConfig.from_dict({"size": 12.0, "iterations_per_generation": 4.0})
    assert isinstance(c.size, int) and isinstance(c.iterations_per_generation, int)
    assert isinstance(c.replaced, int)


def test_from_dict_ignores_unknown_keys_and_roundtrips():
    c = PopulationConfig.from_dict({"size": 20, "nope": 1})
    assert c.size == 20
    assert PopulationConfig.from_dict(c.to_dict()) == c
    assert PopulationConfig.from_dict(None) == PopulationConfig()


# ---------------------------------------------------------- settings + LIMITS
def test_population_is_a_settings_section():
    d = settings.default_settings()
    assert d["population"] == PopulationConfig().to_dict()
    assert set(settings.LIMITS["population"]) <= set(d["population"])


def test_every_default_is_inside_its_limits():
    d = settings.default_settings()
    for section, limits in settings.LIMITS.items():
        for key, (lo, hi) in limits.items():
            assert lo <= d[section][key] <= hi, f"{section}.{key}"


def test_size_limit_goes_up_to_100():
    settings.validate_settings({"population": {"size": 100}})
    settings.validate_settings({"population": {"size": 2}})
    for bad in (101, 1, 0, -5):
        with pytest.raises(ValueError):
            settings.validate_settings({"population": {"size": bad}})


@pytest.mark.parametrize("bad", [
    {"survivor_fraction": 0.04}, {"survivor_fraction": 0.51}, {"survivor_fraction": 1},
    {"mutation_strength": -0.1}, {"mutation_strength": 0.51},
    {"iterations_per_generation": 0}, {"iterations_per_generation": 1001},
    {"size": "10"}, {"size": True}, {"copy_buffer": "yes"}, {"copy_buffer": 1},
    {"unknown": 1},
])
def test_bad_population_values_are_refused(bad):
    with pytest.raises(ValueError):
        settings.validate_settings({"population": bad})


def test_good_population_values_are_accepted():
    settings.validate_settings({"population": {
        "size": 16, "survivor_fraction": 0.5, "iterations_per_generation": 1,
        "mutation_strength": 0, "copy_buffer": False}})


def test_old_saved_settings_still_load():
    """A database written before the population section existed has no such key."""
    old = {"mcts": {"simulations": 200}, "train": {"lr": 0.002}, "ui": {"language": "ar"}}
    m = settings.merged_settings(old)
    assert m["population"] == PopulationConfig().to_dict()
    assert m["mcts"]["simulations"] == 200 and m["ui"]["language"] == "ar"
    m2 = settings.merged_settings({**old, "population": {"size": 30, "bogus": 1}})
    assert m2["population"]["size"] == 30 and "bogus" not in m2["population"]
    assert m2["population"]["survivor_fraction"] == 0.25


# ----------------------------------------------------------- through the API
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
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
        yield c


def test_api_serves_and_saves_the_population_section(client):
    r = client.get("/api/settings").json()
    assert r["values"]["population"]["size"] == 10
    assert r["limits"]["population"]["size"] == [2, 100]
    assert r["defaults"]["population"] == PopulationConfig().to_dict()
    assert client.put("/api/settings", json={"population": {"size": 101}}).status_code == 422
    v = client.put("/api/settings", json={"population": {"size": 40, "copy_buffer": False}}).json()
    assert v["values"]["population"]["size"] == 40
    got = client.get("/api/settings").json()["values"]["population"]
    assert got["size"] == 40 and got["copy_buffer"] is False and got["survivor_fraction"] == 0.25
    client.delete("/api/settings")                       # back to defaults for the other tests
    assert client.get("/api/settings").json()["values"]["population"]["size"] == 10
