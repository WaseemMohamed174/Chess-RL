"""Phase 6: exploit — who stays, who is replaced, who copies whom."""
import pytest
from backend.rl.config import PopulationConfig
from backend.rl.population import rank_agents, select, parents_for


def ranked(n):
    return rank_agents([{"key": i, "score": 1 - i / (n + 1), "loss": 1.0} for i in range(n)])


def test_ten_agents_two_stay_eight_replaced():
    r = select(ranked(10), PopulationConfig().survivors)
    assert [a["key"] for a in r["survivors"]] == [0, 1]
    assert [a["key"] for a in r["replaced"]] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert len(r["pairs"]) == 8


def test_hundred_agents_25_and_75():
    r = select(ranked(100), PopulationConfig(size=100).survivors)
    assert (len(r["survivors"]), len(r["replaced"])) == (25, 75)


def test_input_order_does_not_matter():
    rs = ranked(6)
    a = select(rs, 2)
    b = select(list(reversed(rs)), 2)
    assert [x["key"] for x in a["survivors"]] == [x["key"] for x in b["survivors"]]


def test_champion_is_rank_one_and_survives():
    r = select(ranked(5), 2)
    assert r["champion"]["rank"] == 1 and r["champion"] in r["survivors"]


def test_unranked_agents_are_ignored():
    rs = ranked(4) + [{"key": "gone", "score": None, "loss": None, "rank": None}]
    r = select(rs, 1)
    assert "gone" not in [a["key"] for a in r["survivors"] + r["replaced"]]
    assert len(r["survivors"]) + len(r["replaced"]) == 4


def test_somebody_stays_and_somebody_is_replaced():
    assert len(select(ranked(3), 0)["survivors"]) == 1       # too few asked -> 1
    r = select(ranked(3), 99)                                # too many asked -> n-1
    assert len(r["survivors"]) == 2 and len(r["replaced"]) == 1


def test_single_agent_and_empty():
    r = select(ranked(1), 1)
    assert len(r["survivors"]) == 1 and r["replaced"] == [] and r["pairs"] == []
    assert select([], 2)["champion"] is None


def test_parents_are_shared_best_first():
    r = select(ranked(6), 2)                  # 2 stay, 4 replaced -> 2 each
    counts = {}
    for _, p in r["pairs"]:
        counts[p["key"]] = counts.get(p["key"], 0) + 1
    assert counts == {0: 2, 1: 2}
    r = select(ranked(6), 3)                  # 3 replaced, 3 survivors -> 1 each
    assert sorted(p["key"] for _, p in r["pairs"]) == [0, 1, 2]
    pairs = parents_for([{"k": "a"}, {"k": "b"}, {"k": "c"}], [{"k": "s1"}, {"k": "s2"}])
    assert [p["k"] for _, p in pairs] == ["s1", "s2", "s1"]       # extra goes to the best


def test_parents_for_no_survivors():
    assert parents_for([{"k": 1}], []) == []


def test_works_with_scored_members_and_ties():
    rs = rank_agents([{"key": "a", "score": .5, "loss": 2.0}, {"key": "b", "score": .5, "loss": 1.0},
                      {"key": "c", "score": .5, "loss": None}, {"key": "d", "score": .1, "loss": 0.1}])
    r = select(rs, 2)
    assert [a["key"] for a in r["survivors"]] == ["b", "a"]
    assert [a["key"] for a in r["replaced"]] == ["c", "d"]
