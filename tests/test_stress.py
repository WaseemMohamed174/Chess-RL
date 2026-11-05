"""Phase 10: stress with FakeBackend (real chess, no torch) - several runs, one server,
a server restart in the middle, and stopping while the server is away."""
import time

import pytest

pytest.importorskip("chess")

from backend.inference import make_hub, start_thread
from backend.inference.sources import UniformSource

MCTS = {"simulations": 8, "max_plies": 30}


def _payload(tmp_path, rid, games=4):
    return {"run_id": rid, "model_path": str(tmp_path / f"m{rid}.pt"), "elo": 1000, "iterations": 0,
            "mcts": MCTS,
            "train": {"games_per_iteration": games, "min_buffer": 1, "arena_games": 2,
                      "arena_simulations": 8, "arena_threshold": 0.5, "concurrent_games": 2,
                      "buffer_size": 1000}}


def _start(tmp_path, rid, hub, workers=1):
    from backend.rl.config import TrainConfig
    from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator
    p = _payload(tmp_path, rid)
    ev = EventLog()
    o = Orchestrator(FakeBackend(p["model_path"], "cpu", TrainConfig()), p, ev,
                     remote={"spec": hub.client_spec(), "workers": workers})
    o.start()
    return o, ev


def _metrics(ev):
    return [e for e in ev.since(0)["events"] if e["type"] == "metrics"]


def _wait(cond, secs):
    t0 = time.time()
    while time.time() - t0 < secs:
        if cond():
            return True
        time.sleep(0.2)
    return False


def _stop_all(runs):
    for o, _ in runs:
        o.stop()
    for o, _ in runs:
        o._thread.join(timeout=60)


def test_three_runs_share_one_server_and_all_finish_an_iteration(tmp_path):
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    runs = [_start(tmp_path, i, hub) for i in range(3)]
    try:
        assert _wait(lambda: all(_metrics(ev) for _, ev in runs), 240), [o.error for o, _ in runs]
        assert all(o.error is None for o, _ in runs)
        assert all(m[0]["games"] == 4 for _, ev in runs for m in [_metrics(ev)])
        assert srv.server.stats()["calls"] > 0
    finally:
        _stop_all(runs)
        srv.stop()
    assert all(o.state == "stopped" for o, _ in runs)


def test_runs_survive_the_server_being_restarted_mid_run(tmp_path):
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    runs = [_start(tmp_path, i, hub) for i in range(2)]
    holder = []
    try:
        assert _wait(lambda: srv.server.stats()["positions"] > 0, 60)
        srv.stop()                                         # the server dies while games are running
        time.sleep(1.5)
        holder.append(start_thread(hub, UniformSource()))  # the supervisor would do this
        assert _wait(lambda: all(_metrics(ev) for _, ev in runs), 240), [o.error for o, _ in runs]
        assert all(o.error is None for o, _ in runs)
    finally:
        _stop_all(runs)
        for h in holder:
            h.stop()


def test_stop_returns_even_if_the_server_is_gone(tmp_path):
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    runs = [_start(tmp_path, 0, hub)]
    try:
        assert _wait(lambda: srv.server.stats()["positions"] > 0, 60)
        srv.stop()                                         # and never comes back
        t0 = time.time()
        runs[0][0].stop()
        runs[0][0]._thread.join(timeout=90)
        assert not runs[0][0]._thread.is_alive(), "stop() must not hang on a dead server"
        assert time.time() - t0 < 90
    finally:
        _stop_all(runs)
