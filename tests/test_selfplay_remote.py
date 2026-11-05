"""Phases 5-6: capacity planning, supervisor-managed inference server, sock transport
reconnects, and self-play in separate worker processes."""
import os
import signal
import threading
import time

import numpy as np
import pytest

pytest.importorskip("chess")
pytest.importorskip("fastapi")

from backend.inference import InferenceError, RemoteEvaluator, ServerConfig, make_hub, start_thread
from backend.inference.sources import DebugSource, UniformSource


def _x(n=3, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=(n, 18, 8, 8), dtype=np.uint8)


# ------------------------------------------------------------------ capacity
def test_capacity_plan_is_sane():
    from backend.capacity import plan, workers_for_run
    one = plan(1, 4, "cpu")
    assert one["selfplay_workers"] == 1 and one["max_parallel_runs"] >= 1 and not one["inference_server"]
    big = plan(16, 32, "cpu")
    assert big["selfplay_workers"] == 14 and big["inference_server"]
    assert plan(8, 16, "cpu")["max_parallel_runs"] > plan(8, 5, "cpu")["max_parallel_runs"]
    tight = plan(8, 16, "gpu", safe_vram=3.0, blocks=10, filters=192)
    roomy = plan(8, 16, "gpu", safe_vram=24.0, blocks=10, filters=192)
    assert 1 <= tight["max_parallel_runs"] < roomy["max_parallel_runs"]
    assert plan(8, 0.1, "cpu")["max_parallel_runs"] == 1
    assert workers_for_run(6, 1) == 6 and workers_for_run(6, 2) == 3 and workers_for_run(1, 4) == 1


def test_old_selection_file_gets_capacity_keys(tmp_path, monkeypatch):
    from backend import paths, settings
    f = tmp_path / "selection.json"
    f.write_text('{"version": 1, "device": "cpu", "network": {"blocks": 3, "filters": 64}}')
    monkeypatch.setattr(paths, "SELECTION_FILE", f)
    sel = settings.load_selection()
    assert sel["max_parallel_runs"] >= 1 and sel["selfplay_workers"] >= 1 and "inference_server" in sel


# ------------------------------------------------------------------ sock transport
def test_client_survives_a_server_restart():
    hub = make_hub("sock", 0, max_n=64)
    s1 = start_thread(hub, UniformSource())
    ev = RemoteEvaluator(hub.client_spec(), "m", timeout=5)
    ev.evaluate(_x())
    s1.stop()
    holder = []
    threading.Timer(0.8, lambda: holder.append(start_thread(hub, UniformSource()))).start()
    t0 = time.time()
    logits, values = ev.evaluate(_x())
    assert logits.shape[0] == 3 and 0.5 < time.time() - t0 < 20
    assert ev.port.reconnects >= 2
    holder[0].stop()


def test_wrong_token_is_rejected():
    hub = make_hub("sock", 0, max_n=16)
    srv = start_thread(hub, UniformSource())
    spec = {**hub.client_spec(), "token": "wrong", "retry_s": 0.6}
    with pytest.raises(InferenceError, match="unreachable"):
        RemoteEvaluator(spec, "m").evaluate(_x())
    srv.stop()


def test_refresh_makes_the_server_check_weights_now(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    hub = make_hub("sock", 0, max_n=16)
    srv = start_thread(hub, DebugSource(str(tmp_path)), ServerConfig(reload_check_s=3600, max_wait_ms=1))
    ev = RemoteEvaluator(hub.client_spec(), "a")
    before = ev.evaluate(_x(2))[0][0, 0]
    time.sleep(0.01)
    (tmp_path / "a.txt").write_text("51")
    assert ev.evaluate(_x(2))[0][0, 0] == before
    ev.refresh()
    assert ev.evaluate(_x(2))[0][0, 0] - before == pytest.approx(50)
    srv.stop()


def test_many_clients_join_and_leave():
    hub = make_hub("sock", 0, max_n=32)
    srv = start_thread(hub, UniformSource(), ServerConfig(max_wait_ms=5))
    errs = []

    def client(i):
        try:
            for _ in range(3):
                ev = RemoteEvaluator(hub.client_spec(), "m")
                for _ in range(5):
                    assert ev.evaluate(_x(4, i))[0].shape == (4, 4672)
                ev.close()
        except Exception as e:                              # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=client, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    time.sleep(0.3)
    assert not errs, errs
    assert srv.server.port.n_clients == 0
    srv.stop()


# ------------------------------------------------------------------ supervisor
def test_supervisor_restarts_the_inference_server(tmp_path):
    from backend import paths
    from backend.runtime import Supervisor
    paths.ensure_runtime_dirs()
    sup = Supervisor()
    cfg = {"root": str(tmp_path), "port_file": str(tmp_path / "inf.port"), "max_n": 64,
           "max_batch": 64, "max_wait_ms": 2.0, "shm_prefix": "crltest-"}
    w = sup.ensure_inference(cfg)
    try:
        assert w.wait_ready(60), "inference server did not become READY"
        spec = {"kind": "sock", "token": sup.token, "max_n": 64, "port_file": cfg["port_file"]}
        ev = RemoteEvaluator(spec, "any", timeout=10)
        assert ev.evaluate(_x())[0].shape == (3, 4672)
        os.kill(w.proc.pid, signal.SIGKILL)
        t0 = time.time()
        assert ev.evaluate(_x())[0].shape == (3, 4672)       # the client just waits for the restart
        assert w.restarts == 1 and time.time() - t0 < 60
    finally:
        sup.stop_all()


# ------------------------------------------------------------------ self-play workers
MCTS = {"simulations": 8, "max_plies": 24, "leaves_per_batch": 4}


def _has_torch(workers) -> bool:
    import psutil
    for p in workers.procs:
        if "libtorch" in " ".join(m.path for m in psutil.Process(p.pid).memory_maps()):
            return True
    return False


def test_selfplay_workers_deliver_all_games_without_torch():
    from backend.rl.selfplay_workers import SelfplayWorkers
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    workers = SelfplayWorkers(2, hub.client_spec())
    try:
        got, moves = [], []
        n = workers.run_games("m", 5, MCTS, concurrent=3, seed=1, on_game=got.append, on_move=moves.append)
        assert n == 5 and len(got) == 5
        assert all(g.samples and g.result in (-1, 0, 1) and g.white_eval is None for g in got)
        assert len({g.index for g in got}) == 5
        assert moves and all(m["index"] < 1_000_000 for m in moves)    # only worker 0 reports moves
        assert srv.server.stats()["positions"] > 0
        assert not _has_torch(workers)
    finally:
        workers.close()
        srv.stop()


def test_a_dead_worker_is_replaced_and_its_games_are_replayed():
    from backend.rl.selfplay_workers import SelfplayWorkers
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    workers = SelfplayWorkers(2, hub.client_spec())
    try:
        os.kill(workers.procs[1].pid, signal.SIGKILL)
        workers.procs[1].join(5)
        got = []
        n = workers.run_games("m", 4, MCTS, concurrent=2, seed=2, on_game=got.append)
        assert n == 4 and len(got) == 4
    finally:
        workers.close()
        srv.stop()


def test_stop_interrupts_a_long_task():
    from backend.rl.selfplay_workers import SelfplayWorkers
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    workers = SelfplayWorkers(1, hub.client_spec())
    try:
        t0 = time.time()
        n = workers.run_games("m", 500, {**MCTS, "max_plies": 400}, concurrent=4, seed=3,
                              poll_stop=lambda: time.time() - t0 > 2.0)
        assert n < 500 and time.time() - t0 < 30
    finally:
        workers.close()
        srv.stop()


# ------------------------------------------------------------------ orchestrator end to end
def _payload(tmp_path, rid):
    return {"run_id": rid, "model_path": str(tmp_path / "m.pt"), "elo": 1000, "iterations": 0,
            "mcts": MCTS,
            "train": {"games_per_iteration": 4, "min_buffer": 1, "arena_games": 2,
                      "arena_simulations": 8, "arena_threshold": 0.5, "concurrent_games": 2,
                      "buffer_size": 1000}}


def test_orchestrator_plays_through_remote_workers(tmp_path):
    from backend.rl.config import TrainConfig
    from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    p = _payload(tmp_path, 7)
    ev = EventLog()
    o = Orchestrator(FakeBackend(p["model_path"], "cpu", TrainConfig()), p, ev,
                     remote={"spec": hub.client_spec(), "workers": 2})
    o.set_watch(True)
    o.start()
    try:
        t0 = time.time()
        while time.time() - t0 < 90 and not any(e["type"] == "metrics" for e in ev.since(0)["events"]):
            time.sleep(0.2)
        events = ev.since(0)["events"]
        m = [e for e in events if e["type"] == "metrics"]
        assert m, o.error
        assert m[0]["games"] == 4 and m[0]["infer"]["selfplay"] == {"remote": True}
        assert any(e["type"] == "selfplay_move" for e in events)
        live = [e for e in events if e["type"] == "live"]
        assert live and any(e["ended"] and e["result"] in ("white", "black", "draw") for e in live)
        assert {"selfplay", "learning", "arena"} <= {e["name"] for e in events if e["type"] == "phase"}
        assert srv.server.stats()["positions"] > 0
        assert m[0]["infer"]["arena"] == {"remote": True}                        # phase 7
        assert any(k.endswith(".cand") for k in srv.server.stats()["models"])   # candidate asked the server
        o.pause()
        time.sleep(0.4)
        assert o.state == "paused"
        o.resume()
    finally:
        o.stop()
        o._thread.join(timeout=40)
        srv.stop()
    assert not o._thread.is_alive() and o.state == "stopped"
    assert all(not pr.is_alive() for pr in o.workers.procs)


def test_gateway_trains_two_models_through_one_inference_server(monkeypatch):
    """Whole stack: gateway -> supervisor starts the ONE server -> two train processes,
    each with its own self-play workers, all sharing that server."""
    monkeypatch.setenv("CHESSRL_INFERENCE", "1")
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.state import S
    fast = {"mcts": {"simulations": 50, "max_plies": 40},
            "train": {"games_per_iteration": 4, "min_buffer": 1, "arena_games": 2,
                      "arena_simulations": 10, "concurrent_games": 2}}
    with TestClient(app) as c:
        t0 = time.time()
        while time.time() - t0 < 60:
            if c.get("/api/system/status").json()["workers"].get("play", {}).get("state") == "READY":
                break
            time.sleep(0.3)
        while time.time() - t0 < 60 and not c.get("/api/models").json():
            time.sleep(0.3)
        a = c.get("/api/models").json()[0]
        rb_ = c.post("/api/models", json={"name": "Two", "size": "tiny"})
        assert rb_.status_code in (200, 201), rb_.text
        b = rb_.json()
        rc_ = c.post("/api/models", json={"name": "Three", "size": "tiny"})
        assert rc_.status_code in (200, 201), rc_.text
        third = S.db.get_model(rc_.json()["id"])
        S.selection["selfplay_workers"] = 2
        try:
            ra = c.post("/api/training/start", json={"model_id": a["id"], "overrides": fast, "force": True})
            rb = c.post("/api/training/start", json={"model_id": b["id"], "overrides": fast, "force": True})
            assert ra.status_code == 201 and rb.status_code == 201, (ra.text, rb.text)
            assert S.sup.get("inference") is not None and S.sup.get("inference").is_ready()
            t0, done = time.time(), False
            while time.time() - t0 < 120 and not done:
                time.sleep(0.5)
                runs = c.get("/api/training/status").json()["runs"]
                done = len(runs) == 2 and all(x["metrics"] for x in runs)
            assert done, "both runs should report an iteration through the shared server"
            assert all(x["metrics"][0]["games"] == 4 for x in runs)

            # phase 8: play and hint against the third model while the others train
            g = c.post("/api/games/new", json={"model_id": third["id"], "color": "white"}).json()
            st = c.post("/api/games/play", json={"game_id": g["game_id"], "move_uci": "e2e4"}).json()
            assert len(st["moves"]) == 2 and st["eval"] is not None
            assert c.post(f"/api/games/{g['game_id']}/hint").json()["move"]
            import json as _json
            from backend import paths
            stats_file = paths.RUNTIME_DIR / "inference_stats.json"
            t0, seen = time.time(), set()
            while time.time() - t0 < 10:
                time.sleep(0.5)
                try:
                    seen = set(_json.loads(stats_file.read_text())["models"])
                except Exception:
                    continue
                if os.path.abspath(third["path"]) in seen:
                    break
            assert os.path.abspath(third["path"]) in seen, "Play/hint must be answered by the shared server"
            assert any(k.endswith(".cand") for k in seen), "the arena must use the server too"
        finally:
            c.post("/api/training/stop")
            time.sleep(2)


# ------------------------------------------------------------------ phase 7: learner gate, arena
def test_gate_lets_one_learner_in_and_frees_the_slot_when_its_process_dies(tmp_path):
    import subprocess
    import sys
    from backend.rl.gate import LearnerGate
    base = str(tmp_path / "learner.lock")
    g1, g2 = LearnerGate(base, 1), LearnerGate(base, 1)
    assert g1.acquire() and g1.holding()
    assert g2.acquire(should_stop=lambda: True, poll=0.01) is False      # full: stop while waiting
    g1.release()
    assert g2.acquire() and g2.holding()
    g2.release()
    two = [LearnerGate(base, 2), LearnerGate(base, 2), LearnerGate(base, 2)]
    assert two[0].acquire() and two[1].acquire()                           # 2 slots, 2 learners
    assert two[2].acquire(should_stop=lambda: True, poll=0.01) is False
    for g in two[:2]:
        g.release()
    # another PROCESS takes the slot and is killed without releasing it
    code = ("import sys,time; sys.path.insert(0, %r); from backend.rl.gate import LearnerGate;"
            "g=LearnerGate(%r,1); assert g.acquire(); print('held', flush=True); time.sleep(60)"
            % (os.getcwd(), base))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        assert LearnerGate(base, 1).acquire(should_stop=lambda: True, poll=0.01) is False
    finally:
        p.kill()
        p.wait()
    late = LearnerGate(base, 1)
    assert late.acquire(should_stop=lambda: time.sleep(0.05) or False, poll=0.05)   # the OS freed it
    late.release()


def test_orchestrator_waits_for_a_learner_slot_and_goes_on_when_it_frees(tmp_path):
    from backend.rl.config import TrainConfig
    from backend.rl.gate import LearnerGate
    from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator
    base = str(tmp_path / "gate")
    blocker = LearnerGate(base, 1)
    assert blocker.acquire()                                  # somebody else is learning
    p = _payload(tmp_path, 8)
    ev = EventLog()
    o = Orchestrator(FakeBackend(p["model_path"], "cpu", TrainConfig()), p, ev, gate=LearnerGate(base, 1))
    o.start()
    try:
        time.sleep(4)
        events = ev.since(0)["events"]
        assert any(e["type"] == "phase" and e["name"] == "learning" for e in events)
        assert not any(e["type"] == "metrics" for e in events)          # still queued for a slot
        blocker.release()
        t0 = time.time()
        while time.time() - t0 < 60 and not any(e["type"] == "metrics" for e in ev.since(0)["events"]):
            time.sleep(0.2)
        assert any(e["type"] == "metrics" for e in ev.since(0)["events"]), o.error
    finally:
        o.stop()
        o._thread.join(timeout=30)
    assert o.state == "stopped" and not o.gate.holding()


def test_stop_while_waiting_for_a_learner_slot(tmp_path):
    from backend.rl.config import TrainConfig
    from backend.rl.gate import LearnerGate
    from backend.rl.orchestrator import EventLog, FakeBackend, Orchestrator
    base = str(tmp_path / "gate")
    blocker = LearnerGate(base, 1)
    assert blocker.acquire()
    p = _payload(tmp_path, 9)
    o = Orchestrator(FakeBackend(p["model_path"], "cpu", TrainConfig()), p, EventLog(), gate=LearnerGate(base, 1))
    o.start()
    time.sleep(3)
    o.stop()
    o._thread.join(timeout=20)
    blocker.release()
    assert not o._thread.is_alive() and o.state == "stopped"


def test_play_match_offset_keeps_colours_balanced_across_workers():
    from backend.rl.arena import play_match
    from backend.rl.config import MCTSConfig
    from backend.rl.evaluator import UniformEvaluator
    a, b = UniformEvaluator(), UniformEvaluator()
    cfg = MCTSConfig(simulations=4, max_plies=6)
    r0 = play_match(a, b, 2, cfg, concurrent=2, seed=1, offset=0)
    r1 = play_match(a, b, 2, cfg, concurrent=2, seed=1, offset=1)
    white0 = sorted(g.white_eval is a for g in r0.games)
    white1 = sorted(g.white_eval is a for g in r1.games)
    assert white0 == white1 == [False, True]
    off = play_match(a, b, 1, cfg, concurrent=1, seed=1, offset=1)
    assert off.games[0].white_eval is b                       # offset 1: B has White first


def test_arena_match_over_workers_counts_every_game():
    from backend.rl.selfplay_workers import SelfplayWorkers
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    workers = SelfplayWorkers(2, hub.client_spec())
    try:
        progress = []
        tally = workers.run_match("cand", "best", 7, MCTS, concurrent=3, seed=5, on_progress=progress.append)
        assert sum(tally.values()) == 7 and set(tally) == {"a", "b", "d"}
        assert progress == list(range(1, 8))
        models = srv.server.stats()["models"]
        assert "cand" in models and "best" in models              # both nets were asked
    finally:
        workers.close()
        srv.stop()


def test_a_dead_worker_does_not_lose_arena_games():
    from backend.rl.selfplay_workers import SelfplayWorkers
    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, UniformSource())
    workers = SelfplayWorkers(2, hub.client_spec())
    try:
        os.kill(workers.procs[0].pid, signal.SIGKILL)
        workers.procs[0].join(5)
        tally = workers.run_match("cand", "best", 4, MCTS, concurrent=2, seed=6)
        assert sum(tally.values()) == 4
    finally:
        workers.close()
        srv.stop()


# ------------------------------------------------------------------ phase 8: play worker
def _search_req(hub, path, **kw):
    from backend.workers.play_worker import SearchReq
    return SearchReq(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                     model_path=path, mcts={"simulations": 20, "leaves_per_batch": 4},
                     inference={"port_file": hub.port_file, "max_n": 64}, **kw)


def test_play_worker_uses_the_server_without_the_global_lock(monkeypatch):
    from backend.workers import play_worker
    hub = make_hub("sock", 0, max_n=64)
    monkeypatch.setenv("CHESSRL_TOKEN", hub.token)
    srv = start_thread(hub, UniformSource())
    try:
        out = {}
        with play_worker._lock:                       # a local search would deadlock right here
            t = threading.Thread(target=lambda: out.update(play_worker.do_search(_search_req(hub, "/x/m.pt"))))
            t.start()
            t.join(30)
        assert not t.is_alive(), "a server search must not need the play worker's lock"
        assert out["via"] == "server" and len(out["move"]) >= 4
        assert srv.server.stats()["positions"] > 0
        assert "/x/m.pt" in srv.server.stats()["models"]

        results, errs = [], []

        def ask(i):
            try:
                results.append(play_worker.do_search(_search_req(hub, f"/x/m{i % 2}.pt", seed=i)))
            except Exception as e:                    # noqa: BLE001
                errs.append(repr(e))

        ts = [threading.Thread(target=ask, args=(i,)) for i in range(6)]
        [x.start() for x in ts]
        [x.join(60) for x in ts]
        assert not errs and len(results) == 6 and all(r["via"] == "server" for r in results)
    finally:
        srv.stop()


def test_play_worker_falls_back_to_local_when_the_server_is_gone(monkeypatch):
    from backend.workers import play_worker
    hub = make_hub("sock", 0, max_n=64)
    monkeypatch.setenv("CHESSRL_TOKEN", hub.token)
    srv = start_thread(hub, UniformSource())
    srv.stop()                                         # nothing listens any more
    t0 = time.time()
    res = play_worker.do_search(_search_req(hub, "/x/m.pt"))
    assert res["via"] == "local" and len(res["move"]) >= 4 and time.time() - t0 < 30


def test_play_worker_without_inference_is_unchanged():
    from backend.workers import play_worker
    from backend.workers.play_worker import SearchReq
    r = play_worker.do_search(SearchReq(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                                        mcts={"simulations": 20}))
    assert r["via"] == "local" and r["move"]
