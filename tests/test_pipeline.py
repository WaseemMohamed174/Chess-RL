"""Pipelined self-play: RemoteEvaluator with several connections, the two-half GamePool loop,
and the `urgent` flag that keeps the server from holding a pipelined request back."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from backend.inference import InferenceError, RemoteEvaluator, ServerConfig, make_hub, start_thread
from backend.inference.sources import DebugEvaluator, DebugSource, UniformSource
from backend.rl.config import MCTSConfig
from backend.rl.evaluator import UniformEvaluator
from backend.rl.selfplay import GamePool


def _batch(rng, n):
    return rng.integers(0, 2, size=(n, 18, 8, 8), dtype=np.uint8)


class AsyncUniform:
    """In-process stand-in for a RemoteEvaluator with 2 connections: answers on a thread."""
    pipelined = True

    def __init__(self, delay=0.0):
        self.ex, self.inner, self.delay = ThreadPoolExecutor(2), UniformEvaluator(), delay
        self.calls, self.open = 0, 0
        self.lock = threading.Lock()

    def _run(self, x):
        with self.lock:
            self.open += 1
        time.sleep(self.delay)
        out = self.inner.evaluate(x)
        with self.lock:
            self.open -= 1
        return out

    def evaluate(self, x):
        return self._run(x)

    def evaluate_async(self, x):
        self.calls += 1
        return self.ex.submit(self._run, x)


CFG = MCTSConfig(simulations=8, leaves_per_batch=4, max_plies=24)


# --------------------------------------------------------------- game loop
def test_pipelined_pool_plays_every_game_to_the_end():
    ev = AsyncUniform()
    recs = GamePool(CFG, seed=1).run([(ev, ev)] * 10, concurrent=6)
    assert sorted(r.index for r in recs) == list(range(10))
    assert all(len(r.moves) > 0 and r.result in (-1, 0, 1) for r in recs)
    assert ev.calls > 0                              # it used the async path
    assert all(len(r.samples) > 0 for r in recs)


def test_pipelined_and_serial_agree_on_what_a_game_is():
    a, b = AsyncUniform(), UniformEvaluator()
    p = GamePool(CFG, seed=3).run([(a, a)] * 6, concurrent=4)
    s = GamePool(CFG, seed=3).run([(b, b)] * 6, concurrent=4)       # no evaluate_async: serial
    assert len(p) == len(s) == 6
    assert a.calls > 0
    for r in p + s:
        assert 0 < len(r.moves) <= CFG.max_plies
        assert r.pgn()


def test_the_two_halves_really_overlap():
    ev = AsyncUniform(delay=0.02)
    seen = {"max": 0}
    orig = ev._run

    def spy(x):
        seen["max"] = max(seen["max"], ev.open + 1)
        return orig(x)
    ev._run = spy
    GamePool(CFG, seed=0).run([(ev, ev)] * 8, concurrent=8)
    assert seen["max"] == 2                          # two requests were at the server together


def test_small_pools_and_plain_evaluators_use_the_serial_loop():
    ev = AsyncUniform()
    GamePool(CFG, seed=0).run([(ev, ev)] * 3, concurrent=3)         # < 4 games at once
    assert ev.calls == 0
    plain = UniformEvaluator()
    assert not GamePool._can_pipeline([(plain, plain)] * 8, 8)
    assert GamePool._can_pipeline([(ev, ev)] * 8, 8)
    assert not GamePool._can_pipeline([(ev, plain)] * 8, 8)         # one side cannot: serial


def test_stop_leaves_no_request_in_flight():
    ev = AsyncUniform(delay=0.05)
    n = {"k": 0}

    def stop():
        n["k"] += 1
        return n["k"] > 3

    GamePool(CFG, seed=0).run([(ev, ev)] * 8, concurrent=8, should_stop=stop)
    assert ev.open == 0                              # drained before run() returned


def test_an_error_does_not_leave_requests_in_flight():
    class Boom(AsyncUniform):
        def _run(self, x):
            r = super()._run(x)
            if self.calls == 2:
                raise RuntimeError("boom")
            return r

    ev = Boom(delay=0.03)
    with pytest.raises(RuntimeError, match="boom"):
        GamePool(CFG, seed=0).run([(ev, ev)] * 8, concurrent=8)
    assert ev.open == 0


# ------------------------------------------------- RemoteEvaluator, 2 connections
@pytest.fixture(params=["shm", "queue"])
def served(tmp_path, request):
    (tmp_path / "m.txt").write_text("5")
    hub = make_hub(request.param, n_clients=2, max_n=64)
    srv = start_thread(hub, DebugSource(str(tmp_path)), ServerConfig(max_wait_ms=2))
    yield hub, srv
    srv.stop()
    hub.close()


def test_two_connections_answer_async_calls_row_for_row(served):
    hub, _ = served
    ev = RemoteEvaluator([hub.client_spec(0), hub.client_spec(1)], "m")
    assert ev.pipelined
    rng, ref = np.random.default_rng(0), DebugEvaluator(5)
    xs = [_batch(rng, int(rng.integers(1, 50))) for _ in range(12)]
    futs = [ev.evaluate_async(x) for x in xs]            # more calls than connections
    for x, f in zip(xs, futs):
        got, want = f.result(10), ref.evaluate(x)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_sync_evaluate_and_for_model_work_on_a_pooled_client(served, tmp_path):
    hub, _ = served
    (tmp_path / "n.txt").write_text("9")
    ev = RemoteEvaluator([hub.client_spec(0), hub.client_spec(1)], "m")
    other = ev.for_model("n")
    assert other.pipelined
    x = _batch(np.random.default_rng(1), 10)
    assert np.array_equal(ev.evaluate(x)[0], DebugEvaluator(5).evaluate(x)[0])
    assert np.array_equal(other.evaluate_async(x).result(10)[0], DebugEvaluator(9).evaluate(x)[0])


def test_a_large_async_request_is_split_and_reassembled(served):
    hub, _ = served
    ev = RemoteEvaluator([hub.client_spec(0), hub.client_spec(1)], "m")
    x = _batch(np.random.default_rng(2), 150)            # 150 > max_n = 64
    got, want = ev.evaluate_async(x).result(10), DebugEvaluator(5).evaluate(x)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_async_errors_come_back_as_inference_errors(served):
    hub, _ = served
    ev = RemoteEvaluator([hub.client_spec(0), hub.client_spec(1)], "no-such-model")
    with pytest.raises(InferenceError):
        ev.evaluate_async(_batch(np.random.default_rng(3), 4)).result(10)


def test_single_connection_client_is_unchanged(served):
    hub, _ = served
    ev = RemoteEvaluator(hub.client_spec(0), "m")
    assert not ev.pipelined
    with pytest.raises(RuntimeError):
        ev.evaluate_async(_batch(np.random.default_rng(4), 2))


def test_several_connections_from_one_spec_need_the_sock_transport(served):
    hub, _ = served
    with pytest.raises(ValueError):
        RemoteEvaluator(hub.client_spec(0), "m", connections=2)


def test_full_games_through_a_real_server_with_pipelining():
    hub = make_hub("shm", n_clients=2, max_n=256)
    srv = start_thread(hub, UniformSource(), ServerConfig(max_wait_ms=2, max_batch=256))
    ev = RemoteEvaluator([hub.client_spec(0), hub.client_spec(1)], "any")
    recs = GamePool(CFG, seed=5).run([(ev, ev)] * 8, concurrent=8)
    st = srv.server.stats()
    srv.stop()
    hub.close()
    assert len(recs) == 8 and all(r.moves for r in recs)
    assert st["positions"] > 0


# ------------------------------------------------------------- the urgent flag
@pytest.mark.parametrize("kind", ["shm", "queue", "sock"])
def test_urgent_request_is_not_held_back_by_max_wait(tmp_path, kind):
    """With 3 clients and a 400 ms max_wait, a lone normal request waits for company; an
    urgent one (what evaluate_async sends) is answered at once."""
    (tmp_path / "m.txt").write_text("1")
    hub = make_hub(kind, n_clients=3, max_n=32)
    srv = start_thread(hub, DebugSource(str(tmp_path)), ServerConfig(max_wait_ms=400))
    from backend.inference.transport import open_client_port
    port = open_client_port(hub.client_spec(0))
    x = _batch(np.random.default_rng(0), 4)
    idle = []
    if kind == "sock":                               # sock counts open connections: add 2 idle ones
        for i in (1, 2):
            p = open_client_port(hub.client_spec(i))
            p.call("m", x, 10)
            idle.append(p)
    t0 = time.perf_counter()
    port.finish(port.begin("m", x, 10, urgent=True))
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    port.call("m", x, 10)
    slow = time.perf_counter() - t0
    for p in [port] + idle:
        p.close()
    srv.stop()
    hub.close()
    assert fast < 0.2, fast
    assert slow >= 0.3, slow                         # the normal request really did wait
