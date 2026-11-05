"""Inference server: transports, batching, model cache, errors, and the contract."""
import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from backend.inference import InferenceError, RemoteEvaluator, ServerConfig, make_hub, start_process, start_thread
from backend.inference.sources import DebugEvaluator, DebugSource, UniformSource
from backend.rl.config import MCTSConfig
from backend.rl.evaluator import CountingEvaluator, UniformEvaluator
from backend.rl.selfplay import GamePool

KINDS = ["shm", "queue", "sock"]


def _batch(rng, n):
    return rng.integers(0, 2, size=(n, 18, 8, 8), dtype=np.uint8)


def _write(root, name, offset):
    (root / f"{name}.txt").write_text(str(offset))


@pytest.mark.parametrize("kind", KINDS)
def test_rows_go_back_to_the_right_caller_and_get_batched(tmp_path, kind):
    _write(tmp_path, "m", 5)
    hub = make_hub(kind, n_clients=3, max_n=64)
    srv = start_thread(hub, DebugSource(str(tmp_path)), ServerConfig(max_wait_ms=20))
    errors = []

    def client(i):
        try:
            ev = RemoteEvaluator(hub.client_spec(i), "m")
            rng, ref = np.random.default_rng(i), DebugEvaluator(5)
            for _ in range(15):
                x = _batch(rng, int(rng.integers(1, 40)))
                got, want = ev.evaluate(x), ref.evaluate(x)
                assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
        except Exception as e:                       # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=client, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    st = srv.server.stats()
    srv.stop()
    hub.close()
    assert not errors, errors
    assert st["calls"] < 45                          # 45 requests, fewer forward passes: batching works
    assert st["models"]["m"]["requests"] == 45


@pytest.mark.parametrize("kind", KINDS)
def test_large_request_is_split_and_reassembled(tmp_path, kind):
    _write(tmp_path, "m", 1)
    hub = make_hub(kind, 1, max_n=32)
    srv = start_thread(hub, DebugSource(str(tmp_path)))
    x = _batch(np.random.default_rng(0), 100)        # 100 > max_n=32
    got = RemoteEvaluator(hub.client_spec(0), "m").evaluate(x)
    want = DebugEvaluator(1).evaluate(x)
    srv.stop()
    hub.close()
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_a_lone_request_does_not_hang_when_other_clients_are_idle(tmp_path):
    _write(tmp_path, "m", 0)
    hub = make_hub("shm", 4, max_n=16)               # 4 clients, only one talks
    srv = start_thread(hub, DebugSource(str(tmp_path)), ServerConfig(max_wait_ms=30))
    ev = RemoteEvaluator(hub.client_spec(0), "m", timeout=5)
    t0 = time.perf_counter()
    ev.evaluate(_batch(np.random.default_rng(0), 3))
    dt = time.perf_counter() - t0
    srv.stop()
    hub.close()
    assert dt < 1.0


def test_lru_eviction_and_hot_reload(tmp_path):
    _write(tmp_path, "a", 1)
    _write(tmp_path, "b", 2)
    src = DebugSource(str(tmp_path))
    hub = make_hub("shm", 1, max_n=8)
    srv = start_thread(hub, src, ServerConfig(max_models=1, reload_check_s=0.0, max_wait_ms=1))
    a, b = (RemoteEvaluator(hub.client_spec(0), m) for m in ("a", "b"))
    x = _batch(np.random.default_rng(0), 2)
    a.evaluate(x)
    a.evaluate(x)
    assert src.loads == 1                            # cached, not reloaded
    b.evaluate(x)
    a.evaluate(x)
    assert src.loads == 3                            # max_models=1: b evicted a, a evicted b

    before = a.evaluate(x)[0][0, 0]
    time.sleep(0.01)
    _write(tmp_path, "a", 100)                       # "promote": weights file changed
    after = a.evaluate(x)[0][0, 0]
    srv.stop()
    hub.close()
    assert after - before == pytest.approx(99)       # picked up without restarting anything


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_model_fails_that_call_only(tmp_path, kind):
    _write(tmp_path, "ok", 0)
    hub = make_hub(kind, 1, max_n=8)
    srv = start_thread(hub, DebugSource(str(tmp_path)))
    x = _batch(np.random.default_rng(0), 2)
    with pytest.raises(InferenceError, match="FileNotFoundError"):
        RemoteEvaluator(hub.client_spec(0), "missing").evaluate(x)
    RemoteEvaluator(hub.client_spec(0), "ok").evaluate(x)     # server still alive
    srv.stop()
    hub.close()


def test_client_timeout_when_server_is_gone():
    hub = make_hub("shm", 1, max_n=8)
    ev = RemoteEvaluator(hub.client_spec(0), "m", timeout=0.3)    # no server started
    with pytest.raises(InferenceError, match="no answer"):
        ev.evaluate(_batch(np.random.default_rng(0), 1))
    with pytest.raises(InferenceError, match="timed out earlier"):
        ev.evaluate(_batch(np.random.default_rng(0), 1))
    hub.close()


def test_bad_request_is_rejected_before_it_reaches_the_server():
    hub = make_hub("shm", 1, max_n=8)
    ev = RemoteEvaluator(hub.client_spec(0), "m")
    with pytest.raises(ValueError):
        ev.evaluate(np.zeros((2, 18, 8, 8), np.float32))
    hub.close()


@pytest.mark.parametrize("kind", KINDS)
def test_mcts_games_are_identical_through_the_server(kind):
    """MCTS cannot tell the network lives elsewhere: same seed, same games."""
    cfg = MCTSConfig(simulations=12, max_plies=14)
    local = GamePool(cfg, seed=3).run([(UniformEvaluator(),) * 2] * 3, concurrent=3)

    hub = make_hub(kind, 1, max_n=256)
    srv = start_thread(hub, UniformSource())
    remote = CountingEvaluator(RemoteEvaluator(hub.client_spec(0), "any"), check=True)
    out = GamePool(cfg, seed=3).run([(remote, remote)] * 3, concurrent=3)
    srv.stop()
    hub.close()
    assert [r.moves for r in out] == [r.moves for r in local]
    assert remote.stats().positions > 0


# ---------- the real thing: server and clients in separate processes ----------
def _client_proc(spec, model, seed, out):
    from backend.inference import RemoteEvaluator
    from backend.inference.sources import DebugEvaluator
    ev, ref, rng = RemoteEvaluator(spec, model, timeout=30), DebugEvaluator(7), np.random.default_rng(seed)
    ok = 0
    for _ in range(20):
        x = rng.integers(0, 2, size=(int(rng.integers(1, 30)), 18, 8, 8), dtype=np.uint8)
        a, b = ev.evaluate(x), ref.evaluate(x)
        ok += bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    out.put(ok)


@pytest.mark.parametrize("kind", KINDS)
def test_separate_processes(tmp_path, kind):
    _write(tmp_path, "m", 7)
    ctx = mp.get_context("spawn")
    hub = make_hub(kind, 2, max_n=64, ctx=ctx)
    stats = tmp_path / "stats.json"
    srv = start_process(hub, DebugSource, {"root": str(tmp_path)},
                        ServerConfig(stats_path=str(stats)), ctx=ctx)
    out = ctx.Queue()
    ps = [ctx.Process(target=_client_proc, args=(hub.client_spec(i), "m", i, out)) for i in range(2)]
    [p.start() for p in ps]
    res = [out.get(timeout=90) for _ in ps]
    [p.join(10) for p in ps]
    time.sleep(1.2)                                   # let the server write its stats file
    srv.stop()
    hub.close()
    assert res == [20, 20]
    import json
    assert json.loads(stats.read_text())["positions"] > 0


def test_real_torch_model_through_the_server(tmp_path):
    """Needs PyTorch (skipped here without it). Same code, device is just a string."""
    torch = pytest.importorskip("torch")
    from backend.inference.sources import TorchSource
    from backend.rl.evaluator import TorchEvaluator
    from backend.rl.network import build_network

    net = build_network(2, 16)
    torch.save({"config": net.config(), "model": net.state_dict(), "opt": None, "steps": 0},
               str(tmp_path / "m.pt"))
    hub = make_hub("shm", 1, max_n=64)
    srv = start_thread(hub, TorchSource("cpu", template=str(tmp_path / "{id}.pt")))
    x = _batch(np.random.default_rng(0), 10)
    got = RemoteEvaluator(hub.client_spec(0), "m").evaluate(x)
    want = TorchEvaluator(net, "cpu").evaluate(x)
    srv.stop()
    hub.close()
    assert np.allclose(got[0], want[0], atol=1e-4) and np.allclose(got[1], want[1], atol=1e-4)


def test_stopping_the_server_cuts_connections_so_a_waiting_client_notices_at_once():
    """A client whose request is in flight must not wait out its whole timeout."""
    import threading, time
    import numpy as np
    from backend.inference import make_hub, start_thread, RemoteEvaluator
    from backend.inference.sources import UniformSource

    class Slow(UniformSource):
        def load(self, model_id):
            ev = super().load(model_id)
            inner = ev.evaluate
            def slow(x):
                time.sleep(6)
                return inner(x)
            ev.evaluate = slow
            return ev

    hub = make_hub("sock", 0, max_n=64)
    srv = start_thread(hub, Slow())
    ev = RemoteEvaluator(hub.client_spec(), "m", timeout=60)
    ev.port.retry_s = 1.0
    out = {}

    def call():
        t0 = time.time()
        try:
            ev.evaluate(np.zeros((2, 18, 8, 8), np.uint8))
        except Exception as e:                       # noqa: BLE001
            out["err"] = e
        out["secs"] = time.time() - t0

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.5)
    srv.stop(timeout=0.3)                          # does not wait for the slow forward pass
    t.join(30)
    assert "err" in out and out["secs"] < 4, out          # noticed at once, not after its 60 s timeout
