"""Baseline benchmark: N independent self-play processes (today's architecture).

    python tools/bench.py --runs 1,2,4 --seconds 20 [--evaluator uniform|torch] [--device cpu|cuda]

Each "run" is one process with its own GamePool and its own evaluator, exactly like one
training model today. Reports games/min, positions/s, average batch, and the share of
time spent inside evaluate() ("net %"). Save with --out so phase 10 can compare."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(idx, args, q, spec=None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from backend.rl.config import MCTSConfig
    from backend.rl.evaluator import CountingEvaluator, UniformEvaluator
    from backend.rl.selfplay import GamePool

    if spec is not None:                       # remote: the network lives in the server process
        from backend.inference import RemoteEvaluator
        inner = RemoteEvaluator(spec, "bench")
    elif args["evaluator"] == "torch":
        import torch
        torch.set_num_threads(1)
        from backend.rl.evaluator import TorchEvaluator
        from backend.rl.network import build_network
        inner = TorchEvaluator(build_network(args["blocks"], args["filters"]), args["device"])
    else:
        inner = UniformEvaluator()
    ev = CountingEvaluator(inner)
    cfg = MCTSConfig(simulations=args["sims"], leaves_per_batch=args["leaves"], max_plies=args["max_plies"])
    pool = GamePool(cfg, seed=idx)
    deadline = time.time() + args["seconds"]
    games = 0

    def on_game(_):
        nonlocal games
        games += 1

    t0 = time.time()
    pool.run([(ev, ev)] * 100000, concurrent=args["concurrent"],
             should_stop=lambda: time.time() > deadline, on_game=on_game)
    wall = time.time() - t0
    st = ev.stats()
    q.put({"games": games, "wall": wall, **st.to_dict()})


def make_source(args):
    """Picklable (factory, kwargs) for the server's models; only 'bench' exists."""
    if args["evaluator"] == "torch":
        import tempfile, torch
        from backend.rl.network import build_network
        d = tempfile.mkdtemp(prefix="bench-")
        net = build_network(args["blocks"], args["filters"])
        torch.save({"config": net.config(), "model": net.state_dict(), "opt": None, "steps": 0},
                   f"{d}/bench.pt")
        from backend.inference.sources import TorchSource
        return TorchSource, {"device": args["device"], "template": d + "/{id}.pt"}
    from backend.inference.sources import UniformSource
    return UniformSource, {}


def run_once(n, args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    srv = hub = None
    if args["mode"] == "remote":
        from backend.inference import ServerConfig, make_hub, start_process
        hub = make_hub(args["transport"], n, max_n=max(args["concurrent"] * args["leaves"], 64), ctx=ctx)
        factory, kw = make_source(args)
        srv = start_process(hub, factory, kw, ServerConfig(max_batch=args["max_batch"],
                            max_wait_ms=args["max_wait_ms"]), ctx=ctx)
    ps = [ctx.Process(target=worker, args=(i, args, q, hub.client_spec(i) if hub else None))
          for i in range(n)]
    [p.start() for p in ps]
    res = [q.get() for _ in ps]
    [p.join() for p in ps]
    if srv:
        srv.stop()
        hub.close()
    wall = max(r["wall"] for r in res)
    pos = sum(r["positions"] for r in res)
    calls = sum(r["calls"] for r in res)
    return {"runs": n, "games_per_min": round(sum(r["games"] for r in res) / wall * 60, 1),
            "pos_per_s": round(pos / wall), "avg_batch": round(pos / max(calls, 1), 1),
            "net_pct": round(100 * sum(r["eval_secs"] for r in res) / (wall * n), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="1,2,4")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--evaluator", choices=["uniform", "torch"], default="uniform")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", choices=["local", "remote"], default="local",
                    help="local = one network per process (today); remote = shared inference server")
    ap.add_argument("--transport", choices=["shm", "queue"], default="shm")
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--max-wait-ms", type=float, default=2.0)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--concurrent", type=int, default=32)
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    args = {k: v for k, v in vars(a).items() if k not in ("runs", "out")}
    machine = f"{os.cpu_count()} cores, {platform.platform()}"
    print(f"machine: {machine}\nsetting: {args}\n")
    print("runs  games/min  scaling   pos/s  avg batch  net %")
    print("-" * 56)
    rows, base = [], None
    for n in [int(x) for x in a.runs.split(",")]:
        r = run_once(n, args)
        base = base or r["games_per_min"]
        r["scaling"] = round(r["games_per_min"] / base, 2)
        rows.append(r)
        print(f"{n:>4}  {r['games_per_min']:>9}  {r['scaling']:>6}x  {r['pos_per_s']:>6}  "
              f"{r['avg_batch']:>9}  {r['net_pct']:>5}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"machine": machine, "args": args, "rows": rows}, indent=2))
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
