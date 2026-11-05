"""Does overlapping the CPU work with the network wait help?

    python tools/bench_pipeline.py --latency-ms 6 --games 24 [--max-wait-ms 0]

One self-play worker plays the same games twice against a server that answers after a fixed
delay (a stand-in for a GPU: it sleeps, so the CPU is free meanwhile): once with one blocking
connection (the old loop) and once with two connections (the pipelined loop). Reports time,
games/min and the share of time the worker's CPU was busy."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(mode: str, a) -> dict:
    from backend.inference import RemoteEvaluator, ServerConfig, make_hub, start_process
    from backend.inference.sources import SlowUniformSource
    from backend.rl.config import MCTSConfig
    from backend.rl.selfplay import GamePool

    n = 2 if mode == "pipelined" else 1
    hub = make_hub("shm", n_clients=n, max_n=512)
    srv = start_process(hub, SlowUniformSource, {"latency_ms": a.latency_ms, "per_pos_us": a.per_pos_us},
                        ServerConfig(max_batch=a.max_batch, max_wait_ms=a.max_wait_ms))
    ev = RemoteEvaluator([hub.client_spec(i) for i in range(n)], "m", timeout=60)
    cfg = MCTSConfig(simulations=a.sims, leaves_per_batch=a.leaves, max_plies=a.plies)
    GamePool(cfg, seed=0).run([(ev, ev)] * 2, concurrent=2)           # warm up
    c0, t0 = time.process_time(), time.perf_counter()
    recs = GamePool(cfg, seed=1).run([(ev, ev)] * a.games, concurrent=a.concurrent)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    moves = sum(len(r.moves) for r in recs)
    srv.stop()
    hub.close()
    return {"mode": mode, "games": len(recs), "moves": moves, "seconds": round(wall, 2),
            "sims_per_s": round(moves * a.sims / wall), "cpu_busy_pct": round(100 * cpu / wall)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--concurrent", type=int, default=16)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--plies", type=int, default=40)
    ap.add_argument("--latency-ms", type=float, default=6.0)
    ap.add_argument("--per-pos-us", type=float, default=8.0)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--max-wait-ms", type=float, default=3.0)
    a = ap.parse_args()
    rows = [run(m, a) for m in ("serial", "pipelined")]
    for r in rows:
        print(r)
    print(f"speed-up: x{rows[0]['seconds'] / rows[1]['seconds']:.2f}")


if __name__ == "__main__":
    main()
