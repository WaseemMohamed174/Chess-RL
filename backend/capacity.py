"""How much can this machine run at once? Pure arithmetic, no detection, so it is easy to test.

With the central inference server the work splits into three kinds of process:

    self-play workers   CPU only, no PyTorch, about WORKER_GB each   -> limited by CORES
    one train process   per model: PyTorch + learner + replay buffer -> limited by RAM / VRAM
    inference server    one, owns the networks and the device        -> 1 core, a bit of RAM

All memory numbers are estimates (same spirit as the menu in cli/select.py)."""
from __future__ import annotations

import os

WORKER_GB = 0.2          # a self-play worker: chess + numpy + its MCTS trees
SERVER_GB = 1.0          # the inference server process (PyTorch + CUDA context)
BUFFER_GB = 0.35         # a replay buffer of 200k positions
MAX_RUNS = 16            # only the SUGGESTED number in plan(); never a limit (see estimate())
MAX_WORKERS = 32
MIN_CORES_FOR_SERVER = 3     # below this a shared server only adds overhead


def plan(cores: int, safe_ram: float, device: str, safe_vram: float = 0.0,
         blocks: int = 3, filters: int = 64) -> dict:
    """Capacity settings written to selection.json. device is "gpu" or "cpu"."""
    from .cli.select import need_gb, size_mb          # estimates live next to the menu
    cores = max(int(cores), 1)
    gpu = device == "gpu"
    net_gb = size_mb(blocks, filters) / 1024
    train_gb = need_gb(blocks, filters, True)

    reserved = 1 if cores <= 2 else 2                 # gateway + inference server
    workers = max(1, min(cores - reserved, MAX_WORKERS))

    ram_left = safe_ram - SERVER_GB - workers * WORKER_GB
    if gpu:
        per_run_ram = 1.0 + BUFFER_GB                 # torch in the train process; nets sit in VRAM
        per_run_vram = train_gb + 2 * net_gb          # learner + the server's best/candidate copies
        by_vram = int(safe_vram // per_run_vram)
    else:
        per_run_ram = train_gb + BUFFER_GB            # the learner lives in RAM too
        by_vram = MAX_RUNS
    by_ram = int(ram_left // per_run_ram) if ram_left > 0 else 0
    runs = max(1, min(by_ram, by_vram, MAX_RUNS))

    use_server = cores >= MIN_CORES_FOR_SERVER
    max_batch = max(256, min(256 * workers, 2048 if gpu else 512))
    return {
        "max_parallel_runs": runs,
        "max_learners": 1 if gpu else max(1, cores // 4),   # learners training at the same moment
        "selfplay_workers": workers,                  # total, shared by the runs training now
        "inference_server": use_server,
        "max_batch": max_batch,
        "cpu_threads": 1 if gpu else max(1, cores - workers),   # server forward passes on CPU
        "max_wait_ms": 3.0 if gpu else 5.0,           # a CPU forward is slow: waiting is cheap
    }


def workers_for_run(total_workers: int, active_runs: int) -> int:
    """Self-play workers for a run that starts while `active_runs` (including it) train.
    The first run gets the whole pool, the second half of it, and so on. Runs that started
    earlier are rebalanced by the gateway when runs start or end (see training.push_workers)."""
    return max(1, int(total_workers) // max(int(active_runs), 1))


def inference_enabled(selection: dict) -> bool:
    """CHESSRL_INFERENCE=1/0 overrides the selection (tests, benchmarks)."""
    env = os.environ.get("CHESSRL_INFERENCE")
    if env in ("0", "1"):
        return env == "1"
    return bool(selection.get("inference_server", False))


def from_selection(sel: dict) -> dict:
    """Capacity for an old selection.json that predates these keys (cheap detection only)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = max(min(vm.available / 1024**3 * 0.9, vm.total / 1024**3 * 0.75) - 0.8, 0.0)
    except Exception:
        ram = 4.0
    net = sel.get("network") or {}
    return plan(os.cpu_count() or 1, ram, sel.get("device", "cpu"),
                safe_vram=8.0 if sel.get("device") == "gpu" else 0.0,
                blocks=int(net.get("blocks", 3)), filters=int(net.get("filters", 64)))


def _per_run(device: str, blocks: int, filters: int) -> tuple[float, float]:
    """(RAM, VRAM) in GB that one more training run needs. Same numbers as plan()."""
    from .cli.select import need_gb, size_mb
    net_gb = size_mb(blocks, filters) / 1024
    train_gb = need_gb(blocks, filters, True)
    if device == "gpu":
        return 1.0 + BUFFER_GB, train_gb + 2 * net_gb
    return train_gb + BUFFER_GB, 0.0


def estimate(runs: int, selection: dict, hw: dict | None, blocks: int = 3, filters: int = 64) -> dict:
    """How much memory `runs` training runs at the same moment would need, against what is free.

    It only measures: nothing is refused here. level is "ok" (fits), "warn" (up to 1.5x what is
    free), "severe" (more than that) or "unknown" (the machine could not be measured).
    `fits` is how many runs would fit, at least 1: the size of a batch for "run in batches"."""
    runs = max(int(runs), 1)
    gpu = selection.get("device") == "gpu"
    per_ram, per_vram = _per_run("gpu" if gpu else "cpu", blocks, filters)
    fixed = SERVER_GB + int(selection.get("selfplay_workers", 1)) * WORKER_GB
    need_ram, need_vram = fixed + runs * per_ram, runs * per_vram
    out = {"runs": runs, "device": "gpu" if gpu else "cpu", "per_run_ram": round(per_ram, 2),
           "per_run_vram": round(per_vram, 2), "need_ram": round(need_ram, 2),
           "need_vram": round(need_vram, 2), "avail_ram": None, "avail_vram": None,
           "level": "unknown", "fits": runs, "slower_by": runs}
    if not hw or not hw.get("safe_ram"):
        return out
    ram, vram = float(hw["safe_ram"]), float(hw.get("safe_vram") or 0.0)
    ratio = need_ram / ram
    by = int(max(ram - fixed, 0) // per_ram)
    if gpu:
        ratio = max(ratio, need_vram / vram if vram > 0 else float("inf"))
        by = min(by, int(vram // per_vram))
    fits = max(1, by)
    out.update(avail_ram=round(ram, 2), avail_vram=round(vram, 2) if gpu else None,
               level="ok" if ratio <= 1 else "warn" if ratio <= 1.5 else "severe",
               fits=fits, slower_by=max(1, runs // fits) if runs > fits else 1)
    return out
