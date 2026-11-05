"""Machine selection (selection.json, written by run.py) and user-settings defaults."""
from __future__ import annotations

import json

from . import paths
from .rl.config import MCTSConfig, PopulationConfig, TrainConfig

DEFAULT_SELECTION = {
    "version": 1, "device": "cpu", "cuda_tag": None,
    "network": {"key": "tiny", "name": "Tiny", "blocks": 3, "filters": 64},
    "selfplay_workers": 1, "train_enabled": True,
    "max_parallel_runs": 4,            # how many models may train at the same moment
    "inference_server": False,         # one shared server for all models (see capacity.py)
    "max_batch": 256, "max_wait_ms": 5.0,
    "cpu_threads": 1,                  # torch threads of the inference server when it runs on CPU
    "max_learners": 1,                 # learners that may train at the same moment (rl/gate.py)
}

DEFAULT_UI = {"board_theme": "classic", "piece_set": "cburnett", "show_arrows": True,
              "show_eval": True, "language": "en"}


def load_selection() -> dict:
    try:
        data = json.loads(paths.SELECTION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_SELECTION)
    if "max_parallel_runs" not in data:           # written before capacity planning existed
        from . import capacity
        data = {**capacity.from_selection(data), **data}
    return {**DEFAULT_SELECTION, **data}


def default_settings() -> dict:
    return {"mcts": MCTSConfig().to_dict(), "train": TrainConfig().to_dict(),
            "population": PopulationConfig().to_dict(), "ui": dict(DEFAULT_UI)}


LIMITS = {
    "mcts": {"simulations": (50, 800), "c_puct": (0.1, 10), "dirichlet_alpha": (0.01, 2),
             "dirichlet_eps": (0, 1), "temperature": (0, 5), "temperature_moves": (0, 200),
             "leaves_per_batch": (1, 64), "max_plies": (40, 1000)},
    "train": {"lr": (1e-6, 1), "weight_decay": (0, 0.1), "batch_size": (8, 4096),
              "games_per_iteration": (1, 5000), "epochs_per_iteration": (1, 50),
              "max_steps_per_iteration": (1, 100000), "buffer_size": (1000, 5_000_000),
              "min_buffer": (0, 1_000_000), "concurrent_games": (1, 512),
              "arena_games": (2, 400), "arena_simulations": (10, 800),
              "arena_threshold": (0.5, 1.0)},
    "population": {"size": (2, 100), "survivor_fraction": (0.05, 0.5),
                   "iterations_per_generation": (1, 1000), "mutation_strength": (0, 0.5)},
}


def validate_settings(patch: dict) -> dict:
    """Raise ValueError for unknown keys, wrong types or out-of-range numbers."""
    base = default_settings()
    for section, values in patch.items():
        if section not in base or not isinstance(values, dict):
            raise ValueError(f"unknown settings section: {section}")
        for k, v in values.items():
            if k not in base[section]:
                raise ValueError(f"unknown setting: {section}.{k}")
            want = base[section][k]
            num = isinstance(want, (int, float)) and not isinstance(want, bool)
            if num:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"{section}.{k} must be a number")
            elif type(v) is not type(want):
                raise ValueError(f"{section}.{k} must be {type(want).__name__}")
            lo_hi = LIMITS.get(section, {}).get(k)
            if lo_hi and not lo_hi[0] <= v <= lo_hi[1]:
                raise ValueError(f"{section}.{k} must be between {lo_hi[0]} and {lo_hi[1]}")
    return patch


def merged_settings(stored: dict) -> dict:
    """Defaults overlaid with whatever the user saved."""
    out = default_settings()
    for section, values in (stored or {}).items():
        if section in out and isinstance(values, dict):
            out[section].update({k: v for k, v in values.items() if k in out[section]})
    return out
