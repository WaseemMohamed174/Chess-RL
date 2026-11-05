"""STEP 2 of run.py: the terminal selection screens (Chess-RL).

What the person chooses here is only what depends on THIS machine:
    1. the running device   (CPU / GPU)
    2. the default network  (Tiny / Small / Medium / custom blocks x filters)

Everything that is a tuning knob (MCTS simulations, learning rate, batch size,
games per iteration ...) lives on the website's Settings page, not here.

The result is saved to data/runtime/selection.json. Every other process (the
gateway and the engine workers) reads that file through backend/settings.py, so
no value is ever passed around as a command-line argument.

selection.json
    {
      "version": 1,
      "device": "gpu" | "cpu",
      "cuda_tag": "cu124" | null,
      "network": {"key", "name", "blocks", "filters", "params_m", "size_mb"},
      "selfplay_workers": 3,
      "train_enabled": true
    }
"""
from __future__ import annotations

import json

from .. import paths
from . import ui

SELECTION_FILE = paths.SELECTION_FILE
SELECTION_VERSION = 1

# --------------------------------------------------------------------------
# Network presets. Keep these in sync with the table on the Architecture page.
# --------------------------------------------------------------------------
NETWORKS: list[dict] = [
    {"key": "tiny", "name": "Tiny", "blocks": 3, "filters": 64,
     "cpu_slow": False, "rec_vram": 0.0},
    {"key": "small", "name": "Small", "blocks": 6, "filters": 128,
     "cpu_slow": True, "rec_vram": 2.0},
    {"key": "medium", "name": "Medium", "blocks": 10, "filters": 192,
     "cpu_slow": True, "rec_vram": 6.0},
]

# Shape constants used only for the size / memory ESTIMATE shown in the menu.
# If network.py changes the heads, change these too (the numbers are estimates).
INPUT_PLANES = 18
POLICY_SIZE = 4672
POLICY_CONV = 32      # filters of the policy head 1x1 conv
VALUE_CONV = 1        # filters of the value head 1x1 conv
VALUE_HIDDEN = 256
TRAIN_BATCH = 256     # batch size assumed for the activation-memory estimate
RUNTIME_GB = 1.0      # torch + CUDA context + python overhead

BLOCKS_RANGE = (1, 40)
FILTERS_RANGE = (16, 512)


def count_params(blocks: int, filters: int) -> int:
    f = filters
    p = INPUT_PLANES * 9 * f + 2 * f                       # stem conv + BN
    p += blocks * (2 * 9 * f * f + 4 * f)                  # residual blocks
    p += f * POLICY_CONV + 2 * POLICY_CONV                 # policy 1x1 + BN
    p += POLICY_CONV * 64 * POLICY_SIZE + POLICY_SIZE      # policy FC
    p += f * VALUE_CONV + 2 * VALUE_CONV                   # value 1x1 + BN
    p += VALUE_CONV * 64 * VALUE_HIDDEN + VALUE_HIDDEN     # value FC 1
    p += VALUE_HIDDEN + 1                                  # value FC 2
    return p


def size_mb(blocks: int, filters: int) -> float:
    return count_params(blocks, filters) * 4 / 1024 ** 2   # fp32 weights


def need_gb(blocks: int, filters: int, training: bool = True) -> float:
    """Rough memory needed (RAM on CPU, VRAM on GPU)."""
    params = count_params(blocks, filters)
    if not training:
        return RUNTIME_GB + params * 4 / 1024 ** 3
    weights_opt = params * 16 / 1024 ** 3                  # weights + grads + Adam m, v
    acts = TRAIN_BATCH * blocks * 2 * filters * 64 * 4 * 3 / 1024 ** 3
    return RUNTIME_GB + weights_opt + acts


def _network_entry(blocks: int, filters: int, key: str = "custom",
                   name: str = "Custom") -> dict:
    return {
        "key": key, "name": name, "blocks": blocks, "filters": filters,
        "params_m": round(count_params(blocks, filters) / 1e6, 2),
        "size_mb": round(size_mb(blocks, filters), 1),
    }


# --------------------------------------------------------------------------
# screens
# --------------------------------------------------------------------------

def _show_hardware(hw) -> None:
    ui.section("Hardware")
    lines = [
        f"CPU cores : {hw.cores}",
        f"RAM       : {hw.ram_total:.1f} GB total · {hw.ram_avail:.1f} free",
        f"Safe RAM  : {hw.safe_ram:.1f} GB",
    ]
    if hw.gpu_name:
        lines.append(f"GPU       : {hw.gpu_name}")
        lines.append(f"VRAM      : {hw.vram_total:.1f} GB total · {hw.vram_free:.1f} free")
        lines.append(f"Safe VRAM : {hw.safe_vram:.1f} GB")
        if hw.cuda_tag:
            lines.append(f"CUDA      : {hw.cuda_version} ({hw.cuda_tag})")
        else:
            lines.append(f"CUDA      : {hw.cuda_version or '?'} (need 12.1+)")
    else:
        lines.append("GPU       : not detected (CPU mode)")
    ui.box("YOUR HARDWARE", lines)


def _ask_device(hw, auto: bool) -> tuple[str, str | None]:
    if not hw.gpu_available:
        ui.section("Running device")
        ui.ok("CPU mode (no usable GPU)")
        if getattr(hw, "gpu_note", ""):
            ui.info(hw.gpu_note)
        return "cpu", None
    if auto:
        ui.section("Running device")
        ui.ok(f"GPU mode ({hw.cuda_tag})")
        return "gpu", hw.cuda_tag

    c = ui.menu("Running device", ["Device", "Details"], [
        {"cols": ["GPU", "Much faster training"], "state": "rec",
         "desc": "Use the NVIDIA GPU for self-play inference and training. "
                 "Requires CUDA 12.1 or newer."},
        {"cols": ["CPU", "Slower but safe"],
         "desc": "Use the CPU only. Works everywhere. Fine for the Tiny network."},
    ])
    return ("gpu", hw.cuda_tag) if c == 0 else ("cpu", None)


def _custom_network(budget: float, label: str, training: bool) -> dict | None:
    def ask_int(prompt: str, lo: int, hi: int) -> int | None:
        raw = ui.ask_text(f"{prompt} [{lo}-{hi}]:")
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        ui.fail(f"Enter a whole number from {lo} to {hi}.")
        return None

    blocks = ask_int("Residual blocks", *BLOCKS_RANGE)
    if blocks is None:
        return None
    filters = ask_int("Filters per layer", *FILTERS_RANGE)
    if filters is None:
        return None
    need = need_gb(blocks, filters, training)
    if need > budget:
        ui.fail(f"{blocks} x {filters} needs ~{need:.1f} GB but the safe "
                f"{label} budget is {budget:.1f} GB.")
        return None
    return _network_entry(blocks, filters)


def _ask_network(hw, device: str, train_enabled: bool, auto: bool) -> dict:
    budget = hw.safe_vram if device == "gpu" else hw.safe_ram
    label = "VRAM" if device == "gpu" else "RAM"
    note = f"safe budget: {budget:.1f} GB {label}"

    needs = [need_gb(n["blocks"], n["filters"], train_enabled) for n in NETWORKS]
    fits = [x <= budget for x in needs]
    low_memory = not any(fits)
    if low_memory:
        # Never leave the person with nothing to pick: the smallest stays available.
        fits[0] = True

    if device == "gpu":
        rec = max((i for i, n in enumerate(NETWORKS)
                   if fits[i] and budget >= n["rec_vram"]), default=0)
    else:
        rec = 0  # Tiny is the sensible default on CPU

    if auto:
        n = NETWORKS[rec]
        ui.section("Network size", note)
        ui.ok(f"Network: {n['name']} ({n['blocks']} blocks x {n['filters']} filters)")
        return _network_entry(n["blocks"], n["filters"], n["key"], n["name"])

    if low_memory:
        ui.warn(f"Very little free {label}: even Tiny may be tight. "
                "Close other programs before training.")

    items = []
    for i, n in enumerate(NETWORKS):
        item = {
            "cols": [n["name"], f"{n['blocks']} x {n['filters']}",
                     f"{count_params(n['blocks'], n['filters']) / 1e6:.1f}M",
                     f"{size_mb(n['blocks'], n['filters']):.0f} MB",
                     f"~{needs[i]:.1f} GB"],
            "state": "blocked" if not fits[i] else ("rec" if i == rec else "ok"),
        }
        if item["state"] == "ok" and device == "cpu" and n["cpu_slow"]:
            item["tag"] = "⚠ slow on CPU"
        items.append(item)
    extras = [{"cols": ["Custom"],
               "desc": "Choose your own blocks x filters. Rejected if it does not "
                       "fit the safe memory budget."}]
    explain_lines = [("{range}", "Pick a size, listed from smallest to largest. Bigger "
                                 "plays stronger after training but trains slower and "
                                 "needs more memory. You can create other sizes later "
                                 "from the website.")]

    while True:
        c = ui.menu("Network size", ["Network", "Blocks x Filters", "Params", "Size", "Needs"],
                    items, extras=extras, explain_lines=explain_lines, note=note)
        if c >= len(items):
            custom = _custom_network(budget, label, train_enabled)
            if custom:
                ui.ok(f"Network: custom ({custom['blocks']} x {custom['filters']})")
                return custom
            continue
        n = NETWORKS[c]
        ui.ok(f"Network: {n['name']} ({n['blocks']} blocks x {n['filters']} filters)")
        return _network_entry(n["blocks"], n["filters"], n["key"], n["name"])


def _capacity(hw, device: str, network: dict) -> dict:
    """Auto values (not asked): self-play workers, parallel runs, server batching."""
    from ..capacity import plan
    return plan(hw.cores, hw.safe_ram, device, hw.safe_vram,
                network["blocks"], network["filters"])


def _show_last_selection(sel: dict) -> None:
    net = sel.get("network") or {}
    net_txt = (f"{net.get('name', '?')} "
               f"({net.get('blocks', '?')} x {net.get('filters', '?')})")
    ui.box("LAST SELECTION", [
        f"Device   : {(sel.get('device') or 'auto').upper()}",
        f"Network  : {net_txt}",
        f"Training : {'on' if sel.get('train_enabled', True) else 'off (play only)'}",
        f"Self-play: {sel.get('selfplay_workers', '?')} workers · "
        f"up to {sel.get('max_parallel_runs', '?')} models at once",
    ])


def _load_previous() -> dict | None:
    try:
        data = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") != SELECTION_VERSION or "network" not in data:
        return None
    return data


def save(selection: dict) -> None:
    SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_FILE.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(no_train: bool = False, auto: bool = False) -> dict:
    """Run every selection screen and return (and save) the selection."""
    if not auto:
        existing = _load_previous()
        if existing:
            ui.section("Previous selection")
            _show_last_selection(existing)
            print()
            c = ui.menu(None, ["Option"], [
                {"cols": ["Keep this selection and continue"], "state": "rec",
                 "desc": "Skip the questions and use the previous choices."},
                {"cols": ["Choose again"],
                 "desc": "Show the hardware box and choose from scratch."},
            ])
            if c == 0:
                if no_train and existing.get("train_enabled", True):
                    existing["train_enabled"] = False
                    save(existing)
                return existing

    from ..hardware import detect_hardware
    hw = detect_hardware()
    _show_hardware(hw)

    train_enabled = not no_train
    if no_train:
        ui.section("Training")
        ui.ok("Training disabled (--no-train): play and spectate only")

    device, cuda_tag = _ask_device(hw, auto)
    network = _ask_network(hw, device, train_enabled, auto)

    selection = {
        "version": SELECTION_VERSION,
        "device": device,
        "cuda_tag": cuda_tag,
        "network": network,
        **_capacity(hw, device, network),
        "train_enabled": train_enabled,
    }
    save(selection)

    print()
    ui.ok("Selection saved")
    return selection
