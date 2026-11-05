"""Terminal design for Chess-RL.

One visual language for every screen: rounded boxes, a single indigo accent,
green / amber / red only for status, muted gray for secondary text.
Truecolor when the terminal supports it, plain 16 colors otherwise, and no
color at all when piped or when NO_COLOR is set.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import unicodedata

if os.name == "nt":
    os.system("")                     # enable ANSI escape codes in the Windows console
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_COLOR = bool(sys.stdout.isatty()) and not os.environ.get("NO_COLOR")
_TRUE = _COLOR and (os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
                    or bool(os.environ.get("WT_SESSION")))
_ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")

# name: ((r, g, b), 16-color fallback)
_PAL = {
    "accent": ((129, 140, 248), "36"),
    "ok":     ((74, 222, 128), "32"),
    "warn":   ((250, 204, 21), "33"),
    "err":    ((248, 113, 113), "31"),
    "muted":  ((148, 163, 184), "90"),
}


def _paint(name: str, text: str, bold: bool = False) -> str:
    if not _COLOR:
        return text
    rgb, fb = _PAL[name]
    code = f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}" if _TRUE else fb
    return f"\033[{'1;' if bold else ''}{code}m{text}\033[0m"


def accent(t, b=False): return _paint("accent", t, b)
def green(t):  return _paint("ok", t)
def yellow(t): return _paint("warn", t)
def red(t):    return _paint("err", t)
def gray(t):   return _paint("muted", t)
def bold(t):   return f"\033[1m{t}\033[0m" if _COLOR else t
def dim(t):    return f"\033[2m{t}\033[0m" if _COLOR else t
cyan = accent                         # kept so older call sites still work


def strip(s: str) -> str:
    return _ANSI.sub("", s)


# ---------------------------------------------------------------- measuring

def _w(s: str) -> int:
    n = 0
    for ch in strip(s):
        if unicodedata.combining(ch):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def _pad(s: str, w: int) -> str:
    return s + " " * max(w - _w(s), 0)


def _trunc(s: str, w: int) -> str:
    if _w(s) <= w:
        return s
    while s and _w(s) > w - 1:
        s = s[:-1]
    return s + "\u2026"


def width() -> int:
    return max(78, min(shutil.get_terminal_size((80, 24)).columns - 1, 100))


# ------------------------------------------------------------ one-line msgs

def info(msg: str) -> None: print(f"  {accent('•')} {msg}", flush=True)
def ok(msg: str) -> None:   print(f"  {green('✓')} {msg}", flush=True)
def warn(msg: str) -> None: print(f"  {yellow('!')} {yellow(msg)}", flush=True)
def fail(msg: str) -> None: print(f"  {red('✗')} {red(msg)}", flush=True)


# ------------------------------------------------------------------ banner

_LOGO = ("♞", "CHESS", "RL")


def banner(subtitle: str = "AlphaZero-style chess · play, watch, train") -> None:
    W = width()
    print()
    print("  " + accent(_LOGO[0], True) + "  " + bold(_LOGO[1]) + accent(" · ", True)
          + accent(_LOGO[2], True))
    print("  " + gray(subtitle))
    print(gray("  " + "─" * (W - 4)))


# ------------------------------------------------------- steps and sections

STEPS = ("Libraries", "Selection", "Prepare", "Launch")


def step(n: int, title: str) -> None:
    W = width()
    crumbs = []
    for i, name in enumerate(STEPS, 1):
        if i < n:
            crumbs.append(green("✓ ") + gray(name))
        elif i == n:
            crumbs.append(accent("● " + name, True))
        else:
            crumbs.append(gray("○ " + name))
    print()
    print(accent("━" * W))
    print("  " + accent(f"STEP {n}/{len(STEPS)}", True) + gray("  ·  ") + bold(title.upper()))
    print("  " + gray("  ──  ").join(crumbs))
    print(accent("━" * W))


def section(title: str, note: str = "") -> None:
    W = width()
    left = f"── {title} "
    right = f" {note} ──" if note else ""
    fill = max(W - _w(left) - _w(right), 2)
    line = accent("── ") + bold(title) + " " + gray("─" * fill)
    if note:
        line += gray(" " + note + " ──")
    print()
    print(line)


# ------------------------------------------------------------------- boxes

def box(title: str, lines: list[str], width_: int | None = None, tone: str = "accent") -> None:
    W = width_ or width()
    inner = W - 4
    edge = (lambda s: accent(s)) if tone == "accent" else gray
    flat: list[str] = []
    for ln in lines:
        if strip(ln) == ln and _w(ln) > inner:
            flat.extend(textwrap.wrap(ln, inner) or [""])
        else:
            flat.append(ln)
    t = f" {title} "
    print(edge("╭" + "─" * 1) + bold(accent(t) if tone == "accent" else gray(t))
          + edge("─" * max(W - 3 - _w(t), 0) + "╮"))
    for ln in flat:
        print(edge("│") + " " + ln + " " * max(inner - _w(ln), 0) + " " + edge("│"))
    print(edge("╰" + "─" * (W - 2) + "╯"))


def explain(options: list[tuple[str, str]], title: str = "What each option does") -> None:
    inner = width() - 4
    prefixes = [f"{k}." if k[:1].isdigit() else k for k, _ in options]
    pw = max((_w(p) for p in prefixes), default=0)
    indent = pw + 1
    lines: list[str] = []
    for prefix, (_, text) in zip(prefixes, options):
        wrapped = textwrap.wrap(text, max(inner - indent, 20)) or [""]
        lines.append(accent(_pad(prefix, pw), True) + " " + wrapped[0])
        lines.extend(" " * indent + w for w in wrapped[1:])
    box(title, lines, tone="muted")


def table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    w = [max([_w(h)] + [_w(str(r[i])) for r in rows]) for i, h in enumerate(headers)]

    def rule(l, m, r):
        return gray(l + m.join("─" * (x + 2) for x in w) + r)

    if title:
        print(bold(f" {title}"))
    print(rule("╭", "┬", "╮"))
    print(gray("│") + gray("│").join(f" {accent(_pad(h, w[i]), True)} "
                                     for i, h in enumerate(headers)) + gray("│"))
    print(rule("├", "┼", "┤"))
    for r in rows:
        print(gray("│") + gray("│").join(f" {_pad(str(c), w[i])} "
                                         for i, c in enumerate(r)) + gray("│"))
    print(rule("╰", "┴", "╯"))


# -------------------------------------------------------------------- menus

def _auto() -> bool:
    return not sys.stdin.isatty() or os.environ.get("CHESSRL_AUTO") == "1"


def ask(n: int, default: int = 1) -> int:
    if _auto():
        print(dim(f"  (non-interactive → using default {default})"), flush=True)
        return default
    rng = "1" if n == 1 else f"1-{n}"
    while True:
        try:
            raw = input(f"  {accent('›', True)} Select {gray(f'[{rng}]')} "
                        f"{gray(f'Enter = {default}')}: ").strip()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw)
        print(red(f"    Enter a number from {rng}."), flush=True)


def ask_text(prompt: str) -> str:
    try:
        return input(f"  {prompt}\n  {accent('›', True)} ").strip().strip('"').strip("'")
    except EOFError:
        return ""


def _options_box(headers: list[str], rows: list[dict], numbers: dict[int, int]) -> None:
    W = width()
    inner = W - 4
    gap = 2
    n_data = len(headers)
    has_status = any(r.get("state") in ("rec", "blocked") or r.get("tag") for r in rows)

    head = [""] + list(headers) + ([""] if has_status else [])
    body: list[tuple[list[str], str]] = []
    for i, r in enumerate(rows):
        state = r.get("state", "ok")
        num = f"[{numbers[i]}]" if i in numbers else " ✕ "
        data = [str(x) for x in r["cols"]]
        data += [""] * (n_data - len(data))
        status = ("★ recommended" if state == "rec"
                  else "not available" if state == "blocked" else r.get("tag", ""))
        body.append(([num] + data + ([status] if has_status else []), state))

    nc = len(head)
    widths = [max([_w(head[c])] + [_w(cells[c]) for cells, _ in body]) for c in range(nc)]
    over = sum(widths) + gap * (nc - 1) - inner
    if over > 0 and nc > 1:
        widths[1] = max(widths[1] - over, 8)
    used = sum(widths) + gap * (nc - 1)

    def line(cells: list[str], state: str, header: bool = False) -> str:
        parts = []
        for c, (txt, w) in enumerate(zip(cells, widths)):
            txt = _pad(_trunc(txt, w), w)
            is_status = has_status and c == nc - 1
            if header:
                txt = gray(txt)
            elif state == "blocked":
                txt = gray(txt)
            elif c == 0:
                txt = accent(txt, True) if state != "rec" else green(txt)
            elif state == "rec" and c == 1:
                txt = bold(txt)
            elif is_status and txt.strip():
                txt = green(txt) if state == "rec" else yellow(txt)
            parts.append(txt)
        return (gray("│") + " " + (" " * gap).join(parts)
                + " " * max(inner - used, 0) + " " + gray("│"))

    print(gray("╭" + "─" * (W - 2) + "╮"))
    print(line(head, "ok", header=True))
    print(gray("├" + "─" * (W - 2) + "┤"))
    for cells, state in body:
        print(line(cells, state))
    print(gray("╰" + "─" * (W - 2) + "╯"))


def menu(title: str | None, headers: list[str], items: list[dict], *,
         extras: list[dict] | tuple = (), explain_lines: list[tuple[str, str]] | None = None,
         note: str = "") -> int:
    """Section title, explanation box, numbered options, prompt.

    items / extras are dicts: cols (list[str]), state ("ok" | "rec" | "blocked"),
    desc (explanation), tag (small status for "ok" rows). "blocked" rows are gray,
    unnumbered and cannot be chosen. Returns the 0-based index into items + extras."""
    rows = list(items) + list(extras)
    n_items = len(items)
    numbers: dict[int, int] = {}
    for i, r in enumerate(rows):
        if r.get("state", "ok") != "blocked":
            numbers[i] = len(numbers) + 1
    if not numbers:
        raise RuntimeError("no selectable options")
    rec = [numbers[i] for i, r in enumerate(rows) if r.get("state") == "rec" and i in numbers]
    default = rec[0] if rec else 1

    exp: list[tuple[str, str]] = []
    if explain_lines is None:
        exp = [(str(numbers[i]), r["desc"]) for i, r in enumerate(rows)
               if i in numbers and r.get("desc")]
    else:
        nums = [numbers[i] for i in range(n_items) if i in numbers]
        if nums:
            rng = str(nums[0]) if len(nums) == 1 else f"{nums[0]}-{nums[-1]}"
            exp.extend((k.format(range=rng), t) for k, t in explain_lines)
        exp += [(str(numbers[i]), rows[i]["desc"]) for i in range(n_items, len(rows))
                if i in numbers and rows[i].get("desc")]
    if any(r.get("state") == "blocked" for r in rows):
        exp.append(("✕", "Gray rows need more memory than is safe on this machine."))

    if title:
        section(title, note)
    if exp:
        explain(exp)
        print()
    _options_box(headers, rows, numbers)
    chosen = ask(len(numbers), default)
    return next(i for i, num in numbers.items() if num == chosen)


# ------------------------------------------------------- long-running work

@contextlib.contextmanager
def spinner(msg: str):
    """Animated one-line status for work with no measurable progress."""
    if not _COLOR:
        print(f"  … {msg}", flush=True)
        yield
        return
    stop = threading.Event()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def spin():
        i = 0
        while not stop.is_set():
            sys.stdout.write(f"\r\033[2K  {accent(frames[i % len(frames)])} {msg}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


def run_quiet(cmd: list[str], log_path, on_line=None, env=None) -> int:
    from pathlib import Path
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", env=env)
        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)
            if on_line:
                on_line(line)
        return p.wait()


class InstallProgress:
    W = 30

    def __init__(self, total: int):
        self.total = total
        self.i = 0
        self.name = ""
        self._drawn = False

    def start(self, name: str) -> None:
        self.i += 1
        self.name = name
        self.update(0.0)

    def update(self, frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        if not _COLOR:
            if frac in (0.0, 1.0):
                print(f"({self.i}/{self.total}) {self.name} {int(frac * 100):3d}%", flush=True)
            return
        f = int(self.W * frac)
        l1 = f"  {gray(f'{self.i}/{self.total}')}  {bold(self.name)}"
        l2 = f"  {accent('━' * f)}{gray('─' * (self.W - f))}  {int(frac * 100):3d}%"
        if self._drawn:
            sys.stdout.write("\033[2F")
        sys.stdout.write("\033[2K" + l1 + "\n\033[2K" + l2 + "\n")
        sys.stdout.flush()
        self._drawn = True

    def done(self) -> None:
        self.update(1.0)

    def close(self) -> None:
        self._drawn = False
        ok(f"{self.total}/{self.total} libraries installed")


def pip_progress_hook(prog: InstallProgress):
    state = {"f": 0.0}
    steps = (("Collecting", 0.15), ("Downloading", 0.45), ("Building", 0.60),
             ("Installing collected", 0.85), ("Successfully installed", 1.0))

    def hook(line: str) -> None:
        for key, val in steps:
            if line.startswith(key) and val > state["f"]:
                state["f"] = val
                prog.update(val)
    return hook
