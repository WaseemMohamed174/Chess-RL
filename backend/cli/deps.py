"""Libraries for each virtual environment, and the install flow."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import ui

# gateway: the website server. No PyTorch here, so it starts in about a second.
GATEWAY: list[tuple[str, list[str]]] = [(p, []) for p in (
    "fastapi", "uvicorn[standard]", "numpy", "psutil", "chess",
    "aiosqlite", "httpx", "pydantic",
)]

# engine: everything heavy. torch itself is installed separately in run.py
# (its build depends on the device chosen in step 2).
ENGINE: list[tuple[str, list[str]]] = [(p, []) for p in (
    "numpy", "psutil", "chess", "fastapi", "uvicorn[standard]",
    "onnx", "onnxruntime", "torchinfo",
)]


def base_name(spec: str) -> str:
    return re.split(r"[\[<>=!~ ;]", spec.strip())[0].lower().replace("_", "-")


def missing_by_name(py: Path, specs: list[str]) -> list[str]:
    if not py.exists():
        return list(specs)
    names = [base_name(s) for s in specs]
    code = ("import importlib.metadata as m, json, sys\n"
            "out = []\n"
            "for n in json.loads(sys.argv[1]):\n"
            "    try: m.version(n)\n"
            "    except m.PackageNotFoundError: out.append(n)\n"
            "print(json.dumps(out))\n")
    res = subprocess.run([str(py), "-c", code, json.dumps(names)],
                         capture_output=True, text=True)
    try:
        gone = set(json.loads(res.stdout.strip().splitlines()[-1]))
    except Exception:
        return list(specs)
    return [s for s, n in zip(specs, names) if n in gone]


def find_wheel(central: str, spec: str) -> str | None:
    want = base_name(spec)
    try:
        for f in os.listdir(central):
            if f.endswith(".whl") and f.split("-")[0].lower().replace("_", "-") == want:
                return f
    except OSError:
        pass
    return None


def show_missing(items: list[dict], title: str = "Missing libraries") -> None:
    ui.section(title, note=f"{len(items)} libraries")
    rows = [[str(i), it["lib"], it["component"], it["venv"],
             ui.red(it.get("status", "missing"))] for i, it in enumerate(items, 1)]
    ui.table(["#", "Library", "Component", "Environment", "Status"], rows)


def choose_source() -> tuple[str, str | None]:
    c = ui.menu("How to get the missing libraries", ["Source"], [
        {"cols": ["Download from the internet (PyPI)"], "state": "rec",
         "desc": "Fastest to start. Requires an internet connection."},
        {"cols": ["Install from a central library (folder of .whl files)"],
         "desc": "Works offline. Point to a folder of .whl files. If some wheels "
                 "are missing you will be asked how to get them."},
        {"cols": ["Cancel"], "desc": "Exit. Nothing is installed."},
    ])
    if c == 2:
        raise SystemExit(0)
    if c == 0:
        return "online", None
    while True:
        p = ui.ask_text("Path of the central library folder:")
        if p and os.path.isdir(os.path.expanduser(p)):
            return "central", os.path.expanduser(p)
        ui.fail("Folder not found.")


def install_all(items: list[dict], mode: str, central: str | None, log_path) -> None:
    if mode == "central":
        assert central
        gone = [it for it in items if not find_wheel(central, it["spec"])]
        if gone:
            show_missing([{**g, "status": "not in central"} for g in gone],
                         title="Not in the central library")
            c = ui.menu("Some wheels are not in the central library", ["Action"], [
                {"cols": ["Download them into the central library, then install"],
                 "desc": "Wheels are saved for next time. Slower the first run."},
                {"cols": ["Install them directly from the internet"],
                 "desc": "Faster now, but the central library is not updated."},
                {"cols": ["Cancel"], "desc": "Exit. Nothing is installed."},
            ])
            if c == 2:
                raise SystemExit(0)
            if c == 0:
                for g in gone:
                    ui.run_quiet([sys.executable, "-m", "pip", "wheel", "-w", central,
                                  g["spec"]], log_path)
            else:
                for g in gone:
                    g["force_online"] = True

    ui.section("Installing libraries", note=f"{len(items)} to install")
    prog = ui.InstallProgress(len(items))
    for it in items:
        prog.start(it["lib"])
        py = str(it["py"])
        if mode == "central" and not it.get("force_online"):
            cmd = [py, "-m", "pip", "install", "--no-index",
                   f"--find-links={central}", it["spec"]]
        else:
            cmd = [py, "-m", "pip", "install", *it["extra"], it["spec"]]
        rc = ui.run_quiet(cmd, log_path, ui.pip_progress_hook(prog))
        if rc != 0:
            print()
            ui.fail(f"Failed to install {it['lib']}. See {log_path}")
            raise SystemExit(1)
        prog.done()
    prog.close()
