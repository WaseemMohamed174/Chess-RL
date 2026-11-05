"""Chess-RL launcher.  Usage:  python run.py [options]

Four steps, every time:
    1. Libraries   separate virtual environments (gateway / engine), install what is missing
    2. Selection   device + default network size (terminal screens, backend/cli/select.py)
    3. Prepare     PyTorch build for the chosen device, checkpoint folder, website build
               (Node.js is downloaded automatically if missing; only used for this build)
    4. Launch      start the gateway; it serves the website and starts the engine workers

Options
    --auto          skip the questions, use the recommended choices (or CHESSRL_AUTO=1)
    --no-train      play / spectate only (no training features)
    --no-frontend   do not build the website (the API still works at /docs)
    --no-browser    do not open the browser
    -h, --help      show this text

Environment
    CHESSRL_HOST (default 127.0.0.1)    CHESSRL_PORT (default 8000)
    CHESSRL_NODE_MIRROR   alternative to https://nodejs.org/dist for the Node.js download
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend import paths
from backend.cli import node, ui

HOST = os.environ.get("CHESSRL_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHESSRL_PORT", "8000"))
URL = f"http://{HOST}:{PORT}/"
APP_MODULE = "backend.app.main:app"

TORCH_INDEX = "https://download.pytorch.org/whl/"
TORCH_CUDA_BUILDS = (121, 124, 126, 128)      # CUDA builds PyTorch publishes wheels for

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------- environments

def _create_venv(name):
    py = paths.venv_python(name)
    if py.exists():
        return py
    subprocess.check_call([sys.executable, "-m", "venv", str(paths.VENV_DIR / name)])
    for pkg in ("pip", "wheel", "setuptools"):
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", pkg],
                       capture_output=True, check=False)
    return py


def _ensure_gateway_venv():
    """Re-run this script inside the gateway venv, so everything below uses it."""
    py = paths.venv_python("gateway")
    if Path(sys.executable).absolute() == py.absolute():
        return
    paths.VENV_DIR.mkdir(parents=True, exist_ok=True)
    if not py.exists():
        ui.info("Creating the gateway environment (first run only) ...")
    _create_venv("gateway")
    args = [str(py), str(Path(__file__).resolve())] + sys.argv[1:]
    if os.name == "nt":        # os.execv does not replace the process on Windows
        sys.exit(subprocess.call(args))
    os.execv(str(py), args)


# ---------------------------------------------------------------- step 3 helpers

def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _torch_build(selection):
    """'cpu', or the newest published CUDA build the driver supports ('cu124' ...)."""
    tag = selection.get("cuda_tag")
    if selection.get("device") == "gpu" and tag:
        try:
            usable = [b for b in TORCH_CUDA_BUILDS if b <= int(tag[2:])]
        except ValueError:
            usable = []
        if usable:
            return f"cu{max(usable)}"
    return "cpu"


def _install_torch(deps, py, build, force, mode, central, log):
    extra = ["--index-url", TORCH_INDEX + build] + (["--force-reinstall"] if force else [])
    item = {"spec": "torch", "extra": extra, "py": py, "lib": "torch",
            "component": "Engine", "venv": "engine"}
    try:
        deps.install_all([item], mode, central, log)
        return True
    except SystemExit:
        return False


def _probe_torch(py):
    code = ("import json, torch\n"
            "c = torch.cuda.is_available()\n"
            "print(json.dumps({'version': torch.__version__, 'cuda': c, "
            "'gpu': torch.cuda.get_device_name(0) if c else None}))\n")
    try:
        res = subprocess.run([str(py), "-c", code], capture_output=True,
                             text=True, timeout=180)
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _ensure_torch(deps, select, selection, mode, central, log):
    """Make the engine venv hold the PyTorch build that matches the chosen device.

    Returns the selection (the device falls back to CPU if CUDA does not work)."""
    py = paths.venv_python("engine")
    want = _torch_build(selection)
    ui.section("Deep-learning engine", note=f"PyTorch ({want})")

    state_file = paths.RUNTIME_DIR / "deps.json"
    state = _read_json(state_file)
    have = state.get("torch_build")
    present = not deps.missing_by_name(py, ["torch"])
    # a CUDA build also runs fine on the CPU, so it never needs replacing for "cpu"
    compatible = have == want or (want == "cpu" and (have or "").startswith("cu"))

    if present and compatible:
        ui.ok(f"PyTorch already installed ({have})")
    else:
        ui.info(f"Installing PyTorch ({want}) ...")
        done = _install_torch(deps, py, want, present, mode, central, log)
        if not done and want != "cpu":
            ui.warn("Could not install the GPU build of PyTorch. Falling back to CPU.")
            want = "cpu"
            done = _install_torch(deps, py, want, present, mode, central, log)
        if not done:
            ui.fail(f"PyTorch could not be installed. See {log}")
            ui.warn("Continuing without the engine: play and training will not work "
                    "until PyTorch is installed.")
            return selection
        state["torch_build"] = want
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    probe = _probe_torch(py)
    if not probe:
        ui.warn("PyTorch is installed but could not be imported. See the install log.")
        return selection
    if probe["cuda"]:
        ui.ok(f"PyTorch {probe['version']} · CUDA available ({probe['gpu']})")
    elif selection.get("device") == "gpu":
        ui.warn("PyTorch cannot use the GPU on this machine. Switching to CPU mode.")
        selection = {**selection, "device": "cpu", "cuda_tag": None}
        select.save(selection)
    else:
        ui.ok(f"PyTorch {probe['version']} · CPU")
    return selection


def _frontend_stale():
    index = paths.FRONTEND_DIST / "index.html"
    if not index.exists():
        return True
    built = index.stat().st_mtime
    watch = [paths.FRONTEND_DIR / "package.json", paths.FRONTEND_DIR / "index.html"]
    src = paths.FRONTEND_DIR / "src"
    if src.exists():
        watch += [p for p in src.rglob("*") if p.is_file()]
    return any(p.exists() and p.stat().st_mtime > built for p in watch)


def _ensure_frontend(skip, log, auto=False):
    ui.section("Website", note="React + Vite")
    if skip:
        ui.info("--no-frontend: the website is not built")
        return
    if not (paths.FRONTEND_DIR / "package.json").exists():
        ui.warn("frontend/ was not found. The gateway will serve the API only (/docs).")
        return
    if not _frontend_stale():
        ui.ok("Website is already built")
        return
    tool = node.ensure(auto=auto)
    if not tool:
        ui.warn("Node.js (npm) is not available, so the website cannot be built. "
                "Install Node 18+ from nodejs.org and run again. The API still works at /docs.")
        return
    npm, env = tool["npm"], tool["env"]
    prefix = ["--prefix", str(paths.FRONTEND_DIR)]
    if not (paths.FRONTEND_DIR / "node_modules").exists():
        ui.info("Installing website packages (first run only) ...")
        if ui.run_quiet([npm, *prefix, "install", "--no-audit", "--no-fund"], log, env=env) != 0:
            ui.warn(f"npm install failed. See {log}. The API still works at /docs.")
            return
    ui.info("Building the website ...")
    if ui.run_quiet([npm, *prefix, "run", "build"], log, env=env) != 0:
        ui.warn(f"Website build failed. See {log}. The API still works at /docs.")
        return
    ui.ok("Website built")


# ---------------------------------------------------------------- step 4 helpers

def _port_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def _start_gateway():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    log = open(paths.LOGS_DIR / "gateway.log", "a", encoding="utf-8")
    cmd = [sys.executable, "-m", "uvicorn", APP_MODULE,
           "--host", HOST, "--port", str(PORT), "--log-level", "info"]
    return subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=log,
                            stderr=subprocess.STDOUT, env=env)


def _wait_ready(proc, max_wait=60):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if proc.poll() is not None:
            return False
        try:
            with _NO_PROXY.open(URL + "health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _stop_tree(proc):
    """Stop the gateway AND every worker it started (training must never be orphaned)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    children = []
    if psutil and proc.poll() is None:
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except Exception:
            pass
    if proc.poll() is None:
        proc.terminate()                  # the gateway stops its workers on shutdown
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if psutil and children:               # anything it could not stop (e.g. on Windows)
        alive = [c for c in children if c.is_running()]
        for c in alive:
            try:
                c.terminate()
            except Exception:
                pass
        _, alive = psutil.wait_procs(alive, timeout=3)
        for c in alive:
            try:
                c.kill()
            except Exception:
                pass


# ---------------------------------------------------------------- main

def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        return

    _ensure_gateway_venv()

    from backend.cli import deps, select
    ui.banner()

    auto = "--auto" in sys.argv or os.environ.get("CHESSRL_AUTO") == "1"
    no_train = "--no-train" in sys.argv
    no_frontend = "--no-frontend" in sys.argv
    no_browser = "--no-browser" in sys.argv

    if _port_busy():          # fail early, before installing anything
        ui.fail(f"Port {PORT} is already in use. Close the other program, "
                "or set CHESSRL_PORT to another port.")
        sys.exit(1)

    # ---- 1. Libraries
    ui.step(1, "Libraries")
    paths.ensure_runtime_dirs()
    install_log = paths.LOGS_DIR / "install.log"

    ui.section("Checking environments")
    pending = []
    for venv_name, specs in (("gateway", deps.GATEWAY), ("engine", deps.ENGINE)):
        py = _create_venv(venv_name)
        missing = deps.missing_by_name(py, [s for s, _ in specs])
        for spec, extra in specs:
            if spec in missing:
                pending.append({
                    "spec": spec, "extra": extra, "py": py,
                    "lib": deps.base_name(spec),
                    "component": venv_name.capitalize(), "venv": venv_name,
                })

    mode, central = "online", None
    if pending:
        deps.show_missing(pending)
        mode, central = deps.choose_source()
        deps.install_all(pending, mode, central, install_log)
    else:
        ui.ok("All required libraries are present")

    # ---- 2. Selection
    ui.step(2, "Selection")
    selection = select.run(no_train=no_train, auto=auto)

    # ---- 3. Prepare
    ui.step(3, "Prepare")
    try:
        selection = _ensure_torch(deps, select, selection, mode, central, install_log)
    except Exception as e:
        ui.warn(f"Engine setup failed: {e}")
        ui.warn("Continuing without the engine. Fix it and run again.")
    paths.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_frontend(no_frontend, paths.LOGS_DIR / "frontend.log", auto)

    # ---- 4. Launch
    ui.step(4, "Launch")
    proc = _start_gateway()

    def shutdown(*_):
        ui.info("Stopping ...")
        _stop_tree(proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    with ui.spinner("Starting the gateway ..."):
        ready = _wait_ready(proc)
    if not ready:
        ui.fail("Gateway failed to start. Last log lines:")
        try:
            lines = (paths.LOGS_DIR / "gateway.log").read_text(
                encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-25:]))
        except Exception:
            pass
        shutdown()

    net = selection.get("network", {})
    ui.ok(f"Gateway is up: {URL}")
    ui.box("CHESS-RL IS RUNNING", [
        f"Website  : {URL}",
        f"API docs : {URL}docs",
        f"Device   : {selection.get('device', '?').upper()}",
        f"Network  : {net.get('name', '?')} ({net.get('blocks', '?')} x {net.get('filters', '?')})",
        f"Logs     : {paths.LOGS_DIR}",
    ])
    if not no_browser:
        try:
            webbrowser.open(URL)
        except Exception:
            pass

    ui.info("Running. Press Ctrl+C to stop.")
    try:
        while proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    shutdown()


if __name__ == "__main__":
    main()
