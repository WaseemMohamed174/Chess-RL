"""Find Node.js, or install it. Node is only used to BUILD the website (npm + Vite).
After the build the gateway serves plain files, so Node is not needed to run the app.

Order of attempts, so that `python run.py` works on its own:
    1. A Node 18+ already on this machine (system) or one we installed earlier (data/runtime/node)
    2. Download the official build from nodejs.org into data/runtime/node  (no sudo, no system changes)
    3. The system package manager (apt / dnf / pacman / brew / winget). May ask for the sudo password.

Set CHESSRL_NODE_MIRROR to use another mirror of https://nodejs.org/dist (offline lab, slow link)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .. import paths
from . import ui

MIN_MAJOR = 18                                   # Vite 5 needs Node 18+
DIST = (os.environ.get("CHESSRL_NODE_MIRROR") or "https://nodejs.org/dist").rstrip("/")
LOCAL = paths.RUNTIME_DIR / "node"


def _layout(root: Path):
    if os.name == "nt":
        return root, root / "node.exe", root / "npm.cmd"
    return root / "bin", root / "bin" / "node", root / "bin" / "npm"


def _major(node_exe, env=None) -> int:
    try:
        out = subprocess.run([str(node_exe), "-v"], capture_output=True, text=True,
                             timeout=20, env=env).stdout
        return int(out.strip().lstrip("v").split(".")[0])
    except Exception:
        return 0


def find() -> dict | None:
    """A usable Node 18+ as {"npm", "env", "source", "version"}, or None."""
    cands = []
    bindir, node_exe, npm = _layout(LOCAL)
    if node_exe.exists() and npm.exists():
        cands.append(("local", node_exe, npm, bindir))
    n, p = shutil.which("node"), shutil.which("npm")
    if n and p:
        cands.append(("system", Path(n), Path(p), Path(n).parent))
    if os.name == "nt":                          # winget installs here, PATH is not refreshed yet
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            if base:
                b, ne, npm_ = _layout(Path(base) / "nodejs")
                if ne.exists() and npm_.exists():
                    cands.append(("system", ne, npm_, b))
    for source, node_exe, npm, bindir in cands:
        env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", "")}
        major = _major(node_exe, env)
        if major >= MIN_MAJOR:
            return {"npm": str(npm), "env": env, "source": source, "version": major}
    return None


def _spec() -> dict | None:
    sysname = platform.system().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine().lower())
    if not arch:
        return None
    if sysname == "linux":
        return {"index": f"linux-{arch}", "archive": f"linux-{arch}.tar.gz"}
    if sysname == "darwin":
        return {"index": f"osx-{arch}-tar", "archive": f"darwin-{arch}.tar.gz"}
    if sysname == "windows":
        return {"index": f"win-{arch}-zip", "archive": f"win-{arch}.zip"}
    return None


def pick_release(index: list[dict], key: str, min_major: int = MIN_MAJOR) -> str | None:
    """Newest LTS release that has a build for `key`. index.json is sorted newest first."""
    for e in index:
        try:
            major = int(e["version"].lstrip("v").split(".")[0])
        except (KeyError, ValueError):
            continue
        if e.get("lts") and key in e.get("files", []) and major >= min_major:
            return e["version"]
    return None


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "chess-rl"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, dest: Path, label: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "chess-rl"})
    tty = sys.stdout.isatty()
    if not tty:
        ui.info(f"{label} ...")
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total, done, last = int(r.headers.get("Content-Length") or 0), 0, -1
        while True:
            chunk = r.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if tty and total and done * 100 // total != last:
                last = done * 100 // total
                sys.stdout.write(f"\r\033[2K  • {label} ... {last}%")
                sys.stdout.flush()
    if tty:
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        return
    with tarfile.open(archive) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")
        else:
            root = dest.resolve()
            for m in t.getmembers():
                target = (dest / m.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"unsafe path in archive: {m.name}")
            t.extractall(dest)


def install_portable() -> str:
    """Download the newest LTS into data/runtime/node. Returns its version. Raises on any problem."""
    spec = _spec()
    if not spec:
        raise RuntimeError(f"nodejs.org has no build for {platform.system()} {platform.machine()}")
    index = json.loads(_get(f"{DIST}/index.json"))
    version = pick_release(index, spec["index"])
    if not version:
        raise RuntimeError("no suitable Node.js release found")
    name = f"node-{version}-{spec['archive']}"
    sums = _get(f"{DIST}/{version}/SHASUMS256.txt").decode("utf-8", "replace")
    expected = next((ln.split()[0] for ln in sums.splitlines()
                     if len(ln.split()) == 2 and ln.split()[1] == name), None)
    if not expected:
        raise RuntimeError(f"no checksum published for {name}")
    paths.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="node-", dir=paths.RUNTIME_DIR))
    try:
        archive = tmp / name
        _download(f"{DIST}/{version}/{name}", archive, f"Downloading Node.js {version}")
        if _sha256(archive) != expected:
            raise RuntimeError("the download is corrupt (checksum mismatch)")
        with ui.spinner("Unpacking Node.js"):
            out = tmp / "unpacked"
            _extract(archive, out)
            tops = [p for p in out.iterdir() if p.is_dir()]
            if len(tops) != 1:
                raise RuntimeError("unexpected archive layout")
            if LOCAL.exists():
                shutil.rmtree(LOCAL)
            shutil.move(str(tops[0]), str(LOCAL))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return version


def _system_steps() -> list[list[str]]:
    sysname = platform.system()
    if sysname == "Darwin" and shutil.which("brew"):
        return [["brew", "install", "node"]]
    if sysname == "Windows" and shutil.which("winget"):
        return [["winget", "install", "-e", "--id", "OpenJS.NodeJS.LTS",
                 "--accept-source-agreements", "--accept-package-agreements"]]
    if sysname != "Linux":
        return []
    root = hasattr(os, "geteuid") and os.geteuid() == 0
    if root:
        sudo = []
    elif shutil.which("sudo"):
        sudo = ["sudo"]
    else:
        return []
    if shutil.which("apt-get"):
        return [sudo + ["apt-get", "install", "-y", "ca-certificates", "curl"],
                sudo + ["bash", "-c", "curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -"],
                sudo + ["apt-get", "install", "-y", "nodejs"]]
    if shutil.which("dnf"):
        return [sudo + ["bash", "-c", "curl -fsSL https://rpm.nodesource.com/setup_lts.x | bash -"],
                sudo + ["dnf", "install", "-y", "nodejs"]]
    if shutil.which("pacman"):
        return [sudo + ["pacman", "-S", "--noconfirm", "nodejs", "npm"]]
    return []


def install_system() -> bool:
    """Run the package manager attached to the terminal, so a sudo password prompt works."""
    steps = _system_steps()
    if not steps:
        return False
    for cmd in steps:
        ui.info("$ " + " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            return False
    return True


def ensure(auto: bool = False) -> dict | None:
    """Return a usable Node (see find()), installing one if needed. None if everything failed."""
    found = find()
    if found:
        return found
    ui.info("Node.js 18+ is needed once, to build the website. "
            f"Installing a private copy in {LOCAL} (no sudo, nothing else on this machine changes) ...")
    try:
        version = install_portable()
        found = find()
        if found:
            ui.ok(f"Node.js {version} installed")
            return found
        ui.warn("Node.js was unpacked but does not run on this machine.")
    except Exception as e:
        ui.warn(f"Could not download Node.js: {e}")

    if not _system_steps():
        return None
    if auto or not sys.stdin.isatty():
        ui.info("Skipping the system package manager (non-interactive). Install Node 18+ yourself and run again.")
        return None
    try:
        ans = input("  › Install Node.js with the system package manager instead? "
                    "It may ask for your sudo password. [Y/n]: ").strip().lower()
    except EOFError:
        return None
    if ans not in ("", "y", "yes"):
        return None
    if install_system():
        found = find()
        if found:
            ui.ok(f"Node.js {found['version']} installed")
            return found
    ui.warn("The system install did not give a working Node.js.")
    return None
