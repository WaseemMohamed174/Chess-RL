"""Every path in the project, in one place."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("CHESSRL_DATA_DIR") or BASE_DIR / "data")
RUNTIME_DIR = DATA_DIR / "runtime"
VENV_DIR = RUNTIME_DIR / "venv"
LOGS_DIR = RUNTIME_DIR / "logs"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
SELECTION_FILE = RUNTIME_DIR / "selection.json"
DB_PATH = RUNTIME_DIR / "chess_rl.db"

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"


def venv_python(name: str) -> Path:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return VENV_DIR / name / sub


def ensure_runtime_dirs() -> None:
    for d in (RUNTIME_DIR, VENV_DIR, LOGS_DIR, CHECKPOINTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
