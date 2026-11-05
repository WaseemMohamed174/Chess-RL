"""Shared by both workers: the secret-token check and the device choice."""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException

from .. import settings


def require_token(x_worker_token: str = Header(default="")) -> None:
    expected = os.environ.get("CHESSRL_TOKEN", "")
    if not expected or not hmac.compare_digest(x_worker_token, expected):
        raise HTTPException(status_code=401, detail="bad worker token")


def fake_engine() -> bool:
    return os.environ.get("CHESSRL_FAKE_ENGINE") == "1"


def torch_device() -> str:
    """'cuda' when run.py selected the GPU and PyTorch really sees it, else 'cpu'."""
    if fake_engine() or settings.load_selection().get("device") != "gpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
