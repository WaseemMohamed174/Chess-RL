from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ... import settings as cfg
from ..state import S

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_settings():
    return {"values": cfg.merged_settings(S.db.get_settings()), "limits": cfg.LIMITS,
            "defaults": cfg.default_settings()}


@router.put("")
def put_settings(patch: dict):
    try:
        cfg.validate_settings(patch)
    except ValueError as e:
        raise HTTPException(422, str(e))
    merged = cfg.merged_settings(S.db.get_settings())
    for section, values in patch.items():
        merged[section].update(values)
    S.db.set_settings(merged)
    return {"values": merged}


@router.delete("")
def reset():
    S.db.set_settings(cfg.default_settings())
    return {"values": cfg.default_settings()}
