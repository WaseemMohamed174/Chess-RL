from __future__ import annotations

import os
import re
import tempfile
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from ...rl.arch import architecture
from .. import services
from ..state import S

router = APIRouter(prefix="/api/models", tags=["models"])

SIZES = {"tiny": (3, 64), "small": (6, 128), "medium": (10, 192)}


class NewModel(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    size: str = "tiny"                     # tiny | small | medium | custom
    blocks: int | None = Field(default=None, ge=1, le=40)
    filters: int | None = Field(default=None, ge=8, le=512)


class Rename(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class Bulk(BaseModel):
    ids: list[int] | None = None          # None = every model


@router.get("")
def list_models():
    pops = services.population_model_map()
    return [services.model_view(m, pops) for m in S.db.list_models()]


@router.post("", status_code=201)
async def create(body: NewModel):
    if body.size == "custom":
        if not body.blocks or not body.filters:
            raise HTTPException(400, "custom size needs blocks and filters")
        blocks, filters = body.blocks, body.filters
    elif body.size in SIZES:
        blocks, filters = SIZES[body.size]
    else:
        raise HTTPException(400, "size must be tiny, small, medium or custom")
    return services.model_view(await services.create_model(body.name.strip(), blocks, filters))


def _pick(ids: list[int] | None) -> list[dict]:
    """The models a bulk request is about: the listed ones, or all of them."""
    models = S.db.list_models()
    if ids is None:
        return models
    wanted = set(ids)
    return [m for m in models if m["id"] in wanted]


@router.post("/bulk-delete")
def bulk_delete(body: Bulk):
    """Delete many models at once. A model that is training right now is skipped (not an
    error for the whole request), so the answer lists what was deleted and what was not."""
    deleted, skipped = [], []
    for m in _pick(body.ids):
        try:
            services.delete_model(m)
            deleted.append({"id": m["id"], "name": m["name"]})
        except HTTPException as e:
            skipped.append({"id": m["id"], "name": m["name"], "reason": str(e.detail)})
    return {"deleted": deleted, "skipped": skipped}


@router.get("/download-all")
def download_all(ids: str | None = None):
    """Every model file in one .zip (`ids` = comma separated list to limit it)."""
    want = None
    if ids:
        try:
            want = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "ids must be numbers separated by commas")
    files = [m for m in _pick(want) if os.path.exists(m["path"])]
    if not files:
        raise HTTPException(404, "there are no model files to download")
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    used: set[str] = set()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:        # .pt files do not compress
        for m in files:
            base = re.sub(r'[\\/:*?"<>|]+', "_", m["name"]).strip() or f"model-{m['id']}"
            name = base
            if name.lower() in used:                                  # names are unique, but be safe
                name = f"{base}-{m['id']}"
            used.add(name.lower())
            z.write(m["path"], f"{name}.pt")
    return FileResponse(tmp, media_type="application/zip", filename="models.zip",
                        background=BackgroundTask(os.remove, tmp))


@router.get("/{mid}")
def get_one(mid: int):
    return services.model_view(services.need_model(mid))


@router.get("/{mid}/architecture")
def arch(mid: int):
    m = services.need_model(mid)
    return architecture(m["config"]["blocks"], m["config"]["filters"])


@router.put("/{mid}")
def rename(mid: int, body: Rename):
    services.need_model(mid)
    try:
        return services.model_view(S.db.update_model(mid, name=body.name.strip()))
    except Exception:
        raise HTTPException(409, "that name is already used")


@router.post("/{mid}/clone", status_code=201)
async def clone(mid: int, body: Rename):
    m = services.need_model(mid)
    return services.model_view(await services.create_model(
        body.name.strip(), m["config"]["blocks"], m["config"]["filters"], parent=m))


@router.delete("/{mid}", status_code=204)
def delete(mid: int):
    services.delete_model(services.need_model(mid))


@router.get("/{mid}/download")
def download(mid: int):
    m = services.need_model(mid)
    return FileResponse(m["path"], filename=f"{m['name']}.pt")
