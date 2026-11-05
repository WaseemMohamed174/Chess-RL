from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import services
from ..state import S

router = APIRouter(prefix="/api/games", tags=["games"])


class NewGame(BaseModel):
    model_id: int | None = None            # the model you play against (None = hot-seat)
    color: str = "white"                   # YOUR colour: white | black


class Move(BaseModel):
    game_id: int
    move_uci: str


@router.post("/new", status_code=201)
async def new(body: NewGame):
    me_white = body.color != "black"
    white, black = (None, body.model_id) if me_white else (body.model_id, None)
    return (await services.new_game(white, black)).state()


@router.post("/play")
async def play(body: Move):
    g = services.need_game(body.game_id)
    return (await services.human_move(g, body.move_uci)).state()


@router.get("")
def recent(limit: int = 50):
    return S.db.list_games(min(limit, 200))


@router.get("/{gid}")
def state(gid: int):
    return services.need_game(gid).state()


@router.post("/{gid}/undo")
def undo(gid: int):
    return services.undo(services.need_game(gid)).state()


@router.post("/{gid}/resign")
def resign(gid: int):
    return services.resign(services.need_game(gid)).state()


@router.post("/{gid}/hint")
async def hint(gid: int):
    return await services.hint(services.need_game(gid))


@router.get("/{gid}/pgn", response_class=PlainTextResponse)
def pgn(gid: int):
    g = S.games.get(gid)
    if g:
        return g.pgn()
    row = S.db.get_game(gid)
    return row["pgn"] if row else PlainTextResponse("not found", status_code=404)
