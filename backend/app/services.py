"""Game, model and arena logic of the gateway (no HTTP in here)."""
from __future__ import annotations

import asyncio
import time
import io
import os
import random
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field

import chess
import chess.pgn
from fastapi import HTTPException

from .. import paths
from .. import settings as app_settings
from ..rl.arch import architecture
from ..rl.arena import EloRating
from ..rl.config import PopulationConfig, explorable_values
from ..rl.env import game_result
from ..rl.population import mutate_hyperparams, parents_for, rank_agents, select, total_loss
from .state import S

RESULT = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}


# ------------------------------------------------------------------ models
async def create_model(name: str, blocks: int, filters: int, parent: dict | None = None) -> dict:
    path = paths.CHECKPOINTS_DIR / f"model_{uuid.uuid4().hex[:10]}.pt"
    if parent:
        await asyncio.to_thread(shutil.copyfile, parent["path"], path)
    else:
        await S.play.init_model(str(path), blocks, filters)
    try:
        return S.db.create_model(name, {"blocks": blocks, "filters": filters}, path,
                                 parent_id=parent["id"] if parent else None,
                                 elo=parent["elo"] if parent else 1000.0)
    except sqlite3.IntegrityError:
        path.unlink(missing_ok=True)
        raise HTTPException(409, f"a model named '{name}' already exists")


def population_model_map() -> dict[int, int]:
    """model id -> id of the population it is an agent of (any generation). Such models are
    trained from the Population tab, not as single models."""
    out: dict[int, int] = {}
    for p in S.db.list_populations():
        for mem in S.db.list_members(p["id"]):
            if mem["model_id"] is not None:
                out[mem["model_id"]] = p["id"]
    return out


def model_view(m: dict, pops: dict | None = None) -> dict:
    a = architecture(m["config"]["blocks"], m["config"]["filters"])
    pops = population_model_map() if pops is None else pops
    return {**{k: m[k] for k in ("id", "name", "elo", "iterations", "parent_id", "created_at")},
            "population_id": pops.get(m["id"]),
            "blocks": m["config"]["blocks"], "filters": m["config"]["filters"],
            "params": a["params"], "size_mb": a["size_mb"],
            "file_ok": os.path.exists(m["path"])}


def need_model(mid: int) -> dict:
    m = S.db.get_model(mid)
    if not m:
        raise HTTPException(404, f"model {mid} not found")
    return m


def delete_model(m: dict) -> None:
    if S.db.active_run_for_model(m["id"]):
        raise HTTPException(409, "this model is being trained; stop the training first")
    S.db.delete_model(m["id"])
    for suffix in ("", ".cand", ".buffer.npz", ".tmp"):
        try:
            os.remove(m["path"] + suffix)
        except OSError:
            pass


# ----------------------------------------------------------------- capacity
async def capacity_estimate(runs: int, model: dict | None = None) -> dict:
    """RAM / VRAM that `runs` runs at the same moment need, against what is free now."""
    from .. import capacity, hardware
    try:
        hw = await asyncio.to_thread(lambda: hardware.detect_hardware().to_dict())
    except Exception:
        hw = None
    net = (model or {}).get("config") or S.selection.get("network") or {}
    return capacity.estimate(runs, S.selection, hw, int(net.get("blocks", 3)), int(net.get("filters", 64)))


async def need_capacity(runs: int, model: dict | None, force: bool) -> None:
    """A request above the machine's capacity is answered with 409 and the numbers, once.
    The website shows them in a warning window; "continue anyway" sends the same request with
    force=true. Nothing is printed in the terminal."""
    if force:
        return
    est = await capacity_estimate(runs, model)
    if est["level"] in ("warn", "severe"):
        raise HTTPException(409, {"code": "capacity", **est})


# -------------------------------------------------------------- populations
MAX_POPULATION_NAME = 50                  # an agent is named "<population> A01": stays under 60


def agent_names(population: str, size: int) -> list[str]:
    width = max(2, len(str(size)))
    return [f"{population} A{i:0{width}d}" for i in range(1, size + 1)]


def _copy_buffer(src_path: str, dst_path: str) -> bool:
    """Copy the replay buffer that belongs to a model file. False when there is none."""
    src = src_path + ".buffer.npz"
    if not os.path.exists(src):
        return False
    tmp = dst_path + ".buffer.npz.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst_path + ".buffer.npz")
    return True


async def create_population(name: str, base: dict, overrides: dict | None = None) -> dict:
    """Make a population: one copy of `base` per agent (each with its own weights file and,
    when the settings say so, its own copy of the replay buffer), all registered as the
    generation-0 members of a new population. All or nothing: if anything fails, the agents
    and the population made so far are removed again.

    `overrides` replaces values of the saved `population` settings (e.g. {"size": 4})."""
    name = name.strip()
    if not 1 <= len(name) <= MAX_POPULATION_NAME:
        raise HTTPException(400, f"the population name must be 1 to {MAX_POPULATION_NAME} characters")
    overrides = overrides or {}
    try:
        app_settings.validate_settings({"population": overrides})
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not os.path.exists(base["path"]):
        raise HTTPException(409, f"the file of model '{base['name']}' is missing")
    merged = app_settings.merged_settings(S.db.get_settings())
    merged["population"].update(overrides)
    pc = PopulationConfig.from_dict(merged["population"])

    if any(p["name"] == name for p in S.db.list_populations()):
        raise HTTPException(409, f"a population named '{name}' already exists")
    names = agent_names(name, pc.size)
    taken = {m["name"] for m in S.db.list_models()}
    clash = next((n for n in names if n in taken), None)
    if clash:
        raise HTTPException(409, f"a model named '{clash}' already exists")

    pop = S.db.create_population(name, pc.to_dict(), base_model_id=base["id"])
    start = explorable_values(merged)               # every seed trains with the saved settings
    agents, members = [], []
    try:
        for n in names:
            m = await create_model(n, base["config"]["blocks"], base["config"]["filters"], parent=base)
            agents.append(m)
            if pc.copy_buffer:
                await asyncio.to_thread(_copy_buffer, base["path"], m["path"])
            members.append(S.db.add_member(pop["id"], 0, m["id"], m["name"], role="seed",
                                           parent_model_id=base["id"], hyperparams=start))
    except BaseException:
        for m in agents:
            delete_model(m)                         # the row and every file of the agent
        S.db.delete_population(pop["id"])
        raise
    return {"population": pop, "members": members}


def latest_loss(model_id: int) -> float | None:
    """The training loss of an agent: policy loss + value loss of the last iteration it
    trained (newest run that has any metric). None when it never trained. Inside a
    population this is the loss of the generation that was just played."""
    for run in S.db.list_runs(model_id, limit=10):          # newest run first
        rows = S.db.list_metrics(run["id"])
        if rows:
            return total_loss(rows[-1]["policy_loss"], rows[-1]["value_loss"])
    return None


async def _play_vs_champion(agent_path: str, champ_path: str, agent_white: bool, cfg: dict,
                            search=None) -> float:
    """One game, agent against champion, no database rows and no live view. Returns what
    the AGENT scored: 1 win, 0.5 draw, 0 loss. `search` is how a move is asked for
    (default: the play worker, like every other game of the gateway)."""
    search = search or _search
    board = chess.Board()
    while True:
        r = game_result(board, len(board.move_stack), cfg["max_plies"])
        if r is not None:
            break
        white_to_move = board.turn == chess.WHITE
        path = agent_path if white_to_move == agent_white else champ_path
        res = await search(board.fen(), path, cfg, ply=len(board.move_stack))
        mv = chess.Move.from_uci(res["move"])
        if mv not in board.legal_moves:
            raise HTTPException(502, "engine returned an illegal move")
        board.push(mv)
    if r == 0:
        return 0.5
    return 1.0 if (r == 1) == agent_white else 0.0


async def score_generation(population_id: int, generation: int, champion: dict,
                           games: int | None = None, simulations: int | None = None,
                           concurrent: int = 8, play=None, on_progress=None) -> list[dict]:
    """Score and rank every agent of one generation.

    Each agent plays `games` games against `champion` (a model row; it stays the same for
    the whole generation, so the scores can be compared). Colours alternate. The score is
    the share of points; ties are broken by the lower training loss (see rl.population).
    Score, loss and rank are saved on the member rows, in one go at the end: a cancelled or
    failed scoring leaves the rows untouched.

    `games` and `simulations` default to the saved arena settings. Members whose model was
    deleted are skipped (no score, no rank). An agent that IS the champion gets 0.5 without
    playing. Returns the members of the generation, ranked first.

    `play(agent_path, champ_path, agent_white, cfg) -> score` replaces the real game (tests).
    `on_progress(done, total)` is called after every finished game."""
    pop = S.db.get_population(population_id)
    if not pop:
        raise HTTPException(404, "population not found")
    merged = app_settings.merged_settings(S.db.get_settings())
    games = int(games if games is not None else merged["train"]["arena_games"])
    sims = int(simulations if simulations is not None else merged["train"]["arena_simulations"])
    if games < 1:
        raise HTTPException(422, "games must be at least 1")
    members = [m for m in S.db.list_members(population_id, generation) if m["model_id"] is not None]
    if not members:
        return []
    if not os.path.exists(champion["path"]):
        raise HTTPException(409, f"the file of the champion '{champion['name']}' is missing")
    agents = {}
    for mem in members:
        m = S.db.get_model(mem["model_id"])
        if not m or not os.path.exists(m["path"]):
            raise HTTPException(409, f"the file of agent '{mem['model_name']}' is missing")
        agents[mem["id"]] = m

    cfg = {**merged["mcts"], "simulations": sims, "dirichlet_eps": 0.0, "temperature_moves": 8}
    play = play or (lambda a, c, white, cf: _play_vs_champion(a, c, white, cf))
    gate = asyncio.Semaphore(max(1, int(concurrent)))
    todo = [(mem["id"], i) for mem in members if agents[mem["id"]]["id"] != champion["id"]
            for i in range(games)]
    done = 0
    points: dict[int, float] = {}

    async def one(mem_id: int, i: int) -> None:
        nonlocal done
        async with gate:
            sc = await play(agents[mem_id]["path"], champion["path"], i % 2 == 0, cfg)
        points[mem_id] = points.get(mem_id, 0.0) + sc
        done += 1
        if on_progress:
            on_progress(done, len(todo))

    tasks = [asyncio.ensure_future(one(a, i)) for a, i in todo]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:                                  # one failed or we were cancelled:
            t.cancel()                                   # no game may go on in the background
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    results = []
    for mem in members:
        is_champ = agents[mem["id"]]["id"] == champion["id"]
        results.append({"key": mem["id"],
                        "score": 0.5 if is_champ else points[mem["id"]] / games,
                        "loss": latest_loss(mem["model_id"])})
    for r in rank_agents(results):
        S.db.set_member_result(r["key"], r["score"], r["loss"], r["rank"])
    return S.db.list_members(population_id, generation)


async def make_next_generation(population_id: int, generation: int, rng=None) -> dict:
    """Exploit + explore: build generation `generation + 1` from the scored `generation`.

    The strongest agents stay (same model, same weights, same settings, role "survivor").
    For every other place a child is made from a survivor (survivors take turns, best first):
    a NEW model with a copy of the survivor's weights (and of its replay buffer when the
    population's copy_buffer is on), the survivor's iteration count and Elo, the survivor as
    parent_id, and settings nudged by mutation_strength and cut to the website limits.
    The new generation always has `size` agents, even if an agent of the old one was deleted.

    Nothing of the old generation is touched or deleted, and the population row is not
    changed (generation number, phase and the frozen champion are the coordinator's job).
    All or nothing: if anything fails, the children made so far are removed again.
    Raises 409 if the generation is not scored yet, a survivor's file is missing, or the
    next generation already exists."""
    pop = S.db.get_population(population_id)
    if not pop:
        raise HTTPException(404, "population not found")
    pc = PopulationConfig.from_dict(pop["config"])
    nxt = generation + 1
    if S.db.list_members(population_id, nxt):
        raise HTTPException(409, f"generation {nxt} of '{pop['name']}' already exists")
    scored = [m for m in S.db.list_members(population_id, generation)
              if m["model_id"] is not None and m["rank"] is not None]
    if not scored:
        raise HTTPException(409, f"generation {generation} of '{pop['name']}' is not scored yet")
    stay = select(scored, pc.survivors)["survivors"]
    parents = {}
    for mem in stay:
        m = S.db.get_model(mem["model_id"])
        if not m or not os.path.exists(m["path"]):
            raise HTTPException(409, f"the file of agent '{mem['model_name']}' is missing")
        parents[mem["id"]] = m
    rng = rng or random.Random()
    n_children = max(0, pc.size - len(stay))
    width = max(2, len(str(pc.size)))
    made, rows = [], []
    try:
        for i, (_, mem) in enumerate(parents_for(list(range(n_children)), stay), start=1):
            par = parents[mem["id"]]
            name = f"{pop['name']} G{nxt:02d} C{i:0{width}d}"
            child = await create_model(name, par["config"]["blocks"], par["config"]["filters"], parent=par)
            made.append(child)
            child = S.db.update_model(child["id"], iterations=par["iterations"])
            if pc.copy_buffer:
                await asyncio.to_thread(_copy_buffer, par["path"], child["path"])
            hp = mutate_hyperparams(mem["hyperparams"], pc.mutation_strength, app_settings.LIMITS, rng)
            rows.append((child, par, hp))
        out = [S.db.add_member(population_id, nxt, parents[m["id"]]["id"], m["model_name"],
                               role="survivor", parent_model_id=m["parent_model_id"],
                               hyperparams=m["hyperparams"]) for m in stay]
        out += [S.db.add_member(population_id, nxt, c["id"], c["name"], role="child",
                                parent_model_id=par["id"], hyperparams=hp) for c, par, hp in rows]
    except BaseException:
        S.db.delete_members_of_generation(population_id, nxt)
        for m in made:
            delete_model(m)
        raise
    return {"generation": nxt, "members": out, "survivors": len(stay), "children": len(rows)}


# ------------------------------------------------------------------ games
@dataclass
class LiveGame:
    id: int
    white: int | None                       # model id, or None = human
    black: int | None
    kind: str
    board: chess.Board = field(default_factory=chess.Board)
    result: str = "*"
    last_eval: float | None = None          # White's view, -1..1
    last_top: list = field(default_factory=list)
    move_secs: list = field(default_factory=list)   # seconds each move took, one per ply
    _t_mark: float = field(default_factory=time.monotonic)

    def mark_move(self) -> None:
        """Call right after a move is pushed: records how long that move took."""
        now = time.monotonic()
        self.move_secs.append(round(now - self._t_mark, 1))
        self._t_mark = now

    @property
    def over(self) -> bool:
        return self.result != "*"

    def mover(self) -> int | None:
        return self.white if self.board.turn == chess.WHITE else self.black

    def human_to_move(self) -> bool:
        return self.mover() is None

    def names(self) -> tuple[str, str]:
        def nm(i):
            m = S.db.get_model(i) if i else None
            return m["name"] if m else "Human"
        return nm(self.white), nm(self.black)

    def pgn(self) -> str:
        g = chess.pgn.Game()
        g.headers["White"], g.headers["Black"] = self.names()
        g.headers["Result"] = self.result
        node = g
        for mv in self.board.move_stack:
            node = node.add_variation(mv)
        out = io.StringIO()
        print(g, file=out, end="\n")
        return out.getvalue()

    def state(self) -> dict:
        san, b = [], chess.Board()
        for mv in self.board.move_stack:
            san.append(b.san(mv))
            b.push(mv)
        legal = [m.uci() for m in self.board.legal_moves] if not self.over else []
        return {"game_id": self.id, "fen": self.board.fen(),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "white": self.white, "black": self.black,
                "moves": [m.uci() for m in self.board.move_stack], "san": san,
                "last_move": self.board.move_stack[-1].uci() if self.board.move_stack else None,
                "check": self.board.is_check(), "legal_moves": legal,
                "human_to_move": self.human_to_move() and not self.over,
                "over": self.over, "result": self.result,
                "eval": self.last_eval, "top": self.last_top,
                "move_secs": list(self.move_secs)}


def need_game(gid: int) -> LiveGame:
    g = S.games.get(gid)
    if not g:
        raise HTTPException(404, "game not found (it may belong to an earlier session)")
    return g


def _finish(g: LiveGame, result: str) -> None:
    g.result = result
    S.db.update_game(g.id, g.pgn(), result)


def _check_over(g: LiveGame, max_plies: int) -> None:
    r = game_result(g.board, len(g.board.move_stack), max_plies)
    if r is not None:
        _finish(g, RESULT[r])


async def _search(fen: str, path: str, cfg: dict, ply: int = 0) -> dict:
    """Ask the play worker for a move. While the shared inference server runs, the worker
    uses it, so Play, hints and Arena matches never wait for training (or for each other)."""
    return await S.play.search(fen, path, cfg, ply=ply, inference=S.inference_info())


async def _engine_reply(g: LiveGame) -> None:
    model = need_model(g.mover())
    cfg = S.mcts_settings()
    res = await _search(g.board.fen(), model["path"], cfg, ply=len(g.board.move_stack))
    mv = chess.Move.from_uci(res["move"])
    if mv not in g.board.legal_moves:
        raise HTTPException(502, "engine returned an illegal move")
    to_move_white = g.board.turn == chess.WHITE
    g.last_eval = res["value"] if to_move_white else -res["value"]
    g.last_top = res["top"]
    g.board.push(mv)
    g.mark_move()
    _check_over(g, cfg["max_plies"])
    S.db.update_game(g.id, g.pgn(), g.result)


async def new_game(white: int | None, black: int | None, kind: str = "play") -> LiveGame:
    for i in (white, black):
        if i is not None:
            need_model(i)
    row = S.db.create_game(white, black, kind)
    g = LiveGame(row["id"], white, black, kind)
    S.games[g.id] = g
    if not g.human_to_move():
        await _engine_reply(g)
    return g


async def human_move(g: LiveGame, uci: str) -> LiveGame:
    if g.over:
        raise HTTPException(409, "the game is over")
    if not g.human_to_move():
        raise HTTPException(409, "it is not your turn")
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        raise HTTPException(400, "bad move format")
    if mv not in g.board.legal_moves:
        raise HTTPException(400, "illegal move")
    g.board.push(mv)
    g.mark_move()
    g.last_top = []
    _check_over(g, S.mcts_settings()["max_plies"])
    if not g.over and not g.human_to_move():
        await _engine_reply(g)
    else:
        S.db.update_game(g.id, g.pgn(), g.result)
    return g


def undo(g: LiveGame) -> LiveGame:
    if not g.board.move_stack:
        raise HTTPException(409, "nothing to undo")
    g.board.pop()
    while g.board.move_stack and not g.human_to_move():
        g.board.pop()
    del g.move_secs[len(g.board.move_stack):]
    g._t_mark = time.monotonic()
    g.result, g.last_eval, g.last_top = "*", None, []
    S.db.update_game(g.id, g.pgn(), "*")
    return g


def resign(g: LiveGame) -> LiveGame:
    if g.over:
        raise HTTPException(409, "the game is over")
    side = g.board.turn
    if g.white is None and g.black is not None:
        side = chess.WHITE
    elif g.black is None and g.white is not None:
        side = chess.BLACK
    _finish(g, "0-1" if side == chess.WHITE else "1-0")
    return g


async def hint(g: LiveGame) -> dict:
    """Best move for the side to move, according to the model chosen for this game."""
    mid = g.white if g.white is not None else g.black
    if mid is None:
        raise HTTPException(409, "this game has no model to ask")
    cfg = S.mcts_settings(temperature=0.0)
    cfg["simulations"] = max(50, cfg["simulations"] // 2)
    res = await _search(g.board.fen(), need_model(mid)["path"], cfg)
    white = res["value"] if g.board.turn == chess.WHITE else -res["value"]
    return {"move": res["move"], "eval": white, "top": res["top"]}


# ------------------------------------------------------------------- arena
async def run_match(match_id: int, a: dict, b: dict, n_games: int, sims: int) -> None:
    """Model A and model B play n_games, alternating colours, one move at a time so the
    website can show every move live. In arena_matches, white_wins / black_wins count the
    wins of model A (white_id) / model B (black_id), whatever colour they had."""
    ch = f"match:{match_id}"
    wins = {"a": 0, "b": 0, "d": 0}
    elo = {"a": a["elo"], "b": b["elo"]}
    try:
        for i in range(n_games):
            a_white = i % 2 == 0
            white, black = (a, b) if a_white else (b, a)
            row = S.db.create_game(white["id"], black["id"], "arena", match_id=match_id)
            g = LiveGame(row["id"], white["id"], black["id"], "arena")
            S.games[g.id] = g
            cfg = S.mcts_settings(simulations=sims)
            S.bus.publish(ch, {"type": "game_start", "match_id": match_id, "game_id": g.id,
                               "index": i, "white": white["name"], "black": black["name"]})
            while not g.over:
                mover = white if g.board.turn == chess.WHITE else black
                res = await _search(g.board.fen(), mover["path"], cfg,
                                          ply=len(g.board.move_stack))
                mv = chess.Move.from_uci(res["move"])
                v = res["value"] if g.board.turn == chess.WHITE else -res["value"]
                g.board.push(mv)
                g.mark_move()
                g.last_eval, g.last_top = v, res["top"]
                _check_over(g, cfg["max_plies"])
                S.bus.publish(f"game:{g.id}", {"type": "move", **g.state()})
                S.bus.publish(ch, {"type": "move", **g.state(), "index": i})
                await asyncio.sleep(S.match_delay.get(match_id, 0.4))
            S.db.update_game(g.id, g.pgn(), g.result)
            score_a = {"1-0": 1.0 if a_white else 0.0, "0-1": 0.0 if a_white else 1.0}.get(
                g.result, 0.5)
            wins["a" if score_a == 1 else "b" if score_a == 0 else "d"] += 1
            elo["a"], elo["b"] = EloRating.update(elo["a"], elo["b"], score_a, k=16.0)
            S.db.update_match(match_id, wins["a"], wins["b"], wins["d"])
            S.bus.publish(ch, {"type": "game_end", "match_id": match_id, "index": i,
                               "result": g.result, "a_wins": wins["a"], "b_wins": wins["b"],
                               "draws": wins["d"]})
        S.db.update_model(a["id"], elo=round(elo["a"], 1))
        S.db.update_model(b["id"], elo=round(elo["b"], 1))
        S.bus.publish(ch, {"type": "match_end", "match_id": match_id, "a_wins": wins["a"],
                           "b_wins": wins["b"], "draws": wins["d"],
                           "elo_a": round(elo["a"], 1), "elo_b": round(elo["b"], 1)})
    except asyncio.CancelledError:
        S.bus.publish(ch, {"type": "match_cancelled", "match_id": match_id})
        raise
    except Exception as e:
        S.bus.publish(ch, {"type": "match_error", "match_id": match_id, "error": str(e)})
    finally:
        S.matches.pop(match_id, None)
        S.match_delay.pop(match_id, None)
