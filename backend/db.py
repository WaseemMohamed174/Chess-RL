"""SQLite storage. Schema = the ERD in docs/ARCHITECTURE.md."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS models(
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, config_json TEXT NOT NULL,
    path TEXT NOT NULL, elo REAL NOT NULL DEFAULT 1000, iterations INTEGER NOT NULL DEFAULT 0,
    parent_id INTEGER REFERENCES models(id) ON DELETE SET NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS training_runs(
    id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    status TEXT NOT NULL, hyperparams_json TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT);
CREATE TABLE IF NOT EXISTS metrics(
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL, policy_loss REAL, value_loss REAL, elo REAL,
    games INTEGER, games_per_min REAL, created_at TEXT NOT NULL,
    buffer INTEGER, promoted INTEGER, arena_wins INTEGER, arena_losses INTEGER,
    arena_draws INTEGER, arena_score REAL);
CREATE TABLE IF NOT EXISTS arena_matches(
    id INTEGER PRIMARY KEY, white_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    black_id INTEGER REFERENCES models(id) ON DELETE SET NULL, n_games INTEGER NOT NULL,
    white_wins INTEGER NOT NULL DEFAULT 0, black_wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS games(
    id INTEGER PRIMARY KEY, white_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    black_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    match_id INTEGER REFERENCES arena_matches(id) ON DELETE SET NULL,
    run_id INTEGER REFERENCES training_runs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL, pgn TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '*',
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS user_settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS populations(
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, config_json TEXT NOT NULL,
    base_model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    generation INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'idle',
    phase TEXT NOT NULL DEFAULT 'idle',
    champion_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS population_members(
    id INTEGER PRIMARY KEY,
    population_id INTEGER NOT NULL REFERENCES populations(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL, model_name TEXT NOT NULL,
    parent_model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    role TEXT NOT NULL DEFAULT 'seed', hyperparams_json TEXT NOT NULL DEFAULT '{}',
    score REAL, loss REAL, rank INTEGER, created_at TEXT NOT NULL,
    start_iterations INTEGER,
    UNIQUE(population_id, generation, model_id));
CREATE INDEX IF NOT EXISTS idx_members_pop ON population_members(population_id, generation);
CREATE INDEX IF NOT EXISTS idx_members_model ON population_members(model_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, iteration);
CREATE INDEX IF NOT EXISTS idx_games_created ON games(created_at);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path):
        self._lock = threading.RLock()
        self._c = sqlite3.connect(str(path), check_same_thread=False)
        self._c.row_factory = sqlite3.Row
        self._c.execute("PRAGMA journal_mode = WAL")
        self._c.executescript(SCHEMA)
        self._migrate()
        self._c.execute("PRAGMA foreign_keys = ON")

    def _migrate(self) -> None:
        """Columns added after a first release: an older database gets them here."""
        cols = {r["name"] for r in self._c.execute("PRAGMA table_info(population_members)")}
        if "start_iterations" not in cols:
            self._c.execute("ALTER TABLE population_members ADD COLUMN start_iterations INTEGER")
            self._c.commit()
        mcols = {r["name"] for r in self._c.execute("PRAGMA table_info(metrics)")}
        for name, kind in (("buffer", "INTEGER"), ("promoted", "INTEGER"), ("arena_wins", "INTEGER"),
                           ("arena_losses", "INTEGER"), ("arena_draws", "INTEGER"),
                           ("arena_score", "REAL")):
            if name not in mcols:
                self._c.execute(f"ALTER TABLE metrics ADD COLUMN {name} {kind}")
        self._c.commit()

    def close(self):
        with self._lock:
            self._c.close()

    def _run(self, sql, args=()):
        with self._lock:
            cur = self._c.execute(sql, args)
            self._c.commit()
            return cur

    def _all(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._c.execute(sql, args).fetchall()]

    def _one(self, sql, args=()):
        rows = self._all(sql, args)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- models
    @staticmethod
    def _model(r):
        if r:
            r["config"] = json.loads(r.pop("config_json"))
        return r

    def create_model(self, name, config, path, parent_id=None, elo=1000.0) -> dict:
        cur = self._run("INSERT INTO models(name,config_json,path,elo,parent_id,created_at)"
                        " VALUES(?,?,?,?,?,?)",
                        (name, json.dumps(config), str(path), elo, parent_id, now()))
        return self.get_model(cur.lastrowid)

    def get_model(self, mid):
        return self._model(self._one("SELECT * FROM models WHERE id=?", (mid,)))

    def list_models(self):
        return [self._model(r) for r in self._all("SELECT * FROM models ORDER BY id")]

    def update_model(self, mid, **fields):
        allowed = {"name", "path", "elo", "iterations"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if fields:
            sets = ",".join(f"{k}=?" for k in fields)
            self._run(f"UPDATE models SET {sets} WHERE id=?", (*fields.values(), mid))
        return self.get_model(mid)

    def delete_model(self, mid):
        self._run("DELETE FROM models WHERE id=?", (mid,))

    # ------------------------------------------------------------------ runs
    def create_run(self, model_id, hyperparams) -> dict:
        cur = self._run("INSERT INTO training_runs(model_id,status,hyperparams_json,started_at)"
                        " VALUES(?,?,?,?)", (model_id, "running", json.dumps(hyperparams), now()))
        return self.get_run(cur.lastrowid)

    def get_run(self, rid):
        r = self._one("SELECT * FROM training_runs WHERE id=?", (rid,))
        if r:
            r["hyperparams"] = json.loads(r.pop("hyperparams_json"))
        return r

    def tag_run(self, rid, meta: dict) -> None:
        """Remember who started a run (e.g. {"population_id": 3}) inside its saved settings, so
        the Training page can show it with its population and not among the single models."""
        r = self.get_run(rid)
        if not r:
            return
        hp = {**r["hyperparams"], "meta": {**r["hyperparams"].get("meta", {}), **meta}}
        self._run("UPDATE training_runs SET hyperparams_json=? WHERE id=?", (json.dumps(hp), rid))

    def set_run_status(self, rid, status):
        ended = now() if status in ("stopped", "done", "failed") else None
        self._run("UPDATE training_runs SET status=?, ended_at=COALESCE(?, ended_at) WHERE id=?",
                  (status, ended, rid))

    def active_run(self):
        r = self._one("SELECT id FROM training_runs WHERE status IN ('running','paused')"
                      " ORDER BY id DESC LIMIT 1")
        return self.get_run(r["id"]) if r else None

    def active_runs(self):
        rows = self._all("SELECT id FROM training_runs WHERE status IN ('running','paused')"
                         " ORDER BY id")
        return [self.get_run(r["id"]) for r in rows]

    def active_run_for_model(self, model_id):
        r = self._one("SELECT id FROM training_runs WHERE model_id=? AND status IN"
                      " ('running','paused') ORDER BY id DESC LIMIT 1", (model_id,))
        return self.get_run(r["id"]) if r else None

    def list_runs(self, model_id=None, limit=50):
        if model_id is None:
            rows = self._all("SELECT id FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self._all("SELECT id FROM training_runs WHERE model_id=? ORDER BY id DESC"
                             " LIMIT ?", (model_id, limit))
        return [self.get_run(r["id"]) for r in rows]

    def add_metric(self, run_id, m: dict):
        a = m.get("arena") or {}
        promoted = m.get("promoted")
        self._run("INSERT INTO metrics(run_id,iteration,policy_loss,value_loss,elo,games,"
                  "games_per_min,created_at,buffer,promoted,arena_wins,arena_losses,arena_draws,"
                  "arena_score) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (run_id, m["iteration"], m.get("policy_loss"), m.get("value_loss"),
                   m.get("elo"), m.get("games"), m.get("games_per_min"), now(), m.get("buffer"),
                   None if promoted is None else int(bool(promoted)),
                   a.get("wins"), a.get("losses"), a.get("draws"), a.get("score")))

    def list_metrics(self, run_id):
        """One row per finished iteration, in the same shape as the live `metrics` event
        (so a stopped run shows the same numbers as a running one)."""
        rows = self._all("SELECT iteration,policy_loss,value_loss,elo,games,games_per_min,buffer,"
                         "promoted,arena_wins,arena_losses,arena_draws,arena_score"
                         " FROM metrics WHERE run_id=? ORDER BY iteration", (run_id,))
        for r in rows:
            w, l, d, sc = (r.pop(k) for k in ("arena_wins", "arena_losses", "arena_draws", "arena_score"))
            r["arena"] = ({"wins": w, "losses": l, "draws": d, "score": sc}
                          if w is not None and l is not None and d is not None else None)
            if r["promoted"] is not None:
                r["promoted"] = bool(r["promoted"])
        return rows

    # ----------------------------------------------------------------- games
    def create_game(self, white_id, black_id, kind, match_id=None, run_id=None) -> dict:
        cur = self._run("INSERT INTO games(white_id,black_id,match_id,run_id,kind,created_at)"
                        " VALUES(?,?,?,?,?,?)", (white_id, black_id, match_id, run_id, kind, now()))
        return self.get_game(cur.lastrowid)

    def get_game(self, gid):
        return self._one("SELECT * FROM games WHERE id=?", (gid,))

    def update_game(self, gid, pgn, result):
        self._run("UPDATE games SET pgn=?, result=? WHERE id=?", (pgn, result, gid))

    def list_games(self, limit=50):
        return self._all("SELECT * FROM games ORDER BY id DESC LIMIT ?", (limit,))

    # ----------------------------------------------------------------- arena
    def create_match(self, white_id, black_id, n_games) -> dict:
        cur = self._run("INSERT INTO arena_matches(white_id,black_id,n_games,created_at)"
                        " VALUES(?,?,?,?)", (white_id, black_id, n_games, now()))
        return self.get_match(cur.lastrowid)

    def get_match(self, mid):
        return self._one("SELECT * FROM arena_matches WHERE id=?", (mid,))

    def update_match(self, mid, white_wins, black_wins, draws):
        self._run("UPDATE arena_matches SET white_wins=?,black_wins=?,draws=? WHERE id=?",
                  (white_wins, black_wins, draws, mid))

    # ----------------------------------------------------------- populations
    # A population is a group of agents (models) that train side by side. Every generation
    # leaves one population_members row per agent: how it entered (role), the settings it
    # trained with, and how it ranked. Deleting a model keeps its rows (model_id becomes NULL,
    # model_name stays), so the history of a population never disappears by accident.
    ROLES = ("seed", "survivor", "child")      # seed: first generation, survivor: kept, child: new copy
    POP_STATUS = ("idle", "running", "paused", "stopped", "failed", "done")
    POP_PHASES = ("idle", "training", "evaluating", "selecting")

    @staticmethod
    def _population(r):
        if r:
            r["config"] = json.loads(r.pop("config_json"))
        return r

    @staticmethod
    def _member(r):
        if r:
            r["hyperparams"] = json.loads(r.pop("hyperparams_json"))
        return r

    def create_population(self, name, config, base_model_id=None) -> dict:
        t = now()
        cur = self._run("INSERT INTO populations(name,config_json,base_model_id,created_at,updated_at)"
                        " VALUES(?,?,?,?,?)", (name, json.dumps(config), base_model_id, t, t))
        return self.get_population(cur.lastrowid)

    def get_population(self, pid):
        return self._population(self._one("SELECT * FROM populations WHERE id=?", (pid,)))

    def list_populations(self):
        return [self._population(r) for r in self._all("SELECT * FROM populations ORDER BY id")]

    def update_population(self, pid, **fields):
        allowed = {"name", "status", "phase", "generation", "champion_id", "config"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if fields.get("status") is not None and fields["status"] not in self.POP_STATUS:
            raise ValueError(f"unknown population status: {fields['status']}")
        if fields.get("phase") is not None and fields["phase"] not in self.POP_PHASES:
            raise ValueError(f"unknown population phase: {fields['phase']}")
        if "config" in fields:
            fields["config_json"] = json.dumps(fields.pop("config"))
        if fields:
            sets = ",".join(f"{k}=?" for k in fields)
            self._run(f"UPDATE populations SET {sets}, updated_at=? WHERE id=?",
                      (*fields.values(), now(), pid))
        return self.get_population(pid)

    def delete_population(self, pid):
        """Removes the population and its member history. The models themselves stay."""
        self._run("DELETE FROM populations WHERE id=?", (pid,))

    def add_member(self, population_id, generation, model_id, model_name, role="seed",
                   parent_model_id=None, hyperparams=None) -> dict:
        if role not in self.ROLES:
            raise ValueError(f"unknown member role: {role}")
        cur = self._run("INSERT INTO population_members(population_id,generation,model_id,model_name,"
                        "parent_model_id,role,hyperparams_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                        (population_id, generation, model_id, model_name, parent_model_id, role,
                         json.dumps(hyperparams or {}), now()))
        return self._member(self._one("SELECT * FROM population_members WHERE id=?", (cur.lastrowid,)))

    def delete_members_of_generation(self, population_id, generation) -> None:
        self._run("DELETE FROM population_members WHERE population_id=? AND generation=?",
                  (population_id, generation))

    def set_member_start(self, member_id, iterations) -> None:
        """Iterations the agent had when its generation began (the coordinator needs it to
        know how many are still missing after a restart)."""
        self._run("UPDATE population_members SET start_iterations=? WHERE id=?",
                  (iterations, member_id))

    def get_member(self, member_id):
        return self._member(self._one("SELECT * FROM population_members WHERE id=?", (member_id,)))

    def set_member_result(self, member_id, score, loss=None, rank=None) -> None:
        self._run("UPDATE population_members SET score=?, loss=?, rank=? WHERE id=?",
                  (score, loss, rank, member_id))

    def list_members(self, population_id, generation=None) -> list:
        """Ranked members first (rank 1 = best), the not yet ranked ones last."""
        sql = "SELECT * FROM population_members WHERE population_id=?"
        args = [population_id]
        if generation is not None:
            sql += " AND generation=?"
            args.append(generation)
        rows = self._all(sql + " ORDER BY generation, rank IS NULL, rank, id", args)
        return [self._member(r) for r in rows]

    def generation_summary(self, population_id) -> list:
        """One row per generation: how many agents, the best and the average score.
        This is the data behind the champion-over-generations chart."""
        return self._all("SELECT generation, COUNT(*) AS agents, MAX(score) AS best_score,"
                         " AVG(score) AS mean_score FROM population_members WHERE population_id=?"
                         " GROUP BY generation ORDER BY generation", (population_id,))

    def population_of_model(self, model_id):
        """The population this model is an agent of right now (its newest membership), or None."""
        r = self._one("SELECT population_id FROM population_members WHERE model_id=?"
                      " ORDER BY generation DESC, id DESC LIMIT 1", (model_id,))
        return self.get_population(r["population_id"]) if r else None

    # -------------------------------------------------------------- settings
    def get_settings(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self._all("SELECT * FROM user_settings")}

    def set_settings(self, sections: dict):
        for k, v in sections.items():
            self._run("INSERT INTO user_settings(key,value) VALUES(?,?) ON CONFLICT(key)"
                      " DO UPDATE SET value=excluded.value", (k, json.dumps(v)))
