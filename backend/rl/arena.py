"""Model-vs-model matches and Elo."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import MCTSConfig
from .selfplay import GamePool


class EloRating:
    @staticmethod
    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    @staticmethod
    def update(ra: float, rb: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
        ea = EloRating.expected(ra, rb)
        return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))

    @staticmethod
    def diff_from_score(score: float, n_games: int) -> float:
        """Rating difference implied by a match score (clamped so 100% / 0% stay finite)."""
        eps = 0.5 / max(n_games, 1)
        s = min(max(score, eps), 1 - eps)
        return -400.0 * math.log10(1.0 / s - 1.0)


@dataclass
class MatchResult:
    a_wins: int
    b_wins: int
    draws: int
    games: list

    @property
    def n(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    @property
    def score_a(self) -> float:
        return (self.a_wins + 0.5 * self.draws) / max(self.n, 1)


def play_match(eval_a, eval_b, n_games: int, cfg: MCTSConfig, concurrent: int = 16,
               seed: int | None = None, should_stop=None, on_move=None, on_game=None,
               offset: int = 0) -> MatchResult:
    """A plays White in even games and Black in odd ones. No root noise, no tree sharing.
    offset shifts the colour pattern, so a match split over several workers stays balanced."""
    cfg = MCTSConfig.from_dict({**cfg.to_dict(), "dirichlet_eps": 0.0, "temperature_moves": 8})
    specs = [(eval_a, eval_b) if (i + offset) % 2 == 0 else (eval_b, eval_a)
             for i in range(n_games)]
    records = GamePool(cfg, seed).run(specs, concurrent, should_stop, on_move, on_game,
                                      collect_samples=False)
    a_w = b_w = d = 0
    for r in records:
        a_is_white = r.white_eval is eval_a
        if r.result == 0:
            d += 1
        elif (r.result == 1) == a_is_white:
            a_w += 1
        else:
            b_w += 1
    return MatchResult(a_w, b_w, d, records)
