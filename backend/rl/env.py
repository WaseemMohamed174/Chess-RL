"""Game-end rules, kept in one place."""
from __future__ import annotations

import chess


def leaf_value(board: chess.Board, n_legal: int) -> float | None:
    """Value for the side to move if the position is over, else None. Cheap: used in the tree."""
    if n_legal == 0:
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return 0.0
    return None


def game_result(board: chess.Board, plies: int, max_plies: int) -> int | None:
    """Result of a REAL game from White's view (1 / 0 / -1), or None while it goes on.

    Claims draws by threefold repetition and the 50-move rule, and adjudicates
    games longer than max_plies as draws so self-play never runs forever."""
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0
        return 1 if outcome.winner == chess.WHITE else -1
    return 0 if plies >= max_plies else None
