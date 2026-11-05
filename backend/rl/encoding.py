"""Board <-> tensor and move <-> index (AlphaZero encoding).

Everything is seen from the side to move: for Black the board is flipped
vertically, so the network never has to learn two colours separately.

Input  (18, 8, 8)  uint8
    0-5    my pieces      P N B R Q K
    6-11   their pieces
    12     colour (1 = White to move)
    13-14  my castling rights    (king side, queen side)
    15-16  their castling rights (king side, queen side)
    17     en-passant square
Output 4672 = 64 squares x 73 move planes
    0-55   queen-like slides: 8 directions x 7 distances (also queen promotions)
    56-63  knight jumps
    64-72  under-promotions: 3 directions x (knight, bishop, rook)
"""
from __future__ import annotations

import chess
import numpy as np

INPUT_PLANES = 18
MOVE_PLANES = 73
POLICY_SIZE = 64 * MOVE_PLANES      # 4672

_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # (d_file, d_rank)
_DIR_IDX = {d: i for i, d in enumerate(_DIRS)}
_KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KNIGHT_IDX = {d: i for i, d in enumerate(_KNIGHT)}
_UNDER = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def board_to_tensor_slow(board: chess.Board) -> np.ndarray:
    """Reference version (square by square). Kept for the tests; the fast one must match it."""
    me = board.turn
    flip = 56 if me == chess.BLACK else 0
    x = np.zeros((INPUT_PLANES, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        sq ^= flip
        plane = piece.piece_type - 1 + (0 if piece.color == me else 6)
        x[plane, sq >> 3, sq & 7] = 1
    if me == chess.WHITE:
        x[12] = 1
    if board.has_kingside_castling_rights(me):
        x[13] = 1
    if board.has_queenside_castling_rights(me):
        x[14] = 1
    if board.has_kingside_castling_rights(not me):
        x[15] = 1
    if board.has_queenside_castling_rights(not me):
        x[16] = 1
    if board.ep_square is not None:
        sq = board.ep_square ^ flip
        x[17, sq >> 3, sq & 7] = 1
    return x


def move_to_index_slow(move: chess.Move, board: chess.Board) -> int:
    """Reference version. Kept for the tests and to fill the cache of the fast one."""
    frm, to = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        frm ^= 56
        to ^= 56
    df = (to & 7) - (frm & 7)
    dr = (to >> 3) - (frm >> 3)
    if move.promotion and move.promotion != chess.QUEEN:
        plane = 64 + (df + 1) * 3 + _UNDER.index(move.promotion)
    elif (df, dr) in _KNIGHT_IDX:
        plane = 56 + _KNIGHT_IDX[(df, dr)]
    else:
        dist = max(abs(df), abs(dr))
        direction = ((df > 0) - (df < 0), (dr > 0) - (dr < 0))
        plane = _DIR_IDX[direction] * 7 + dist - 1
    return frm * MOVE_PLANES + plane


# ---------------------------------------------------------------- fast versions
# Profiling a self-play worker showed these two taking a third of its CPU time: the old
# board_to_tensor asked python-chess for every piece one by one (piece_map -> piece_at), and
# move_to_index redid the same arithmetic for the same (from, to) pairs again and again.
_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Same result as board_to_tensor_slow, built from python-chess's bitboards: each piece
    set is one 64-bit integer, and numpy unpacks 12 of them into planes in one call."""
    me = board.turn
    them = not me
    pm = board.pieces_mask
    raw = b"".join([pm(t, me).to_bytes(8, "little") for t in _PIECE_TYPES]
                   + [pm(t, them).to_bytes(8, "little") for t in _PIECE_TYPES])
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little").reshape(12, 8, 8)
    x = np.zeros((INPUT_PLANES, 8, 8), dtype=np.uint8)
    # bit k of a mask is square k, so row = rank. Black sees the board flipped vertically.
    x[:12] = bits[:, ::-1, :] if me == chess.BLACK else bits
    if me == chess.WHITE:
        x[12] = 1
    if board.has_kingside_castling_rights(me):
        x[13] = 1
    if board.has_queenside_castling_rights(me):
        x[14] = 1
    if board.has_kingside_castling_rights(them):
        x[15] = 1
    if board.has_queenside_castling_rights(them):
        x[16] = 1
    if board.ep_square is not None:
        sq = board.ep_square ^ (56 if me == chess.BLACK else 0)
        x[17, sq >> 3, sq & 7] = 1
    return x


_MOVE_CACHE: dict = {}


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Same result as move_to_index_slow; each (side, from, to, promotion) is worked out once."""
    key = (board.turn << 16) | ((move.promotion or 0) << 12) | (move.from_square << 6) | move.to_square
    i = _MOVE_CACHE.get(key)
    if i is None:
        i = _MOVE_CACHE[key] = move_to_index_slow(move, board)
    return i


def legal_move_map(board: chess.Board) -> dict[int, chess.Move]:
    """{policy index: move} for every legal move (this is also the legal-move mask)."""
    return {move_to_index(m, board): m for m in board.legal_moves}


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    return legal_move_map(board).get(index)


def legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    mask[list(legal_move_map(board))] = True
    return mask
