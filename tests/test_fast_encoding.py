"""The fast encoders must give exactly what the reference ones give, on real game positions."""
import chess
import numpy as np

from backend.rl.encoding import (board_to_tensor, board_to_tensor_slow, move_to_index,
                                 move_to_index_slow)


def _positions(n_games=40, plies=120, seed=0):
    rng = np.random.default_rng(seed)
    extra = ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",   # castling
             "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
             "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",           # en passant
             "r3k2r/8/8/8/8/8/p6p/R3K2R b KQkq - 0 1",                                  # promotions near
             "8/P6k/8/8/8/8/p6K/8 w - - 0 1"]
    for f in extra:
        yield chess.Board(f)
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            yield b.copy()
            b.push(moves[int(rng.integers(len(moves)))])


def test_board_to_tensor_matches_reference():
    n = 0
    for b in _positions():
        a, r = board_to_tensor(b), board_to_tensor_slow(b)
        assert a.dtype == r.dtype == np.uint8 and a.shape == r.shape
        assert np.array_equal(a, r), b.fen()
        n += 1
    assert n > 1000


def test_move_to_index_matches_reference_including_promotions():
    seen_under = 0
    for b in _positions():
        for m in b.legal_moves:
            assert move_to_index(m, b) == move_to_index_slow(m, b), (b.fen(), m)
            seen_under += bool(m.promotion and m.promotion != chess.QUEEN)
    assert seen_under > 0                       # the under-promotion branch was really exercised
