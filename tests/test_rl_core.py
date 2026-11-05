"""Tests for the RL core. Run:  python -m pytest tests -q"""
import random
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rl import encoding as E
from backend.rl.arena import EloRating, play_match
from backend.rl.config import MCTSConfig, TrainConfig
from backend.rl.evaluator import UniformEvaluator
from backend.rl.mcts import search
from backend.rl.replay_buffer import ReplayBuffer
from backend.rl.selfplay import GamePool


def test_every_legal_move_gets_a_unique_index():
    rng = random.Random(7)
    for _ in range(60):
        b = chess.Board()
        while not b.is_game_over() and b.ply() < 160:
            m = E.legal_move_map(b)
            assert len(m) == b.legal_moves.count()
            assert all(0 <= i < E.POLICY_SIZE for i in m)
            b.push(rng.choice(list(b.legal_moves)))


def test_underpromotions_and_castling_are_encoded():
    b = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    promos = [m for m in b.legal_moves if m.promotion]
    assert len({E.move_to_index(m, b) for m in promos}) == 4     # Q, R, B, N all distinct
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert len(E.legal_move_map(b)) == b.legal_moves.count()


def test_black_is_seen_from_its_own_side():
    w, bl = chess.Board(), chess.Board()
    bl.push_san("e4")
    # black to move after 1.e4 must look like white at the start, apart from the e-pawns
    a, c = E.board_to_tensor(w), E.board_to_tensor(bl)
    assert a[0].sum() == c[0].sum() == 8 and a[5, 0, 4] == c[5, 0, 4] == 1
    assert a[12].all() and not c[12].any()


def test_mcts_finds_mate_in_one():
    b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    job = search(b, UniformEvaluator(), MCTSConfig(simulations=200), seed=0)
    assert job.top_moves(1)[0]["uci"] == "d8h4"


def test_batched_pool_fills_batches_and_records_valid_games():
    class Counting(UniformEvaluator):
        sizes = []

        def evaluate(self, x):
            self.sizes.append(len(x))
            return super().evaluate(x)

    ev = Counting()
    cfg = MCTSConfig(simulations=16, max_plies=20, leaves_per_batch=4)
    recs = GamePool(cfg, seed=1).run([(ev, ev)] * 4, concurrent=4)
    assert len(recs) == 4 and np.mean(ev.sizes) > 12            # ~4 games x 4 leaves
    for r in recs:
        b = chess.Board()
        for u in r.moves:
            b.push_uci(u)                                         # every move is legal
        st, idx, p, z = r.samples[0]
        assert st.shape == (18, 8, 8) and abs(p.sum() - 1) < 1e-2 and z in (-1.0, 0.0, 1.0)


def test_replay_buffer_roundtrip(tmp_path):
    buf = ReplayBuffer(5)
    for i in range(8):                                            # wraps around
        buf.add(np.full((18, 8, 8), i % 2, np.uint8), np.array([3, 9]), np.array([.25, .75]), 1.0)
    assert len(buf) == 5
    x, pi, z = buf.sample(4, np.random.default_rng(0))
    assert x.shape == (4, 18, 8, 8) and np.allclose(pi.sum(1), 1, atol=1e-2)
    buf.save(tmp_path / "b.npz")
    other = ReplayBuffer(5)
    other.load(tmp_path / "b.npz")
    assert len(other) == 5


def test_elo():
    assert abs(EloRating.expected(1500, 1500) - 0.5) < 1e-9
    assert EloRating.diff_from_score(0.5, 20) == pytest.approx(0)
    assert EloRating.diff_from_score(1.0, 20) > 0


# ---------------------------------------------------------------- need PyTorch


def test_param_count_matches_the_estimate_in_select_py():
    torch = pytest.importorskip("torch")
    from backend.cli import select
    from backend.rl.network import build_network
    for blocks, filters in ((3, 64), (6, 128), (10, 192)):
        assert build_network(blocks, filters).n_params() == select.count_params(blocks, filters)


def test_network_shapes_and_layer_table():
    torch = pytest.importorskip("torch")
    from backend.rl.network import build_network, layer_table
    net = build_network(2, 32).eval()
    p, v = net(torch.zeros(3, 18, 8, 8))
    assert p.shape == (3, E.POLICY_SIZE) and v.shape == (3,) and (v.abs() <= 1).all()
    rows = layer_table(net)
    assert sum(r["params"] for r in rows) == net.n_params()


def test_training_reduces_loss_and_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from backend.rl.evaluator import TorchEvaluator
    from backend.rl.network import build_network
    from backend.rl.trainer import Trainer, load_network
    net = build_network(2, 32)
    cfg = TrainConfig(batch_size=32, lr=3e-3)
    ev = TorchEvaluator(net)
    recs = GamePool(MCTSConfig(simulations=8, max_plies=24), 0).run([(ev, ev)] * 3, concurrent=3)
    buf = ReplayBuffer(1000)
    for r in recs:
        buf.add_game(r)
    tr = Trainer(net, cfg)
    rng = np.random.default_rng(0)
    first = tr.train(buf, 5, rng)
    last = tr.train(buf, 60, rng)
    assert last["policy_loss"] + last["value_loss"] < first["policy_loss"] + first["value_loss"]
    tr.save(tmp_path / "m.pt")
    again = load_network(tmp_path / "m.pt")
    x = torch.from_numpy(buf.states[:2]).float()
    net.eval()
    assert torch.allclose(net(x)[0], again(x)[0], atol=1e-5)


def test_arena_match_between_two_networks():
    torch = pytest.importorskip("torch")
    from backend.rl.evaluator import TorchEvaluator
    from backend.rl.network import build_network
    a, b = TorchEvaluator(build_network(1, 16)), TorchEvaluator(build_network(1, 16))
    res = play_match(a, b, 2, MCTSConfig(simulations=6, max_plies=16), concurrent=2, seed=0)
    assert res.n == 2 and 0 <= res.score_a <= 1


# ---------------------------------------------------------------- evaluator contract
def test_counting_evaluator_counts_and_checks_the_contract():
    from backend.rl.evaluator import CountingEvaluator, check_request

    ev = CountingEvaluator(UniformEvaluator(), check=True)
    x = np.zeros((5, 18, 8, 8), np.uint8)
    logits, values = ev.evaluate(x)
    assert logits.shape == (5, E.POLICY_SIZE) and values.shape == (5,)
    ev.evaluate(x[:2])
    s = ev.stats()
    assert (s.calls, s.positions, s.max_batch) == (2, 7, 5)
    assert ev.stats(reset=True).positions == 7 and ev.stats().positions == 0
    with pytest.raises(ValueError):
        check_request(x.astype(np.float32))          # wrong dtype
    with pytest.raises(ValueError):
        check_request(x[:, :17])                     # wrong planes


def test_counting_evaluator_keeps_identity_for_selfplay_grouping():
    from backend.rl.evaluator import CountingEvaluator
    w = CountingEvaluator(UniformEvaluator())
    recs = GamePool(MCTSConfig(simulations=8, max_plies=6), seed=1).run([(w, w)] * 2, concurrent=2)
    assert len(recs) == 2 and w.stats().positions > 0
