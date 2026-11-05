"""Backend tests run with the FAKE engine: real chess and real MCTS, but no PyTorch
(CHESSRL_FAKE_ENGINE=1). The env vars must be set before any backend module is imported."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CHESSRL_DATA_DIR", tempfile.mkdtemp(prefix="chessrl-test-"))
os.environ["CHESSRL_FAKE_ENGINE"] = "1"
os.environ["CHESSRL_ENGINE_PYTHON"] = sys.executable
