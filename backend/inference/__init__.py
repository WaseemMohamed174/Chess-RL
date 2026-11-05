"""Central inference server: one process owns the networks and the device
("cpu" / "cuda"), many light workers send it positions and get logits back.

    transport.py  how bytes move (shared memory + semaphores, or plain queues)
    client.py     RemoteEvaluator: the same evaluate(x) call MCTS already uses
    server.py     batching loop, model cache (LRU), hot reload of changed weights
    sources.py    where models come from (real PyTorch files, or debug fakes)
"""
from .client import InferenceError, RemoteEvaluator   # noqa: F401
from .server import InferenceServer, ServerConfig, start_process, start_thread   # noqa: F401
from .transport import make_hub   # noqa: F401
