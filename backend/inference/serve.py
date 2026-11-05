"""Entry point of the inference server process (started and watched by the Supervisor).

    python -m backend.inference.serve

Reads its settings from the JSON file named by CHESSRL_INFERENCE_CONFIG and the secret
from CHESSRL_TOKEN. Writes the port it listens on to cfg["port_file"] once it is ready."""
from __future__ import annotations

import glob
import json
import os
import signal
import sys
import threading

from .server import InferenceServer, ServerConfig
from .sources import TorchSource, UniformSource
from .transport import open_server_port


def _clear_stale_blocks(prefix: str) -> None:
    """A crashed server cannot unlink its shared-memory blocks; remove what it left (Linux)."""
    for f in glob.glob(f"/dev/shm/{prefix}*"):
        try:
            os.unlink(f)
        except OSError:
            pass


def main() -> int:
    with open(os.environ["CHESSRL_INFERENCE_CONFIG"], encoding="utf-8") as f:
        cfg = json.load(f)
    token = os.environ["CHESSRL_TOKEN"]
    prefix = cfg.get("shm_prefix", "crl-")
    _clear_stale_blocks(prefix)
    try:
        os.unlink(cfg["port_file"])               # nobody may read a dead server's port
    except OSError:
        pass

    from ..workers.common import fake_engine, torch_device      # same rules as the workers
    device = "cpu" if fake_engine() else torch_device()
    if fake_engine():
        source = UniformSource()
    else:
        if device == "cpu" and cfg.get("cpu_threads"):
            import torch
            torch.set_num_threads(int(cfg["cpu_threads"]))
        source = TorchSource(device, template="{id}", root=cfg["root"])

    port = open_server_port({"kind": "sock", "host": "127.0.0.1", "port": 0, "token": token,
                             "max_n": cfg.get("max_n", 512), "port_file": cfg["port_file"],
                             "shm_prefix": prefix})
    server = InferenceServer(port, source, ServerConfig(
        max_batch=cfg.get("max_batch", 256), max_wait_ms=cfg.get("max_wait_ms", 2.0),
        max_models=cfg.get("max_models", 4), reload_check_s=cfg.get("reload_check_s", 1.0),
        stats_path=cfg.get("stats_path")))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    print(f"inference server on 127.0.0.1:{port.port} device={device}", flush=True)
    try:
        server.serve(stop)
    finally:
        port.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
