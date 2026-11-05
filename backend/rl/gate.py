"""LearnerGate: at most N learners train at the same moment, across ALL train processes.

Each training run lives in its own process, so a normal threading.Semaphore cannot count
them. This uses N lock files instead: a learner must hold one of them while it trains.
The operating system drops a lock when its process dies, so a crashed learner can never
leave the gate closed.

Why: two learners at once only fight for the same GPU and each needs its own copy of the
weights, gradients and Adam state in VRAM. One after the other is as fast and fits."""
from __future__ import annotations

import os
import time

try:
    import fcntl

    def _try_lock(fh) -> bool:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        fcntl.flock(fh, fcntl.LOCK_UN)
except ImportError:                                   # Windows
    import msvcrt

    def _try_lock(fh) -> bool:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


class LearnerGate:
    def __init__(self, base_path: str, slots: int = 1):
        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
        self.paths = [f"{base_path}.{i}" for i in range(max(1, int(slots)))]
        self._fh = None

    def acquire(self, should_stop=None, poll: float = 0.2) -> bool:
        """Wait for a free slot. Returns False if should_stop() said stop while waiting."""
        while True:
            for p in self.paths:
                fh = open(p, "a+b")
                if _try_lock(fh):
                    self._fh = fh
                    return True
                fh.close()
            if should_stop is not None and should_stop():
                return False
            time.sleep(poll)

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                _unlock(fh)
            finally:
                fh.close()

    def holding(self) -> bool:
        return self._fh is not None
