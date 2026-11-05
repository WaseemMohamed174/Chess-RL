"""In-process publish / subscribe. Threads publish; WebSocket handlers subscribe."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class EventBus:
    def __init__(self):
        self._subs: dict[str, set] = {}
        self._last: dict[str, dict] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop) -> None:
        self._loop = loop

    def publish(self, channel: str, event: dict) -> None:
        """Safe to call from any thread."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._dispatch, channel, event)
        except RuntimeError:                      # loop closed during shutdown
            pass

    def _dispatch(self, channel, event):
        self._last[channel] = event
        for q in list(self._subs.get(channel, ())):
            if q.full():
                q.get_nowait()                    # slow client: drop the oldest event
            q.put_nowait(event)

    def last(self, channel: str):
        return self._last.get(channel)

    @asynccontextmanager
    async def subscribe(self, channel: str):
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs.setdefault(channel, set()).add(q)
        try:
            yield q
        finally:
            self._subs[channel].discard(q)
