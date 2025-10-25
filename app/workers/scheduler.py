from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from app.core.logger import get_logger

log = get_logger(__name__)


PeriodicCallable = Callable[[], Awaitable[None]]


class PeriodicScheduler:
    """Very small periodic task scheduler.

    schedule(coro_func, interval) will run the coroutine indefinitely at
    approximately the given interval until stop() is called.
    """

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except Exception:
                pass
        self._tasks.clear()

    def schedule(self, func: PeriodicCallable, interval_sec: float) -> None:
        async def _loop() -> None:
            while self._running:
                start = time.time()
                try:
                    await func()
                except Exception:
                    log.exception("Periodic task failed")
                # maintain approximate interval
                elapsed = time.time() - start
                await asyncio.sleep(max(0.0, interval_sec - elapsed))

        task = asyncio.create_task(_loop())
        self._tasks.append(task)


_scheduler: Optional[PeriodicScheduler] = None


def get_scheduler() -> PeriodicScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = PeriodicScheduler()
    return _scheduler


# Housekeeping: remove temp files older than max_age_sec from .tmp
def _tmp_dir() -> Path:
    return Path(os.getenv("SENTRIAI_TMP", "./.tmp"))


async def cleanup_tmp(max_age_sec: float = 24 * 3600) -> None:
    now = time.time()
    base = _tmp_dir()
    if not base.exists():
        return
    removed = 0
    for p in base.rglob("*"):
        try:
            if p.is_file():
                age = now - p.stat().st_mtime
                if age > max_age_sec:
                    p.unlink(missing_ok=True)
                    removed += 1
        except Exception:
            # Ignore errors during cleanup
            pass
    if removed:
        log.info("Cleaned %d temp files from %s", removed, base)
