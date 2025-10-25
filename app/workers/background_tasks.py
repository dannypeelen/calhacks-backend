from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, Optional

from app.core.logger import get_logger
from app.services.analyzer import analyze_video

log = get_logger(__name__)


JobCallable = Callable[[], Awaitable[Any]]


@dataclass
class TaskItem:
    job_id: str
    func: JobCallable


class BackgroundTaskRunner:
    """Lightweight async background task queue.

    - submit(callable) enqueues an async callable to be executed.
    - start() runs a worker loop consuming tasks until stop() is called.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TaskItem] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        async def _worker() -> None:
            log.info("BackgroundTaskRunner started")
            while self._running:
                item: TaskItem = await self._queue.get()
                try:
                    await item.func()
                except Exception:
                    log.exception("Background task %s failed", item.job_id)
                finally:
                    self._queue.task_done()
            log.info("BackgroundTaskRunner stopped")

        self._task = asyncio.create_task(_worker())

    async def stop(self) -> None:
        self._running = False
        # Put a sentinel to unblock queue.get if needed
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            self._task = None

    async def submit(self, job_id: str, func: JobCallable) -> None:
        await self._queue.put(TaskItem(job_id=job_id, func=func))


_runner: Optional[BackgroundTaskRunner] = None


def get_task_runner() -> BackgroundTaskRunner:
    global _runner
    if _runner is None:
        _runner = BackgroundTaskRunner()
    return _runner


# Convenience job: analyze uploaded frames offline and persist JSON
def _session_results_path(session_id: str) -> Path:
    base = Path(os.getenv("SENTRIAI_TMP", "./.tmp")) / "sessions" / session_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "results.json"


async def enqueue_analyze_frames(session_id: str, frame_paths: list[str]) -> str:
    """Enqueue analysis of extracted frames for a session.

    Writes results to .tmp/sessions/<session_id>/results.json for later retrieval.
    Returns the job_id assigned to this task.
    """
    job_id = f"analyze-{session_id}"

    async def _job() -> None:
        try:
            payload = {"session_id": session_id, "frames": {"paths": frame_paths}}
            result = await analyze_video(payload)
            out = _session_results_path(session_id)
            out.write_text(json.dumps(result, indent=2))
            log.info("Analysis completed for session %s -> %s", session_id, out)
        except Exception:
            log.exception("Session analysis failed: %s", session_id)

    runner = get_task_runner()
    await runner.submit(job_id, _job)
    return job_id
