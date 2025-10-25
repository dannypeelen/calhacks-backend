from __future__ import annotations

from app.core.logger import get_logger

log = get_logger(__name__)


async def summarize_detections(session_id: str) -> str:
    """Minimal stub that returns a placeholder summary.

    Integrate an LLM or external service to summarize session detections.
    """
    log.info("summarize_detections called for session %s", session_id)
    return "No notable events detected."
