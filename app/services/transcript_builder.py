from __future__ import annotations

from typing import List, Dict, Any

from app.core.logger import get_logger

log = get_logger(__name__)


async def build_transcript(session_id: str) -> List[Dict[str, Any]]:
    """Minimal stub that returns an empty transcript.

    Replace with logic that reads detections per frame and creates a
    chronological list of human-readable events.
    """
    log.info("build_transcript called for session %s", session_id)
    return []
