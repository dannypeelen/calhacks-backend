from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.logger import get_logger

log = get_logger(__name__)


async def compute_threat_score(session_id: str) -> int:
    """Minimal stub: read saved analysis JSON and return max threat if available.

    Falls back to 0 when not found.
    """
    base = Path(os.getenv("SENTRIAI_TMP", "./.tmp")) / "sessions" / session_id / "results.json"
    try:
        if base.exists():
            data = json.loads(base.read_text())
            return int(data.get("summary", {}).get("max_threat", 0))
    except Exception:
        log.exception("Failed to read threat score for session %s", session_id)
    return 0
