from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TranscriptItem(BaseModel):
    ts: float  # epoch seconds
    event: str
    metadata: Optional[Dict[str, Any]] = None


class Transcript(BaseModel):
    session_id: str
    items: List[TranscriptItem]
