from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ThreatScore(BaseModel):
    score: int
    level: Optional[str] = None  # e.g., low/medium/high
    reasons: Optional[List[str]] = None
    details: Optional[Dict[str, Any]] = None


class ThreatSummary(BaseModel):
    frames: int
    max_threat: int
    avg_threat: float
