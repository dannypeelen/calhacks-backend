from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class Detection(BaseModel):
    label: Optional[str] = None
    score: Optional[float] = None
    bbox: Optional[List[float]] = None  # [x, y, w, h] or [x1, y1, x2, y2]


class ModelResult(BaseModel):
    ok: bool = True
    model: Optional[str] = None
    # Some models return a list, some a dict; allow flexible payloads
    detections: Optional[Union[Detection, List[Detection], Dict[str, Any]]] = None
    raw: Optional[Dict[str, Any]] = None


class FrameAnalysis(BaseModel):
    frame_id: str
    path: Optional[str] = None
    theft: Optional[Dict[str, Any]] = None
    weapon: Optional[Dict[str, Any]] = None
    threat: Optional[Dict[str, Any]] = None
