"""
Analyzer: orchestrates per-frame model calls and fuses outputs.

Minimal async implementation that:
- Accepts flexible input describing frames (base64, path list from video_processor, or single path).
- Calls Baseten-backed theft detector (model_theft.detect_theft).
- Optionally calls a Baseten weapon endpoint if configured.
- Produces a compact, analyzer-friendly result per frame plus a naive threat score.

Notes:
- Baseten client is synchronous (requests); we offload calls to a thread to avoid blocking the event loop.
- This is intentionally light; plug in Fetch.AI agents and a richer fusion/scoring scheme later.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.baseten_client import get_baseten_client
from app.models.model_theft import detect_theft, format_for_analysis

log = get_logger(__name__)


@dataclass
class FrameItem:
    frame_id: str
    path: Optional[str] = None
    b64: Optional[str] = None


def _gather_frames_from_input(video: Any, limit: int = 10) -> List[FrameItem]:
    """Collect up to `limit` frames from flexible input structures.

    Supported patterns:
    - dict or model with attribute `frame_b64` or `image_b64`
    - dict or model with attribute `frame_path`
    - dict or model with `.frames` being:
        - list[str] of paths, or
        - dict with key `paths` containing list[str]
    """
    items: List[FrameItem] = []

    def _get(attr: str) -> Any:
        if isinstance(video, dict):
            return video.get(attr)
        return getattr(video, attr, None)

    # Single base64
    for key in ("frame_b64", "image_b64"):
        val = _get(key)
        if isinstance(val, str) and val:
            items.append(FrameItem(frame_id="0", b64=val))
            return items[:limit]

    # Single path
    frame_path = _get("frame_path")
    if isinstance(frame_path, str) and frame_path:
        items.append(FrameItem(frame_id="0", path=frame_path))
        return items[:limit]

    # Frames collection
    frames = _get("frames")
    paths: Optional[List[str]] = None
    if isinstance(frames, dict) and isinstance(frames.get("paths"), list):
        paths = [str(p) for p in frames["paths"]]
    elif isinstance(frames, list):
        paths = [str(p) for p in frames]

    if paths:
        for i, p in enumerate(paths[:limit]):
            items.append(FrameItem(frame_id=str(i), path=p))

    return items[:limit]


async def _call_baseten_weapon(image_b64: str, extra_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optional weapon detection via Baseten if endpoint is configured."""
    endpoint = os.getenv("BASETEN_WEAPON_ENDPOINT", "")
    if not endpoint:
        return {"ok": False, "skipped": True, "reason": "endpoint_missing"}

    client = get_baseten_client()
    return await asyncio.to_thread(client.predict_image, endpoint, image_b64, extra_input or {})


def _to_b64_from_path(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def _threat_score_simple(theft: Dict[str, Any], weapon: Dict[str, Any]) -> int:
    """Very naive score 0-10 based on model scores if present."""
    score = 0.0
    # Try common nested paths
    theft_score = None
    weapon_score = None
    for src, name in ((theft, "theft"), (weapon, "weapon")):
        if not src or not src.get("ok", True):
            continue
        det = src.get("detections")
        if isinstance(det, dict) and "score" in det:
            val = float(det.get("score", 0))
        elif isinstance(det, list) and det and isinstance(det[0], dict) and "score" in det[0]:
            val = float(det[0].get("score", 0))
        else:
            val = 0.0
        if name == "theft":
            theft_score = val
        else:
            weapon_score = val

    # Combine with simple weights
    if theft_score is not None:
        score += min(max(theft_score, 0.0), 1.0) * 6.0
    if weapon_score is not None:
        score += min(max(weapon_score, 0.0), 1.0) * 4.0

    return int(round(min(score, 10.0)))


async def analyze_video(video: Any) -> Dict[str, Any]:
    """Analyze frames from provided video input.

    Returns a dict containing per-frame results and a simple summary.
    """
    frames = _gather_frames_from_input(video, limit=10)
    if not frames:
        log.warning("analyze_video: no frames provided")
        return {"ok": False, "error": "No frames"}

    results: List[Dict[str, Any]] = []

    for item in frames:
        # Obtain base64 for weapon model (optional) and bytes for theft
        if item.b64:
            image_b64 = item.b64.split(",", 1)[1] if item.b64.strip().startswith("data:") and "," in item.b64 else item.b64
        else:
            image_b64 = _to_b64_from_path(str(item.path))

        # Theft detection (sync -> offload)
        theft_res = await asyncio.to_thread(detect_theft, base64.b64decode(image_b64))
        theft_for_analysis = format_for_analysis(theft_res)

        # Weapon detection (optional Baseten endpoint)
        weapon_res = await _call_baseten_weapon(image_b64)

        # Simple threat score
        score = _threat_score_simple(theft_res, weapon_res)

        results.append(
            {
                "frame_id": item.frame_id,
                "path": item.path,
                "theft": theft_for_analysis,
                "weapon": weapon_res,
                "threat": {"score": score},
            }
        )

    summary = {
        "frames": len(results),
        "max_threat": max((r["threat"]["score"] for r in results), default=0),
        "avg_threat": round(sum((r["threat"]["score"] for r in results)) / max(len(results), 1), 2),
    }

    return {"ok": True, "results": results, "summary": summary}
