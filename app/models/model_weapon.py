"""
Baseten-backed weapon detection over single frames.

Provides sync and async helpers similar to model_theft to keep analyzer
integration consistent.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio

import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from app.core.logger import get_logger
from app.core.config import get_settings
from app.services.baseten_client import get_baseten_client

log = get_logger(__name__)
_settings = get_settings()


def _to_jpeg_bytes(frame: Any) -> bytes:
    if isinstance(frame, (str, Path)):
        return Path(frame).read_bytes()
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        ok, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("Failed to encode frame to JPEG")
        return enc.tobytes()
    if Image is not None and isinstance(frame, Image.Image):
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    raise TypeError("Unsupported frame type")


def detect_weapon(frame: Any, endpoint: Optional[str] = None, **extra) -> Dict[str, Any]:
    try:
        jpeg = _to_jpeg_bytes(frame)
    except Exception as e:
        log.exception("weapon: frame prep failed: %s", e)
        return {"ok": False, "error": "Failed to prepare frame"}
    image_b64 = base64.b64encode(jpeg).decode()
    endpoint_url = endpoint or _settings.BASETEN_WEAPON_ENDPOINT or os.getenv("BASETEN_WEAPON_ENDPOINT", "")
    client = get_baseten_client()
    resp = client.predict_image(endpoint_url, image_b64, extra or None)
    det = resp.get("detections") or resp.get("output") or resp.get("result")
    return {"ok": bool(resp.get("ok", True)), "model": "baseten:weapon", "detections": det, "raw": resp}


async def async_detect_weapon(frame: Any, endpoint: Optional[str] = None, **extra) -> Dict[str, Any]:
    try:
        jpeg = await asyncio.to_thread(_to_jpeg_bytes, frame)
    except Exception as e:
        log.exception("weapon: frame prep failed (async): %s", e)
        return {"ok": False, "error": "Failed to prepare frame"}
    image_b64 = base64.b64encode(jpeg).decode()
    endpoint_url = endpoint or _settings.BASETEN_WEAPON_ENDPOINT or os.getenv("BASETEN_WEAPON_ENDPOINT", "")
    client = get_baseten_client()
    resp = await client.apredict_image(endpoint_url, image_b64, extra or None)
    det = resp.get("detections") or resp.get("output") or resp.get("result")
    return {"ok": bool(resp.get("ok", True)), "model": "baseten:weapon", "detections": det, "raw": resp}
