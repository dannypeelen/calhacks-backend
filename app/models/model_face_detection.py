"""
Baseten-backed face detection over single frames.

Mirrors theft/weapon adapters for consistency.
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


def detect_face(frame: Any, endpoint: Optional[str] = None, **extra) -> Dict[str, Any]:
    try:
        jpeg = _to_jpeg_bytes(frame)
    except Exception as e:
        log.exception("face: frame prep failed: %s", e)
        return {"ok": False, "error": "Failed to prepare frame"}
    image_b64 = base64.b64encode(jpeg).decode()
    # Prefer env-configured endpoint via client helper if endpoint not given
    client = get_baseten_client()
    resolved = endpoint or _settings.BASETEN_FACE_ENDPOINT or os.getenv("BASETEN_FACE_ENDPOINT", "")
    if resolved:
        resp = asyncio.run(client.apredict_image(resolved, image_b64, extra or None))
    else:
        resp = asyncio.run(client.apredict_face(image_b64, **extra))
    det = resp.get("detections") or resp.get("output") or resp.get("result")
    return {"ok": bool(resp.get("ok", True)), "model": "baseten:face", "detections": det, "raw": resp}


async def async_detect_face(frame: Any, endpoint: Optional[str] = None, **extra) -> Dict[str, Any]:
    try:
        jpeg = await asyncio.to_thread(_to_jpeg_bytes, frame)
    except Exception as e:
        log.exception("face: frame prep failed (async): %s", e)
        return {"ok": False, "error": "Failed to prepare frame"}
    image_b64 = base64.b64encode(jpeg).decode()
    client = get_baseten_client()
    resolved = endpoint or _settings.BASETEN_FACE_ENDPOINT or os.getenv("BASETEN_FACE_ENDPOINT", "")
    if resolved:
        resp = await client.apredict_image(resolved, image_b64, extra or None)
    else:
        resp = await client.apredict_face(image_b64, **extra)
    det = resp.get("detections") or resp.get("output") or resp.get("result")
    return {"ok": bool(resp.get("ok", True)), "model": "baseten:face", "detections": det, "raw": resp}
