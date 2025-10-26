"""
Baseten-backed theft detection over single video frames.

Minimal API:
- detect_theft(frame, conf_thresh=0.5, endpoint=env)
  - frame can be a file path, bytes, numpy array (BGR/RGB), or PIL.Image.
  - encodes to JPEG, base64, and calls Baseten via BasetenClient.
- format_for_analysis(result)
  - simple adapter to feed analyzer once implemented.

Env:
- BASETEN_API_KEY (required by BasetenClient)
- BASETEN_THEFT_ENDPOINT (full model inference URL)
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio

import cv2  # type: ignore
import numpy as np  # type: ignore

try:  # optional pillow
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from app.core.logger import get_logger
from app.core.config import get_settings
from app.services.baseten_client import get_baseten_client

log = get_logger(__name__)
_settings = get_settings()


def _to_jpeg_bytes(frame: Any) -> bytes:
    """Convert various frame inputs into JPEG bytes.

    Accepts:
    - str/Path: file path to an image
    - bytes: raw image bytes (assumed already JPEG/PNG)
    - numpy.ndarray: image array; encodes to JPEG
    - PIL.Image.Image: encodes to JPEG
    """
    # Path-like
    if isinstance(frame, (str, Path)):
        p = Path(frame)
        data = p.read_bytes()
        return data

    # Raw bytes
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)

    # numpy array
    if isinstance(frame, np.ndarray):
        # Ensure 3-channel uint8
        arr = frame
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # If grayscale, promote to 3 channels
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        ok, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("Failed to encode frame to JPEG")
        return enc.tobytes()

    # PIL image
    if Image is not None and isinstance(frame, Image.Image):
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    raise TypeError("Unsupported frame type; provide path, bytes, numpy array, or PIL image")


def detect_theft(
    frame: Any,
    conf_thresh: float = 0.5,
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run theft detection on a single frame via Baseten.

    Returns a normalized dict with `ok`, `model`, `raw`, and `detections` keys.
    """
    try:
        jpeg_bytes = _to_jpeg_bytes(frame)
    except Exception as e:
        log.exception("Failed to prepare frame: %s", e)
        return {"ok": False, "error": "Failed to prepare frame"}

    image_b64 = base64.b64encode(jpeg_bytes).decode()
    endpoint_url = endpoint or _settings.BASETEN_THEFT_ENDPOINT or os.getenv("BASETEN_THEFT_ENDPOINT", "")
    client = get_baseten_client()

    extra_input = {"conf_thresh": float(conf_thresh)}
    resp = client.predict_image(endpoint_url, image_b64, extra_input=extra_input)

    # Try to map to a common detection format; keep raw for analyzer
    detections = resp.get("detections") or resp.get("output") or resp.get("result")
    result = {
        "ok": bool(resp.get("ok", True)),
        "model": "baseten:theft",
        "detections": detections,
        "raw": resp,
    }
    return result


async def async_detect_theft(
    frame: Any,
    conf_thresh: float = 0.5,
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Async theft detection via Baseten.

    Offloads CPU-bound JPEG encoding to a thread and uses the async Baseten
    client to avoid blocking the event loop.
    """
    try:
        jpeg_bytes = await asyncio.to_thread(_to_jpeg_bytes, frame)
    except Exception as e:
        log.exception("Failed to prepare frame (async): %s", e)
        return {"ok": False, "error": "Failed to prepare frame"}

    image_b64 = base64.b64encode(jpeg_bytes).decode()
    endpoint_url = endpoint or _settings.BASETEN_THEFT_ENDPOINT or os.getenv("BASETEN_THEFT_ENDPOINT", "")
    client = get_baseten_client()

    extra_input = {"conf_thresh": float(conf_thresh)}
    resp = await client.apredict_image(endpoint_url, image_b64, extra_input=extra_input)

    # DEBUG log the raw Baseten JSON response
    log.debug("Theft detection raw Baseten response: %s", json.dumps(resp, indent=2))

    detections = resp.get("detections") or resp.get("output") or resp.get("result")
    return {
        "ok": bool(resp.get("ok", True)),
        "model": "baseten:theft",
        "detections": detections,
        "raw": resp,
    }

def format_for_analysis(theft_result: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter to analyzer input shape once defined.

    For now, return a simple namespaced payload with type hints.
    """
    return {
        "type": "theft",
        "ok": theft_result.get("ok", False),
        "data": theft_result.get("detections"),
        "meta": {"model": theft_result.get("model", "baseten:theft")},
        "raw": theft_result.get("raw"),
    }
