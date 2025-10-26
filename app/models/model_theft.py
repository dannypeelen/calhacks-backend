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
import binascii
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    if isinstance(frame, Path):
        return frame.read_bytes()

    if isinstance(frame, str):
        data_str = frame.strip()
        maybe_data = data_str
        if maybe_data.startswith("data:") and "," in maybe_data:
            maybe_data = maybe_data.split(",", 1)[1]
        try:
            return base64.b64decode(maybe_data, validate=True)
        except (binascii.Error, ValueError):
            pass  # Not base64; fall back to filesystem

        p = Path(data_str)
        if not p.exists():
            raise TypeError("String input must be a valid path or base64 data URI")
        return p.read_bytes()

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

    extra_input = {
        "conf_thresh": float(conf_thresh),
        "conf": float(conf_thresh),
    }
    resp = client.predict_image(endpoint_url, image_b64, extra_input=extra_input)

    return _normalize_response(resp)


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

    extra_input = {
        "conf_thresh": float(conf_thresh),
        "conf": float(conf_thresh),
    }
    resp = await client.apredict_image(endpoint_url, image_b64, extra_input=extra_input)

    # DEBUG log the raw Baseten JSON response
    log.debug("Theft detection raw Baseten response: %s", json.dumps(resp, indent=2))

    return _normalize_response(resp)


def _normalize_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure downstream consumers receive consistent theft payloads."""
    detections = resp.get("detections") or resp.get("output") or resp.get("result") or []
    meta = resp.get("meta") or {}

    if detections is None:
        detections = []

    normalized = _normalize_detections(detections)
    confidence = _max_confidence(normalized)
    num_detections = meta.get("num_detections")
    if num_detections is None:
        num_detections = len(normalized)

    return {
        "ok": bool(resp.get("ok", True)),
        "model": "baseten:theft",
        "detections": normalized,
        "num_detections": num_detections,
        "confidence": confidence if confidence is not None else 0.0,
        "image_base64": resp.get("image_base64"),
        "coordinates": _coordinate_summary(normalized),
        "raw": resp,
    }


def _normalize_detections(detections: Any) -> List[Dict[str, Any]]:
    """Force detections into a predictable list."""
    if isinstance(detections, dict):
        detections = [detections]
    elif isinstance(detections, (list, tuple)):
        detections = list(detections)
    else:
        return []

    normed: List[Dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        box = det.get("box") or {}
        center = det.get("center")
        size = det.get("size")
        normed.append(
            {
                "box": {
                    "x1": _to_int(box.get("x1")),
                    "y1": _to_int(box.get("y1")),
                    "x2": _to_int(box.get("x2")),
                    "y2": _to_int(box.get("y2")),
                },
                "center": {
                    "cx": _to_int(center.get("cx")),
                    "cy": _to_int(center.get("cy")),
                }
                if isinstance(center, dict)
                else None,
                "size": {
                    "w": _to_int(size.get("w")),
                    "h": _to_int(size.get("h")),
                }
                if isinstance(size, dict)
                else None,
                "confidence": _to_float(det.get("confidence") or det.get("conf") or det.get("score")),
                "class_id": _to_int(det.get("class_id")),
                "class_name": _class_name(det),
            }
        )
    return normed


def _class_name(det: Dict[str, Any]) -> str:
    if det.get("class_name"):
        return str(det["class_name"])
    class_id = det.get("class_id")
    if class_id is None:
        return "object"
    return str(class_id)


def _max_confidence(detections: Any) -> Optional[float]:
    """Derive the highest detection confidence/score from normalized payloads."""
    best = None
    for det in detections:
        if not isinstance(det, dict):
            continue
        conf = det.get("confidence")
        if conf is None:
            continue
        try:
            val = float(conf)
        except (TypeError, ValueError):
            continue
        if best is None or val > best:
            best = val
    return best


def _coordinate_summary(detections: list) -> Dict[str, Any]:
    boxes = []
    centers = []
    sizes = []
    for det in detections:
        box = det.get("box")
        if box and all(v is not None for v in box.values()):
            boxes.append(box)
        center = det.get("center")
        if center and all(v is not None for v in center.values()):
            centers.append(center)
        size = det.get("size")
        if size and all(v is not None for v in size.values()):
            sizes.append(size)
    return {"boxes": boxes, "centers": centers, "sizes": sizes}


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def format_for_analysis(theft_result: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter to analyzer input shape once defined.

    For now, return a simple namespaced payload with type hints.
    """
    return {
        "type": "theft",
        "ok": theft_result.get("ok", False),
        "data": theft_result.get("detections"),
        "meta": {
            "model": theft_result.get("model", "baseten:theft"),
            "num_detections": theft_result.get("num_detections"),
            "confidence": theft_result.get("confidence"),
            "coordinates": theft_result.get("coordinates"),
            "image_base64": theft_result.get("image_base64"),
        },
        "raw": theft_result.get("raw"),
    }
