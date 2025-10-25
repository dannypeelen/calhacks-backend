"""
Lightweight video processing helpers used by /video endpoints.

Minimal goals:
- Save uploaded videos to a temporary folder for later analysis.
- Decode base64 webcam frames and return simple metadata.

Deliberately minimal: no heavy analysis performed here. This module is a
staging point for later integration with model clients (Baseten/FetchAI)
and the analyzer pipeline.
"""

from __future__ import annotations

import base64
import io
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import UploadFile
import time
import cv2  # OpenCV for video frame extraction (in requirements)

try:
    from PIL import Image  # pillow (listed in requirements.txt)
except Exception:  # pragma: no cover - optional at runtime
    Image = None  # type: ignore

# Workspace-local temp directory for uploads/frames
from app.core.logger import get_logger
from app.workers.background_tasks import enqueue_analyze_frames

TMP_DIR = Path(os.getenv("SENTRIAI_TMP", "./.tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger(__name__)

# For simple 1 FPS throttling on live frames
_last_stream_process_ts: float = 0.0


async def process_uploaded_video(file: UploadFile) -> Dict[str, Any]:
    """Persist the uploaded video and return basic metadata.

    - Stores the file under `./.tmp/<uuid>_<filename>`.
    - Returns a payload with a generated session_id and file info.
    """

    session_id = str(uuid.uuid4())
    safe_name = f"{session_id}_{Path(file.filename or 'video').name}"
    dest = TMP_DIR / safe_name

    # Stream to disk in chunks to avoid memory spikes
    size = 0
    # Read in chunks to avoid loading entire file in memory
    try:
        with dest.open("ab") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)
    except Exception as e:
        log.exception("Failed saving uploaded video: %s", e)
        return {"ok": False, "error": "Failed to save video"}

    # Extract up to N frames at ~1 FPS for downstream model reads
    frames_info: Dict[str, Any] = {"count": 0, "paths": []}
    job_id = None
    try:
        frames = _extract_frames_1fps(str(dest), session_id=session_id, max_frames=10)
        frames_info = {"count": len(frames), "paths": frames}
        if frames:
            # Enqueue offline analysis of sampled frames
            job_id = await enqueue_analyze_frames(session_id, frames)
    except Exception as e:
        # Log but do not fail the upload; analysis can still run later
        log.exception("Frame extraction or enqueue failed: %s", e)

    return {
        "ok": True,
        "session_id": session_id,
        "filename": file.filename,
        "saved_path": str(dest),
        "bytes": size,
        "frames": frames_info,
        "job_id": job_id,
        "note": "Saved and sampled at ~1 FPS for model ingestion; analysis enqueued if frames available.",
    }


async def process_webcam_frame(frame_data: str) -> Dict[str, Any]:
    """Decode an incoming base64 frame string and return simple metadata.

    Accepts common data URLs (e.g., "data:image/jpeg;base64,<...>") or raw base64.
    Saves the frame as `./.tmp/frame_<uuid>.jpg` for potential downstream use.
    """

    # Strip potential data URL header
    if "," in frame_data and frame_data.lstrip().startswith("data:"):
        frame_data = frame_data.split(",", 1)[1]

    try:
        blob = base64.b64decode(frame_data, validate=False)
    except Exception as e:
        log.warning("Invalid base64 frame: %s", e)
        return {"ok": False, "error": "Invalid base64 frame"}

    img_size = None
    inferred_format = "jpg"

    if Image is not None:
        try:
            with Image.open(io.BytesIO(blob)) as im:
                img_size = {"width": im.width, "height": im.height}
                inferred_format = (im.format or "JPEG").lower()
        except Exception:
            # If Pillow can't parse, still persist raw bytes
            pass

    # Persist the raw frame for later analysis
    frame_id = str(uuid.uuid4())
    # 1 FPS throttle for live stream; only save/process when not throttled
    global _last_stream_process_ts
    now = time.time()
    throttled = now - _last_stream_process_ts < 1.0

    out_path = None
    if not throttled:
        out_path = TMP_DIR / f"frame_{frame_id}.{inferred_format}"
        try:
            out_path.write_bytes(blob)
        except Exception as e:
            log.exception("Failed to write webcam frame: %s", e)
            return {"ok": False, "error": "Failed to persist frame"}
        _last_stream_process_ts = now

    return {
        "ok": True,
        "frame_id": frame_id,
        "saved_path": str(out_path) if out_path else None,
        "image_size": img_size,  # None when not parsable
        "throttled": throttled,
        "hint": "1 FPS processing. Models can read from saved_path.",
    }


def _extract_frames_1fps(video_path: str, session_id: str, max_frames: int = 10) -> list[str]:
    """Extract frames at approximately 1 FPS and save to session folder.

    - Determines FPS using OpenCV; falls back to 30 if unavailable.
    - Saves frames as JPEGs under `./.tmp/frames/<session_id>/`.
    - Returns list of saved file paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for reading")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    frame_interval = max(int(round(fps)), 1)

    out_dir = TMP_DIR / "frames" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            # Write JPEG
            fname = out_dir / f"frame_{saved_count:05d}.jpg"
            try:
                ok = cv2.imwrite(str(fname), frame)
                if ok:
                    saved.append(str(fname))
                    saved_count += 1
                else:
                    log.warning("cv2.imwrite returned False for %s", fname)
            except Exception as e:
                log.exception("Failed writing frame %s: %s", fname, e)

            if saved_count >= max_frames:
                break
        idx += 1

    cap.release()
    return saved
