from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class VideoInput(BaseModel):
    """Flexible input for analysis and streaming control.

    Supports multiple shapes used across the codebase:
    - Single base64 frame: `frame_b64` or `image_b64`
    - Single image path on disk: `frame_path`
    - Multiple frames: `frames` can be either a list[str] of paths or
      an object like `{ "paths": [ ... ] }` (as returned by /video/upload).
    - Optional `session_id` and arbitrary `metadata` for correlation.
    """

    frame_b64: Optional[str] = None
    image_b64: Optional[str] = None
    frame_path: Optional[str] = None
    frames: Optional[Any] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
