from fastapi import APIRouter, HTTPException, Query
from app.services.analyzer import analyze_video
from app.schemas.video import VideoInput
from app.models.model_theft import async_detect_theft
from app.models.model_weapon import async_detect_weapon
from pathlib import Path
import base64

router = APIRouter()

@router.post("/run")
async def run_analysis(video: VideoInput):
    """
    Send video frames or URLs to remote models (Baseten + FetchAI).
    Returns aggregated detections.
    """
    results = await analyze_video(video)
    return {"detections": results}


@router.get("/models")
def list_models():
    """Return currently available models and their endpoints."""
    return {
        "models": {
            "weapons": "Baseten ID 12345",
            "robbery": "Baseten ID 67890",
            "face": "FetchAI agent abc123"
        }
    }


def _first_image_bytes(video: VideoInput) -> bytes:
    # Base64 fields
    b64 = video.image_b64 or video.frame_b64
    if isinstance(b64, str) and b64:
        if b64.lstrip().startswith("data:") and "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            return base64.b64decode(b64, validate=False)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid base64 image")

    # Single path
    if isinstance(video.frame_path, str) and video.frame_path:
        p = Path(video.frame_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="frame_path not found")
        return p.read_bytes()

    # Frames list
    frames = video.frames
    path = None
    if isinstance(frames, dict) and isinstance(frames.get("paths"), list) and frames["paths"]:
        path = str(frames["paths"][0])
    elif isinstance(frames, list) and frames:
        path = str(frames[0])
    if path:
        p = Path(path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="frame path not found")
        return p.read_bytes()

    raise HTTPException(status_code=422, detail="No image provided")


@router.post("/theft")
async def detect_theft_only(video: VideoInput, conf_thresh: float = Query(0.5, ge=0.0, le=1.0)):
    """Run only the theft detector on the given frame and return its raw result."""
    img = _first_image_bytes(video)
    result = await async_detect_theft(img, conf_thresh=conf_thresh)
    return result


@router.post("/weapon")
async def detect_weapon_only(video: VideoInput):
    """Run only the weapon detector on the given frame and return its raw result."""
    img = _first_image_bytes(video)
    result = await async_detect_weapon(img)
    return result
