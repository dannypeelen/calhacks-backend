from fastapi import APIRouter
from app.services.analyzer import analyze_video
from app.schemas.video import VideoInput

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

