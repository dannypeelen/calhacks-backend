from fastapi import APIRouter, HTTPException, Query
from app.services.analyzer import analyze_video
from app.schemas.video import VideoInput
from app.models import model_theft as theft_model
from app.models import model_weapon as weapon_model
from app.models import model_face_detection as face_model
from app.core.logger import get_logger
from pathlib import Path
import base64
import numpy as np
import cv2
from datetime import datetime
from app.services.face_embedder import process_faces_from_frame

router = APIRouter()
log = get_logger(__name__)

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
    result = await theft_model.async_detect_theft(img, conf_thresh=conf_thresh)
    return result


@router.post("/weapon")
async def detect_weapon_only(video: VideoInput):
    """Run only the weapon detector on the given frame and return its raw result."""
    img = _first_image_bytes(video)
    result = await weapon_model.async_detect_weapon(img)
    return result


@router.post("/face")
async def detect_face_only(video: VideoInput):
    """Run only the face detector on the given frame and return its raw result."""
    img = _first_image_bytes(video)
    result = await face_model.async_detect_face(img)
    return result


@router.post("/debug/baseten")
async def debug_baseten_models(video: VideoInput, conf_thresh: float = Query(0.5, ge=0.0, le=1.0)):
    """Call all Baseten-backed models, print their raw responses, and return them."""
    img = _first_image_bytes(video)

    theft_res = await theft_model.async_detect_theft(img, conf_thresh=conf_thresh)
    weapon_res = await weapon_model.async_detect_weapon(img)
    face_res = await face_model.async_detect_face(img)

    log.info("[Baseten Debug] Theft ok=%s raw=%s", theft_res.get("ok"), theft_res.get("raw"))
    log.info("[Baseten Debug] Weapon ok=%s raw=%s", weapon_res.get("ok"), weapon_res.get("raw"))
    log.info("[Baseten Debug] Face ok=%s raw=%s", face_res.get("ok"), face_res.get("raw"))

    all_ok = all(res.get("ok") for res in (theft_res, weapon_res, face_res))
    message = "All Baseten models responded successfully" if all_ok else "One or more Baseten models reported errors"

    return {
        "message": message,
        "results": {
            "theft": theft_res,
            "weapon": weapon_res,
            "face": face_res,
        }
    }

@router.post("/event_faces")
async def detect_and_store_faces(video: VideoInput):
    """
    Run theft + weapon detectors, and if either triggers,
    detect faces, embed them, and store embeddings in ChromaDB.
    Returns structured data matching the Convex schema format.
    """
    img_bytes = _first_image_bytes(video)

    # Convert bytes to OpenCV frame
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run theft and weapon detection
    theft_res = await theft_model.async_detect_theft(img_bytes)
    weapon_res = await weapon_model.async_detect_weapon(img_bytes)

    theft_ok = theft_res.get("ok") and theft_res.get("detections")
    weapon_ok = weapon_res.get("ok") and weapon_res.get("detections")

    if not (theft_ok or weapon_ok):
        return {
            "alerts": [],
            "faces": [],
            "footageAnalysis": {
                "detectionType": "normal",
                "timestamps": [],
                "faces_detected": {
                    "status": "completed",
                    "faceID": None
                }
            },
            "message": "No theft or weapon event detected.",
            "faces_stored": 0
        }

    event_type = "theft" if theft_ok else "weapon"
    theft_conf = theft_res.get("confidence") if theft_ok else None
    weapon_conf = weapon_res.get("confidence") if weapon_ok else None
    theft_boxes = (theft_res.get("coordinates") or {}).get("boxes", []) if theft_ok else []
    weapon_boxes = (weapon_res.get("coordinates") or {}).get("boxes", []) if weapon_ok else []

    stored_faces = process_faces_from_frame(
        frame,
        event_type=event_type,
        theft_conf=theft_conf,
        weapon_conf=weapon_conf
    )

    # Structure response to match Convex schema
    faces_data = []
    alerts_data = []

    for face in stored_faces:
        # Format face data for Convex faces table
        face_record = {
            "vectors": face["vectors"],
            "faceID": face["faceID"],
            "faceUrl": face["faceUrl"],
            "createdAt": face["createdAt"],
            "threatType": face["threatType"]
        }
        faces_data.append(face_record)

        # Create corresponding alert record
        alert_record = {
            "type": face["threatType"],
            "faceID": face["faceID"],
            "summary": f"Face detected during {face['threatType']} event",
            "imageUrl": face["faceUrl"],  # Can be populated with actual image URL
            "createdAt": face["createdAt"]
        }
        alerts_data.append(alert_record)

    # Create footage analysis record
    footage_analysis = {
        "detectionType": f"{event_type.title()} Detection",
        "timestamps": [
            {
                "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
                "action": event_type,
                "description": f"{event_type.title()} event detected with {len(stored_faces)} faces identified",
                "severity": "high" if len(stored_faces) > 0 else "medium"
            }
        ],
        "faces_detected": {
            "status": "completed",
            "faceID": faces_data[0]["faceID"] if faces_data else None
        },
        "detection_regions": {
            "theft": theft_boxes,
            "weapon": weapon_boxes,
        }
    }

    return {
        "alerts": alerts_data,
        "faces": faces_data,
        "footageAnalysis": footage_analysis,
        "event_triggered": True,
        "event_type": event_type,
        "faces_stored": len(stored_faces),
        "detection_boxes": {
            "theft": theft_boxes,
            "weapon": weapon_boxes,
        },
        "message": f"Successfully processed {len(stored_faces)} faces for {event_type} event"
    }
