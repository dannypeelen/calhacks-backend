from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import asyncio
import json
from app.core.config import get_settings
import logging
from app.services.baseten_client import get_baseten_client
import os

router = APIRouter()
log = logging.getLogger(__name__)


async def detect_theft_ws(image_b64: str, conf_thresh: float = 0.5) -> Dict[str, Any]:
    """Async theft detection for WebSocket."""
    try:
        endpoint_url = get_settings().BASETEN_THEFT_ENDPOINT or os.getenv("BASETEN_THEFT_ENDPOINT", "")
        client = get_baseten_client()
        extra_input = {"conf_thresh": float(conf_thresh)}
        resp = await client.apredict_image(endpoint_url, image_b64, extra_input=extra_input)

        detections = resp.get("detections") or resp.get("output") or resp.get("result") or []
        return {
            "ok": True,
            "model": "theft",
            "detections": detections,
        }
    except Exception as e:
        log.exception(f"Theft detection error: {e}")
        return {"ok": False, "model": "theft", "detections": [], "error": str(e)}


async def detect_weapon_ws(image_b64: str, conf_thresh: float = 0.5) -> Dict[str, Any]:
    """Async weapon detection for WebSocket."""
    try:
        endpoint_url = get_settings().BASETEN_WEAPON_ENDPOINT or os.getenv("BASETEN_WEAPON_ENDPOINT", "")
        client = get_baseten_client()
        extra_input = {"conf_thresh": float(conf_thresh)}
        resp = await client.apredict_image(endpoint_url, image_b64, extra_input=extra_input)

        detections = resp.get("detections") or resp.get("output") or resp.get("result") or []
        return {
            "ok": True,
            "model": "weapon",
            "detections": detections,
        }
    except Exception as e:
        log.exception(f"Weapon detection error: {e}")
        return {"ok": False, "model": "weapon", "detections": [], "error": str(e)}


async def detect_face_ws(image_b64: str, conf_thresh: float = 0.4) -> Dict[str, Any]:
    """Async face detection for WebSocket with configurable threshold (default 40%)."""
    try:
        client = get_baseten_client()
        endpoint_url = get_settings().BASETEN_FACE_ENDPOINT or os.getenv("BASETEN_FACE_ENDPOINT", "")

        if endpoint_url:
            extra_input = {"conf_thresh": float(conf_thresh)}
            resp = await client.apredict_image(endpoint_url, image_b64, extra_input=extra_input)
        else:
            resp = await client.apredict_face(image_b64)

        # Handle the nested structure from your example
        detections_data = resp.get("detections", {})

        # Normalize the face detection response
        if isinstance(detections_data, dict) and "faces" in detections_data:
            # Format: {"count": 2, "faces": [...]}
            faces = detections_data.get("faces", [])
            # Filter by confidence threshold
            filtered_faces = [f for f in faces if isinstance(f, dict) and f.get("conf", 0) >= conf_thresh]
            normalized_detections = filtered_faces
        else:
            # Already a list
            if isinstance(detections_data, list):
                filtered_detections = [d for d in detections_data if isinstance(d, dict) and d.get("conf", 0) >= conf_thresh]
                normalized_detections = filtered_detections
            else:
                normalized_detections = []

        return {
            "ok": True,
            "model": "face",
            "detections": normalized_detections,
            "count": len(normalized_detections),
        }
    except Exception as e:
        log.exception(f"Face detection error: {e}")
        return {"ok": False, "model": "face", "detections": [], "error": str(e), "count": 0}


async def run_all_detections(image_b64: str, conf_thresh: float = 0.4) -> Dict[str, Any]:
    """Run all three detections in parallel with configurable thresholds."""
    results = await asyncio.gather(
        detect_theft_ws(image_b64, conf_thresh=0.5),  # Theft uses 50% threshold
        detect_weapon_ws(image_b64, conf_thresh=0.5),  # Weapon uses 50% threshold
        detect_face_ws(image_b64, conf_thresh=conf_thresh),  # Face uses passed threshold (default 40%)
        return_exceptions=True
    )

    theft_result, weapon_result, face_result = results

    # Handle any exceptions
    if isinstance(theft_result, Exception):
        theft_result = {"ok": False, "model": "theft", "detections": [], "error": str(theft_result)}
    if isinstance(weapon_result, Exception):
        weapon_result = {"ok": False, "model": "weapon", "detections": [], "error": str(weapon_result)}
    if isinstance(face_result, Exception):
        face_result = {"ok": False, "model": "face", "detections": [], "error": str(face_result), "count": 0}

    return {
        "theft": theft_result,
        "weapon": weapon_result,
        "face": face_result,
    }


@router.websocket("/ws")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time detection across all models."""
    await websocket.accept()
    log.info("WebSocket connection established")

    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                image_b64 = message.get("image")
                conf_thresh = message.get("conf_thresh", 0.4)  # Default 40% for face detection

                if not image_b64:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No image data provided"
                    })
                    continue

                # Run all detections in parallel
                results = await run_all_detections(image_b64, conf_thresh=conf_thresh)

                # Send results back to client
                await websocket.send_json({
                    "type": "detections",
                    "data": results,
                    "timestamp": message.get("timestamp")
                })

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        log.info("WebSocket connection closed")
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        finally:
            await websocket.close()