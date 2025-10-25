from fastapi import APIRouter, File, UploadFile, WebSocket
from app.services.video_processor import process_uploaded_video, process_webcam_frame
from app.services.livekit_manager import get_livekit_manager
from app.core.logger import get_logger

router = APIRouter()
log = get_logger(__name__)

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handle video uploads for offline analysis."""
    results = await process_uploaded_video(file)
    return {"message": "Video received", "results": results}


@router.websocket("/stream")
async def video_stream(websocket: WebSocket):
    """
    WebSocket control plane for LiveKit streaming.

    Usage:
    - Send a JSON message {"action": "start", "room": "<room>", "identity": "<id>"}
      to start a LiveKit-backed session. The backend subscribes to the room
      and forwards detections back over this WebSocket and the LiveKit data track.
    - Send {"action": "stop", "session_id": "..."} to stop.

    For legacy clients still sending base64 frames, send
    {"action": "frame", "data": "data:image/jpeg;base64,..."} and the server
    will process at ~1 FPS.
    """
    await websocket.accept()
    lk = get_livekit_manager()
    active_session: str | None = None

    async def ws_sink(payload):
        try:
            await websocket.send_json(payload)
        except Exception:
            log.exception("Failed sending WS payload")

    try:
        while True:
            msg = await websocket.receive_text()
            # Simple protocol: try to parse JSON first, else treat as raw frame
            action = None
            payload = None
            try:
                payload = __import__("json").loads(msg)
                action = payload.get("action")
            except Exception:
                payload = None

            if action == "start":
                room = payload.get("room")
                identity = payload.get("identity")
                if not room:
                    await websocket.send_json({"ok": False, "error": "room required"})
                    continue
                session_id = await lk.start_room_session(room, identity=identity, result_sink=ws_sink)
                active_session = session_id
                await websocket.send_json({"ok": True, "session_id": session_id})
            elif action == "stop":
                sid = payload.get("session_id") or active_session
                if not sid:
                    await websocket.send_json({"ok": False, "error": "session_id required"})
                    continue
                await lk.stop_room_session(sid)
                if active_session == sid:
                    active_session = None
                await websocket.send_json({"ok": True})
            elif action == "frame":
                data = payload.get("data") if payload else msg
                results = await process_webcam_frame(data)
                await websocket.send_json(results)
            else:
                # Legacy path: treat raw text as base64 frame
                if payload is None:
                    results = await process_webcam_frame(msg)
                    await websocket.send_json(results)
                else:
                    await websocket.send_json({"ok": False, "error": "unknown action"})
    except Exception:
        log.exception("/video/stream handler crashed")
    finally:
        if active_session:
            try:
                await lk.stop_room_session(active_session)
            except Exception:
                log.exception("Failed to stop LiveKit session %s", active_session)
