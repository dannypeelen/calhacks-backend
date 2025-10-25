"""
LiveKit integration for real-time streaming and bidirectional event data.

Responsibilities:
- Connect to a LiveKit room using env-configured credentials.
- Subscribe to remote participants' video tracks.
- Capture frames at ~1 FPS and pass them to analyzer/model clients.
- Publish real-time detection results back to clients via a data track.

Notes:
- This is a modular, asyncio-friendly scaffold. The media-plane subscription
  APIs differ between LiveKit Python libs; production systems typically use
  LiveKit Agents SDK for media processing. Per request, we assume
  `livekit-server-sdk` is available for server auth/token handling.
  Replace placeholders with concrete subscription code depending on your
  deployment and SDK choice.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.analyzer import analyze_video

# Optional imports; keep module import-safe even if SDK not installed yet
try:
    from livekit import api as lk_api  # token, room/server mgmt (server SDK)
except Exception:  # pragma: no cover
    lk_api = None  # type: ignore

try:
    from livekit import rtc as lk_rtc  # media-plane RTC client (Python)
except Exception:  # pragma: no cover
    lk_rtc = None  # type: ignore


log = get_logger(__name__)


DetectionSink = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class LiveKitSession:
    room_name: str
    identity: str
    room: Optional["lk_rtc.Room"] = None
    data_publish_fn: Optional[Callable[[str], Awaitable[None]]] = None
    per_participant_ts: Dict[str, float] = field(default_factory=dict)
    running: bool = False


class LiveKitManager:
    """Encapsulates LiveKit auth and media subscription lifecycle."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._sessions: Dict[str, LiveKitSession] = {}

    def create_join_token(self, room_name: str, identity: Optional[str] = None) -> str:
        """Create an access token for a client or backend bot to join a room.

        Supports different livekit python SDK variants:
        - Newer style: set `at.grants = VideoGrants(...)`
        - Older style: `at.add_grant(VideoGrant(...))`
        """
        if lk_api is None:
            raise RuntimeError("livekit.api not available; ensure 'livekit'/'livekit-api' is installed")

        identity = identity or f"sentri-backend-{uuid.uuid4()}"
        at = lk_api.AccessToken(
            api_key=self._settings.LIVEKIT_API_KEY,
            api_secret=self._settings.LIVEKIT_API_SECRET,
        )
        at.identity = identity

        # Try modern grants API first
        Grants = getattr(lk_api, "VideoGrants", None)
        if Grants is not None:
            at.grants = Grants(room=room_name, room_join=True)  # type: ignore[attr-defined]
        else:
            # Fallback to legacy add_grant API
            Grant = getattr(lk_api, "VideoGrant", None)
            if Grant is None or not hasattr(at, "add_grant"):
                raise RuntimeError("Unsupported livekit token API: missing grant methods")
            at.add_grant(Grant(room=room_name, room_join=True))  # type: ignore[attr-defined]

        token = at.to_jwt()
        return token

    async def start_room_session(
        self,
        room_name: str,
        identity: Optional[str] = None,
        result_sink: Optional[DetectionSink] = None,
    ) -> str:
        """Start backend LiveKit session and subscribe to video tracks.

        Returns a session_id to manage lifecycle from the API route.
        """
        if lk_rtc is None:
            raise RuntimeError("livekit.rtc not available; install 'livekit-rtc'")

        session_id = str(uuid.uuid4())
        identity = identity or f"sentri-backend-{session_id}"
        session = LiveKitSession(room_name=room_name, identity=identity)
        self._sessions[session_id] = session

        token = self.create_join_token(room_name=room_name, identity=identity)

        room = lk_rtc.Room()

        # Bind event handlers
        @room.on("participant_connected")
        def _on_participant_connected(participant):  # type: ignore[no-redef]
            log.info("Participant connected: %s", getattr(participant, "identity", "?"))

        @room.on("participant_disconnected")
        def _on_participant_disconnected(participant):  # type: ignore[no-redef]
            pid = getattr(participant, "sid", "?")
            session.per_participant_ts.pop(pid, None)
            log.info("Participant disconnected: %s", pid)

        @room.on("track_subscribed")
        def _on_track_subscribed(track, publication, participant):  # type: ignore[no-redef]
            try:
                if isinstance(track, lk_rtc.RemoteVideoTrack):
                    # Attach a frame callback
                    def _on_frame(frame: "lk_rtc.VideoFrame") -> None:
                        asyncio.create_task(self._handle_video_frame(session_id, participant, frame, result_sink))

                    # API differences exist; try common registration options
                    try:
                        track.add_listener(on_video_frame=_on_frame)  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            track.on("frame_received", _on_frame)  # type: ignore[attr-defined]
                        except Exception:
                            log.error("Unable to attach video frame listener to track")
            except Exception:
                log.exception("track_subscribed handler failed")

        @room.on("track_unsubscribed")
        def _on_track_unsubscribed(track, publication, participant):  # type: ignore[no-redef]
            pid = getattr(participant, "sid", "?")
            session.per_participant_ts.pop(pid, None)

        await room.connect(self._settings.LIVEKIT_URL, token)

        # Prepare data channel publisher
        async def _publish(msg: str) -> None:
            try:
                payload = msg.encode("utf-8")
                await room.local_participant.publish_data(
                    payload, lk_rtc.DataPacketKind.RELIABLE
                )
            except Exception:
                log.exception("Failed to publish data packet")

        session.room = room
        session.data_publish_fn = _publish
        session.running = True
        log.info("LiveKit session %s started for room=%s", session_id, room_name)
        return session_id

    async def _handle_video_frame(self, session_id: str, participant: Any, frame: Any, result_sink: Optional[DetectionSink]) -> None:
        """Handle incoming frames with 1 FPS throttling and analysis."""
        session = self._sessions.get(session_id)
        if not session or not session.running:
            return
        pid = getattr(participant, "sid", "?")
        now = time.time()
        last = session.per_participant_ts.get(pid, 0.0)
        if now - last < 1.0:
            return
        session.per_participant_ts[pid] = now

        # Convert frame to JPEG base64
        try:
            # frame.to_ndarray(format="bgr24") is common in livekit-rtc
            import cv2  # local import to avoid global dependency
            import numpy as np  # type: ignore

            img = frame.to_ndarray(format="bgr24")  # type: ignore[attr-defined]
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                log.warning("cv2.imencode failed for participant %s", pid)
                return
            image_b64 = base64.b64encode(enc.tobytes()).decode()
        except Exception:
            log.exception("Failed to convert frame to JPEG")
            return

        # Call analyzer on a single frame
        try:
            payload = {"image_b64": image_b64}
            result = await analyze_video(payload)
        except Exception:
            log.exception("Analyzer failed for participant %s", pid)
            return

        # Broadcast back via data channel and optional sink
        try:
            out = {"type": "detection", "session_id": session_id, "participant": pid, "result": result}
            await self.send_data(session_id, out)
            if result_sink:
                await result_sink(out)
        except Exception:
            log.exception("Failed to forward analysis result")

    async def stop_room_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        session.running = False
        try:
            if session.room is not None:
                await session.room.disconnect()
        except Exception:
            log.exception("Error disconnecting LiveKit room for session %s", session_id)
        self._sessions.pop(session_id, None)
        log.info("LiveKit session %s stopped", session_id)

    async def send_data(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Send payload to connected clients over a data track (if available)."""
        session = self._sessions.get(session_id)
        if not session:
            return
        msg = json.dumps(payload)
        if session.data_publish_fn:
            await session.data_publish_fn(msg)
        else:
            log.debug("[Loopback] Data message: %s", msg)


# Singleton accessor
_manager: Optional[LiveKitManager] = None


def get_livekit_manager() -> LiveKitManager:
    global _manager
    if _manager is None:
        _manager = LiveKitManager()
    return _manager
