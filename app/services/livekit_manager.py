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
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from app.core.config import get_settings
from app.core.logger import get_logger

# Optional imports; keep module import-safe even if SDK not installed yet
try:
    from livekit import api as lk_api  # token, room/server mgmt (server SDK)
except Exception:  # pragma: no cover
    lk_api = None  # type: ignore


log = get_logger(__name__)


DetectionSink = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class LiveKitSession:
    room_name: str
    identity: str
    data_publish_fn: Optional[Callable[[str], Awaitable[None]]] = None
    last_frame_ts: float = 0.0
    running: bool = False


class LiveKitManager:
    """Encapsulates LiveKit auth and media subscription lifecycle."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._sessions: Dict[str, LiveKitSession] = {}

    def create_join_token(self, room_name: str, identity: Optional[str] = None) -> str:
        """Create an access token for a client or backend bot to join a room."""
        if lk_api is None:
            raise RuntimeError("livekit-server-sdk not available")

        identity = identity or f"sentri-backend-{uuid.uuid4()}"
        at = lk_api.AccessToken(
            api_key=self._settings.LIVEKIT_API_KEY,
            api_secret=self._settings.LIVEKIT_API_SECRET,
        )
        at.add_grant(lk_api.VideoGrant(room=room_name, room_join=True))
        at.identity = identity
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
        session_id = str(uuid.uuid4())
        identity = identity or f"sentri-backend-{session_id}"
        session = LiveKitSession(room_name=room_name, identity=identity)
        self._sessions[session_id] = session

        # In a production implementation, you would:
        # - Use a media-capable SDK to join and subscribe to tracks (Agents SDK)
        # - Register callbacks for on_track_subscribed / on_track_unsubscribed
        # - For each video frame, throttle to ~1FPS and forward to analyzers

        session.running = True
        asyncio.create_task(self._mock_media_loop(session_id, result_sink))
        log.info("LiveKit session %s started for room=%s", session_id, room_name)
        return session_id

    async def stop_room_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        session.running = False
        self._sessions.pop(session_id, None)
        log.info("LiveKit session %s stopped", session_id)

    async def send_data(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Send payload to connected clients over a data track.

        Here, we model it as an async function stored in the session; your
        concrete implementation should publish on a LiveKit data track.
        """
        session = self._sessions.get(session_id)
        if not session:
            return
        msg = json.dumps(payload)
        if session.data_publish_fn:
            await session.data_publish_fn(msg)
        else:
            log.debug("[Loopback] Data message: %s", msg)

    async def _mock_media_loop(
        self, session_id: str, result_sink: Optional[DetectionSink]
    ) -> None:
        """Placeholder media loop.

        Replace this with real LiveKit track subscription and frame callbacks.
        This mock emits a tick every second to demonstrate end-to-end wiring.
        """
        session = self._sessions.get(session_id)
        if not session:
            return
        counter = 0
        while session.running:
            now = time.time()
            if now - session.last_frame_ts >= 1.0:
                session.last_frame_ts = now
                # Simulate a frame buffer; in real code, you would get raw
                # frame bytes from the LiveKit video track.
                fake_frame_bytes = f"frame-{counter}".encode()
                fake_b64 = base64.b64encode(fake_frame_bytes).decode()

                # Send to baseten/fetch agents via your analyzer pipeline
                result = await self._run_analysis_stub(fake_b64)

                # Emit results back to clients
                payload = {"type": "detection", "seq": counter, "result": result}
                await self.send_data(session_id, payload)

                # Also forward to external sink if provided (e.g., websocket)
                if result_sink:
                    await result_sink(payload)

                counter += 1
            await asyncio.sleep(0.05)

    async def _run_analysis_stub(self, frame_b64: str) -> Dict[str, Any]:
        """Call into analyzer/model clients here.

        Integrate app/services/baseten_client.py and fetchai_agent.py once
        they expose async APIs. Returning a synthetic structure for now.
        """
        # Example placeholder output
        return {
            "weapons": {"score": 0.02},
            "robbery": {"score": 0.01},
            "threat": {"level": "low", "score": 2},
        }


# Singleton accessor
_manager: Optional[LiveKitManager] = None


def get_livekit_manager() -> LiveKitManager:
    global _manager
    if _manager is None:
        _manager = LiveKitManager()
    return _manager

