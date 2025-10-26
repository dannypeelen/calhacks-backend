1. StreamProcessor service (new file app/services/stream_processor.py)

Class StreamProcessor with:
set_frame(np.ndarray): push into asyncio.Queue(maxsize=1) (drop oldest if full).
start() / stop() to manage two asyncio.Tasks.
Loops:
face_loop: every face_interval seconds:
grab newest frame (or reuse last), call async_detect_face.
drop detections with conf/confidence/score < 0.5.
update latest_results["face"].
await publish_callback({"face": latest_face, "meta": {...}}).
threat_loop: every threat_interval seconds:
similar pattern; gather theft + weapon tasks if endpoints exist.
update latest_results["theft"], ["weapon"].
publish payload.
Accept publish_callbacks list (async callables). When results arrive, iterate and await cb(payload).
Accept optional interval overrides and default to env (FACE_INTERVAL default 0.2, THREAT_INTERVAL default 0.5).
Provide latest_results property for legacy WS queries if needed.

2. Config update (app/core/config.py)

Add FACE_INTERVAL: float = 0.2 and THREAT_INTERVAL: float = 0.5.
Use Settings to pull from .env.

3. LiveKitManager wiring

On start_room_session:
Create StreamProcessor with intervals from settings or per-message overrides.
Register publisher callback that:
Publishes JSON to LiveKit data channel
await room.local_participant.publish_data(json.dumps(payload).encode(), lk_rtc.DataPacketKind.RELIABLE)
Forwards to ws sink (existing result_sink parameter).
Store processor in the session state.
On track_subscribed for remote video:
For each frame, convert to ndarray (already doing) and call processor.set_frame(ndarray).
On session stop or disconnect:
await processor.stop().
Disconnect room as before.

4. Legacy /video/stream “frame” action

Instead of process_webcam_frame, push frame bytes into the same StreamProcessor queue.
Optionally keep process_webcam_frame for immediate responses, but do not block loops; the result sink will get asynchronous updates.

5. JSON Payload format

For each publish, send something like:
{
  "type": "detection",
  "session_id": "...",
  "participant": "...",
  "results": {
    "face": { ... }, 
    "theft": { ... },
    "weapon": { ... }
  },
  "meta": {
    "face_interval": 0.2,
    "threat_interval": 0.5,
    "timestamp": "...",
    "source": "livekit"
  }
}
On face updates, include filtered faces list; on threat updates, include latest theft/weapon scores. Combine them when both loops produce results (or publish separate messages with partial data; just be consistent).

6. Logging

INFO log when publishing each payload, e.g., log.info("Face results ok=%s detections=%d", result["ok"], len(...)).
DEBUG log the raw Baseten JSON if needed.

7. Tests

Unit-test StreamProcessor by mocking async_detect_face/async_detect_theft/async_detect_weapon and verifying:
Face loop drops low-confidence boxes.
Threat loop runs independently (e.g., use fake intervals of 0.01/0.02 and ensure both callbacks fire).
Test /video/stream by simulating a start message with interval overrides; confirm that a frame pushed via "frame" action eventually triggers the publish callback.

8. Frontend docs summary

Join + publish:
GET /video/livekit/join → {url, token}; connect with @livekit/client; publish camera track.
Open WS /video/stream, send {"action":"start","room":"<room>","identity":"<id>","face_interval":0.2,"threat_interval":0.5}.
Receive detections:
Listen to LiveKit data channel (RoomEvent.DataReceived); fallback to WS messages.
Each message contains results.face, results.theft, results.weapon.
Render loop:
Keep latestResults state.
Use requestAnimationFrame to draw camera video; overlay face boxes where conf ≥ 0.5.
Show threat labels when score >= 0.5.
Stopping:
Send {"action":"stop","session_id":"..."} or close WS; disconnect LiveKit as needed.