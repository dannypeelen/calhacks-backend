# Frontend Integration Guide: Real-time Video Analysis

This document outlines how to integrate with the CalHacks backend for real-time face, theft, and weapon detection using LiveKit and WebSocket connections.

## Setup

### 1. Join LiveKit Room

```javascript
import { Room, RoomEvent } from '@livekit/client';

// Get connection details from backend
const response = await fetch('/video/livekit/join', {
  method: 'GET',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    room: 'my-analysis-room',
    identity: 'user-123'
  })
});

const { url, token } = await response.json();

// Connect to LiveKit room
const room = new Room();
await room.connect(url, token);

// Publish camera track
await room.localParticipant.setCameraEnabled(true);
const cameraTrack = room.localParticipant.videoTracks.values().next().value;
```

### 2. Initialize WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/video/stream');

// Send start message with detection intervals
ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'start',
    room: 'my-analysis-room',
    identity: 'user-123',
    face_interval: 0.2,    // Face detection every 200ms
    threat_interval: 0.5   // Threat detection every 500ms
  }));
};
```

## Receiving Detections

### Option 1: LiveKit Data Channel (Recommended)

```javascript
room.on(RoomEvent.DataReceived, (payload, participant, kind) => {
  const data = JSON.parse(new TextDecoder().decode(payload));
  handleDetectionResults(data);
});
```

### Option 2: WebSocket Messages (Fallback)

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleDetectionResults(data);
};

function handleDetectionResults(data) {
  if (data.type === 'detection') {
    const { results, session_id, participant, meta } = data;

    // Update state with latest detection results
    setLatestResults(prev => ({
      ...prev,
      face: results.face || prev.face,
      theft: results.theft || prev.theft,
      weapon: results.weapon || prev.weapon,
      session_id,
      participant,
      meta
    }));
  }
}
```

## Render Loop

### Canvas Setup

```javascript
const canvas = document.getElementById('video-canvas');
const ctx = canvas.getContext('2d');
let latestResults = {};

function renderLoop() {
  // Draw camera feed
  if (cameraTrack) {
    const videoElement = cameraTrack.attachedElements[0];
    if (videoElement && videoElement.readyState >= 2) {
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    }
  }

  // Overlay face detection boxes
  if (latestResults.face?.ok && latestResults.face.detections?.faces) {
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;

    latestResults.face.detections.faces.forEach(face => {
      if (face.conf >= 0.5) {  // Confidence threshold
        const [x1, y1, x2, y2] = face.bbox || face.box;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
    });
  }

  // Show threat detection labels
  if (latestResults.theft?.ok && latestResults.theft.detections > 0.5) {
    ctx.fillStyle = '#ff0000';
    ctx.font = '24px Arial';
    ctx.fillText(`🚨 THEFT DETECTED: ${latestResults.theft.detections.toFixed(2)}`, 10, 30);
  }

  if (latestResults.weapon?.ok && latestResults.weapon.detections > 0.5) {
    ctx.fillStyle = '#ff0000';
    ctx.font = '24px Arial';
    ctx.fillText(`⚠️ WEAPON DETECTED: ${latestResults.weapon.detections.toFixed(2)}`, 10, 60);
  }

  requestAnimationFrame(renderLoop);
}

// Start render loop
renderLoop();
```

## State Management

```javascript
const [latestResults, setLatestResults] = useState({
  face: { ok: false, detections: null },
  theft: { ok: false, detections: null },
  weapon: { ok: false, detections: null },
  session_id: null,
  participant: null,
  meta: null
});

// Update results when new data arrives
function handleDetectionResults(data) {
  if (data.type === 'detection') {
    setLatestResults(prev => ({
      ...prev,
      face: data.results.face || prev.face,
      theft: data.results.theft || prev.theft,
      weapon: data.results.weapon || prev.weapon,
      session_id: data.session_id,
      participant: data.participant,
      meta: data.meta
    }));
  }
}
```

## Stopping Analysis

### WebSocket Method

```javascript
function stopAnalysis() {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      action: 'stop',
      session_id: latestResults.session_id
    }));
  }
  ws.close();
}
```

### LiveKit Method

```javascript
async function stopAnalysis() {
  // Disconnect from LiveKit room
  await room.disconnect();

  // Also stop via WebSocket if connected
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      action: 'stop',
      session_id: latestResults.session_id
    }));
    ws.close();
  }
}
```

## Error Handling

```javascript
// WebSocket error handling
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
  // Fallback to LiveKit-only mode or show error UI
};

// LiveKit error handling
room.on(RoomEvent.ConnectionStateChanged, (state) => {
  if (state === 'disconnected') {
    console.log('LiveKit disconnected');
    // Handle reconnection or show offline state
  }
});

// Detection service errors
function handleDetectionResults(data) {
  if (data.type === 'detection') {
    // Check individual service health
    if (!data.results.face?.ok) {
      console.warn('Face detection service unavailable');
    }
    if (!data.results.theft?.ok) {
      console.warn('Theft detection service unavailable');
    }
    if (!data.results.weapon?.ok) {
      console.warn('Weapon detection service unavailable');
    }

    // Update UI state accordingly
    setLatestResults(prev => ({
      ...prev,
      face: data.results.face || prev.face,
      theft: data.results.theft || prev.theft,
      weapon: data.results.weapon || prev.weapon,
      session_id: data.session_id,
      participant: data.participant,
      meta: data.meta
    }));
  }
}
```

## Performance Considerations

1. **Frame Rate**: The backend processes face detection at 0.2s intervals and threat detection at 0.5s intervals by default
2. **Canvas Rendering**: Use `requestAnimationFrame` for smooth 60fps rendering
3. **Memory Management**: Clear canvas before each frame to prevent memory leaks
4. **Connection Management**: Implement reconnection logic for both LiveKit and WebSocket connections

## Example Response Format

```json
{
  "type": "detection",
  "session_id": "abc123",
  "participant": "participant_xyz",
  "results": {
    "face": {
      "ok": true,
      "model": "baseten:face",
      "detections": {
        "faces": [
          {
            "bbox": [100, 100, 200, 200],
            "conf": 0.95,
            "landmarks": [...]
          }
        ]
      }
    },
    "theft": {
      "ok": true,
      "model": "baseten:theft",
      "detections": 0.85
    },
    "weapon": {
      "ok": false,
      "model": "baseten:weapon",
      "error": "Service temporarily unavailable"
    }
  },
  "meta": {
    "source": "threat",
    "face_interval": 0.2,
    "threat_interval": 0.5,
    "timestamp": 1699123456.789
  }
}
```
