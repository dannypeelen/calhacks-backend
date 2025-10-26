from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models import model_face_detection as face_model
from app.models import model_theft as theft_model
from app.models import model_weapon as weapon_model

PublishCallback = Callable[[Dict[str, Any]], Awaitable[None]]

log = get_logger(__name__)
settings = get_settings()

CONF_THRESHOLD = 0.5


class StreamProcessor:
    """Runs face/theft/weapon detectors on incoming frames at independent cadences."""

    def __init__(
        self,
        face_interval: Optional[float] = None,
        threat_interval: Optional[float] = None,
        publish_callbacks: Optional[List[PublishCallback]] = None,
        session_id: Optional[str] = None,
        participant_sid: Optional[str] = None,
    ) -> None:
        self.face_interval = face_interval or settings.FACE_INTERVAL or 0.2
        self.threat_interval = threat_interval or settings.THREAT_INTERVAL or 0.5
        self.publish_callbacks = publish_callbacks or []
        self.session_id = session_id
        self.participant_sid = participant_sid

        self._frame_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(maxsize=1)
        self._face_task: Optional[asyncio.Task] = None
        self._threat_task: Optional[asyncio.Task] = None
        self._running = False

        empty = {"ok": False, "error": "No data", "detections": None}
        self._latest_results: Dict[str, Dict[str, Any]] = {
            "face": {**empty, "model": "baseten:face"},
            "theft": {**empty, "model": "baseten:theft"},
            "weapon": {**empty, "model": "baseten:weapon"},
        }

    @property
    def latest_results(self) -> Dict[str, Dict[str, Any]]:
        return self._latest_results

    def set_frame(self, frame: np.ndarray) -> None:
        if not self._running:
            return
        try:
            if self._frame_queue.full():
                # drop oldest
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._face_task = asyncio.create_task(self._face_loop(), name="stream-face-loop")
        self._threat_task = asyncio.create_task(self._threat_loop(), name="stream-threat-loop")

    async def stop(self) -> None:
        self._running = False
        await self._frame_queue.put(None)
        tasks = [t for t in (self._face_task, self._threat_task) if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._face_task = None
        self._threat_task = None

    async def _face_loop(self) -> None:
        last_frame: Optional[np.ndarray] = None
        while self._running:
            start = time.perf_counter()
            frame = await self._get_frame(last_frame)
            if frame is not None:
                last_frame = frame
                try:
                    result = await face_model.async_detect_face(frame)
                except Exception as exc:  # pragma: no cover
                    log.exception("Face detection failed: %s", exc)
                    result = {"ok": False, "error": str(exc), "model": "baseten:face"}
                self._latest_results["face"] = self._filter_face_result(result)
                await self._publish(
                    {
                        "type": "detection",
                        "session_id": self.session_id,
                        "participant": self.participant_sid,
                        "results": {"face": self._latest_results["face"]},
                        "meta": self._meta("face"),
                    }
                )
            await self._sleep_remaining(start, self.face_interval)

    async def _threat_loop(self) -> None:
        last_frame: Optional[np.ndarray] = None
        while self._running:
            start = time.perf_counter()
            frame = await self._get_frame(last_frame)
            if frame is not None:
                last_frame = frame
                payload: Dict[str, Dict[str, Any]] = {}
                tasks = []
                labels = []
                if settings.BASETEN_THEFT_ENDPOINT:
                    tasks.append(theft_model.async_detect_theft(frame))
                    labels.append("theft")
                if settings.BASETEN_WEAPON_ENDPOINT:
                    tasks.append(weapon_model.async_detect_weapon(frame))
                    labels.append("weapon")
                if tasks:
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    for label, resp in zip(labels, responses):
                        if isinstance(resp, Exception):  # pragma: no cover
                            log.exception("%s detection failed: %s", label, resp)
                            result = {"ok": False, "error": str(resp), "model": f"baseten:{label}"}
                        else:
                            result = resp
                        self._latest_results[label] = result
                        payload[label] = result
                    if payload:
                        await self._publish(
                            {
                                "type": "detection",
                                "session_id": self.session_id,
                                "participant": self.participant_sid,
                                "results": payload,
                                "meta": self._meta("threat"),
                            }
                        )
            await self._sleep_remaining(start, self.threat_interval)

    async def _get_frame(self, fallback: Optional[np.ndarray]) -> Optional[np.ndarray]:
        try:
            frame = await asyncio.wait_for(self._frame_queue.get(), timeout=0.01)
            if frame is None:
                return fallback
            return frame
        except asyncio.TimeoutError:
            return fallback

    async def _sleep_remaining(self, start: float, interval: float) -> None:
        elapsed = time.perf_counter() - start
        remaining = max(0.0, interval - elapsed)
        await asyncio.sleep(remaining)

    def _filter_face_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        detections = result.get("detections")
        filtered = []
        faces = None
        if isinstance(detections, dict):
            faces = detections.get("faces") or detections.get("detections")
        elif isinstance(detections, list):
            faces = detections
        if isinstance(faces, list):
            for det in faces:
                if not isinstance(det, dict):
                    continue
                score = det.get("conf") or det.get("confidence") or det.get("score") or 0.0
                if float(score) >= CONF_THRESHOLD:
                    filtered.append(det)
        new_det = {"faces": filtered}
        return {**result, "detections": new_det}

    async def _publish(self, payload: Dict[str, Any]) -> None:
        if not self.publish_callbacks:
            return
        
        # INFO log when publishing each payload
        results = payload.get("results", {})
        face_results = results.get("face", {})
        theft_results = results.get("theft", {})
        weapon_results = results.get("weapon", {})
        
        face_ok = face_results.get("ok", False)
        face_detections = len(face_results.get("detections", {}).get("faces", [])) if face_results.get("detections") else 0
        theft_ok = theft_results.get("ok", False)
        weapon_ok = weapon_results.get("ok", False)
        
        log.info(
            "Publishing detection payload: face_ok=%s face_detections=%d theft_ok=%s weapon_ok=%s session=%s participant=%s",
            face_ok, face_detections, theft_ok, weapon_ok, 
            self.session_id, self.participant_sid
        )
        
        for cb in self.publish_callbacks:
            try:
                await cb(payload)
            except Exception as exc:  # pragma: no cover
                log.exception("Publish callback failed: %s", exc)

    def _meta(self, source: str) -> Dict[str, Any]:
        return {
            "source": source,
            "face_interval": self.face_interval,
            "threat_interval": self.threat_interval,
            "timestamp": time.time(),
        }
