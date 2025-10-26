"""
Live camera tester for the Baseten-backed face, theft, and weapon models.

Usage:
1. Activate your venv and ensure `.env` has BASETEN_* endpoints and BASETEN_API_KEY.
2. Run: `python tools/face_cam_test/face_cam_test.py`
   Optional flags:
     --camera 0            # webcam index
     --max-width 640       # resize frame width before sending to model
     --face-interval 0.2   # seconds between Baseten face inferences
     --threat-interval 0.5 # seconds between theft/weapon inferences

If the Baseten face endpoint is missing, the script automatically falls back to
OpenCV's Haar cascade detector (local only). Theft/weapon overlays are disabled
when their endpoints are missing.

Press `q` or `Esc` to stop the preview window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
import base64

# Ensure repo root is on sys.path so `import app` works when running this script directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import model_face_detection as face_model  # noqa: E402
from app.models import model_theft as theft_model  # noqa: E402
from app.models import model_weapon as weapon_model  # noqa: E402
from app.core.config import get_settings  # noqa: E402

_SETTINGS = get_settings()


CONF_THRESHOLD = 0.5
THEFT_MAX_WIDTH = 480
THEFT_JPEG_QUALITY = 80


def parse_detections(
    detections: Any,
    default_label: str = "object",
    min_score: float = CONF_THRESHOLD,
) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """Convert detection payloads into (bbox, label, score) tuples."""
    results: List[Tuple[Tuple[int, int, int, int], str, float]] = []

    det_list: Optional[List[Any]] = None
    if isinstance(detections, dict):
        if isinstance(detections.get("faces"), list):
            det_list = detections["faces"]
        elif isinstance(detections.get("detections"), list):
            det_list = detections["detections"]
        else:
            det_list = [detections]
    elif isinstance(detections, list):
        det_list = detections

    if not det_list:
        return results

    for det in det_list:
        if not isinstance(det, dict):
            continue
        bbox = det.get("box") or det.get("bbox") or det.get("rect")
        normalized = _normalize_box(bbox)
        if normalized is None:
            continue
        label = det.get("class_name") or det.get("label") or det.get("class") or default_label
        raw_score = det.get("confidence")
        if raw_score is None:
            raw_score = det.get("conf", det.get("score", 0.0))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score < min_score:
            continue
        results.append((normalized, str(label), score))

    return results


def threat_detections_from_result(result: Dict[str, Any], default_label: str = "threat") -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """Extract boxes for theft/weapon overlays, falling back to coordinates when needed."""
    if not result or not result.get("ok"):
        return []

    detections = parse_detections(result.get("detections"), default_label=default_label, min_score=0.05)
    if detections:
        return detections

    coords = (result.get("coordinates") or {}).get("boxes") or []
    confidence = float(result.get("confidence") or 0.0)
    normalized = []
    for box in coords:
        norm = _normalize_box(box)
        if norm is not None:
            normalized.append((norm, default_label, confidence))
    return normalized


def _normalize_box(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
    """Convert bbox to x1,y1,x2,y2 integers."""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            return int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x1, y1 = bbox["x"], bbox["y"]
            x2, y2 = x1 + bbox["w"], y1 + bbox["h"]
            return int(x1), int(y1), int(x2), int(y2)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
        return int(x1), int(y1), int(x2), int(y2)
    return None


def _prepare_theft_frame(frame: np.ndarray) -> np.ndarray:
    """Resize/copy frame for theft model to keep payload small."""
    theft_frame = frame
    if THEFT_MAX_WIDTH and theft_frame.shape[1] > THEFT_MAX_WIDTH:
        scale = THEFT_MAX_WIDTH / theft_frame.shape[1]
        theft_frame = cv2.resize(theft_frame, None, fx=scale, fy=scale)
    return theft_frame


def frame_to_base64(frame: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR frame into base64 JPEG string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(quality, 95))]
    success, encoded = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise ValueError("Failed to encode frame to JPEG")
    return base64.b64encode(encoded).decode()


async def run_async(camera: int, max_width: int, face_interval: float, threat_interval: float) -> None:
    load_dotenv(override=False)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera}")

    latest_results = {
        "face": {"ok": False, "detections": None, "model": "face"},
        "theft": {"ok": False, "detections": None, "model": "baseten:theft"},
        "weapon": {"ok": False, "detections": None, "model": "baseten:weapon"},
    }

    face_available = bool(_SETTINGS.BASETEN_FACE_ENDPOINT or os.getenv("BASETEN_FACE_ENDPOINT"))
    theft_available = bool(_SETTINGS.BASETEN_THEFT_ENDPOINT or os.getenv("BASETEN_THEFT_ENDPOINT"))
    weapon_available = bool(_SETTINGS.BASETEN_WEAPON_ENDPOINT or os.getenv("BASETEN_WEAPON_ENDPOINT"))

    cascade = None
    if not face_available:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")
        print("[FaceCam] Baseten face endpoint missing; using local Haar cascade detector.")
    else:
        print("[FaceCam] Using Baseten face detection endpoint.")
    if not theft_available:
        print("[FaceCam] Theft endpoint missing; theft alerts disabled.")
    if not weapon_available:
        print("[FaceCam] Weapon endpoint missing; weapon alerts disabled.")

    try:
        last_face_infer = 0.0
        last_threat_infer = 0.0
        face_task: Optional[asyncio.Task] = None
        threat_task: Optional[asyncio.Task] = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame grab failed, exiting.")
                break

            if max_width > 0 and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            if face_task and face_task.done():
                try:
                    latest_results["face"] = face_task.result()
                    print("[FaceCam] Face raw result:\n" + json.dumps(latest_results["face"], indent=2))
                    print_detection_summary("face", latest_results["face"])
                except Exception as exc:  # pragma: no cover
                    print(f"[FaceCam] Face inference failed: {exc}")
                face_task = None

            if threat_task and threat_task.done():
                try:
                    threat_results = threat_task.result()
                    latest_results["theft"] = threat_results.get("theft", latest_results["theft"])
                    latest_results["weapon"] = threat_results.get("weapon", latest_results["weapon"])
                    if theft_available:
                        print_detection_summary("theft", latest_results["theft"])
                    if weapon_available:
                        print_detection_summary("weapon", latest_results["weapon"])
                except Exception as exc:  # pragma: no cover
                    print(f"[FaceCam] Threat inference failed: {exc}")
                threat_task = None

            now = time.time()
            if face_available and (face_task is None) and (now - last_face_infer >= face_interval):
                last_face_infer = now
                face_task = asyncio.create_task(run_face_model(frame.copy()))
            if (theft_available or weapon_available) and (threat_task is None) and (
                now - last_threat_infer >= threat_interval
            ):
                last_threat_infer = now
                threat_task = asyncio.create_task(
                    run_threat_models(frame.copy(), theft_available, weapon_available)
                )

            face_result = (
                latest_results["face"] if face_available else local_haar_detect(frame, cascade)
            )
            if not face_available:
                latest_results["face"] = face_result

            overlay = frame.copy()
            status = "OK" if face_result.get("ok") else face_result.get("error", "Error")
            detections = face_result.get("detections")
            det_infos = (
                parse_detections(detections, default_label="face", min_score=CONF_THRESHOLD)
                if face_result.get("ok")
                else []
            )

            for (x1, y1, x2, y2), label, score in det_infos:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"{label}: {score:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if theft_available and latest_results["theft"].get("ok"):
                theft_boxes = threat_detections_from_result(latest_results["theft"], default_label="theft")
                for (x1, y1, x2, y2), label, score in theft_boxes:
                    color = (0, 0, 255) if score >= 0.5 else (0, 165, 255)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        overlay,
                        f"{label}: {score:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                overlay,
                f"Face model: {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if face_result.get("ok") else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            face_text, face_color = format_face_text(face_result)
            cv2.putText(
                overlay,
                face_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                face_color,
                2,
                cv2.LINE_AA,
            )

            status_lines: List[Tuple[str, Tuple[int, int, int]]] = []
            if theft_available:
                status_lines.append(
                    format_status("Theft", latest_results["theft"], extract_score(latest_results["theft"]))
                )
            if weapon_available:
                status_lines.append(
                    format_status("Weapon", latest_results["weapon"], extract_score(latest_results["weapon"]))
                )
            for idx, (txt, col) in enumerate(status_lines, start=2):
                cv2.putText(
                    overlay,
                    txt,
                    (10, 30 + idx * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    col,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Baseten Face Detection (press q to quit)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            await asyncio.sleep(0)  # yield control to event loop
    finally:
        cap.release()
        cv2.destroyAllWindows()


def print_detection_summary(label: str, result: Dict[str, Any]) -> None:
    """Print a concise summary of the latest model response."""
    model_name = result.get("model", label)
    if not result.get("ok"):
        print(f"[{model_name}] Error: {result.get('error', 'unknown error')}")
        return
    if label.lower() == "face":
        detections = parse_detections(result.get("detections"), default_label="face", min_score=CONF_THRESHOLD)
    else:
        detections = threat_detections_from_result(result, default_label=label)
    if not detections:
        print(f"[{model_name}] ok=True but no detections reported.")
        return
    items = ", ".join(f"{lbl}@{score:.2f}" for _, lbl, score in detections)
    print(f"[{model_name}] Detections: {items}")


def local_haar_detect(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> Dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    detections = [
        {"bbox": [int(x), int(y), int(w), int(h)], "label": "face_local", "score": 1.0}
        for (x, y, w, h) in faces
    ]
    return {
        "ok": True,
        "model": "local:haar",
        "detections": detections,
        "raw": {"detector": "haar", "count": len(detections)},
    }


async def run_face_model(frame: np.ndarray) -> Dict[str, Any]:
    try:
        return await face_model.async_detect_face(frame)
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc), "model": "baseten:face"}


async def run_threat_models(
    frame: np.ndarray,
    theft_available: bool,
    weapon_available: bool,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    tasks = []
    labels = []
    if theft_available:
        theft_frame = _prepare_theft_frame(frame.copy())
        frame_b64 = frame_to_base64(theft_frame, quality=THEFT_JPEG_QUALITY)
        tasks.append(theft_model.async_detect_theft(frame_b64))
        labels.append("theft")
    if weapon_available:
        tasks.append(weapon_model.async_detect_weapon(frame))
        labels.append("weapon")

    if tasks:
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for label, resp in zip(labels, responses):
            if isinstance(resp, Exception):  # pragma: no cover
                results[label] = {"ok": False, "error": str(resp), "model": f"baseten:{label}"}
            else:
                results[label] = resp

    if theft_available:
        results.setdefault("theft", {"ok": False, "error": "No result", "model": "baseten:theft"})
    if weapon_available:
        results.setdefault("weapon", {"ok": False, "error": "No result", "model": "baseten:weapon"})

    return results


def extract_score(result: Dict[str, Any]) -> Optional[float]:
    if not result or not result.get("ok"):
        return None
    detections = result.get("detections")
    if isinstance(detections, dict) and "score" in detections:
        return float(detections.get("score", 0.0))
    if isinstance(detections, list):
        scores = [float(det.get("score", det.get("confidence", 0.0))) for det in detections if isinstance(det, dict)]
        if scores:
            return max(scores)
    return None


def format_status(prefix: str, result: Dict[str, Any], score: Optional[float]) -> Tuple[str, Tuple[int, int, int]]:
    if not result.get("ok"):
        return (f"{prefix}: ERROR", (0, 0, 255))
    if score is None:
        return (f"{prefix}: no data", (0, 255, 255))
    if score >= 0.5:
        return (f"{prefix}: ALERT ({score:.2f})", (0, 0, 255))
    return (f"{prefix}: clear ({score:.2f})", (0, 255, 0))


def format_face_text(result: Dict[str, Any]) -> Tuple[str, Tuple[int, int, int]]:
    if not result.get("ok"):
        return ("Face: no detection", (0, 0, 255))
    detections = parse_detections(result.get("detections"), default_label="face", min_score=CONF_THRESHOLD)
    if not detections:
        return ("Face: none", (0, 255, 255))
    top = ", ".join(f"{lbl}@{score:.2f}" for _, lbl, score in detections[:2])
    return (f"Face: {top}", (0, 255, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Baseten face detection with webcam feed.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--max-width", type=int, default=640, help="Resize frame to this width before sending (default: 640)")
    parser.add_argument("--face-interval", type=float, default=0.2, help="Seconds between Baseten face inferences (default: 0.2)")
    parser.add_argument(
        "--threat-interval",
        type=float,
        default=0.5,
        help="Seconds between Baseten theft/weapon inferences (default: 0.5)",
    )
    args = parser.parse_args()
    asyncio.run(
        run_async(
            camera=args.camera,
            max_width=args.max_width,
            face_interval=args.face_interval,
            threat_interval=args.threat_interval,
        )
    )


if __name__ == "__main__":
    main()
