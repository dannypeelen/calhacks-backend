import base64
import pytest

from app.services.analyzer import analyze_video


@pytest.mark.asyncio
async def test_analyze_single_base64(monkeypatch):
    async def fake_theft(_bytes):
        detection = {
            "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "center": {"cx": 5, "cy": 5},
            "size": {"w": 10, "h": 10},
            "confidence": 0.6,
            "class_id": 1,
            "class_name": "shoplifting",
        }
        return {
            "ok": True,
            "detections": [detection],
            "confidence": 0.6,
            "coordinates": {
                "boxes": [detection["box"]],
                "centers": [detection["center"]],
                "sizes": [detection["size"]],
            },
            "model": "baseten:theft",
            "raw": {},
        }

    async def fake_weapon(_bytes, **kwargs):
        return {"ok": True, "detections": {"score": 0.2}, "model": "baseten:weapon", "raw": {}}

    monkeypatch.setattr("app.models.model_theft.async_detect_theft", fake_theft)
    monkeypatch.setattr("app.models.model_weapon.async_detect_weapon", fake_weapon)

    img_bytes = b"not-an-image-but-ok-for-base64"
    b64 = base64.b64encode(img_bytes).decode()
    out = await analyze_video({"image_b64": b64})
    assert out["ok"] is True
    assert out["summary"]["frames"] == 1
    # naive score: 0.6*6 + 0.2*4 = 3.6 + 0.8 = 4.4 -> 4
    assert out["results"][0]["threat"]["score"] == 4


@pytest.mark.asyncio
async def test_analyze_path_list(tmp_path, monkeypatch):
    async def fake_theft(_bytes):
        detection = {
            "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "center": {"cx": 5, "cy": 5},
            "size": {"w": 10, "h": 10},
            "confidence": 0.0,
            "class_id": 0,
            "class_name": "normal",
        }
        return {
            "ok": True,
            "detections": [detection],
            "confidence": 0.0,
            "coordinates": {
                "boxes": [detection["box"]],
                "centers": [detection["center"]],
                "sizes": [detection["size"]],
            },
            "model": "baseten:theft",
            "raw": {},
        }

    async def fake_weapon(_bytes, **kwargs):
        return {"ok": True, "detections": {"score": 1.0}, "model": "baseten:weapon", "raw": {}}

    monkeypatch.setattr("app.models.model_theft.async_detect_theft", fake_theft)
    monkeypatch.setattr("app.models.model_weapon.async_detect_weapon", fake_weapon)

    p1 = tmp_path / "f1.jpg"
    p2 = tmp_path / "f2.jpg"
    p1.write_bytes(b"a")
    p2.write_bytes(b"b")

    payload = {"frames": {"paths": [str(p1), str(p2)]}}
    out = await analyze_video(payload)
    assert out["ok"] is True
    assert out["summary"]["frames"] == 2
    # weapon 1.0 -> 4; theft 0 -> 0; score=4
    assert all(r["threat"]["score"] == 4 for r in out["results"])  
