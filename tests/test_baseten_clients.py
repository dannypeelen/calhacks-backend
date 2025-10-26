import pytest


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    async def apredict_image(self, endpoint_url, image_b64, extra_input=None):
        return dict(self.payload)

    async def apredict_face(self, image_b64, **extra):
        return dict(self.payload)


@pytest.mark.asyncio
async def test_model_theft_adapter_returns_normalized(monkeypatch):
    from app.models import model_theft

    payload = {
        "detections": [
            {
                "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                "center": {"cx": 5, "cy": 5},
                "size": {"w": 10, "h": 10},
                "confidence": 0.66,
                "class_id": 1,
                "class_name": "shoplifting",
            }
        ],
        "meta": {"num_detections": 1},
    }
    monkeypatch.setattr(model_theft, "get_baseten_client", lambda: FakeClient(payload))

    # Provide raw bytes; adapter handles base64 + client call
    out = await model_theft.async_detect_theft(b"bytes")
    assert out["ok"] is True
    assert out["model"] == "baseten:theft"
    assert out["detections"] == payload["detections"]
    assert out["confidence"] == 0.66
    assert out["num_detections"] == 1
    assert out["coordinates"]["boxes"][0]["x1"] == 0
    assert out["raw"] == payload


@pytest.mark.asyncio
async def test_model_weapon_adapter_returns_normalized(monkeypatch):
    from app.models import model_weapon

    payload = {"result": {"score": 0.25, "label": "weapon"}}
    monkeypatch.setattr(model_weapon, "get_baseten_client", lambda: FakeClient(payload))

    out = await model_weapon.async_detect_weapon(b"bytes")
    assert out["ok"] is True
    assert out["model"] == "baseten:weapon"
    assert out["detections"] == payload["result"]
    assert out["raw"] == payload


@pytest.mark.asyncio
async def test_model_face_adapter_returns_normalized(monkeypatch):
    from app.models import model_face_detection as model_face

    payload = {"result": {"score": 0.9, "label": "face"}}
    monkeypatch.setattr(model_face, "get_baseten_client", lambda: FakeClient(payload))

    out = await model_face.async_detect_face(b"bytes")
    assert out["ok"] is True
    assert out["model"] == "baseten:face"
    assert out["detections"] == payload["result"]
    assert out["raw"] == payload
