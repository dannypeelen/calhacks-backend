import asyncio
import pytest

from app.services.baseten_client import BasetenClient


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _AsyncClientStub:
    def __init__(self, payload):
        self._payload = payload

    async def post(self, *args, **kwargs):
        return _Resp(self._payload)


@pytest.mark.asyncio
async def test_apredict_image_success(monkeypatch):
    payload = {"result": {"score": 0.9}}
    c = BasetenClient(api_key="x")
    monkeypatch.setattr(c, "_get_async_client", lambda: _AsyncClientStub(payload))
    out = await c.apredict_image("https://example.com/infer", "abcd")
    assert out.get("ok") is True
    assert out.get("result", {}).get("score") == 0.9


@pytest.mark.asyncio
async def test_apredict_image_missing_endpoint():
    c = BasetenClient(api_key="x")
    out = await c.apredict_image("", "abcd")
    assert out["ok"] is False
    assert "endpoint" in out["error"]
