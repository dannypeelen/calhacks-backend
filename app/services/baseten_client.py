"""
Baseten client wrapper (async, httpx) with convenience helpers per model.

Supports generic image prediction and typed helpers for specific models in
this app: theft, weapon, and face detection. Endpoints and API key are loaded
from environment variables via app.core.config.get_settings.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

import asyncio
import httpx

from app.core.logger import get_logger
from app.core.config import get_settings

log = get_logger(__name__)


class BasetenClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0) -> None:
        self._settings = get_settings()
        # Prefer explicit arg, then settings, then env var (fallback)
        self.api_key = api_key or self._settings.BASETEN_API_KEY or os.getenv("BASETEN_API_KEY", "")
        self._timeout = timeout
        self._aclient: Optional[httpx.AsyncClient] = None

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self._timeout)
        return self._aclient

    async def apredict_image(
        self, endpoint_url: str, image_b64: str, extra_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async: Send a single image to a Baseten model endpoint using httpx.

        Args:
            endpoint_url: Full Baseten model inference URL (e.g., from env).
            image_b64: Base64-encoded image data (JPEG/PNG).
            extra_input: Additional input fields required by the specific model.

        Returns:
            A dict with the JSON response or an error payload.
        """
        if not endpoint_url:
            return {"ok": False, "error": "Baseten endpoint URL missing"}

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"input": {"image": image_b64}}
        if extra_input:
            payload["input"].update(extra_input)

        try:
            client = self._get_async_client()
            resp = await client.post(endpoint_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            data.setdefault("ok", True)
            return data
        except Exception as e:
            log.exception("Baseten request failed: %s", e)
            return {"ok": False, "error": "Baseten request failed"}

    # -------------------------
    # Convenience wrappers
    # -------------------------
    async def apredict_theft(self, image_b64: str, conf_thresh: float = 0.5) -> Dict[str, Any]:
        url = self._settings.BASETEN_THEFT_ENDPOINT or os.getenv("BASETEN_THEFT_ENDPOINT", "")
        extra = {"conf_thresh": float(conf_thresh)}
        return await self.apredict_image(url, image_b64, extra)

    async def apredict_weapon(self, image_b64: str, **extra: Any) -> Dict[str, Any]:
        url = self._settings.BASETEN_WEAPON_ENDPOINT or os.getenv("BASETEN_WEAPON_ENDPOINT", "")
        return await self.apredict_image(url, image_b64, extra or None)

    async def apredict_face(self, image_b64: str, **extra: Any) -> Dict[str, Any]:
        url = self._settings.BASETEN_FACE_ENDPOINT or os.getenv("BASETEN_FACE_ENDPOINT", "")
        return await self.apredict_image(url, image_b64, extra or None)

    async def apredict_image_bytes(self, endpoint_url: str, image_bytes: bytes, extra_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper that base64-encodes raw bytes before calling apredict_image."""
        image_b64 = base64.b64encode(image_bytes).decode()
        return await self.apredict_image(endpoint_url, image_b64, extra_input)

    def predict_image(
        self, endpoint_url: str, image_b64: str, extra_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Blocking wrapper around apredict_image.

        Safe to call from worker threads (e.g., asyncio.to_thread). Avoid calling
        from the main event loop context directly.
        """
        return asyncio.run(self.apredict_image(endpoint_url, image_b64, extra_input))

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None


_client: Optional[BasetenClient] = None


def get_baseten_client() -> BasetenClient:
    global _client
    if _client is None:
        _client = BasetenClient()
    return _client
