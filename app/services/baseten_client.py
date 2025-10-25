"""
Lightweight Baseten client wrapper.

Provides a minimal, production-leaning API to send image frames to a Baseten
model endpoint using REST. The actual endpoint URL and API key are supplied via
environment variables, allowing different models (e.g., theft, weapon) to have
independent endpoints.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

import requests

from app.core.logger import get_logger

log = get_logger(__name__)


class BasetenClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("BASETEN_API_KEY", "")

    def predict_image(
        self, endpoint_url: str, image_b64: str, extra_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a single image to a Baseten model endpoint.

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
            resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            data.setdefault("ok", True)
            return data
        except Exception as e:
            log.exception("Baseten request failed: %s", e)
            return {"ok": False, "error": "Baseten request failed"}


_client: Optional[BasetenClient] = None


def get_baseten_client() -> BasetenClient:
    global _client
    if _client is None:
        _client = BasetenClient()
    return _client
