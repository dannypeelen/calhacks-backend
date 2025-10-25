"""SentriAI application package.

This package contains FastAPI routes, services, models, and utilities for the
SentriAI backend. Subpackages include:
- api: FastAPI route definitions
- core: configuration and logging
- services: integrations (LiveKit, Baseten, etc.)
- models: model adapters
- schemas: Pydantic models
- workers: background queue and scheduler
"""

__all__ = [
    "api",
    "core",
    "services",
    "models",
    "schemas",
    "workers",
]

__version__ = "1.0.0"
