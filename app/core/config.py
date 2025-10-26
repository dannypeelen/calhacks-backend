import os
from functools import lru_cache
from dotenv import load_dotenv
try:
    # Pydantic v2+ location
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # pragma: no cover
    # Fallback for older environments (v1) if needed
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    # LiveKit
    LIVEKIT_URL: str = "wss://first-project-f84ekuzh.livekit.cloud"
    LIVEKIT_API_KEY: str = "APIfD4sebB8mLTJ"
    LIVEKIT_API_SECRET: str = "fdPTWk30DnkOSECLZmJXP4fTi0blogF6WeDt1nu0CZYA"

    # Baseten
    BASETEN_API_KEY: str = "9KudnXAF.iEOJDfr5AH3HxAP2OSWTnWPVGP7IbYNf"
    BASETEN_THEFT_ENDPOINT: str = "https://model-zq8n19pq.api.baseten.co/development/predict"
    BASETEN_WEAPON_ENDPOINT: str = "https://model-232y9o0q.api.baseten.co/development/predict"
    BASETEN_FACE_ENDPOINT: str = "https://model-vq0n8m4w.api.baseten.co/development/predict"


    # Optional LLMs
    CLAUDE_KEY: str = ""

    # Streaming intervals
    FACE_INTERVAL: float = 0.2
    THREAT_INTERVAL: float = 0.5

    # General
    ENV: str = os.getenv("ENV", "development")
    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Ensure .env is loaded once
    load_dotenv(override=False)
    return Settings()
