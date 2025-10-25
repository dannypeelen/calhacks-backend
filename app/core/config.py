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
    LIVEKIT_URL: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""

    # General
    ENV: str = os.getenv("ENV", "development")
    # Pydantic v2 config
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Ensure .env is loaded once
    load_dotenv(override=False)
    return Settings()
