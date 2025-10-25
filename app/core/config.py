import os
from functools import lru_cache
from pydantic import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    # LiveKit
    LIVEKIT_URL: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""

    # General
    ENV: str = os.getenv("ENV", "development")

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Ensure .env is loaded once
    load_dotenv(override=False)
    return Settings()
