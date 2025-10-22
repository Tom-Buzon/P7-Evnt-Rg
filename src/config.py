from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
    mistral_embed_model: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    data_dir: Path = Path(os.getenv("DATA_DIR", "data/raw"))
    clean_dir: Path = Path(os.getenv("CLEAN_DIR", "data/clean"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", "artifacts/index"))
    region_filter: str | None = os.getenv("REGION_FILTER") or None
    city_filter: str | None = os.getenv("CITY_FILTER") or None
    date_min: str = os.getenv("DATE_MIN", "2024-01-01")
    date_max: str = os.getenv("DATE_MAX", "2025-12-31")

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.clean_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
