from pathlib import Path
from src.config import settings

def test_index_folder_exists():
    assert settings.index_dir.exists(), "Index folder does not exist — run build-index first."
