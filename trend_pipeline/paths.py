from __future__ import annotations

from pathlib import Path

try:
    from utils.env_loader import load_project_dotenv
except Exception:  # pragma: no cover
    load_project_dotenv = None


if load_project_dotenv is not None:
    load_project_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TREND_DATA_DIR = PROJECT_ROOT / "data" / "trend_pipeline"
RAW_DATA_DIR = TREND_DATA_DIR / "raw"
PROCESSED_DATA_DIR = TREND_DATA_DIR / "processed"
CHROMA_DIR = TREND_DATA_DIR / "chromadb"
ANALYSIS_DIR = TREND_DATA_DIR / "analysis"


def ensure_directories() -> None:
    for path in [TREND_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DIR, ANALYSIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
