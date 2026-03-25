from __future__ import annotations

from pathlib import Path

try:
    from utils.env_loader import load_project_dotenv
except Exception:  # pragma: no cover
    load_project_dotenv = None


if load_project_dotenv is not None:
    load_project_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "rag"
RAG_SOURCE_DIR = RAG_DIR / "sources"
NCS_SOURCE_DIR = RAG_SOURCE_DIR / "ncs"
RAW_DATA_DIR = RAG_DIR / "raw"
TREND_RAW_DIR = RAW_DATA_DIR / "trends"
PROCESSED_DATA_DIR = RAG_DIR / "processed"
TREND_PROCESSED_DIR = PROCESSED_DATA_DIR / "trends"
NCS_PROCESSED_DIR = PROCESSED_DATA_DIR / "ncs"
BENCHMARK_DIR = RAG_DIR / "benchmarks"
ANALYSIS_DIR = RAG_DIR / "analysis"
RAG_STORE_DIR = RAG_DIR / "stores"
CHROMA_TRENDS_DIR = RAG_STORE_DIR / "chromadb_trends"
CHROMA_NCS_DIR = RAG_STORE_DIR / "chromadb_ncs"
CHROMA_DIR = CHROMA_TRENDS_DIR


def ensure_directories() -> None:
    for path in [
        RAG_DIR,
        RAG_SOURCE_DIR,
        NCS_SOURCE_DIR,
        RAW_DATA_DIR,
        TREND_RAW_DIR,
        PROCESSED_DATA_DIR,
        TREND_PROCESSED_DIR,
        NCS_PROCESSED_DIR,
        BENCHMARK_DIR,
        ANALYSIS_DIR,
        RAG_STORE_DIR,
        CHROMA_TRENDS_DIR,
        CHROMA_NCS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
