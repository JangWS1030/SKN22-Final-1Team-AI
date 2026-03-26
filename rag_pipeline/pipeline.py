from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .analyze_trends import KeywordAnalyzer
from .data_refiner import DataRefiner
from .llm_refiner import LLMRefiner
from .universal_crawler import UniversalCrawler
from .vectorize_chromadb import build_collection

logger = logging.getLogger(__name__)

# 실행 가능한 단계 정의
VALID_STEPS = ("crawl", "refine", "llm_refine", "vectorize", "rebuild_styles", "analyze")
DEFAULT_REFRESH_STEPS = ("crawl", "refine", "llm_refine", "vectorize", "rebuild_styles")


def run_pipeline(*, with_llm: bool = False, with_vectorize: bool = False, with_analysis: bool = False) -> None:
    print("====== 헤어 트렌드 통합 파이프라인 시작 ======")
    UniversalCrawler().crawl()
    DataRefiner().refine()

    if with_llm:
        LLMRefiner().refine_with_llm()
    if with_vectorize:
        build_collection()
    if with_analysis:
        KeywordAnalyzer().analyze_and_visualize()

    print("====== 헤어 트렌드 통합 파이프라인 완료 ======")


def refresh_trends(
    steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    트렌드 데이터 최신화 파이프라인.

    Django 서버 또는 RunPod 핸들러에서 호출하여
    크롤링 → 정제 → LLM 정제 → 벡터DB 갱신 → 스타일 추천 컬렉션 리빌드를 실행.

    Args:
        steps: 실행할 단계 리스트. None이면 전체 실행.
            가능한 값: crawl, refine, llm_refine, vectorize, rebuild_styles, analyze

    Returns:
        각 단계의 실행 결과 및 소요 시간
    """
    if steps is None:
        steps = list(DEFAULT_REFRESH_STEPS)

    # 유효성 검증
    invalid = [s for s in steps if s not in VALID_STEPS]
    if invalid:
        return {"error": f"잘못된 단계: {invalid}. 가능한 값: {VALID_STEPS}"}

    results: Dict[str, Any] = {
        "steps_requested": steps,
        "steps_completed": [],
        "steps_failed": [],
        "details": {},
    }
    t0 = time.time()

    for step in steps:
        step_t0 = time.time()
        logger.info(f"[refresh_trends] === {step} 시작 ===")
        try:
            detail = _run_step(step)
            elapsed = round(time.time() - step_t0, 2)
            results["steps_completed"].append(step)
            results["details"][step] = {"status": "ok", "elapsed_seconds": elapsed, **detail}
            logger.info(f"[refresh_trends] === {step} 완료 ({elapsed}s) ===")
        except Exception as e:
            elapsed = round(time.time() - step_t0, 2)
            results["steps_failed"].append(step)
            results["details"][step] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_seconds": elapsed,
            }
            logger.error(f"[refresh_trends] === {step} 실패: {e} ===")

    results["total_elapsed_seconds"] = round(time.time() - t0, 2)
    results["success"] = len(results["steps_failed"]) == 0
    return results


def _run_step(step: str) -> Dict[str, Any]:
    """개별 단계 실행."""
    if step == "crawl":
        crawler = UniversalCrawler()
        crawler.crawl()
        return {"description": "트렌드 웹 크롤링 완료"}

    if step == "refine":
        refiner = DataRefiner()
        refiner.refine()
        return {"description": "크롤링 데이터 정제 완료"}

    if step == "llm_refine":
        llm_refiner = LLMRefiner()
        llm_refiner.refine_with_llm()
        return {"description": "LLM 기반 정제 완료"}

    if step == "vectorize":
        collection = build_collection()
        count = collection.count() if collection else 0
        return {"description": "ChromaDB 트렌드 벡터DB 갱신 완료", "document_count": count}

    if step == "rebuild_styles":
        from style_recommender import build_style_collection
        collection = build_style_collection()
        count = collection.count() if collection else 0
        return {"description": "스타일 추천 컬렉션 리빌드 완료", "style_count": count}

    if step == "analyze":
        analyzer = KeywordAnalyzer()
        analyzer.analyze_and_visualize()
        return {"description": "키워드 분석 완료"}

    return {}
