from __future__ import annotations

from .analyze_trends import KeywordAnalyzer
from .data_refiner import DataRefiner
from .llm_refiner import LLMRefiner
from .universal_crawler import UniversalCrawler
from .vectorize_chromadb import build_collection


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
