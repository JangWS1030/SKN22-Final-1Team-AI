from __future__ import annotations

import argparse

from .analyze_trends import KeywordAnalyzer
from .data_refiner import DataRefiner
from .llm_refiner import LLMRefiner
from .pipeline import run_pipeline
from .rag_query import ask, interactive_chat
from .universal_crawler import UniversalCrawler
from .vectorize_chromadb import build_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated hair trend crawling/RAG pipeline")
    parser.add_argument(
        "command",
        choices=["crawl", "refine", "llm-refine", "vectorize", "rag", "analyze", "full"],
        help="실행할 파이프라인 단계",
    )
    parser.add_argument("--query", help="rag 명령에서 사용할 질문")
    parser.add_argument("--top-k", type=int, default=10, help="rag 검색 결과 개수")
    parser.add_argument("--no-expand", action="store_true", help="rag 쿼리 확장을 비활성화")
    parser.add_argument("--with-llm", action="store_true", help="full 실행 시 LLM 정제를 포함")
    parser.add_argument("--with-vectorize", action="store_true", help="full 실행 시 ChromaDB 생성을 포함")
    parser.add_argument("--with-analysis", action="store_true", help="full 실행 시 키워드 분석을 포함")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "crawl":
        UniversalCrawler().crawl()
        return
    if args.command == "refine":
        DataRefiner().refine()
        return
    if args.command == "llm-refine":
        LLMRefiner().refine_with_llm()
        return
    if args.command == "vectorize":
        build_collection()
        return
    if args.command == "analyze":
        KeywordAnalyzer().analyze_and_visualize()
        return
    if args.command == "rag":
        if args.query:
            print(ask(args.query, n_results=args.top_k, expand=not args.no_expand))
        else:
            interactive_chat()
        return

    run_pipeline(
        with_llm=args.with_llm,
        with_vectorize=args.with_vectorize,
        with_analysis=args.with_analysis,
    )


if __name__ == "__main__":
    main()
