from __future__ import annotations

import argparse
from pathlib import Path

from .analyze_trends import KeywordAnalyzer
from .data_refiner import DataRefiner
from .evaluate_stylist_rag import run_stylist_evaluation
from .llm_refiner import LLMRefiner
from .ncs_pdf_ingest import NcsPdfIngestor
from .ncs_llm_refiner import NcsLlmRefiner
from .ncs_rag_query import ask_ncs, interactive_chat as interactive_ncs_chat
from .ncs_vectorize_chromadb import build_ncs_collection
from .pipeline import run_pipeline
from .rag_query import ask, interactive_chat
from .stylist_rag_query import ask_stylist, interactive_chat as interactive_stylist_chat
from .universal_crawler import UniversalCrawler
from .vectorize_chromadb import build_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated hair stylist RAG pipeline")
    parser.add_argument(
        "command",
        choices=[
            "crawl",
            "refine",
            "llm-refine",
            "vectorize",
            "rag",
            "analyze",
            "full",
            "ncs-extract",
            "ncs-llm-refine",
            "ncs-vectorize",
            "ncs-rag",
            "stylist-rag",
            "stylist-eval",
        ],
        help="실행할 파이프라인 단계",
    )
    parser.add_argument("--query", help="rag 명령에서 사용할 질문")
    parser.add_argument("--top-k", type=int, default=10, help="rag 검색 결과 개수")
    parser.add_argument("--no-expand", action="store_true", help="rag 쿼리 확장을 비활성화")
    parser.add_argument("--front-matter-pages", type=int, default=13, help="ncs-extract에서 건너뛸 앞부분 페이지 수")
    parser.add_argument("--min-chars", type=int, default=120, help="ncs-extract에서 유지할 최소 문자 수")
    parser.add_argument("--delay-seconds", type=float, default=0.6, help="LLM 요청 간 대기 시간")
    parser.add_argument("--save-every", type=int, default=20, help="LLM 정제 중간 저장 주기")
    parser.add_argument("--batch-size", type=int, default=5, help="ncs-llm-refine에서 한 번에 보낼 청크 수")
    parser.add_argument("--limit", type=int, help="개발/검증용 처리 개수 제한")
    parser.add_argument("--benchmark-file", help="stylist-eval에서 사용할 벤치마크 JSON 파일")
    parser.add_argument("--output-dir", help="stylist-eval 결과를 저장할 디렉터리")
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
    if args.command == "ncs-extract":
        NcsPdfIngestor(front_matter_pages=args.front_matter_pages, min_chars=args.min_chars).extract()
        return
    if args.command == "ncs-llm-refine":
        NcsLlmRefiner().refine_with_llm(
            delay_seconds=args.delay_seconds,
            save_every=args.save_every,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        return
    if args.command == "ncs-vectorize":
        build_ncs_collection()
        return
    if args.command == "rag":
        if args.query:
            print(ask(args.query, n_results=args.top_k, expand=not args.no_expand))
        else:
            interactive_chat()
        return
    if args.command == "ncs-rag":
        if args.query:
            print(ask_ncs(args.query, n_results=args.top_k, expand=not args.no_expand))
        else:
            interactive_ncs_chat()
        return
    if args.command == "stylist-rag":
        if args.query:
            print(ask_stylist(args.query, trend_results=args.top_k, ncs_results=args.top_k, expand=not args.no_expand))
        else:
            interactive_stylist_chat()
        return
    if args.command == "stylist-eval":
        run_stylist_evaluation(
            benchmark_file=Path(args.benchmark_file) if args.benchmark_file else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            top_k=args.top_k,
            limit=args.limit,
        )
        return

    run_pipeline(
        with_llm=args.with_llm,
        with_vectorize=args.with_vectorize,
        with_analysis=args.with_analysis,
    )


if __name__ == "__main__":
    main()
