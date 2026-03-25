from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import Dataset
import numpy as np
from openai import OpenAI
from ragas import evaluate
from ragas.llms.base import llm_factory
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
from ragas.metrics._context_recall import LLMContextRecall
from sentence_transformers import SentenceTransformer

from .paths import ANALYSIS_DIR, BENCHMARK_DIR, ensure_directories
from .stylist_rag_query import (
    GPT_MODEL,
    SERVICE_RULES,
    SYSTEM_PROMPT,
    build_response_guide,
    build_stylist_user_prompt,
    retrieve_bundle,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_BENCHMARK_FILE = BENCHMARK_DIR / "stylist_eval_set.json"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "stylist_ragas_eval"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
JUDGE_MODEL = os.environ.get("STYLIST_EVAL_JUDGE_MODEL", GPT_MODEL)
NO_RAG_SYSTEM_PROMPT = """당신은 현업 미용사를 위한 헤어 트렌드 + 시술 어시스턴트입니다.

외부 자료나 검색 결과는 제공되지 않습니다.
일반적인 미용 실무 지식만으로 답변하고, 확실하지 않은 세부 절차나 숫자는 일반론으로만 설명하세요.
출처를 꾸며내지 마세요.
"""
EVAL_COMPACT_RULE = (
    "평가용 답변이므로 5개 이하 번호 목록으로, 700자 이내로 압축하세요. "
    "핵심 스타일명, 핵심 시술 단계, 핵심 수치만 남기고 불필요한 수식은 줄이세요."
)
JUDGE_SYSTEM_PROMPT = """당신은 미용 실무용 RAG 평가자입니다.

두 답변을 동일한 검색 근거와 동일한 단계 체크리스트로 평가하세요.

평가 기준:
1. faithfulness_label
- fully_supported: 구체적 주장 대부분이 검색 근거에 직접 뒷받침됨
- mostly_supported: 일부 표현 차이는 있지만 핵심 주장은 근거로 지지됨
- mixed: 근거 있는 내용과 근거 없는 내용이 섞임
- mostly_unsupported: 핵심 절차나 구체 수치가 근거와 잘 맞지 않음
- unsupported: 실질적으로 근거 없이 답했거나 검색 근거와 모순됨

2. covered_steps
- 반드시 제공된 expected_steps 중 실제로 답변에 의미 있게 포함된 항목만 넣으세요.
- expected_steps의 원문 문자열을 그대로 복사하세요.
- 포함되지 않은 항목은 넣지 마세요.

출력 규칙:
- JSON만 출력하세요.
- 최상위 키는 rag, no_rag 두 개만 사용하세요.
- 각 키 아래에는 faithfulness_label, covered_steps만 넣으세요.
"""
FAITHFULNESS_LABEL_SCORES = {
    "fully_supported": 1.0,
    "mostly_supported": 0.75,
    "mixed": 0.5,
    "mostly_unsupported": 0.25,
    "unsupported": 0.0,
}
TREND_SERVICE_HINTS = {
    "커트": ["bob", "lob", "pixie", "bixie", "fringe", "bang", "shag", "wolf", "cut", "layered cut"],
    "펌": ["wave", "curl", "perm", "blowout", "soft waves", "volume"],
    "컬러": ["color", "colour", "blonde", "brown", "copper", "highlight", "brunette", "ash", "auburn", "red"],
    "업스타일": ["updo", "bun", "braid", "ponytail", "chignon", "twist", "crown"],
    "스타일링": ["styling", "blowout", "blow dry", "volume", "texture", "wet look", "slick"],
    "샴푸/클리닉": ["care", "clinic", "treatment", "repair", "damage", "healthy hair", "scalp", "shampoo"],
}


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


def _get_ragas_llm():
    return llm_factory(model=GPT_MODEL, provider="openai", client=_get_openai_client())


@lru_cache(maxsize=1)
def _get_similarity_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"벤치마크 파일 형식이 잘못되었습니다: {path}")
    return data


def normalize_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    required_fields = ("id", "user_input", "reference")
    list_fields = (
        "required_terms",
        "expected_trend_terms",
        "expected_sources",
        "expected_pages",
        "expected_source_pages",
        "expected_steps",
    )

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"벤치마크 {index}번째 항목이 객체가 아닙니다.")
        missing = [field for field in required_fields if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"벤치마크 {index}번째 항목에 필수 필드가 비어 있습니다: {', '.join(missing)}")

        normalized = dict(row)
        for field in list_fields:
            value = normalized.get(field, [])
            normalized[field] = value if isinstance(value, list) else []
        normalized_rows.append(normalized)

    return normalized_rows


def _trend_context_text(doc: dict[str, Any]) -> str:
    return (
        f"트렌드 제목: {doc.get('title', '')}\n"
        f"카테고리: {doc.get('category', '')}\n"
        f"요약: {doc.get('summary', '')}\n"
        f"스타일 태그: {doc.get('style_tags', '')}\n"
        f"출처: {doc.get('source', '')} ({doc.get('year', '')})"
    )


def _ncs_context_text(doc: dict[str, Any]) -> str:
    return (
        f"시술 제목: {doc.get('title', '')}\n"
        f"카테고리: {doc.get('category', '')}\n"
        f"서비스 유형: {doc.get('service_type', '')}\n"
        f"도구: {doc.get('tools', '')}\n"
        f"단계: {doc.get('steps', '')}\n"
        f"주의사항: {doc.get('cautions', '')}\n"
        f"요약: {doc.get('summary', '')}\n"
        f"출처: {doc.get('source_document_name', '')} p.{doc.get('source_page', '')}"
    )


def _build_retrieved_contexts(bundle: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    contexts.extend(_trend_context_text(doc) for doc in bundle["trend_docs"])
    contexts.extend(_ncs_context_text(doc) for doc in bundle["ncs_docs"])
    return contexts


def _normalize_space(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_term(text: str) -> str:
    return "".join(_normalize_space(text).lower().split())


def _normalize_source(text: str) -> str:
    return _normalize_term(text)


def _normalize_service(text: str) -> str:
    return _normalize_space(text).lower()


def _expected_source_page_pairs(row: dict[str, Any]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    source_pages = row.get("expected_source_pages", [])
    if isinstance(source_pages, list) and source_pages:
        for item in source_pages:
            if not isinstance(item, dict):
                continue
            source = _normalize_source(item.get("source", ""))
            for page in item.get("pages", []):
                try:
                    pairs.add((source, int(page)))
                except (TypeError, ValueError):
                    continue
        return pairs

    sources = [_normalize_source(source) for source in row.get("expected_sources", []) if str(source).strip()]
    pages: list[int] = []
    for page in row.get("expected_pages", []):
        try:
            pages.append(int(page))
        except (TypeError, ValueError):
            continue
    for source in sources:
        for page in pages:
            pairs.add((source, page))
    return pairs


def _trend_doc_blob(doc: dict[str, Any]) -> str:
    style_tags = doc.get("style_tags", [])
    color_tags = doc.get("color_tags", [])
    style_text = ", ".join(style_tags) if isinstance(style_tags, list) else str(style_tags or "")
    color_text = ", ".join(color_tags) if isinstance(color_tags, list) else str(color_tags or "")
    return _normalize_term(
        " ".join(
            [
                str(doc.get("title", "")),
                str(doc.get("category", "")),
                style_text,
                color_text,
                str(doc.get("summary", "")),
                str(doc.get("source", "")),
            ]
        )
    )


def _ncs_doc_blob(doc: dict[str, Any]) -> str:
    return _normalize_term(
        " ".join(
            [
                str(doc.get("title", "")),
                str(doc.get("category", "")),
                str(doc.get("service_type", "")),
                str(doc.get("summary", "")),
                str(doc.get("source_document_name", "")),
                str(doc.get("steps", "")),
                str(doc.get("tools", "")),
            ]
        )
    )


def generate_no_rag_answer(query: str, service_types: list[str]) -> str:
    response = _get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": f"{NO_RAG_SYSTEM_PROMPT}\n{EVAL_COMPACT_RULE}"},
            {
                "role": "user",
                "content": (
                    f"[질문]\n{query}\n\n"
                    f"[답변 형식 가이드]\n{build_response_guide(service_types)}\n\n"
                    "[규칙]\n"
                    "- 검색 결과나 문서 근거 없이 일반적인 실무 지식으로 답하세요.\n"
                    "- 확실하지 않은 절차나 숫자는 단정하지 말고 일반론으로 표현하세요.\n"
                    "- 답변 마지막에는 `근거 자료: 없음 (no-rag baseline)`를 적으세요.\n"
                ),
            },
        ],
        temperature=0.25,
        max_tokens=1800,
    )
    return response.choices[0].message.content or ""


def generate_rag_eval_answer(query: str, bundle: dict[str, Any]) -> str:
    response = _get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n{EVAL_COMPACT_RULE}"},
            {"role": "user", "content": build_stylist_user_prompt(query, bundle)},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""


def collect_predictions(benchmark_rows: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rag_rows: list[dict[str, Any]] = []
    no_rag_rows: list[dict[str, Any]] = []

    for row in benchmark_rows:
        query = row["user_input"]
        bundle = retrieve_bundle(query, trend_results=top_k, ncs_results=top_k, expand=True)
        service_types = bundle["service_types"]
        rag_response = generate_rag_eval_answer(query, bundle)
        no_rag_response = generate_no_rag_answer(query, service_types)
        retrieved_contexts = _build_retrieved_contexts(bundle)

        common = {
            "id": row["id"],
            "user_input": query,
            "reference": row["reference"],
            "service_type": row.get("service_type", ""),
            "required_terms": row.get("required_terms", []),
            "expected_trend_terms": row.get("expected_trend_terms", []),
            "expected_sources": row.get("expected_sources", []),
            "expected_pages": row.get("expected_pages", []),
            "expected_source_pages": row.get("expected_source_pages", []),
            "expected_steps": row.get("expected_steps", []),
        }

        rag_rows.append(
            {
                **common,
                "response": rag_response,
                "retrieved_contexts": retrieved_contexts,
                "retrieved_trend_docs": bundle["trend_docs"],
                "retrieved_ncs_docs": bundle["ncs_docs"],
                "inferred_service_types": service_types,
            }
        )
        no_rag_rows.append(
            {
                **common,
                "response": no_rag_response,
                "retrieved_contexts": retrieved_contexts,
                "retrieved_trend_docs": bundle["trend_docs"],
                "retrieved_ncs_docs": bundle["ncs_docs"],
                "inferred_service_types": service_types,
            }
        )

    return rag_rows, no_rag_rows


def _to_dataset(rows: list[dict[str, Any]], include_contexts: bool) -> Dataset:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            "id": row["id"],
            "user_input": row["user_input"],
            "response": row["response"],
            "reference": row["reference"],
        }
        if include_contexts:
            record["retrieved_contexts"] = row["retrieved_contexts"]
        records.append(record)
    return Dataset.from_list(records)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _attach_semantic_similarity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model = _get_similarity_model()
    texts = [row["response"] for row in rows] + [row["reference"] for row in rows]
    embeddings = model.encode(texts, normalize_embeddings=False)
    half = len(rows)
    response_embeddings = embeddings[:half]
    reference_embeddings = embeddings[half:]

    enriched: list[dict[str, Any]] = []
    for row, response_embedding, reference_embedding in zip(rows, response_embeddings, reference_embeddings):
        enriched.append(
            {
                **row,
                "semantic_similarity": _cosine_similarity(np.asarray(response_embedding), np.asarray(reference_embedding)),
            }
        )
    return enriched


def _attach_required_term_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        required_terms = row.get("required_terms", [])
        normalized_response = _normalize_term(row["response"])
        if required_terms:
            hits = sum(1 for term in required_terms if _normalize_term(str(term)) in normalized_response)
            coverage = hits / len(required_terms)
        else:
            coverage = 0.0
        enriched.append({**row, "required_term_coverage": coverage})
    return enriched


def _attach_retrieval_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        expected_pairs = _expected_source_page_pairs(row)
        expected_sources = {source for source, _ in expected_pairs}
        retrieved_pairs: set[tuple[str, int]] = set()
        retrieved_sources: set[str] = set()

        for doc in row.get("retrieved_ncs_docs", []):
            source = _normalize_source(doc.get("source_document_name", ""))
            retrieved_sources.add(source)
            try:
                page = int(doc.get("source_page", 0))
            except (TypeError, ValueError):
                continue
            retrieved_pairs.add((source, page))

        matched_pairs = expected_pairs & retrieved_pairs
        enriched.append(
            {
                **row,
                "retrieval_source_hit_at_k": 1.0 if expected_sources and (expected_sources & retrieved_sources) else 0.0,
                "retrieval_hit_at_k": 1.0 if matched_pairs else 0.0,
                "retrieval_page_recall_at_k": (len(matched_pairs) / len(expected_pairs)) if expected_pairs else 0.0,
            }
        )
    return enriched


def _attach_trend_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        expected_terms = [
            _normalize_term(term)
            for term in row.get("expected_trend_terms", [])
            if _normalize_term(term)
        ]
        trend_blobs = [_trend_doc_blob(doc) for doc in row.get("retrieved_trend_docs", [])]
        trend_hit = 0.0
        if expected_terms and trend_blobs:
            trend_hit = 1.0 if any(term in blob for term in expected_terms for blob in trend_blobs) else 0.0
        enriched.append({**row, "trend_hit_at_k": trend_hit})
    return enriched


def _service_alias_hit(blob: str, aliases: list[str]) -> float:
    normalized_blob = _normalize_term(blob)
    return 1.0 if any(_normalize_term(alias) in normalized_blob for alias in aliases if str(alias).strip()) else 0.0


def _attach_integrated_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        expected_service = str(row.get("service_type", "")).strip()
        normalized_expected_service = _normalize_service(expected_service)
        inferred_services = [_normalize_service(service) for service in row.get("inferred_service_types", [])]
        inferred_hit = 1.0 if normalized_expected_service and normalized_expected_service in inferred_services else 0.0

        trend_aliases = TREND_SERVICE_HINTS.get(expected_service, [])
        trend_service_hit = (
            1.0
            if any(_service_alias_hit(_trend_doc_blob(doc), trend_aliases) for doc in row.get("retrieved_trend_docs", []))
            else 0.0
        )

        ncs_aliases = SERVICE_RULES.get(expected_service, {}).get("aliases", [expected_service])
        ncs_service_hit = (
            1.0
            if any(_service_alias_hit(_ncs_doc_blob(doc), ncs_aliases) for doc in row.get("retrieved_ncs_docs", []))
            else 0.0
        )

        service_consistency = mean([inferred_hit, trend_service_hit, ncs_service_hit])
        trend_hit = float(row.get("trend_hit_at_k", 0.0) or 0.0)
        ncs_hit = float(row.get("retrieval_hit_at_k", 0.0) or 0.0)
        dual_source_hit = 1.0 if trend_hit > 0 and ncs_hit > 0 else 0.0

        enriched.append(
            {
                **row,
                "dual_source_hit": dual_source_hit,
                "service_consistency": service_consistency,
            }
        )
    return enriched


def _evaluate_rag_grounding(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [LLMContextPrecisionWithoutReference(), LLMContextRecall()]
    result = evaluate(
        dataset=_to_dataset(rows, include_contexts=True),
        metrics=metrics,
        llm=_get_ragas_llm(),
        show_progress=False,
    )
    return result.to_pandas().to_dict(orient="records")


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _match_expected_steps(expected_steps: list[str], covered_steps: list[str]) -> tuple[list[str], list[str], float]:
    if not isinstance(expected_steps, list):
        expected_steps = []
    if not isinstance(covered_steps, list):
        covered_steps = []
    normalized_expected = {_normalize_term(step): step for step in expected_steps}
    matched: list[str] = []
    for step in covered_steps:
        normalized = _normalize_term(step)
        if normalized in normalized_expected and normalized_expected[normalized] not in matched:
            matched.append(normalized_expected[normalized])

    missing = [step for step in expected_steps if step not in matched]
    score = len(matched) / len(expected_steps) if expected_steps else 0.0
    return matched, missing, score


def _judge_answer_pair(row: dict[str, Any], rag_answer: str, no_rag_answer: str) -> dict[str, Any]:
    expected_steps = row.get("expected_steps", [])
    contexts = "\n\n".join(row.get("retrieved_contexts", []))
    response = _get_openai_client().chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[질문]\n{row['user_input']}\n\n"
                    f"[expected_steps]\n{json.dumps(expected_steps, ensure_ascii=False)}\n\n"
                    f"[검색 근거]\n{contexts}\n\n"
                    f"[rag 답변]\n{rag_answer}\n\n"
                    f"[no_rag 답변]\n{no_rag_answer}\n"
                ),
            },
        ],
        temperature=0,
        max_tokens=1200,
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(_strip_code_fence(content))
    except json.JSONDecodeError:
        parsed = {}

    rag_block = parsed.get("rag", {}) if isinstance(parsed, dict) else {}
    no_rag_block = parsed.get("no_rag", {}) if isinstance(parsed, dict) else {}

    rag_faithfulness = FAITHFULNESS_LABEL_SCORES.get(str(rag_block.get("faithfulness_label", "")).strip(), 0.0)
    no_rag_faithfulness = FAITHFULNESS_LABEL_SCORES.get(str(no_rag_block.get("faithfulness_label", "")).strip(), 0.0)

    rag_covered, rag_missing, rag_step_score = _match_expected_steps(expected_steps, rag_block.get("covered_steps", []))
    no_rag_covered, no_rag_missing, no_rag_step_score = _match_expected_steps(expected_steps, no_rag_block.get("covered_steps", []))

    return {
        "rag": {
            "faithfulness": rag_faithfulness,
            "faithfulness_label": rag_block.get("faithfulness_label", ""),
            "covered_steps": rag_covered,
            "missing_steps": rag_missing,
            "step_completeness": rag_step_score,
        },
        "no_rag": {
            "faithfulness": no_rag_faithfulness,
            "faithfulness_label": no_rag_block.get("faithfulness_label", ""),
            "covered_steps": no_rag_covered,
            "missing_steps": no_rag_missing,
            "step_completeness": no_rag_step_score,
        },
    }


def _attach_judge_metrics(
    rag_rows: list[dict[str, Any]],
    no_rag_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enriched_rag_rows: list[dict[str, Any]] = []
    enriched_no_rag_rows: list[dict[str, Any]] = []

    for rag_row, no_rag_row in zip(rag_rows, no_rag_rows):
        judged = _judge_answer_pair(rag_row, rag_row["response"], no_rag_row["response"])
        rag_metrics = judged["rag"]
        no_rag_metrics = judged["no_rag"]
        enriched_rag_rows.append({**rag_row, **rag_metrics})
        enriched_no_rag_rows.append({**no_rag_row, **no_rag_metrics})

    return enriched_rag_rows, enriched_no_rag_rows


def _merge_metrics(rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if metric_rows and "id" in metric_rows[0]:
        metric_map = {row["id"]: row for row in metric_rows}
        return [{**row, **metric_map.get(row["id"], {})} for row in rows]
    return [{**row, **metric_row} for row, metric_row in zip(rows, metric_rows, strict=False)]


def _metric_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None else float("nan")


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else float("nan")


def build_summary(
    rag_rows: list[dict[str, Any]],
    no_rag_rows: list[dict[str, Any]],
    rag_grounding_summary: dict[str, float],
    benchmark_path: Path,
) -> dict[str, Any]:
    common_keys = [
        "semantic_similarity",
        "required_term_coverage",
        "faithfulness",
        "step_completeness",
    ]

    common_summary: dict[str, Any] = {}
    for key in common_keys:
        rag_mean = _mean_metric(rag_rows, key)
        no_rag_mean = _mean_metric(no_rag_rows, key)
        common_summary[key] = {
            "rag": rag_mean,
            "no_rag": no_rag_mean,
            "delta": rag_mean - no_rag_mean,
            "rag_win_count": sum(
                1
                for rag_row, no_rag_row in zip(rag_rows, no_rag_rows)
                if _metric_value(rag_row, key) > _metric_value(no_rag_row, key)
            ),
        }

    trend_summary = {
        "trend_hit_at_k": _mean_metric(rag_rows, "trend_hit_at_k"),
    }
    ncs_summary = {
        "retrieval_source_hit_at_k": _mean_metric(rag_rows, "retrieval_source_hit_at_k"),
        "retrieval_hit_at_k": _mean_metric(rag_rows, "retrieval_hit_at_k"),
        "retrieval_page_recall_at_k": _mean_metric(rag_rows, "retrieval_page_recall_at_k"),
    }
    integrated_rag_only = {
        "dual_source_hit": _mean_metric(rag_rows, "dual_source_hit"),
        "service_consistency": _mean_metric(rag_rows, "service_consistency"),
        **rag_grounding_summary,
    }

    return {
        "samples": len(rag_rows),
        "model": GPT_MODEL,
        "judge_model": JUDGE_MODEL,
        "benchmark_file": str(benchmark_path),
        "trend_metrics": trend_summary,
        "ncs_metrics": ncs_summary,
        "integrated_rag_only_metrics": integrated_rag_only,
        "common_metrics": common_summary,
    }


def build_markdown_report(
    summary: dict[str, Any],
    rag_rows: list[dict[str, Any]],
    no_rag_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Stylist RAG Evaluation")
    lines.append("")
    lines.append(f"- Samples: `{summary['samples']}`")
    lines.append(f"- Model: `{summary['model']}`")
    lines.append(f"- Judge Model: `{summary['judge_model']}`")
    lines.append(f"- Benchmark: `{summary['benchmark_file']}`")
    lines.append("- Comparison: `stylist-rag` vs `no-rag baseline`")
    lines.append(
        "- Metrics: `trend_hit_at_k`, `retrieval_source_hit_at_k`, `retrieval_hit_at_k`, `retrieval_page_recall_at_k`, `dual_source_hit`, `service_consistency`, `semantic_similarity`, `required_term_coverage`, `faithfulness`, `step_completeness`, `context_precision`, `context_recall`"
    )
    lines.append("")
    lines.append("## Trend Retrieval")
    lines.append("")
    lines.append("| metric | rag |")
    lines.append("| --- | ---: |")
    for key, value in summary["trend_metrics"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.append("")
    lines.append("## NCS Retrieval")
    lines.append("")
    lines.append("| metric | rag |")
    lines.append("| --- | ---: |")
    for key, value in summary["ncs_metrics"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.append("")
    lines.append("## Integrated")
    lines.append("")
    lines.append("| metric | rag | no_rag | delta | rag_win_count |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for key, value in summary["integrated_rag_only_metrics"].items():
        lines.append(f"| {key} | {value:.4f} | - | - | - |")
    for key, values in summary["common_metrics"].items():
        lines.append(
            f"| {key} | {values['rag']:.4f} | {values['no_rag']:.4f} | {values['delta']:.4f} | {values['rag_win_count']} |"
        )
    lines.append("")
    lines.append("## Per Sample")
    lines.append("")
    lines.append(
        "| id | query | trend_hit | source_hit | retrieval_hit | dual_source | service_consistency | page_recall | rag_faith | no_rag_faith | rag_steps | no_rag_steps |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rag_row, no_rag_row in zip(rag_rows, no_rag_rows):
        query = rag_row["user_input"].replace("|", "/")
        lines.append(
            f"| {rag_row['id']} | {query} | {rag_row.get('trend_hit_at_k', 0.0):.4f} | {rag_row['retrieval_source_hit_at_k']:.4f} | {rag_row['retrieval_hit_at_k']:.4f} | {rag_row.get('dual_source_hit', 0.0):.4f} | {rag_row.get('service_consistency', 0.0):.4f} | {rag_row['retrieval_page_recall_at_k']:.4f} | {rag_row['faithfulness']:.4f} | {no_rag_row['faithfulness']:.4f} | {rag_row['step_completeness']:.4f} | {no_rag_row['step_completeness']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def save_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    rag_rows: list[dict[str, Any]],
    no_rag_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "rag_rows.json").write_text(json.dumps(rag_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "no_rag_rows.json").write_text(json.dumps(no_rag_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(build_markdown_report(summary, rag_rows, no_rag_rows), encoding="utf-8")


def refresh_saved_stylist_evaluation(
    benchmark_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    benchmark_path = benchmark_file or DEFAULT_BENCHMARK_FILE
    output_path = output_dir or DEFAULT_OUTPUT_DIR

    benchmark_rows = normalize_benchmark_rows(load_benchmark(benchmark_path))
    benchmark_map = {row["id"]: row for row in benchmark_rows}

    rag_rows = json.loads((output_path / "rag_rows.json").read_text(encoding="utf-8"))
    no_rag_rows = json.loads((output_path / "no_rag_rows.json").read_text(encoding="utf-8"))

    def merge_benchmark_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for row in rows:
            benchmark_row = benchmark_map.get(row["id"], {})
            merged.append(
                {
                    **row,
                    "service_type": benchmark_row.get("service_type", row.get("service_type", "")),
                    "required_terms": benchmark_row.get("required_terms", row.get("required_terms", [])),
                    "expected_trend_terms": benchmark_row.get("expected_trend_terms", row.get("expected_trend_terms", [])),
                    "expected_sources": benchmark_row.get("expected_sources", row.get("expected_sources", [])),
                    "expected_pages": benchmark_row.get("expected_pages", row.get("expected_pages", [])),
                    "expected_source_pages": benchmark_row.get("expected_source_pages", row.get("expected_source_pages", [])),
                    "expected_steps": benchmark_row.get("expected_steps", row.get("expected_steps", [])),
                }
            )
        return merged

    rag_rows = merge_benchmark_fields(rag_rows)
    no_rag_rows = merge_benchmark_fields(no_rag_rows)

    rag_rows = _attach_required_term_coverage(rag_rows)
    no_rag_rows = _attach_required_term_coverage(no_rag_rows)
    rag_rows = _attach_retrieval_metrics(rag_rows)
    rag_rows = _attach_trend_metrics(rag_rows)
    rag_rows = _attach_integrated_metrics(rag_rows)

    rag_grounding_summary = {
        "llm_context_precision_without_reference": _mean_metric(rag_rows, "llm_context_precision_without_reference"),
        "context_recall": _mean_metric(rag_rows, "context_recall"),
    }
    summary = build_summary(rag_rows, no_rag_rows, rag_grounding_summary, benchmark_path)
    save_outputs(output_path, summary, rag_rows, no_rag_rows)
    return summary


def run_stylist_evaluation(
    benchmark_file: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_directories()
    benchmark_path = benchmark_file or DEFAULT_BENCHMARK_FILE
    output_path = output_dir or DEFAULT_OUTPUT_DIR

    benchmark_rows = load_benchmark(benchmark_path)
    benchmark_rows = normalize_benchmark_rows(benchmark_rows)
    if limit is not None:
        benchmark_rows = benchmark_rows[:limit]

    rag_rows, no_rag_rows = collect_predictions(benchmark_rows, top_k=top_k)
    rag_rows = _attach_semantic_similarity(rag_rows)
    no_rag_rows = _attach_semantic_similarity(no_rag_rows)
    rag_rows = _attach_required_term_coverage(rag_rows)
    no_rag_rows = _attach_required_term_coverage(no_rag_rows)
    rag_rows = _attach_retrieval_metrics(rag_rows)
    rag_rows = _attach_trend_metrics(rag_rows)
    rag_rows = _attach_integrated_metrics(rag_rows)

    rag_rows, no_rag_rows = _attach_judge_metrics(rag_rows, no_rag_rows)

    rag_grounding_rows = _evaluate_rag_grounding(rag_rows)
    rag_rows = _merge_metrics(rag_rows, rag_grounding_rows)

    rag_grounding_summary = {
        "llm_context_precision_without_reference": _mean_metric(rag_rows, "llm_context_precision_without_reference"),
        "context_recall": _mean_metric(rag_rows, "context_recall"),
    }
    summary = build_summary(rag_rows, no_rag_rows, rag_grounding_summary, benchmark_path)
    save_outputs(output_path, summary, rag_rows, no_rag_rows)

    print("=== stylist ragas eval ===")
    print(f"samples: {summary['samples']}")
    print(f"output: {output_path}")
    for key, value in summary["trend_metrics"].items():
        print(f"{key}: rag={value:.4f}")
    for key, value in summary["ncs_metrics"].items():
        print(f"{key}: rag={value:.4f}")
    for key, value in summary["integrated_rag_only_metrics"].items():
        print(f"{key}: rag={value:.4f}")
    for key, values in summary["common_metrics"].items():
        print(f"{key}: rag={values['rag']:.4f} | no_rag={values['no_rag']:.4f} | delta={values['delta']:.4f}")

    return summary


if __name__ == "__main__":
    run_stylist_evaluation()
