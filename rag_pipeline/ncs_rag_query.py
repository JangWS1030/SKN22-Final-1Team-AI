from __future__ import annotations

from collections import Counter
import json
import math
import os
import re
import unicodedata
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from .paths import CHROMA_NCS_DIR, NCS_PROCESSED_DIR, ensure_directories


COLLECTION_NAME = "hair_ncs_manuals"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GPT_MODEL = os.environ.get("NCS_RAG_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("NCS_RAG_TOP_K", "8"))
DENSE_FETCH_MULTIPLIER = int(os.environ.get("NCS_RAG_FETCH_MULTIPLIER", "8"))
DENSE_FETCH_MIN = int(os.environ.get("NCS_RAG_FETCH_MIN", "60"))
LEXICAL_FETCH_MIN = int(os.environ.get("NCS_RAG_LEXICAL_FETCH_MIN", "40"))
RRF_K = int(os.environ.get("NCS_RAG_RRF_K", "50"))
ADJACENT_PAGE_WINDOW = int(os.environ.get("NCS_RAG_ADJACENT_PAGE_WINDOW", "2"))
ADJACENT_PAGE_BONUS = float(os.environ.get("NCS_RAG_ADJACENT_PAGE_BONUS", "0.012"))
CORPUS_FILE = NCS_PROCESSED_DIR / "ncs_rag_ready.json"
STOPWORDS = {
    "어떻게",
    "해주세요",
    "해줘",
    "알려줘",
    "방법",
    "진행",
    "진행해",
    "설명",
    "추천",
    "뭐야",
    "인가요",
    "있나요",
    "전체",
    "순서",
    "정리",
    "포인트",
    "기준",
    "관련",
    "요즘",
    "유행",
}
QUERY_SERVICE_HINTS = {
    "커트": ["커트", "컷", "단발", "레이어", "보브", "bob", "bixie", "pixie"],
    "펌": ["펌", "웨이브", "c컬", "s컬", "컬", "perm", "wave", "curl", "볼륨매직", "매직"],
    "컬러": ["컬러", "염색", "탈색", "color", "colour", "highlight", "브라운", "코퍼"],
    "업스타일": ["업스타일", "브레이드", "번", "updo", "bun", "braid", "가발"],
    "스타일링": ["스타일링", "드라이", "블로우", "블로우드라이", "blow", "손질", "스타일", "볼륨"],
    "샴푸/클리닉": ["샴푸", "클리닉", "트리트먼트", "손상모", "clinic", "treatment", "케어"],
}
SERVICE_DOC_HINTS = {
    "커트": ["커트", "컷", "원렝스"],
    "펌": ["펌", "웨이브", "매직", "볼륨매직", "컬"],
    "컬러": ["컬러", "염색", "탈색"],
    "업스타일": ["업스타일", "브레이드", "번", "가발"],
    "스타일링": ["스타일", "스타일링", "드라이", "블로우"],
    "샴푸/클리닉": ["샴푸", "클리닉", "트리트먼트", "케어"],
}

SYSTEM_PROMPT = """당신은 NCS 헤어미용 교육자료 기반의 시술 가이드 어시스턴트입니다.

규칙:
1. 반드시 [참고 자료]에 근거해서만 답변하세요.
2. 미용사가 고객에게 설명하거나 실제 시술 순서를 정리하듯 한국어로 답변하세요.
3. 가능하면 준비, 시술 순서, 주의사항, 마무리/홈케어 순서로 답변하세요.
4. 자료에 있는 도구, 제품, 단계, 주의사항을 구체적으로 활용하세요.
5. 답변 끝에 반드시 근거 문서명과 페이지를 함께 적으세요.
6. 자료에 없으면 없다고 명확히 말하세요.
7. 의료적 진단이나 피부 질환 치료처럼 미용 범위를 벗어나는 내용은 단정하지 말고 전문가 확인이 필요하다고 적으세요.
"""


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def _get_collection():
    ensure_directories()
    client = chromadb.PersistentClient(path=str(CHROMA_NCS_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception as exc:
        raise RuntimeError(
            f"NCS ChromaDB 컬렉션을 찾을 수 없습니다. 먼저 `python -m rag_pipeline.main ncs-vectorize`를 실행하세요. ({exc})"
        ) from exc


def expand_query(query: str) -> str:
    response = _get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "사용자의 미용 시술 질문을 NCS 매뉴얼 검색용 키워드로 확장하세요.\n"
                    "서비스 유형, 시술 단계, 도구, 주의사항을 포함하세요.\n"
                    "키워드만 쉼표로 구분해 출력하세요.\n"
                    "예시: '손상모 클리닉 어떻게 해?' -> '손상모, 클리닉, 준비, 도포, 자연처치, 마무리, 홈케어, 손상모 특수관리, 트리트먼트, 주의사항'"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=200,
    )
    expanded = response.choices[0].message.content or ""
    return f"{query}, {expanded.strip()}"


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", _normalize_text(text))


def _join_value(value: object, separator: str = ", ") -> str:
    if isinstance(value, list):
        return separator.join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _doc_key(doc: dict) -> tuple[str, str, str]:
    return (
        _normalize_text(doc.get("title", "")),
        _normalize_text(doc.get("source_document_name", "")),
        str(doc.get("source_page", "")).strip(),
    )


def _safe_page(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _build_doc(meta: dict, distance: float | None = None, search_text: str = "", source_alias_names: str = "") -> dict:
    return {
        "title": meta.get("display_title", ""),
        "category": meta.get("category", ""),
        "service_type": _join_value(meta.get("service_type", "")),
        "target_conditions": _join_value(meta.get("target_conditions", "")),
        "tools": _join_value(meta.get("tools", "")),
        "steps": _join_value(meta.get("steps", ""), " | "),
        "cautions": _join_value(meta.get("cautions", ""), " | "),
        "summary": meta.get("summary", ""),
        "stylist_answer": meta.get("stylist_answer", ""),
        "source_document_name": meta.get("source_document_name", ""),
        "source_page": meta.get("source_page", ""),
        "distance": distance if distance is not None else 999.0,
        "search_text": search_text or meta.get("search_text", ""),
        "source_alias_names": source_alias_names or _join_value(meta.get("source_alias_names", "")),
    }


def _doc_search_blob(doc: dict) -> str:
    return " ".join(
        part
        for part in [
            str(doc.get("title", "")),
            str(doc.get("search_text", "")),
            str(doc.get("summary", "")),
            str(doc.get("steps", "")),
            str(doc.get("cautions", "")),
            str(doc.get("tools", "")),
            str(doc.get("target_conditions", "")),
            str(doc.get("service_type", "")),
            str(doc.get("source_document_name", "")),
            str(doc.get("source_alias_names", "")),
        ]
        if part
    )


def _doc_tokens(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", _normalize_text(text))


@lru_cache(maxsize=1)
def _get_lexical_index() -> dict:
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"NCS 검색 원본 파일이 없습니다: {CORPUS_FILE}")

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        raw_rows = json.load(file)

    docs: list[dict] = []
    doc_term_freqs: list[Counter[str]] = []
    doc_lengths: list[int] = []
    doc_freq: Counter[str] = Counter()
    docs_by_source_page: dict[tuple[str, int], list[dict]] = {}
    seen: set[tuple[str, str, str]] = set()

    for row in raw_rows:
        doc = _build_doc(
            meta={
                "display_title": row.get("display_title", ""),
                "category": row.get("category", ""),
                "service_type": row.get("service_type", []),
                "target_conditions": row.get("target_conditions", []),
                "tools": row.get("tools", []),
                "steps": row.get("steps", []),
                "cautions": row.get("cautions", []),
                "summary": row.get("summary", ""),
                "stylist_answer": row.get("stylist_answer", ""),
                "source_document_name": row.get("source_document_name", ""),
                "source_page": str(row.get("source_page", "")),
            },
            search_text=row.get("search_text", ""),
            source_alias_names=_join_value(row.get("source_alias_names", [])),
        )
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)

        docs.append(doc)
        blob_tokens = _doc_tokens(_doc_search_blob(doc))
        doc_lengths.append(len(blob_tokens))
        term_freq = Counter(blob_tokens)
        doc_term_freqs.append(term_freq)
        for token in term_freq:
            doc_freq[token] += 1

        page = _safe_page(doc.get("source_page"))
        if page is not None:
            source_key = (_normalize_text(doc.get("source_document_name", "")), page)
            docs_by_source_page.setdefault(source_key, []).append(doc)

    avg_doc_length = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
    return {
        "docs": docs,
        "doc_term_freqs": doc_term_freqs,
        "doc_lengths": doc_lengths,
        "doc_freq": dict(doc_freq),
        "avg_doc_length": avg_doc_length,
        "docs_by_source_page": docs_by_source_page,
    }


def _dense_candidates(search_query: str, fetch_k: int) -> list[dict]:
    results = _get_collection().query(query_texts=[search_query], n_results=fetch_k)

    docs: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for index in range(len(results["ids"][0])):
        meta = results["metadatas"][0][index]
        doc = _build_doc(meta=meta, distance=results["distances"][0][index])
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        docs.append(doc)
    return docs


def _bm25_score(query_tokens: list[str], term_freq: Counter[str], doc_length: int, *, total_docs: int, doc_freq: dict[str, int], avg_doc_length: float) -> float:
    if not query_tokens or not doc_length or not avg_doc_length:
        return 0.0

    k1 = 1.5
    b = 0.75
    score = 0.0
    for token in query_tokens:
        token_frequency = term_freq.get(token, 0)
        if token_frequency <= 0:
            continue
        frequency = doc_freq.get(token, 0)
        if frequency <= 0:
            continue
        idf = math.log(1.0 + (total_docs - frequency + 0.5) / (frequency + 0.5))
        denominator = token_frequency + k1 * (1.0 - b + b * (doc_length / avg_doc_length))
        score += idf * ((token_frequency * (k1 + 1.0)) / denominator)
    return score


def _lexical_score(query: str, doc: dict, *, bm25_score: float = 0.0) -> float:
    compact_query = _compact_text(query)
    compact_blob = _compact_text(_doc_search_blob(doc))
    substring_bonus = 2.5 if compact_query and compact_query in compact_blob else 0.0
    return bm25_score + substring_bonus + (0.12 * _keyword_score(query, doc))


def _lexical_candidates(query: str, limit: int) -> list[dict]:
    query_tokens = _query_tokens(query)
    index = _get_lexical_index()
    docs = index["docs"]
    doc_term_freqs = index["doc_term_freqs"]
    doc_lengths = index["doc_lengths"]
    doc_freq = index["doc_freq"]
    avg_doc_length = index["avg_doc_length"]
    total_docs = len(docs)

    scored: list[tuple[float, dict]] = []
    for doc, term_freq, doc_length in zip(docs, doc_term_freqs, doc_lengths):
        bm25 = _bm25_score(
            query_tokens,
            term_freq,
            doc_length,
            total_docs=total_docs,
            doc_freq=doc_freq,
            avg_doc_length=avg_doc_length,
        )
        score = _lexical_score(query, doc, bm25_score=bm25)
        if score <= 0:
            continue
        scored.append((score, {**doc, "lexical_score": score}))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]["distance"]))
    return [doc for _, doc in ranked[:limit]]


def _adjacent_page_candidates(seed_docs: list[dict]) -> dict[tuple[str, str, str], tuple[dict, float]]:
    page_index = _get_lexical_index()["docs_by_source_page"]
    boosted: dict[tuple[str, str, str], tuple[dict, float]] = {}
    for seed_doc in seed_docs:
        source_key = _normalize_text(seed_doc.get("source_document_name", ""))
        page = _safe_page(seed_doc.get("source_page"))
        if page is None:
            continue
        for delta in range(1, ADJACENT_PAGE_WINDOW + 1):
            bonus = ADJACENT_PAGE_BONUS / delta
            for candidate_page in (page - delta, page + delta):
                if candidate_page < 1:
                    continue
                for candidate in page_index.get((source_key, candidate_page), []):
                    key = _doc_key(candidate)
                    current = boosted.get(key)
                    if current is None or bonus > current[1]:
                        boosted[key] = ({**candidate, "distance": 999.0}, bonus)
    return boosted


def _infer_query_services(query: str) -> list[str]:
    normalized_query = _normalize_text(query)
    inferred: list[str] = []
    for service, hints in QUERY_SERVICE_HINTS.items():
        if any(hint in normalized_query for hint in hints):
            inferred.append(service)
    return inferred


def _service_rank_bonus(query: str, doc: dict) -> int:
    inferred_services = _infer_query_services(query)
    if not inferred_services:
        return 0

    doc_text = " ".join(
        [
            _normalize_text(doc.get("service_type", "")),
            _normalize_text(doc.get("source_document_name", "")),
            _normalize_text(doc.get("title", "")),
            _normalize_text(doc.get("target_conditions", "")),
        ]
    )
    score = 0
    for index, service in enumerate(inferred_services[:2]):
        weight = 10 if index == 0 else 6
        if any(token in doc_text for token in SERVICE_DOC_HINTS.get(service, [])):
            score += weight

    normalized_query = _normalize_text(query)
    if "가발" in doc_text and "가발" not in normalized_query:
        score -= 10
    if "남성" in doc_text and not any(token in normalized_query for token in ("남성", "남자", "men", "male")):
        score -= 3
    return score


def _hybrid_rank(query: str, dense_docs: list[dict], lexical_docs: list[dict], n_results: int) -> list[dict]:
    merged: dict[tuple[str, str, str], dict] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = _doc_key(doc)
        entry = merged.setdefault(key, {**doc, "hybrid_score": 0.0, "lexical_score": 0.0, "adjacent_bonus": 0.0})
        entry["hybrid_score"] += 1.0 / (RRF_K + rank)
        entry["distance"] = min(float(entry.get("distance", 999.0)), float(doc.get("distance", 999.0)))

    for rank, doc in enumerate(lexical_docs, start=1):
        key = _doc_key(doc)
        entry = merged.setdefault(key, {**doc, "hybrid_score": 0.0, "lexical_score": 0.0, "adjacent_bonus": 0.0})
        entry["hybrid_score"] += 1.0 / (RRF_K + rank)
        entry["lexical_score"] = max(float(entry.get("lexical_score", 0.0)), float(doc.get("lexical_score", 0.0)))
        entry.setdefault("distance", float(doc.get("distance", 999.0)))

    seed_docs = sorted(
        merged.values(),
        key=lambda doc: (-float(doc.get("hybrid_score", 0.0)), -float(doc.get("lexical_score", 0.0)), float(doc.get("distance", 999.0))),
    )[: max(n_results * 3, 12)]

    for key, (doc, bonus) in _adjacent_page_candidates(seed_docs).items():
        entry = merged.setdefault(key, {**doc, "hybrid_score": 0.0, "lexical_score": 0.0, "adjacent_bonus": 0.0})
        entry["adjacent_bonus"] = max(float(entry.get("adjacent_bonus", 0.0)), bonus)
        entry["hybrid_score"] += bonus
        entry["lexical_score"] = max(float(entry.get("lexical_score", 0.0)), _lexical_score(query, entry))

    ranked = sorted(
        merged.values(),
        key=lambda doc: (
            -float(doc.get("hybrid_score", 0.0)),
            -float(doc.get("lexical_score", 0.0)),
            -_keyword_score(query, doc),
            float(doc.get("distance", 999.0)),
        ),
    )
    return ranked[: max(n_results * 6, 24)]


def retrieve(query: str, n_results: int = TOP_K, expand: bool = True) -> list[dict]:
    search_query = expand_query(query) if expand else query
    fetch_k = max(n_results * DENSE_FETCH_MULTIPLIER, DENSE_FETCH_MIN)
    dense_docs = _dense_candidates(search_query, fetch_k=fetch_k)
    lexical_docs = _lexical_candidates(query=search_query, limit=max(fetch_k, LEXICAL_FETCH_MIN))
    docs = _hybrid_rank(query=query, dense_docs=dense_docs, lexical_docs=lexical_docs, n_results=n_results)

    ranked = sorted(
        docs,
        key=lambda doc: (
            -_service_rank_bonus(query, doc),
            -_keyword_score(query, doc),
            -float(doc.get("lexical_score", 0.0)),
            -float(doc.get("hybrid_score", 0.0)),
            float(doc.get("distance", 999.0)),
        ),
    )
    return ranked[:n_results]


def _keyword_score(query: str, doc: dict) -> int:
    tokens = _query_tokens(query)
    if not tokens:
        return 0

    title = _normalize_text(str(doc.get("title", "")))
    high_priority = " ".join(
        [
            title,
            _normalize_text(str(doc.get("service_type", ""))),
            _normalize_text(str(doc.get("target_conditions", ""))),
            _normalize_text(str(doc.get("source_document_name", ""))),
        ]
    )
    medium_priority = " ".join(
        [
            _normalize_text(str(doc.get("summary", ""))),
            _normalize_text(str(doc.get("steps", ""))),
            _normalize_text(str(doc.get("cautions", ""))),
            _normalize_text(str(doc.get("stylist_answer", ""))),
            _normalize_text(str(doc.get("tools", ""))),
            _normalize_text(str(doc.get("search_text", ""))),
        ]
    )

    score = 0
    compact_query = re.sub(r"\s+", "", _normalize_text(query))
    compact_title = re.sub(r"\s+", "", title)
    if compact_query and compact_query in compact_title:
        score += 20

    for token in tokens:
        if token in high_priority:
            score += 8
        if token in medium_priority:
            score += 3
    return score


def _query_tokens(query: str) -> list[str]:
    raw_tokens = re.findall(r"[0-9A-Za-z가-힣]+", _normalize_text(query))
    return [token for token in raw_tokens if len(token) >= 2 and token not in STOPWORDS]


def build_context(docs: list[dict]) -> str:
    context_parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        context_parts.append(
            f"""[자료 {index}]
제목: {doc['title']}
카테고리: {doc['category']}
서비스 유형: {doc['service_type']}
대상 조건: {doc['target_conditions']}
도구: {doc['tools']}
단계: {doc['steps']}
주의사항: {doc['cautions']}
요약: {doc['summary']}
실무 답변 초안: {doc['stylist_answer']}
출처: {doc['source_document_name']} p.{doc['source_page']}
"""
        )
    return "\n".join(context_parts)


def ask_ncs(query: str, n_results: int = TOP_K, expand: bool = True) -> str:
    docs = retrieve(query, n_results=n_results, expand=expand)
    if not docs:
        return "관련 NCS 시술 데이터를 찾지 못했습니다."

    response = _get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[참고 자료]\n{build_context(docs)}\n\n[질문]\n{query}"},
        ],
        temperature=0.2,
        max_tokens=1800,
    )
    return response.choices[0].message.content or ""


def interactive_chat() -> None:
    collection = _get_collection()
    print("=" * 60)
    print("✂️ NCS 헤어 시술 RAG 챗봇")
    print(f"   DB: {collection.count()}건 | 모델: {GPT_MODEL}")
    print("   종료: quit / exit / q")
    print("=" * 60)

    while True:
        query = input("\n🔍 질문: ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            print("👋 종료합니다.")
            break

        print("\n검색 중...")
        docs = retrieve(query)
        print(f"📄 관련 자료 {len(docs)}건 검색 완료 (최소 거리: {docs[0]['distance']:.4f})")

        print("\n💬 답변 생성 중...\n")
        print(ask_ncs(query))
        print("\n" + "-" * 60)


if __name__ == "__main__":
    interactive_chat()
