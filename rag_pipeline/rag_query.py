"""
ChromaDB + OpenAI 기반 헤어 트렌드 RAG 파이프라인.
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from .paths import CHROMA_TRENDS_DIR, ensure_directories


COLLECTION_NAME = "hair_trends"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GPT_MODEL = os.environ.get("TREND_RAG_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TREND_RAG_TOP_K", "10"))
FETCH_K = int(os.environ.get("TREND_RAG_FETCH_K", "30"))
LEXICAL_FETCH_K = int(os.environ.get("TREND_RAG_LEXICAL_FETCH_K", "30"))
RRF_K = int(os.environ.get("TREND_RAG_RRF_K", "60"))

ALIAS_GROUPS = {
    "french_bob": ["french bob", "french-bob", "프렌치 밥", "프렌치밥"],
    "blunt_bob": ["blunt bob", "blunt-bob", "블런트 밥", "칼단발", "일자 단발"],
    "pageboy_bob": ["pageboy bob", "pageboy-bob", "page boy bob", "페이지보이 밥", "페이지 보이 밥"],
    "power_bob": ["power bob", "power-bob", "파워 밥", "파워 보브"],
    "graduated_bob": ["graduated bob", "graduated-bob", "그래쥬에이티드 밥", "그래듀에이티드 밥"],
    "bob": ["bob", "밥", "보브", "단발", "단발머리"],
    "lob": ["lob", "long bob", "long-bob", "롱 밥", "롱밥", "롭"],
    "bixie": ["bixie", "빅시"],
    "pixie": ["pixie", "픽시"],
    "face_framing": ["face framing", "face-framing", "페이스 프레이밍", "페이스프레이밍"],
    "layered_cut": ["layered cut", "layered cuts", "layer cut", "layer cuts", "레이어 컷", "레이어드 컷", "레이어드"],
    "wolf_cut": ["wolf cut", "wolf-cut", "울프컷", "울프 컷"],
    "shag": ["shag", "shaggy", "샤그", "샤기"],
    "perm": [
        "perm",
        "perms",
        "펌",
        "웨이브",
        "wave",
        "waves",
        "curl",
        "curls",
        "컬",
        "c컬",
        "s컬",
        "볼륨매직",
        "디지털펌",
        "히팅펌",
    ],
    "soft_wave": ["soft wave", "soft waves", "소프트 웨이브", "소프트웨이브"],
    "effortless_hair": ["effortless hair", "에포트리스 헤어", "내추럴 무드 헤어"],
    "blowout_with_volume": [
        "blowout with volume",
        "volume blowout",
        "볼륨 블로우아웃",
        "볼륨 블로우 아웃",
        "볼륨 드라이",
        "볼륨매직 c컬",
    ],
    "wet_look": ["wet look", "wet-look", "웨트 룩", "웻 룩", "wet hair"],
    "ash_brown": ["ash brown", "ash-brown", "애쉬 브라운", "애쉬브라운"],
    "copper": ["copper", "코퍼"],
    "bronde": ["bronde", "브론드"],
    "blonde": ["blonde", "blond", "블론드"],
    "highlight": ["highlight", "highlights", "하이라이트"],
    "braid_crown": ["braid crown", "braid-crown", "crown braid", "crown braids", "브레이드 크라운", "크라운 브레이드"],
    "updo": ["updo", "up do", "업스타일"],
    "ponytail": ["ponytail", "ponytail", "포니테일"],
    "bun": ["bun", "번"],
    "clinic": ["clinic", "클리닉", "treatment", "트리트먼트", "care", "케어"],
    "repair": ["repair", "repairing", "복구", "리페어"],
    "damage_care": ["damage care", "손상모 케어", "손상모 관리", "damage treatment"],
}

TREND_PRIORITY_RULES = (
    ({"bob", "french_bob", "blunt_bob", "pageboy_bob", "power_bob", "graduated_bob"}, {"blunt_bob", "pageboy_bob", "power_bob", "french_bob", "graduated_bob", "bob", "lob"}),
    ({"bixie", "pixie"}, {"bixie", "pixie", "face_framing"}),
    ({"perm", "soft_wave", "blowout_with_volume"}, {"soft_wave", "effortless_hair", "blowout_with_volume", "perm"}),
    ({"layered_cut", "shag", "wolf_cut"}, {"layered_cut", "soft_wave", "effortless_hair", "shag", "wolf_cut"}),
    ({"ash_brown", "copper", "bronde", "blonde", "highlight"}, {"ash_brown", "copper", "bronde", "blonde", "highlight"}),
    ({"braid_crown", "updo", "ponytail", "bun"}, {"braid_crown", "updo", "ponytail", "bun"}),
    ({"clinic", "repair", "damage_care"}, {"clinic", "repair", "damage_care"}),
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "look",
    "style",
    "styles",
    "trend",
    "trends",
    "hair",
    "hairstyle",
    "hairstyles",
    "추천",
    "정리",
    "설명",
    "포인트",
    "시술",
    "방법",
    "기준",
    "유행",
    "요즘",
    "관련",
    "스타일",
    "헤어",
    "머리",
}

SYSTEM_PROMPT = """당신은 전문 헤어 트렌드 컨설턴트입니다.
아래 제공되는 [참고 자료]는 2025-2026년 글로벌·한국 패션 매거진에서 수집한 최신 헤어 트렌드 데이터입니다.

중요 배경 정보:
- 데이터의 대부분은 여성 헤어 트렌드입니다. "남자", "men", "male", "grooming" 등이 명시되지 않은 자료는 여성 대상입니다.
- 따라서 "여자 머리", "여성 헤어" 관련 질문에는 제공된 자료 대부분이 해당됩니다.

규칙:
1. 반드시 [참고 자료]에 기반하여 답변하세요.
2. 참고 자료의 요약(summary)과 스타일 태그를 적극 활용하여 구체적으로 답변하세요.
3. 각 추천 스타일마다 출처(source)를 반드시 함께 언급해주세요.
4. 한국어로 답변하세요.
5. 구체적인 스타일링 팁이 자료에 있으면 반드시 포함해주세요.
6. 자료에 완전히 관련 없는 주제(헤어가 아닌 질문)에만 "해당 정보는 현재 데이터에 없습니다"라고 안내하세요.
7. 답변은 3~5개 스타일을 추천하되, 각 스타일별로 어떤 사람에게 어울리는지, 연출 방법도 포함하세요.
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
    client = chromadb.PersistentClient(path=str(CHROMA_TRENDS_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB 컬렉션을 찾을 수 없습니다. 먼저 `python -m rag_pipeline.main vectorize`를 실행하세요. ({exc})"
        ) from exc


def _normalize_basic_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = re.sub(r"[-/|+,]", " ", normalized)
    normalized = re.sub(r"[^0-9a-z가-힣_ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@lru_cache(maxsize=1)
def _alias_patterns() -> list[tuple[str, str]]:
    patterns: list[tuple[str, str]] = []
    for canonical, variants in ALIAS_GROUPS.items():
        all_variants = {canonical.replace("_", " "), canonical, *variants}
        for variant in all_variants:
            normalized_variant = _normalize_basic_text(variant.replace("_", " "))
            if normalized_variant:
                patterns.append((normalized_variant, canonical))
    patterns.sort(key=lambda item: len(item[0]), reverse=True)
    return patterns


def _apply_aliases(text: str) -> str:
    normalized = f" {_normalize_basic_text(text)} "
    for variant, canonical in _alias_patterns():
        normalized = normalized.replace(f" {variant} ", f" {canonical} ")
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_token(token: str) -> str:
    if "_" in token:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[0-9a-z가-힣_]+", _apply_aliases(text))
    normalized_tokens: list[str] = []
    for token in tokens:
        token = _normalize_token(token)
        if len(token) <= 1 or token in STOPWORDS:
            continue
        normalized_tokens.append(token)
    return normalized_tokens


def _rrf_score(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (RRF_K + rank)


def _field_overlap_score(query_tokens: set[str], entry: dict, weight_scale: float = 1.0) -> float:
    if not query_tokens:
        return 0.0

    score = 0.0
    matched_tokens = 0
    field_weights = (
        ("title_tokens", 4.0),
        ("style_tokens", 5.0),
        ("color_tokens", 5.0),
        ("search_tokens", 3.0),
        ("summary_tokens", 2.0),
        ("category_tokens", 1.0),
        ("source_tokens", 1.0),
    )
    for token in query_tokens:
        token_matched = False
        for field_name, weight in field_weights:
            if token in entry[field_name]:
                score += weight * weight_scale
                token_matched = True
        if token_matched:
            matched_tokens += 1
            if "_" in token:
                score += 2.5 * weight_scale
    if matched_tokens:
        score += (matched_tokens / max(len(query_tokens), 1)) * 4.0 * weight_scale
    return score


def _theme_boost(query_tokens: set[str], entry: dict) -> float:
    doc_tokens = (
        entry["title_tokens"]
        | entry["style_tokens"]
        | entry["color_tokens"]
        | entry["search_tokens"]
        | entry["summary_tokens"]
    )
    score = 0.0
    for triggers, preferred_tokens in TREND_PRIORITY_RULES:
        if query_tokens & triggers:
            matched = len(doc_tokens & preferred_tokens)
            if matched:
                score += 4.0 * matched
    return score


@lru_cache(maxsize=1)
def _get_trend_corpus() -> list[dict]:
    payload = _get_collection().get(include=["documents", "metadatas"])
    entries: list[dict] = []
    for doc_id, document, metadata in zip(payload["ids"], payload["documents"], payload["metadatas"]):
        meta = metadata or {}
        title = str(meta.get("display_title", ""))
        category = str(meta.get("category", ""))
        summary = str(meta.get("summary", ""))
        style_tags = str(meta.get("style_tags", ""))
        color_tags = str(meta.get("color_tags", ""))
        source = str(meta.get("source", ""))
        year = str(meta.get("year", ""))
        search_text = str(document or "")
        entries.append(
            {
                "id": doc_id,
                "title": title,
                "category": category,
                "summary": summary,
                "style_tags": style_tags,
                "color_tags": color_tags,
                "source": source,
                "year": year,
                "search_text": search_text,
                "title_tokens": set(_tokenize(title)),
                "category_tokens": set(_tokenize(category)),
                "summary_tokens": set(_tokenize(summary)),
                "style_tokens": set(_tokenize(style_tags)),
                "color_tokens": set(_tokenize(color_tags)),
                "source_tokens": set(_tokenize(source)),
                "search_tokens": set(_tokenize(search_text)),
            }
        )
    return entries


def expand_query(query: str) -> str:
    response = _get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "사용자의 헤어 관련 질문을 벡터 검색에 최적화된 키워드로 확장하세요.\n"
                    "한국어와 영어 키워드를 모두 포함하세요.\n"
                    "키워드만 쉼표로 구분하여 출력하세요. 설명은 하지 마세요.\n"
                    "예시: '여자 봄 머리' → '2026 spring women hairstyle trend, 여성 봄 헤어스타일, bob, lob, layers, bangs, 단발, 레이어드, 뱅'"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=200,
    )
    expanded = response.choices[0].message.content or ""
    return f"{query}, {expanded.strip()}"


def retrieve(query: str, n_results: int = TOP_K, expand: bool = True) -> list[dict]:
    search_query = expand_query(query) if expand else query
    collection = _get_collection()
    corpus = _get_trend_corpus()
    corpus_by_id = {entry["id"]: entry for entry in corpus}

    fetch_count = min(max(n_results * 4, FETCH_K), max(collection.count(), 1))
    dense_results = collection.query(query_texts=[search_query], n_results=fetch_count)

    dense_rank_map: dict[str, int] = {}
    dense_distance_map: dict[str, float] = {}
    for rank, (doc_id, distance) in enumerate(zip(dense_results["ids"][0], dense_results["distances"][0]), start=1):
        dense_rank_map[doc_id] = rank
        dense_distance_map[doc_id] = float(distance)

    primary_query_tokens = set(_tokenize(query))
    expanded_query_tokens = set(_tokenize(search_query)) - primary_query_tokens

    lexical_candidates: list[tuple[float, str]] = []
    for entry in corpus:
        lexical_score = _field_overlap_score(primary_query_tokens, entry, weight_scale=1.0)
        lexical_score += _field_overlap_score(expanded_query_tokens, entry, weight_scale=0.6)
        lexical_score += _theme_boost(primary_query_tokens | expanded_query_tokens, entry)
        if lexical_score > 0:
            lexical_candidates.append((lexical_score, entry["id"]))
    lexical_candidates.sort(key=lambda item: (-item[0], item[1]))
    lexical_candidates = lexical_candidates[: min(max(n_results * 4, LEXICAL_FETCH_K), len(lexical_candidates))]

    lexical_rank_map = {doc_id: rank for rank, (_, doc_id) in enumerate(lexical_candidates, start=1)}
    lexical_score_map = {doc_id: score for score, doc_id in lexical_candidates}

    candidate_ids = set(dense_rank_map) | set(lexical_rank_map)
    ranked_docs: list[tuple[float, dict]] = []
    for doc_id in candidate_ids:
        entry = corpus_by_id[doc_id]
        dense_rrf = _rrf_score(dense_rank_map.get(doc_id))
        lexical_rrf = _rrf_score(lexical_rank_map.get(doc_id))
        field_rerank = _field_overlap_score(primary_query_tokens, entry, weight_scale=1.0)
        field_rerank += _field_overlap_score(expanded_query_tokens, entry, weight_scale=0.4)
        field_rerank += _theme_boost(primary_query_tokens | expanded_query_tokens, entry)
        hybrid_score = (dense_rrf * 1.2) + (lexical_rrf * 1.0) + min(field_rerank, 40.0) / 100.0

        ranked_docs.append(
            (
                hybrid_score,
                {
                    "title": entry["title"],
                    "category": entry["category"],
                    "summary": entry["summary"],
                    "style_tags": entry["style_tags"],
                    "color_tags": entry["color_tags"],
                    "source": entry["source"],
                    "year": entry["year"],
                    "distance": dense_distance_map.get(doc_id, 1.5),
                    "lexical_score": round(float(lexical_score_map.get(doc_id, 0.0)), 4),
                    "hybrid_score": round(float(hybrid_score), 6),
                },
            )
        )

    ranked_docs.sort(
        key=lambda item: (
            -item[0],
            item[1]["distance"],
            -float(item[1].get("lexical_score", 0.0) or 0.0),
            item[1]["title"],
        )
    )
    return [doc for _, doc in ranked_docs[:n_results]]


def build_context(docs: list[dict]) -> str:
    context_parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        context_parts.append(
            f"""[자료 {index}]
제목: {doc['title']}
카테고리: {doc['category']}
요약: {doc['summary']}
스타일 태그: {doc['style_tags']}
컬러 태그: {doc['color_tags']}
출처: {doc['source']} ({doc['year']})
"""
        )
    return "\n".join(context_parts)


def ask(query: str, n_results: int = TOP_K, expand: bool = True) -> str:
    docs = retrieve(query, n_results=n_results, expand=expand)
    if not docs:
        return "관련 트렌드 데이터를 찾지 못했습니다."

    response = _get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[참고 자료]\n{build_context(docs)}\n\n[질문]\n{query}"},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content or ""


def interactive_chat() -> None:
    collection = _get_collection()
    print("=" * 60)
    print("💇 헤어 트렌드 RAG 챗봇 (ChromaDB + GPT)")
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

        print("\n💬 GPT 답변 생성 중...\n")
        print(ask(query))
        print("\n" + "-" * 60)


if __name__ == "__main__":
    interactive_chat()
