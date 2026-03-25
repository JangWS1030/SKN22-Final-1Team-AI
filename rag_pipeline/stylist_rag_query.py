from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache

from openai import OpenAI

from .ncs_rag_query import retrieve as retrieve_ncs
from .rag_query import retrieve as retrieve_trends


GPT_MODEL = os.environ.get(
    "STYLIST_RAG_MODEL",
    os.environ.get("NCS_RAG_MODEL", os.environ.get("TREND_RAG_MODEL", "gpt-4o-mini")),
)
TREND_TOP_K = int(os.environ.get("STYLIST_TREND_TOP_K", "4"))
NCS_TOP_K = int(os.environ.get("STYLIST_NCS_TOP_K", "4"))

SYSTEM_PROMPT = """당신은 현업 미용사를 위한 헤어 트렌드 + 시술 통합 어시스턴트입니다.

[트렌드 자료]는 최신 스타일 방향과 무드, [시술 자료]는 NCS 기반 작업 절차와 주의사항입니다.

규칙:
1. 미용사가 바로 참고할 수 있는 실무용 답변으로 작성하세요.
2. 먼저 질문과 가장 잘 맞는 최신 트렌드 방향 2~3개를 정리하세요.
3. 그 다음 해당 트렌드를 구현하기 위한 시술 접근을 준비, 작업 포인트, 주의사항, 마무리/홈케어 순으로 설명하세요.
4. 시술 단계는 반드시 [시술 자료]에 나온 내용만 사용하세요. 트렌드 자료만 보고 절차를 추정하지 마세요.
5. 트렌드 자료와 시술 자료가 완전히 일치하지 않으면, 어떤 부분이 트렌드 추천이고 어떤 부분이 시술 근거인지 구분해서 적으세요.
6. 자료에 없는 세부는 "자료 기준으로는 확인되지 않음"이라고 적으세요.
7. 답변 마지막에 `트렌드 출처`, `시술 출처`를 분리해서 적으세요.
8. 한국어로 답변하세요.
9. NCS 시술 자료가 정확히 같은 디자인명이 아니라 같은 시술 축의 기본 문서라면, "해당 트렌드에 적용 가능한 기본 시술 포인트"라고 명시하세요.
"""

SERVICE_HINTS = {
    "커트": [
        "bob",
        "lob",
        "pixie",
        "bixie",
        "bang",
        "fringe",
        "layer",
        "shag",
        "wolf",
        "crop",
        "cut",
        "단발",
        "컷",
        "커트",
        "레이어",
        "뱅",
        "숏컷",
    ],
    "펌": [
        "perm",
        "wave",
        "curl",
        "curly",
        "curls",
        "웨이브",
        "펌",
        "컬",
        "s-컬",
        "c컬",
        "히피",
        "볼륨매직",
        "셋팅",
    ],
    "컬러": [
        "blonde",
        "bronde",
        "balayage",
        "highlight",
        "color",
        "colour",
        "bleach",
        "tone",
        "염색",
        "컬러",
        "탈색",
        "하이라이트",
        "브라운",
        "블론드",
    ],
    "업스타일": [
        "updo",
        "braid",
        "braids",
        "ponytail",
        "bun",
        "chignon",
        "twist",
        "업스타일",
        "묶음",
        "번",
        "브레이드",
    ],
    "스타일링": [
        "blowout",
        "blow dry",
        "blow-dry",
        "slick",
        "sleek",
        "wet look",
        "style",
        "styling",
        "드라이",
        "스타일링",
        "웨트",
    ],
    "샴푸/클리닉": [
        "shampoo",
        "clinic",
        "treatment",
        "care",
        "손상모",
        "클리닉",
        "트리트먼트",
        "케어",
        "샴푸",
        "모발관리",
        "헤어케어",
    ],
}

SERVICE_QUERY_HINTS = {
    "커트": [
        "헤어커트",
        "컷 시술",
        "원랭스 커트",
        "커트 베이스",
        "커트 질감 처리",
    ],
    "펌": [
        "헤어펌",
        "펌 시술",
        "와인딩",
        "볼륨매직",
        "C컬",
        "매직스트레이트펌",
        "아이론",
        "펌 마무리",
        "펌 후 홈케어",
    ],
    "컬러": [
        "헤어컬러",
        "염색 시술",
        "컬러 도포",
        "탈색 주의사항",
        "컬러 마무리",
    ],
    "업스타일": [
        "업스타일 시술",
        "블로킹",
        "핀 고정",
        "업스타일 마무리",
        "헤어스타일 디자인",
    ],
    "스타일링": [
        "헤어스타일링",
        "블로우드라이",
        "아이론 스타일링",
        "스타일 마무리",
        "질감 표현",
    ],
    "샴푸/클리닉": [
        "손상모 클리닉",
        "샴푸 클리닉",
        "모발 클리닉",
        "손상모 관리",
        "샴푸 시술",
    ],
}

SERVICE_RULES = {
    "커트": {
        "aliases": ["커트", "컷", "헤어커트", "cut", "bob", "lob", "pixie", "단발", "레이어"],
        "preferred_sources": ["헤어커트"],
        "secondary_sources": ["헤어스타일", "샴푸와 클리닉", "헤어펌+디자인"],
        "discouraged_sources": ["특수머리", "헤어컬러"],
    },
    "펌": {
        "aliases": ["펌", "perm", "웨이브", "wave", "curl", "컬", "히팅펌", "콜드펌", "볼륨매직", "매직"],
        "preferred_sources": ["헤어펌"],
        "secondary_sources": ["헤어스타일", "샴푸와 클리닉"],
        "discouraged_sources": ["특수머리", "헤어컬러"],
    },
    "컬러": {
        "aliases": ["컬러", "염색", "탈색", "color", "colour", "bleach", "highlight", "하이라이트"],
        "preferred_sources": ["헤어컬러"],
        "secondary_sources": ["샴푸와 클리닉"],
        "discouraged_sources": ["특수머리", "헤어펌"],
    },
    "업스타일": {
        "aliases": ["업스타일", "updo", "브레이드", "braid", "번", "bun", "포니테일", "ponytail"],
        "preferred_sources": ["특수머리", "헤어스타일"],
        "secondary_sources": ["샴푸와 클리닉"],
        "discouraged_sources": ["헤어컬러"],
    },
    "스타일링": {
        "aliases": ["스타일링", "style", "styling", "드라이", "blow", "아이론", "텍스처", "웨트"],
        "preferred_sources": ["헤어스타일", "샴푸와 클리닉", "헤어펌"],
        "secondary_sources": ["헤어커트"],
        "discouraged_sources": [],
    },
    "샴푸/클리닉": {
        "aliases": ["샴푸", "클리닉", "트리트먼트", "손상모", "헤어케어", "모발관리"],
        "preferred_sources": ["샴푸와 클리닉"],
        "secondary_sources": ["헤어펌", "헤어컬러"],
        "discouraged_sources": ["특수머리"],
    },
}

PRIMARY_SERVICE_TOKENS = {
    "커트": ["커트", "컷", "헤어커트"],
    "펌": ["펌", "웨이브", "perm"],
    "컬러": ["컬러", "염색", "탈색", "color", "colour"],
    "업스타일": ["업스타일", "브레이드", "번", "bun", "updo"],
    "스타일링": ["스타일링", "드라이", "아이론", "styling"],
    "샴푸/클리닉": ["샴푸", "클리닉", "트리트먼트", "손상모", "treatment", "clinic"],
}

SERVICE_RESPONSE_GUIDES = {
    "커트": """주요 답변 형식:
1. `추천 트렌드`: 현재 유행 스타일 2~3개와 어울리는 모질/얼굴형 포인트
2. `커트 설계`: 길이감, 아웃라인, 섹션/베이스, 질감 처리 포인트
3. `작업 순서`: 준비 -> 베이스 설정 -> 커트 -> 질감 처리 -> 마무리 스타일링
4. `주의사항`: 길이 편차, 베이스 폭, 과한 질감 처리, 얼굴형 보정 포인트
5. `손질/유지`: 드라이 방향, 텍스처 제품, 재방문 주기

커트 답변 규칙:
- 시술 자료에 `베이스`, `시술각`, `질감 처리`가 있으면 우선적으로 설명하세요.
- 트렌드 자료의 밥/레이어/픽시 추천은 `커트 설계`와 연결해서 풀어주세요.
- 펌 자료를 인용할 때는 "해당 커트에 응용 가능한 기본 시술 포인트"라고 구분하세요.""",
    "펌": """주요 답변 형식:
1. `추천 트렌드`: 현재 유행 웨이브/컬 방향 2~3개
2. `디자인 해석`: 컬 굵기, 볼륨 위치, 레이어/커트 연계 포인트
3. `시술 순서`: 상담 -> 사전 준비 -> 약제/와인딩 -> 처리 -> 중화/헹굼 -> 건조/마무리
4. `주의사항`: 손상도, 열/시간 관리, 연화/중화 리스크
5. `손질/홈케어`: 컬 말리기, 제품, 홈케어, 재시술 주기

펌 답변 규칙:
- `와인딩`, `롯드`, `연화`, `중화`, `건조` 관련 NCS 근거를 우선 사용하세요.
- 트렌드 자료는 컬의 무드와 실루엣 설명에 쓰고, 절차는 반드시 시술 자료 기준으로 설명하세요.
- 커트가 함께 필요한 경우 `사전 커트` 또는 `레이어 연계`를 별도 한 줄로 구분하세요.""",
    "컬러": """주요 답변 형식:
1. `추천 트렌드`: 유행 컬러 톤 2~3개와 피부톤/무드 포인트
2. `컬러 설계`: 베이스 톤, 밝기, 보색/톤다운, 탈색 필요 여부
3. `시술 순서`: 상담 -> 진단 -> 제품 선택 -> 도포/방치 -> 헹굼/후처리 -> 마무리
4. `주의사항`: 두피/손상모, 톤 편차, 탈색 리스크, 색 빠짐 관리
5. `유지 관리`: 컬러 샴푸, 열 관리, 다음 방문 포인트

컬러 답변 규칙:
- `염색`, `탈색`, `제품 선택`, `도포`, `후처리` 관련 근거를 우선 사용하세요.
- 자료에 정확한 컬러 레시피가 없으면 추정하지 말고 "자료 기준으로는 확인되지 않음"이라고 적으세요.
- 트렌드 컬러 추천과 실제 NCS 시술 단계는 반드시 구분해서 쓰세요.""",
    "업스타일": """주요 답변 형식:
1. `추천 트렌드`: 현재 유행 업스타일 방향 2~3개
2. `디자인 포인트`: 볼륨 위치, 질감, 묶는 위치, 장식/가닥 처리
3. `작업 순서`: 준비 도구 -> 블로킹/베이스 -> 고정 -> 형태 정리 -> 마무리
4. `주의사항`: 고정력, 핀 사용, 무게 중심, 장시간 유지 포인트
5. `유지 안내`: 고정 제품, 수정 포인트, 행사 후 정리

업스타일 답변 규칙:
- `블로킹`, `핀 고정`, `업스타일 절차`, `가체/보조재료`가 있으면 우선적으로 활용하세요.
- 가발/특수머리 자료를 쓸 때는 일반 업스타일에 바로 적용 가능한 부분만 추려서 설명하세요.
- 실무에서 바로 참고할 수 있게 고정 포인트와 무너짐 방지 포인트를 짧게 적으세요.""",
    "스타일링": """주요 답변 형식:
1. `추천 트렌드`: 현재 유행 질감/마감 방향 2~3개
2. `스타일링 설계`: 필요한 수분 상태, 제품군, 도구 선택
3. `작업 순서`: 건조 상태 확인 -> 제품 도포 -> 드라이/아이론 -> 텍스처 정리 -> 고정
4. `주의사항`: 과열, 과도한 제품 사용, 볼륨 붕괴 포인트
5. `고객 손질`: 집에서 재현하는 방법과 제품 사용 순서

스타일링 답변 규칙:
- `블로우드라이`, `아이론`, `에센스`, `세럼`, `왁스`, `스프레이` 같은 제품/도구 근거를 적극 사용하세요.
- 단순 트렌드 소개가 아니라 재현 방법 중심으로 설명하세요.""",
    "샴푸/클리닉": """주요 답변 형식:
1. `추천 트렌드`: 요즘 많이 찾는 케어 방향이나 고객 니즈
2. `진단 포인트`: 손상도, 건조도, 열/화학 시술 이력
3. `시술 순서`: 상담 -> 샴푸/전처리 -> 도포/핸들링 -> 열처리/침투 -> 후처리 -> 마무리
4. `주의사항`: 과도한 마찰, 열 손상, 제품 선택 미스, 두피 민감도
5. `홈케어`: 추천 제품군, 주기, 피해야 할 습관

클리닉 답변 규칙:
- `손상모`, `전처리`, `핸들링`, `스티머/미스트`, `후처리`, `홈케어` 근거를 우선 사용하세요.
- 단순 샴푸 절차와 손상모 클리닉 절차를 구분해서 써주세요.
- 자료에 있는 단계가 많으면 핵심 순서만 짧게 요약하고, 고객 설명 문장처럼 정리하세요.""",
}

DEFAULT_CATEGORY_WEIGHTS = {
    "procedure": 18,
    "preparation": 14,
    "finishing": 11,
    "aftercare": 10,
    "consultation": 7,
    "safety": 6,
    "theory": 3,
}

CATEGORY_INTENT_WEIGHTS = [
    (
        ["홈케어", "손질", "유지", "유지법", "관리법", "집에서", "애프터", "aftercare"],
        {
            "aftercare": 20,
            "finishing": 16,
            "procedure": 12,
            "consultation": 6,
            "preparation": 5,
            "safety": 5,
            "theory": 2,
        },
    ),
    (
        ["주의", "주의사항", "주의점", "위험", "손상", "화상", "부작용", "조심"],
        {
            "safety": 20,
            "procedure": 14,
            "preparation": 10,
            "finishing": 8,
            "aftercare": 7,
            "consultation": 5,
            "theory": 3,
        },
    ),
    (
        ["상담", "디자인", "추천", "어울리는", "결정"],
        {
            "consultation": 18,
            "procedure": 12,
            "preparation": 9,
            "finishing": 8,
            "aftercare": 6,
            "safety": 5,
            "theory": 4,
        },
    ),
]


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _infer_service_types(query: str, trend_docs: list[dict]) -> list[str]:
    query_text = _normalize_text(query)
    trend_text_parts: list[str] = []
    for doc in trend_docs:
        trend_text_parts.extend(
            [
                str(doc.get("title", "")),
                str(doc.get("category", "")),
                str(doc.get("summary", "")),
                str(doc.get("style_tags", "")),
                str(doc.get("color_tags", "")),
            ]
        )

    trend_text = _normalize_text(" ".join(trend_text_parts))
    scores: dict[str, int] = {}
    for service_type, keywords in SERVICE_HINTS.items():
        score = 0
        for keyword in keywords:
            if keyword in query_text:
                score += 5
            if keyword in trend_text:
                score += 1
        for primary_token in PRIMARY_SERVICE_TOKENS.get(service_type, []):
            if _normalize_text(primary_token) in query_text:
                score += 6
        if score > 0:
            scores[service_type] = score

    if not scores:
        return ["스타일링"]

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_score = ranked[0][1]
    return [service_type for service_type, score in ranked if score >= max(2, top_score - 2)][:3]


def _extract_trend_keywords(trend_docs: list[dict], limit: int = 8) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()

    for doc in trend_docs:
        raw_tags = str(doc.get("style_tags", ""))
        for raw_tag in raw_tags.split(","):
            tag = raw_tag.strip()
            normalized = _normalize_text(tag)
            if not tag or normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(tag)
            if len(keywords) >= limit:
                return keywords
    return keywords


def build_ncs_query(query: str, trend_docs: list[dict]) -> tuple[str, list[str], list[str]]:
    service_types = _infer_service_types(query, trend_docs)
    style_keywords = _extract_trend_keywords(trend_docs)

    pieces = [query]
    if service_types:
        pieces.append("관련 시술: " + ", ".join(service_types))
    if style_keywords:
        pieces.append("스타일 키워드: " + ", ".join(style_keywords))

    return " | ".join(pieces), service_types, style_keywords


def _service_match_score(doc: dict, service_types: list[str]) -> int:
    if not service_types:
        return 0

    target = _doc_text(doc)
    score = 0
    for index, service_type in enumerate(service_types):
        rule = SERVICE_RULES.get(service_type, {})
        aliases = rule.get("aliases", [service_type])
        weight = 10 if index == 0 else 5
        if any(_normalize_text(alias) in target for alias in aliases):
            score += weight
    return score


def _doc_text(doc: dict) -> str:
    return _normalize_text(
        " ".join(
            [
                str(doc.get("service_type", "")),
                str(doc.get("category", "")),
                str(doc.get("title", "")),
                str(doc.get("summary", "")),
                str(doc.get("stylist_answer", "")),
                str(doc.get("source_document_name", "")),
                str(doc.get("search_text", "")),
                str(doc.get("source_alias_names", "")),
                str(doc.get("tools", "")),
                str(doc.get("steps", "")),
                str(doc.get("cautions", "")),
            ]
        )
    )


def _query_tokens(query: str) -> list[str]:
    raw_tokens = re.findall(r"[0-9A-Za-z가-힣]+", _normalize_text(query))
    stopwords = {
        "요즘",
        "유행",
        "추천",
        "정리",
        "포인트",
        "방법",
        "어떻게",
        "해주세요",
        "해줘",
        "알려줘",
        "그",
        "스타일",
        "시술",
    }
    return [token for token in raw_tokens if len(token) >= 2 and token not in stopwords]


def _primary_service(service_types: list[str]) -> str:
    return service_types[0] if service_types else "스타일링"


def build_response_guide(service_types: list[str]) -> str:
    primary_service = _primary_service(service_types)
    primary_guide = SERVICE_RESPONSE_GUIDES.get(primary_service, SERVICE_RESPONSE_GUIDES["스타일링"])
    secondary = [service for service in service_types[1:] if service != primary_service]

    parts = [f"주요 시술 축: {primary_service}", primary_guide]
    if secondary:
        parts.append("보조 시술 축: " + ", ".join(secondary))
        parts.append(
            "보조 시술 축은 주 시술 흐름을 해치지 않는 범위에서만 보완적으로 설명하고, 주 절차는 주요 시술 축 기준으로 유지하세요."
        )
    return "\n\n".join(parts)


def _category_weights_for_query(query: str) -> dict[str, int]:
    normalized_query = _normalize_text(query)
    for trigger_tokens, weights in CATEGORY_INTENT_WEIGHTS:
        if any(_normalize_text(token) in normalized_query for token in trigger_tokens):
            return weights
    return DEFAULT_CATEGORY_WEIGHTS


def _source_score(doc: dict, service_types: list[str]) -> int:
    source_text = _normalize_text(str(doc.get("source_document_name", "")))
    if not service_types:
        return 0

    total = 0
    for index, service_type in enumerate(service_types):
        rule = SERVICE_RULES.get(service_type, {})
        scale = 1.0 if index == 0 else 0.6
        for pattern in rule.get("preferred_sources", []):
            if _normalize_text(pattern) in source_text:
                total += int(14 * scale)
        for pattern in rule.get("secondary_sources", []):
            if _normalize_text(pattern) in source_text:
                total += int(6 * scale)
        for pattern in rule.get("discouraged_sources", []):
            if _normalize_text(pattern) in source_text:
                total -= int(8 * scale)
    return total


def _actionability_score(doc: dict) -> int:
    score = 0
    if str(doc.get("steps", "")).strip():
        score += 8
    if str(doc.get("cautions", "")).strip():
        score += 4
    if str(doc.get("tools", "")).strip():
        score += 2
    if str(doc.get("category", "")) == "theory" and not str(doc.get("steps", "")).strip():
        score -= 4
    return score


def _keyword_overlap_score(query: str, doc: dict, style_keywords: list[str]) -> int:
    doc_text = _doc_text(doc)
    score = 0
    for token in _query_tokens(query):
        if token in doc_text:
            score += 3
    for keyword in style_keywords:
        normalized_keyword = _normalize_text(keyword)
        if len(normalized_keyword) >= 4 and normalized_keyword in doc_text:
            score += 2
    return min(score, 18)


def _retrieval_signal_score(doc: dict) -> int:
    lexical_score = float(doc.get("lexical_score", 0.0) or 0.0)
    hybrid_score = float(doc.get("hybrid_score", 0.0) or 0.0)
    adjacent_bonus = float(doc.get("adjacent_bonus", 0.0) or 0.0)

    score = 0
    score += min(10, int(round(lexical_score)))
    score += min(6, int(round(hybrid_score * 200)))
    score += min(3, int(round(adjacent_bonus * 200)))
    return score


def _ncs_doc_score(query: str, doc: dict, service_types: list[str], style_keywords: list[str]) -> tuple[int, float]:
    category = str(doc.get("category", ""))
    category_score = _category_weights_for_query(query).get(category, 0)
    total = 0
    total += _service_match_score(doc, service_types)
    total += _source_score(doc, service_types)
    total += category_score
    total += _actionability_score(doc)
    total += _keyword_overlap_score(query, doc, style_keywords)
    total += _retrieval_signal_score(doc)
    return total, float(doc.get("distance", 999.0))


def _dedupe_ncs_docs(docs: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in docs:
        key = (
            str(doc.get("title", "")),
            str(doc.get("source_document_name", "")),
            str(doc.get("source_page", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _safe_page(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _doc_identity_key(doc: dict) -> tuple[str, str, str]:
    return (
        str(doc.get("title", "")),
        str(doc.get("source_document_name", "")),
        str(doc.get("source_page", "")),
    )


def _source_cluster_window(source_name: str, primary_service: str) -> int:
    normalized_source = _normalize_text(source_name)
    if primary_service in {"펌", "업스타일"}:
        return 2
    if any(token in normalized_source for token in ("특수머리", "헤어펌")):
        return 2
    return 1


def _select_ncs_docs(ranked_docs: list[dict], limit: int, service_types: list[str]) -> list[dict]:
    if not ranked_docs:
        return []

    selected: list[dict] = []
    used: set[tuple[str, str, str]] = set()
    primary_service = _primary_service(service_types)

    def push(doc: dict) -> None:
        key = _doc_identity_key(doc)
        if key in used or len(selected) >= limit:
            return
        used.add(key)
        selected.append(doc)

    anchor = ranked_docs[0]
    push(anchor)

    anchor_source = _normalize_text(str(anchor.get("source_document_name", "")))
    anchor_page = _safe_page(anchor.get("source_page"))
    if anchor_page is not None:
        anchor_lexical_score = float(anchor.get("lexical_score", 0.0) or 0.0)
        adjacent_lexical_threshold = min(4.0, anchor_lexical_score * 0.5) if anchor_lexical_score > 0 else 4.0
        page_cluster_window = _source_cluster_window(str(anchor.get("source_document_name", "")), primary_service)
        adjacent_docs = [
            doc
            for doc in ranked_docs[1:]
            if _normalize_text(str(doc.get("source_document_name", ""))) == anchor_source
            and (page := _safe_page(doc.get("source_page"))) is not None
            and 0 < abs(page - anchor_page) <= page_cluster_window
            and (
                float(doc.get("adjacent_bonus", 0.0) or 0.0) > 0
                or float(doc.get("lexical_score", 0.0) or 0.0) >= adjacent_lexical_threshold
            )
        ]
        adjacent_docs.sort(
            key=lambda doc: (
                abs((_safe_page(doc.get("source_page")) or anchor_page) - anchor_page),
                -float(doc.get("lexical_score", 0.0) or 0.0),
                -float(doc.get("hybrid_score", 0.0) or 0.0),
                float(doc.get("distance", 999.0)),
            )
        )
        for doc in adjacent_docs:
            push(doc)

    for doc in ranked_docs[1:]:
        push(doc)

    return selected


def _retrieve_ncs_candidates(query: str, service_types: list[str], n_results: int) -> list[dict]:
    candidate_queries = [query]
    for service_type in service_types[:2]:
        query_hints = SERVICE_QUERY_HINTS.get(service_type, [])
        for query_hint in query_hints:
            candidate_queries.append(f"{query} | {query_hint}")
            candidate_queries.append(query_hint)
        candidate_queries.append(f"{service_type} 시술")

    candidates: list[dict] = []
    for candidate_query in candidate_queries:
        candidates.extend(retrieve_ncs(candidate_query, n_results=max(n_results, 6), expand=False))
    return _dedupe_ncs_docs(candidates)


def retrieve_bundle(
    query: str,
    trend_results: int = TREND_TOP_K,
    ncs_results: int = NCS_TOP_K,
    expand: bool = True,
) -> dict:
    trend_docs = retrieve_trends(query, n_results=trend_results, expand=expand)
    ncs_query, service_types, style_keywords = build_ncs_query(query, trend_docs)
    ncs_docs = _retrieve_ncs_candidates(ncs_query, service_types, n_results=max(ncs_results * 2, ncs_results))
    filtered_ncs_docs = [doc for doc in ncs_docs if _service_match_score(doc, service_types) > 0] or ncs_docs
    scored_ncs_docs = [
        (_ncs_doc_score(query, doc, service_types, style_keywords), doc) for doc in filtered_ncs_docs
    ]
    ranked_ncs_docs = [
        doc
        for _, doc in sorted(
            scored_ncs_docs,
            key=lambda item: (-item[0][0], item[0][1]),
        )
    ]

    return {
        "trend_docs": trend_docs,
        "ncs_docs": _select_ncs_docs(ranked_ncs_docs, limit=ncs_results, service_types=service_types),
        "ncs_query": ncs_query,
        "service_types": service_types,
        "style_keywords": style_keywords,
    }


def _build_trend_context(docs: list[dict]) -> str:
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        parts.append(
            f"""[트렌드 자료 {index}]
제목: {doc['title']}
카테고리: {doc['category']}
요약: {doc['summary']}
스타일 태그: {doc['style_tags']}
컬러 태그: {doc['color_tags']}
출처: {doc['source']} ({doc['year']})
"""
        )
    return "\n".join(parts)


def _build_ncs_context(docs: list[dict]) -> str:
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        parts.append(
            f"""[시술 자료 {index}]
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
    return "\n".join(parts)


def build_stylist_user_prompt(query: str, bundle: dict) -> str:
    return (
        f"[질문]\n{query}\n\n"
        f"[답변 형식 가이드]\n{build_response_guide(bundle['service_types'])}\n\n"
        f"[추론된 시술 힌트]\n{', '.join(bundle['service_types']) or '없음'}\n\n"
        f"[추론된 스타일 키워드]\n{', '.join(bundle['style_keywords']) or '없음'}\n\n"
        f"[NCS 검색어]\n{bundle['ncs_query']}\n\n"
        f"[트렌드 자료]\n{_build_trend_context(bundle['trend_docs']) or '없음'}\n\n"
        f"[시술 자료]\n{_build_ncs_context(bundle['ncs_docs']) or '없음'}"
    )


def generate_stylist_answer(query: str, bundle: dict) -> str:
    response = _get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_stylist_user_prompt(query, bundle)},
        ],
        temperature=0.25,
        max_tokens=2200,
    )
    return response.choices[0].message.content or ""


def ask_stylist(
    query: str,
    trend_results: int = TREND_TOP_K,
    ncs_results: int = NCS_TOP_K,
    expand: bool = True,
) -> str:
    bundle = retrieve_bundle(query, trend_results=trend_results, ncs_results=ncs_results, expand=expand)
    if not bundle["trend_docs"] and not bundle["ncs_docs"]:
        return "관련 트렌드와 시술 데이터를 찾지 못했습니다."
    return generate_stylist_answer(query, bundle)


def interactive_chat() -> None:
    print("=" * 60)
    print("💇 미용사용 통합 RAG 챗봇")
    print(f"   모델: {GPT_MODEL} | 트렌드 {TREND_TOP_K}건 + 시술 {NCS_TOP_K}건")
    print("   종료: quit / exit / q")
    print("=" * 60)

    while True:
        query = input("\n🔍 질문: ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            print("👋 종료합니다.")
            break

        print("\n검색 및 답변 생성 중...\n")
        print(ask_stylist(query))
        print("\n" + "-" * 60)


if __name__ == "__main__":
    interactive_chat()
