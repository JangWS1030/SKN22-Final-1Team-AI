from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREND_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "llm_refined_trends.json",
    PROJECT_ROOT / "data" / "trend_pipeline" / "processed" / "llm_refined_trends.json",
)

TOKEN_RE = re.compile(r"[a-z0-9가-힣]+")
STOP_TOKENS = frozenset(
    {
        "hair",
        "hairstyle",
        "style",
        "trend",
        "look",
        "머리",
        "헤어",
        "스타일",
        "트렌드",
        "추천",
        "요즘",
        "유행",
    }
)
STYLE_ALIASES = {
    "단발": ("bob", "lob", "short bob"),
    "보브": ("bob",),
    "밥": ("bob",),
    "숏컷": ("pixie", "short crop"),
    "픽시": ("pixie",),
    "레이어드": ("layered", "layers"),
    "레이어컷": ("layered cut", "layers"),
    "허쉬컷": ("hush cut",),
    "울프컷": ("wolf cut", "soft mullet"),
    "멀릿": ("mullet", "soft mullet"),
    "태슬컷": ("tassel cut", "blunt bob"),
    "샤기": ("shag", "shaggy"),
    "샤기컷": ("shag", "shaggy cut"),
    "커튼뱅": ("curtain bangs",),
    "시스루뱅": ("see-through bangs", "wispy bangs"),
    "뱅": ("bangs",),
    "앞머리": ("bangs",),
    "포니테일": ("ponytail",),
    "브레이드": ("braid", "braided"),
    "번": ("bun", "updo"),
    "업스타일": ("updo",),
    "웨이브": ("waves", "wavy"),
    "컬": ("curls", "curly"),
    "펌": ("perm",),
    "히피펌": ("perm", "curly"),
    "가일컷": ("side part", "classic crop"),
    "리프컷": ("leaf cut", "layered medium cut"),
    "댄디컷": ("dandy cut", "soft crop"),
    "리젠트": ("regent cut", "slicked back"),
}
COLOR_ALIASES = {
    "애쉬브라운": ("ash brown",),
    "애쉬": ("ash", "ash brown"),
    "블랙": ("black",),
    "흑발": ("black",),
    "다크브라운": ("dark brown",),
    "브라운": ("brown",),
    "초코브라운": ("chocolate brown", "brown"),
    "레드브라운": ("auburn", "red brown"),
    "레드": ("red", "auburn"),
    "카퍼": ("copper",),
    "블론드": ("blonde",),
    "금발": ("blonde",),
    "실버": ("silver",),
    "그레이": ("gray", "silver"),
    "핑크": ("pink",),
    "베이지": ("beige", "ash beige"),
}


@dataclass(frozen=True)
class TrendMatch:
    trend_name: str
    hairstyle_text: str
    color_text: str
    description: str
    source: str
    year: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_name": self.trend_name,
            "hairstyle_text": self.hairstyle_text,
            "color_text": self.color_text,
            "description": self.description,
            "source": self.source,
            "year": self.year,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class ResolvedTrendRequest:
    trend_data_path: Optional[str]
    requested_hairstyle_text: str
    requested_color_text: str
    resolved_hairstyle_text: str
    resolved_color_text: str
    recommended_color_text: str
    prompt_hint: str
    matches: tuple[TrendMatch, ...]

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "trend_data_path": self.trend_data_path,
            "requested_hairstyle_text": self.requested_hairstyle_text,
            "requested_color_text": self.requested_color_text,
            "resolved_hairstyle_text": self.resolved_hairstyle_text,
            "resolved_color_text": self.resolved_color_text,
            "recommended_color_text": self.recommended_color_text,
            "prompt_hint": self.prompt_hint,
            "matches": [match.to_dict() for match in self.matches],
        }


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _normalize_text(text: str) -> str:
    return _normalize_space(text).lower()


def _split_phrases(text: str) -> list[str]:
    normalized = _normalize_text(text).replace("·", ",")
    return [part.strip() for part in re.split(r"[,/;|]+", normalized) if part.strip()]


def _tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(_normalize_text(text))
    return [token for token in tokens if len(token) > 1 and token not in STOP_TOKENS]


def _expand_phrases(text: str, alias_map: dict[str, Sequence[str]]) -> list[str]:
    phrases = _split_phrases(text)
    expanded: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase not in seen:
            expanded.append(phrase)
            seen.add(phrase)
        for alias_key, alias_values in alias_map.items():
            if alias_key in phrase:
                for alias in alias_values:
                    normalized_alias = _normalize_text(alias)
                    if normalized_alias and normalized_alias not in seen:
                        expanded.append(normalized_alias)
                        seen.add(normalized_alias)
    return expanded


def _token_overlap_score(query_tokens: Sequence[str], blob: str) -> float:
    if not query_tokens:
        return 0.0
    hits = sum(1 for token in set(query_tokens) if token in blob)
    coverage = hits / max(len(set(query_tokens)), 1)
    return coverage * 3.0


def _phrase_score(query_phrases: Sequence[str], blob: str) -> float:
    score = 0.0
    for phrase in query_phrases:
        if len(phrase) < 2:
            continue
        if phrase in blob:
            score += 1.8 if " " in phrase else 1.1
    return score


def _similarity_score(query_text: str, *candidates: str) -> float:
    normalized_query = _normalize_text(query_text)
    if not normalized_query:
        return 0.0
    best = 0.0
    for candidate in candidates:
        normalized_candidate = _normalize_text(candidate)
        if not normalized_candidate:
            continue
        best = max(best, SequenceMatcher(None, normalized_query, normalized_candidate).ratio())
    if best < 0.45:
        return 0.0
    return (best - 0.45) * 2.4


def _first_non_empty(values: Iterable[str]) -> str:
    for value in values:
        normalized = _normalize_space(value)
        if normalized:
            return normalized
    return ""


def _resolve_trend_data_path() -> Optional[Path]:
    env_path = _normalize_space(os.environ.get("LLM_REFINED_TRENDS_PATH", ""))
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path

    for path in DEFAULT_TREND_DATA_CANDIDATES:
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=1)
def _load_trend_records() -> tuple[Optional[str], list[dict[str, str]]]:
    path = _resolve_trend_data_path()
    if path is None:
        return None, []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return str(path), []

    records: list[dict[str, str]] = []
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            record = {
                "trend_name": _normalize_space(str(row.get("trend_name", ""))),
                "hairstyle_text": _normalize_space(str(row.get("hairstyle_text", ""))),
                "color_text": _normalize_space(str(row.get("color_text", ""))),
                "description": _normalize_space(str(row.get("description", ""))),
                "source": _normalize_space(str(row.get("source", ""))),
                "year": _normalize_space(str(row.get("year", ""))),
            }
            record["blob"] = _normalize_text(
                " ".join(
                    [
                        record["trend_name"],
                        record["hairstyle_text"],
                        record["color_text"],
                        record["description"],
                        record["source"],
                    ]
                )
            )
            records.append(record)
    return str(path), records


def _rank_trend_records(hairstyle_text: str, color_text: str) -> list[TrendMatch]:
    trend_data_path, records = _load_trend_records()
    if not records:
        return []

    style_phrases = _expand_phrases(hairstyle_text, STYLE_ALIASES)
    style_tokens = _tokenize(" ".join(style_phrases))
    color_phrases = _expand_phrases(color_text, COLOR_ALIASES)
    color_tokens = _tokenize(" ".join(color_phrases))

    matches: list[TrendMatch] = []
    for record in records:
        blob = record["blob"]
        style_score = 0.0
        color_score = 0.0
        if hairstyle_text:
            style_score += _phrase_score(style_phrases, blob)
            style_score += _token_overlap_score(style_tokens, blob)
            style_score += _similarity_score(
                hairstyle_text,
                record["hairstyle_text"],
                record["trend_name"],
            )
            if record["hairstyle_text"] and _normalize_text(record["hairstyle_text"]) in _normalize_text(hairstyle_text):
                style_score += 1.2
        if color_text:
            color_score += _phrase_score(color_phrases, blob)
            color_score += _token_overlap_score(color_tokens, blob) * 0.9
            color_score += _similarity_score(color_text, record["color_text"])

        score = style_score + color_score
        if score < 1.2:
            continue

        matches.append(
            TrendMatch(
                trend_name=record["trend_name"],
                hairstyle_text=record["hairstyle_text"],
                color_text=record["color_text"],
                description=record["description"],
                source=record["source"],
                year=record["year"],
                score=score,
            )
        )

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches


def _build_resolved_hairstyle_text(requested: str, matches: Sequence[TrendMatch]) -> str:
    phrases: list[str] = []
    seen: set[str] = set()

    for match in matches[:3]:
        for part in re.split(r"[,/;|]+", match.hairstyle_text):
            phrase = _normalize_space(part)
            normalized = _normalize_text(phrase)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            phrases.append(phrase)

    requested_clean = _normalize_space(requested)
    if requested_clean and not phrases:
        return requested_clean

    if requested_clean:
        requested_norm = _normalize_text(requested_clean)
        if requested_norm and not any(requested_norm in _normalize_text(phrase) for phrase in phrases):
            if all(ord(char) < 128 for char in requested_clean) and len(requested_clean) <= 64:
                phrases.insert(0, requested_clean)

    if not phrases and matches:
        fallback = _first_non_empty([matches[0].trend_name, matches[0].hairstyle_text])
        if fallback:
            phrases.append(fallback)

    return ", ".join(phrases[:4]) if phrases else requested_clean


def _build_prompt_hint(matches: Sequence[TrendMatch]) -> str:
    hints: list[str] = []
    for match in matches[:2]:
        source = _first_non_empty([match.hairstyle_text, match.trend_name])
        if source:
            hints.append(source)
    return ", ".join(dict.fromkeys(hints))


def resolve_generation_request(
    hairstyle_text: str,
    color_text: str,
    *,
    top_k: int = 3,
) -> ResolvedTrendRequest:
    trend_data_path, _ = _load_trend_records()
    requested_hairstyle_text = _normalize_space(hairstyle_text)
    requested_color_text = _normalize_space(color_text)
    matches = tuple(_rank_trend_records(requested_hairstyle_text, requested_color_text)[:top_k])
    resolved_hairstyle_text = _build_resolved_hairstyle_text(requested_hairstyle_text, matches)
    recommended_color_text = _first_non_empty(match.color_text for match in matches)

    return ResolvedTrendRequest(
        trend_data_path=trend_data_path,
        requested_hairstyle_text=requested_hairstyle_text,
        requested_color_text=requested_color_text,
        resolved_hairstyle_text=resolved_hairstyle_text or requested_hairstyle_text,
        resolved_color_text=requested_color_text,
        recommended_color_text=recommended_color_text,
        prompt_hint=_build_prompt_hint(matches),
        matches=matches,
    )
