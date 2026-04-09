"""
헤어스타일 추천 엔진 (Style Recommender)

3개 벡터 가중 결합 → 인메모리 코사인 유사도 → Top-5 추천

┌───────────────────┬────────────────────────────┬────────┐
│ 입력 벡터          │ 구성 요소                    │ 가중치  │
├───────────────────┼────────────────────────────┼────────┤
│ 얼굴 비율 벡터     │ 비율 측정값 + 얼굴형 분류 결과 │ 40%    │
│ 황금비율 근접도     │ 황금비 편차 점수              │ 20%    │
│ user_preference   │ 길이·분위기·모발·컬러·예산     │ 40%    │
└───────────────────┴────────────────────────────┴────────┘

→ 3개 벡터를 가중 결합 → 로컬 스타일 카탈로그 코사인 유사도 비교 → Top-5 추천
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.trend_prompt import _load_trend_records

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STYLE_SOURCE_NAME = "llm_refined_trends"

GOLDEN_RATIO = 1.618

FACE_SHAPES = ("oval", "round", "square", "heart", "oblong")
LENGTHS = ("short", "medium", "long")
MOODS = ("natural", "trendy", "classic", "edgy", "cute")
HAIR_TYPES = ("straight", "wavy", "curly")
COLOR_TEMPS = ("warm", "cool", "neutral")
BUDGETS = ("low", "medium", "high")

# 가중치
W_FACE = 0.40
W_GOLDEN = 0.20
W_PREF = 0.40

# 벡터 차원 구성:
#   face_shape  : 5  (oval, round, square, heart, oblong)
#   golden      : 1
#   length      : 3  (short, medium, long)
#   mood        : 5  (natural, trendy, classic, edgy, cute)
#   hair_type   : 3  (straight, wavy, curly)
#   color_temp  : 3  (warm, cool, neutral)
#   budget      : 3  (low, medium, high)
# total = 23
VEC_DIM = 23

# 각 그룹의 시작 인덱스
_IDX_FACE = 0       # 0-4
_IDX_GOLDEN = 5     # 5
_IDX_LENGTH = 6     # 6-8
_IDX_MOOD = 9       # 9-13
_IDX_HAIR = 14      # 14-16
_IDX_COLOR = 17     # 17-19
_IDX_BUDGET = 20    # 20-22

# ---------------------------------------------------------------------------
# 나이 → 분위기 매핑
# ---------------------------------------------------------------------------
_AGE_MOOD_MAP: Dict[str, List[str]] = {
    "10s": ["cute", "trendy"],
    "20s": ["trendy", "edgy"],
    "30s": ["natural", "trendy"],
    "40s": ["classic", "natural"],
    "50s": ["classic", "natural"],
    "60s": ["classic"],
}

# ---------------------------------------------------------------------------
# 한글 → 영문 취향 키워드 매핑
# ---------------------------------------------------------------------------
_PREF_KO_MAP: Dict[str, Dict[str, str]] = {
    "length": {
        "숏": "short", "짧은": "short", "단발": "short", "숏컷": "short",
        "미디엄": "medium", "중간": "medium", "어깨": "medium", "쇄골": "medium",
        "롱": "long", "긴": "long", "긴머리": "long", "허리": "long",
    },
    "mood": {
        "자연스러운": "natural", "내추럴": "natural", "편한": "natural",
        "트렌디": "trendy", "유행": "trendy", "힙한": "trendy",
        "클래식": "classic", "단정한": "classic", "깔끔한": "classic", "오피스": "classic",
        "엣지": "edgy", "개성": "edgy", "파격": "edgy", "펑키": "edgy",
        "귀여운": "cute", "큐트": "cute", "사랑스러운": "cute", "러블리": "cute",
    },
    "hair_type": {
        "직모": "straight", "생머리": "straight", "스트레이트": "straight",
        "웨이브": "wavy", "물결": "wavy",
        "곱슬": "curly", "컬": "curly", "펌": "curly",
    },
    "color_temp": {
        "따뜻한": "warm", "웜톤": "warm", "브라운": "warm", "레드": "warm",
        "차가운": "cool", "쿨톤": "cool", "애쉬": "cool", "블루": "cool",
        "자연색": "neutral", "검정": "neutral", "블랙": "neutral",
    },
    "budget": {
        "저렴": "low", "저예산": "low", "부담없": "low",
        "보통": "medium", "적당": "medium",
        "고급": "high", "프리미엄": "high", "비싸도": "high",
    },
}


# ---------------------------------------------------------------------------
# 얼굴형 분류
# ---------------------------------------------------------------------------
def classify_face_shape(ratios: Dict[str, Optional[float]]) -> Tuple[str, Dict[str, float]]:
    """
    MediaPipe 비율 데이터 → 얼굴형 분류 + 각 얼굴형 확률 스코어.

    분류 기준:
    - oval:   cheekbone > jaw, 세로 길이 > 가로 (cheekbone_to_height < 0.75)
    - round:  cheekbone ≈ height, jaw ≈ cheekbone
    - square:  jaw ≈ cheekbone, temple ≈ cheekbone (각진 비율)
    - heart:  temple/cheekbone 넓고 jaw 좁음
    - oblong: 세로가 가로 대비 길음 (cheekbone_to_height < 0.65)
    """
    c2h = ratios.get("cheekbone_to_height") or 0.72
    j2h = ratios.get("jaw_to_height") or 0.62
    t2h = ratios.get("temple_to_height") or 0.68
    j2c = ratios.get("jaw_to_cheekbone") or 0.85

    scores: Dict[str, float] = {}

    # Oval: 이마~광대 넓고, 턱 자연스럽게 좁아짐, 적당히 긴 얼굴
    scores["oval"] = (
        _clamp(1.0 - abs(c2h - 0.72) * 5.0) * 0.4
        + _clamp(1.0 - abs(j2c - 0.82) * 4.0) * 0.3
        + _clamp(1.0 - abs(t2h - c2h) * 8.0) * 0.3
    )

    # Round: 가로세로 비슷, 광대 넓고, 턱도 비교적 넓음
    scores["round"] = (
        _clamp(1.0 - abs(c2h - 0.82) * 4.0) * 0.4
        + _clamp(1.0 - abs(j2c - 0.90) * 4.0) * 0.3
        + _clamp(c2h - 0.75) * 3.0 * 0.3
    )

    # Square: 턱이 광대만큼 넓고, 관자놀이도 비슷 (각진 느낌)
    scores["square"] = (
        _clamp(1.0 - abs(j2c - 0.95) * 5.0) * 0.4
        + _clamp(1.0 - abs(t2h - c2h) * 6.0) * 0.3
        + _clamp(j2c - 0.88) * 4.0 * 0.3
    )

    # Heart: 이마/관자놀이 넓고, 턱 좁음
    scores["heart"] = (
        _clamp(1.0 - j2c) * 2.0 * 0.4
        + _clamp(t2h - j2h) * 3.0 * 0.3
        + _clamp(1.0 - abs(c2h - 0.73) * 5.0) * 0.3
    )

    # Oblong: 세로가 가로 대비 길음
    scores["oblong"] = (
        _clamp(1.0 - c2h) * 2.5 * 0.4
        + _clamp(1.0 - abs(j2c - 0.83) * 4.0) * 0.3
        + _clamp(0.70 - c2h) * 5.0 * 0.3
    )

    # 정규화
    total = sum(scores.values())
    if total > 1e-6:
        scores = {k: v / total for k, v in scores.items()}

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best, scores


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 황금비율 근접도
# ---------------------------------------------------------------------------
def golden_ratio_score(ratios: Dict[str, Optional[float]]) -> float:
    """
    얼굴 비율의 황금비(1.618) 근접도를 0~1 스코어로 반환.

    주요 비교:
    - face_height / cheekbone_width 가 1.618에 가까울수록 높은 점수
    - cheekbone / jaw 비율이 1.618에 가까울수록 보너스
    """
    c2h = ratios.get("cheekbone_to_height")
    j2c = ratios.get("jaw_to_cheekbone")

    if c2h is None or c2h < 1e-6:
        return 0.5  # 데이터 없으면 중립

    # height / cheekbone_width = 1 / c2h
    h_to_w = 1.0 / c2h
    dev_main = abs(h_to_w - GOLDEN_RATIO) / GOLDEN_RATIO
    score_main = _clamp(1.0 - dev_main * 2.0)

    score_sub = 0.5
    if j2c is not None and j2c > 1e-6:
        # cheekbone / jaw 비율
        c_to_j = 1.0 / j2c
        dev_sub = abs(c_to_j - GOLDEN_RATIO) / GOLDEN_RATIO
        score_sub = _clamp(1.0 - dev_sub * 2.0)

    return score_main * 0.7 + score_sub * 0.3


# ---------------------------------------------------------------------------
# 취향 텍스트 → 구조화된 preference dict
# ---------------------------------------------------------------------------
def parse_preference_text(
    text: str,
    age: Optional[int] = None,
) -> Dict[str, Any]:
    """
    자연어 취향 텍스트를 구조화된 preference dict로 변환.

    Returns:
        {
            "length": "medium",
            "mood": ["trendy", "natural"],
            "hair_type": "wavy",
            "color_temp": "warm",
            "budget": "medium",
            "age_group": "20s",
        }
    """
    result: Dict[str, Any] = {
        "length": None,
        "mood": [],
        "hair_type": None,
        "color_temp": None,
        "budget": None,
        "age_group": None,
    }

    text_lower = text.lower().strip()

    for category, mapping in _PREF_KO_MAP.items():
        for ko_key, en_val in mapping.items():
            if ko_key in text_lower:
                if category == "mood":
                    if en_val not in result["mood"]:
                        result["mood"].append(en_val)
                else:
                    if result[category] is None:
                        result[category] = en_val

    if age is not None:
        age_group = f"{(age // 10) * 10}s"
        result["age_group"] = age_group
        if not result["mood"]:
            result["mood"] = _AGE_MOOD_MAP.get(age_group, ["natural"])

    # 기본값 채우기
    if result["length"] is None:
        result["length"] = "medium"
    if not result["mood"]:
        result["mood"] = ["natural"]
    if result["hair_type"] is None:
        result["hair_type"] = "straight"
    if result["color_temp"] is None:
        result["color_temp"] = "neutral"
    if result["budget"] is None:
        result["budget"] = "medium"

    return result


# ---------------------------------------------------------------------------
# llm_refined_trends -> recommendation style normalization
# ---------------------------------------------------------------------------
_STYLE_SPLIT_RE = re.compile(r"[,/;|]+")


def _normalize_blob(*values: str) -> str:
    return " ".join(str(value or "").strip().lower() for value in values if str(value or "").strip())


def _contains_any(blob: str, keywords: Sequence[str]) -> bool:
    return any(keyword in blob for keyword in keywords)


def _split_style_terms(text: str) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for part in _STYLE_SPLIT_RE.split(str(text or "")):
        normalized = part.strip()
        if normalized and normalized not in seen:
            terms.append(normalized)
            seen.add(normalized)
    return terms


def _slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z가-힣]+", "-", str(text or "").strip().lower()).strip("-")
    return slug or fallback


def _dedupe_preserve(values: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _infer_length(hairstyle_text: str, description: str, trend_name: str) -> str:
    blob = _normalize_blob(hairstyle_text, description, trend_name)
    if _contains_any(blob, ("long hair", "ponytail", "braids", "micro braids", "goddess braids", "twists", "updo", "bun", "chignon", "french twist")):
        return "long"
    if _contains_any(blob, ("buzz cut", "fade buzz cut", "burr buzz cut", "pixie", "crop", "crew cut", "french bob", "bob")):
        return "short"
    if _contains_any(blob, ("lob", "shoulder-length", "shoulder length", "mid-length", "mid length", "mullet", "wolf cut", "shag", "layered cut", "leaf cut", "medium")):
        return "medium"
    return "medium"


def _infer_hair_types(hairstyle_text: str, description: str, trend_name: str) -> List[str]:
    blob = _normalize_blob(hairstyle_text, description, trend_name)
    hair_types: List[str] = []
    if _contains_any(blob, ("curly", "curls", "curl", "perm", "afro", "coily", "coil", "곱슬", "컬")):
        hair_types.append("curly")
    if _contains_any(blob, ("wavy", "waves", "wave", "웨이브")):
        hair_types.append("wavy")
    if not hair_types:
        hair_types.append("straight")
    return hair_types


def _infer_mood(hairstyle_text: str, description: str, trend_name: str) -> List[str]:
    blob = _normalize_blob(hairstyle_text, description, trend_name)
    moods: List[str] = []
    if _contains_any(blob, ("natural", "soft", "manageable", "casual", "low-key", "effortless", "loose", "air dry", "easy", "자연", "편안", "부드")):
        moods.append("natural")
    if _contains_any(blob, ("classic", "tailored", "wall street", "sleek", "slicked-back", "french twist", "polished", "glamour", "단정", "클래식", "우아")):
        moods.append("classic")
    if _contains_any(blob, ("mullet", "wolf", "buzz", "fade", "shag", "pixie", "edgy", "punk", "dramatic", "art school", "개성", "엣지", "강렬")):
        moods.append("edgy")
    if _contains_any(blob, ("cute", "playful", "girly", "발랄", "귀여")):
        moods.append("cute")
    if _contains_any(blob, ("trend", "trendy", "modern", "fashion", "celebrity", "유행", "트렌드", "모던", "세련")) or not moods:
        moods.append("trendy")
    return _dedupe_preserve(moods)


def _infer_color_temp(color_text: str, description: str) -> str:
    blob = _normalize_blob(color_text, description)
    if _contains_any(blob, ("ash", "silver", "gray", "grey", "platinum", "icy", "cool")):
        return "cool"
    if _contains_any(blob, ("auburn", "copper", "brown", "blonde", "gold", "golden", "honey", "beige", "red", "chocolate", "warm")):
        return "warm"
    return "neutral"


def _infer_maintenance(hairstyle_text: str, description: str) -> str:
    blob = _normalize_blob(hairstyle_text, description)
    if _contains_any(blob, ("buzz cut", "burr buzz cut")):
        return "low"
    if _contains_any(blob, ("braids", "micro braids", "goddess braids", "twists", "updo", "bun", "perm", "afro")):
        return "high"
    return "medium"


def _infer_face_shapes(length: str, hairstyle_text: str, description: str) -> List[str]:
    blob = _normalize_blob(hairstyle_text, description)
    if _contains_any(blob, ("buzz", "fade", "slicked-back", "ponytail", "updo")):
        return ["oval", "square", "oblong"]
    if _contains_any(blob, ("bob", "blunt", "french bob", "lob")):
        return ["oval", "heart", "oblong"]
    if _contains_any(blob, ("curtain bangs", "face-framing", "layers", "waves")):
        return ["oval", "round", "square", "heart"]
    if _contains_any(blob, ("mullet", "wolf cut", "shag")):
        return ["oval", "heart", "oblong"]
    if length == "long":
        return ["oval", "round", "heart", "oblong"]
    return list(FACE_SHAPES)


def _build_keywords(trend_name: str, hairstyle_text: str, color_text: str, description: str) -> List[str]:
    keywords = _split_style_terms(hairstyle_text)
    token_blob = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", _normalize_blob(trend_name, color_text, description))
    for token in token_blob.split():
        if len(token) >= 3:
            keywords.append(token)
    return _dedupe_preserve(keywords)[:18]


def _estimate_freshness(year_text: str) -> float:
    years = [int(value) for value in re.findall(r"\d{4}", str(year_text or ""))]
    if not years:
        return 0.6
    newest = max(years)
    return _clamp(0.45 + max(0, min(newest, 2026) - 2022) * 0.12, lo=0.45, hi=0.93)


def _estimate_popularity(keyword_count: int) -> float:
    return _clamp(0.45 + min(keyword_count, 6) * 0.06, lo=0.45, hi=0.81)


def _build_style_from_trend_record(record: Dict[str, str], index: int) -> Dict[str, Any]:
    trend_name = str(record.get("trend_name", "")).strip()
    hairstyle_text = str(record.get("hairstyle_text", "")).strip() or trend_name
    color_text = str(record.get("color_text", "")).strip()
    description = str(record.get("description", "")).strip() or hairstyle_text
    source = str(record.get("source", "")).strip()
    year = str(record.get("year", "")).strip()

    length = _infer_length(hairstyle_text, description, trend_name)
    hair_types = _infer_hair_types(hairstyle_text, description, trend_name)
    keywords = _build_keywords(trend_name, hairstyle_text, color_text, description)

    prompt_parts = [hairstyle_text]
    if color_text:
        prompt_parts.append(f"{color_text} hair")
    if description:
        prompt_parts.append(description)

    return {
        "id": _slugify(f"{trend_name or hairstyle_text}-{index + 1}", f"trend-{index + 1}"),
        "style_name": hairstyle_text,
        "description": description,
        "face_shapes": _infer_face_shapes(length, hairstyle_text, description),
        "length": length,
        "mood": _infer_mood(hairstyle_text, description, trend_name),
        "hair_types": hair_types,
        "color_temp": _infer_color_temp(color_text, description),
        "maintenance": _infer_maintenance(hairstyle_text, description),
        "popularity_score": _estimate_popularity(len(keywords)),
        "freshness_score": _estimate_freshness(year),
        "sd_positive": ", ".join(part for part in prompt_parts if part),
        "sd_negative": "",
        "sd_guidance": 8.5,
        "keywords": keywords,
        "trend_name": trend_name,
        "hairstyle_text": hairstyle_text,
        "color_text": color_text,
        "source": source,
        "year": year,
        "source_dataset": _STYLE_SOURCE_NAME,
    }


# ---------------------------------------------------------------------------
# 벡터 인코딩
# ---------------------------------------------------------------------------
def _one_hot(value: str, categories: Sequence[str]) -> np.ndarray:
    """단일 값 → one-hot."""
    vec = np.zeros(len(categories), dtype=np.float32)
    if value in categories:
        vec[categories.index(value)] = 1.0
    return vec


def _multi_hot(values: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    """복수 값 → multi-hot (정규화)."""
    vec = np.zeros(len(categories), dtype=np.float32)
    for v in values:
        if v in categories:
            vec[categories.index(v)] = 1.0
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec


def encode_style_vector(style: Dict[str, Any]) -> np.ndarray:
    """
    llm_refined_trends 기반 추천 스타일 1개 → 23차원 피처 벡터.
    """
    vec = np.zeros(VEC_DIM, dtype=np.float32)

    # Face shape compatibility (multi-hot)
    face_vec = _multi_hot(style.get("face_shapes", []), FACE_SHAPES)
    vec[_IDX_FACE:_IDX_FACE + len(FACE_SHAPES)] = face_vec

    # Golden ratio: 스타일은 중립값 0.5 (모든 스타일에 동일)
    vec[_IDX_GOLDEN] = 0.5

    # Length
    vec[_IDX_LENGTH:_IDX_LENGTH + len(LENGTHS)] = _one_hot(
        style.get("length", "medium"), LENGTHS
    )

    # Mood (multi-hot)
    vec[_IDX_MOOD:_IDX_MOOD + len(MOODS)] = _multi_hot(
        style.get("mood", []), MOODS
    )

    # Hair type (multi-hot)
    vec[_IDX_HAIR:_IDX_HAIR + len(HAIR_TYPES)] = _multi_hot(
        style.get("hair_types", []), HAIR_TYPES
    )

    # Color temp
    vec[_IDX_COLOR:_IDX_COLOR + len(COLOR_TEMPS)] = _one_hot(
        style.get("color_temp", "neutral"), COLOR_TEMPS
    )

    # Budget ← maintenance 매핑
    vec[_IDX_BUDGET:_IDX_BUDGET + len(BUDGETS)] = _one_hot(
        style.get("maintenance", "medium"), BUDGETS
    )

    return vec


def encode_user_vector(
    face_scores: Dict[str, float],
    golden_score: float,
    preference: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    유저 3개 벡터를 가중 결합하여 23차원 쿼리 벡터 생성.

    Args:
        weights: {"face": 0.4, "golden": 0.2, "preference": 0.4}
                 서버에서 동적으로 가중치 조절 가능. 미전달 시 기본값 사용.
                 합이 1.0이 아니면 정규화됨.
    """
    # 가중치 결정
    w_face = W_FACE
    w_golden = W_GOLDEN
    w_pref = W_PREF
    if weights:
        w_face = float(weights.get("face", W_FACE))
        w_golden = float(weights.get("golden", W_GOLDEN))
        w_pref = float(weights.get("preference", W_PREF))
        # 정규화: 합이 1.0이 되도록 (모두 0이면 기본값 복원)
        total = w_face + w_golden + w_pref
        if total > 0:
            w_face /= total
            w_golden /= total
            w_pref /= total
        else:
            w_face, w_golden, w_pref = W_FACE, W_GOLDEN, W_PREF
            logger.warning("All weights are 0, falling back to defaults")
        logger.info("Custom weights: face=%.2f, golden=%.2f, pref=%.2f", w_face, w_golden, w_pref)

    vec = np.zeros(VEC_DIM, dtype=np.float32)

    # --- Face shape ---
    face_vec = np.array(
        [face_scores.get(s, 0.0) for s in FACE_SHAPES], dtype=np.float32
    )
    norm = np.linalg.norm(face_vec)
    if norm > 1e-6:
        face_vec = face_vec / norm
    vec[_IDX_FACE:_IDX_FACE + len(FACE_SHAPES)] = face_vec * math.sqrt(w_face)

    # --- Golden ratio ---
    vec[_IDX_GOLDEN] = golden_score * math.sqrt(w_golden)

    # --- Preferences ---
    pref_scale = math.sqrt(w_pref)

    # Length
    length_vec = _one_hot(preference.get("length", "medium"), LENGTHS)
    vec[_IDX_LENGTH:_IDX_LENGTH + len(LENGTHS)] = length_vec * pref_scale

    # Mood
    moods = preference.get("mood", ["natural"])
    if isinstance(moods, str):
        moods = [moods]
    mood_vec = _multi_hot(moods, MOODS)
    vec[_IDX_MOOD:_IDX_MOOD + len(MOODS)] = mood_vec * pref_scale

    # Hair type
    hair_vec = _one_hot(preference.get("hair_type", "straight"), HAIR_TYPES)
    vec[_IDX_HAIR:_IDX_HAIR + len(HAIR_TYPES)] = hair_vec * pref_scale

    # Color temp
    color_vec = _one_hot(preference.get("color_temp", "neutral"), COLOR_TEMPS)
    vec[_IDX_COLOR:_IDX_COLOR + len(COLOR_TEMPS)] = color_vec * pref_scale

    # Budget
    budget_vec = _one_hot(preference.get("budget", "medium"), BUDGETS)
    vec[_IDX_BUDGET:_IDX_BUDGET + len(BUDGETS)] = budget_vec * pref_scale

    return vec


def _apply_weight_scaling(vec: np.ndarray) -> np.ndarray:
    """스타일 벡터에 가중치 스케일링 적용 (유저 벡터와 동일 공간)."""
    scaled = vec.copy()
    scaled[_IDX_FACE:_IDX_FACE + len(FACE_SHAPES)] *= math.sqrt(W_FACE)
    scaled[_IDX_GOLDEN] *= math.sqrt(W_GOLDEN)
    pref_scale = math.sqrt(W_PREF)
    scaled[_IDX_LENGTH:_IDX_LENGTH + len(LENGTHS)] *= pref_scale
    scaled[_IDX_MOOD:_IDX_MOOD + len(MOODS)] *= pref_scale
    scaled[_IDX_HAIR:_IDX_HAIR + len(HAIR_TYPES)] *= pref_scale
    scaled[_IDX_COLOR:_IDX_COLOR + len(COLOR_TEMPS)] *= pref_scale
    scaled[_IDX_BUDGET:_IDX_BUDGET + len(BUDGETS)] *= pref_scale
    return scaled


@lru_cache(maxsize=1)
def _load_hairstyles() -> List[Dict[str, Any]]:
    trend_data_path, records = _load_trend_records()
    if not records:
        raise RuntimeError("llm_refined_trends recommendation data could not be loaded.")

    styles = [
        _build_style_from_trend_record(record, index)
        for index, record in enumerate(records)
    ]
    logger.info(
        "Loaded %d recommendation styles from %s",
        len(styles),
        trend_data_path or _STYLE_SOURCE_NAME,
    )
    return styles


def list_available_hairstyles() -> List[str]:
    """현재 llm_refined_trends에서 생성에 활용 가능한 hairstyle_text 목록."""
    available: List[str] = []
    seen: set[str] = set()
    for style in _load_hairstyles():
        hairstyle_text = str(style.get("hairstyle_text") or style.get("style_name") or "").strip()
        if hairstyle_text and hairstyle_text not in seen:
            available.append(hairstyle_text)
            seen.add(hairstyle_text)
    return available


def _style_metadata(style: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "style_name": style["style_name"],
        "description": style["description"],
        "face_shapes": ",".join(style.get("face_shapes", [])),
        "length": style.get("length", "medium"),
        "mood": ",".join(style.get("mood", [])),
        "hair_types": ",".join(style.get("hair_types", [])),
        "maintenance": style.get("maintenance", "medium"),
        "popularity_score": style.get("popularity_score", 0.5),
        "freshness_score": style.get("freshness_score", 0.5),
        "sd_positive": style.get("sd_positive", ""),
        "sd_negative": style.get("sd_negative", ""),
        "sd_guidance": style.get("sd_guidance", 8.5),
        "trend_name": style.get("trend_name", ""),
        "hairstyle_text": style.get("hairstyle_text", style["style_name"]),
        "color_text": style.get("color_text", ""),
        "source": style.get("source", ""),
        "year": style.get("year", ""),
        "source_dataset": style.get("source_dataset", _STYLE_SOURCE_NAME),
    }


def _style_document(style: Dict[str, Any]) -> str:
    return (
        f"{style['style_name']}: {style['description']} "
        f"Keywords: {', '.join(style.get('keywords', []))}"
    )


def _query_styles_in_memory(user_vec: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict[str, Any], str]]:
    user_norm = float(np.linalg.norm(user_vec))
    scored: List[Tuple[str, float, Dict[str, Any], str]] = []
    for style in _load_hairstyles():
        style_vec = _apply_weight_scaling(encode_style_vector(style))
        denom = user_norm * float(np.linalg.norm(style_vec))
        similarity = float(np.dot(user_vec, style_vec) / denom) if denom > 1e-6 else 0.0
        scored.append((style["id"], similarity, _style_metadata(style), _style_document(style)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Top-5 추천
# ---------------------------------------------------------------------------
@dataclass
class StyleRecommendation:
    """추천 결과 1건."""
    rank: int
    style_id: str
    style_name: str
    score: float  # 코사인 유사도 (0~1, 높을수록 좋음)
    face_shapes: List[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def recommend_top_k(
    face_ratios: Dict[str, Optional[float]],
    preference: Optional[Dict[str, Any]] = None,
    preference_text: Optional[str] = None,
    age: Optional[int] = None,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> List[StyleRecommendation]:
    """
    얼굴 비율 + 취향 → Top-K 헤어스타일 추천.

    Args:
        face_ratios: MediaPipe _build_face_mesh_analysis()의 ratios 딕셔너리
            {"cheekbone_to_height", "jaw_to_height", "temple_to_height", "jaw_to_cheekbone"}
        preference: 구조화된 취향 (length, mood, hair_type, color_temp, budget)
        preference_text: 자연어 취향 텍스트 (preference 없을 때 파싱)
        age: 나이 (preference_text 파싱 시 분위기 추론에 사용)
        top_k: 추천 개수 (기본 5)
        weights: {"face": 0.4, "golden": 0.2, "preference": 0.4}
                 Django 서버에서 동적으로 가중치 조절 가능.

    Returns:
        List[StyleRecommendation] 상위 K개
    """
    # 1. 얼굴형 분류
    face_shape, face_scores = classify_face_shape(face_ratios)
    logger.info("Face shape: %s (scores: %s)", face_shape, face_scores)

    # 2. 황금비율 근접도
    g_score = golden_ratio_score(face_ratios)
    logger.info("Golden ratio score: %.3f", g_score)

    # 3. 취향 벡터 준비
    if preference is None:
        if preference_text:
            preference = parse_preference_text(preference_text, age=age)
        else:
            preference = parse_preference_text("", age=age)

    # 4. 유저 벡터 인코딩 (가중치 적용)
    user_vec = encode_user_vector(face_scores, g_score, preference, weights=weights)

    # 5. 로컬 스타일 카탈로그 코사인 유사도 랭킹
    ranked_rows = _query_styles_in_memory(user_vec, top_k)

    # 6. 결과 매핑
    recommendations = []
    for rank, (sid, similarity, meta, doc) in enumerate(ranked_rows):
        recommendations.append(StyleRecommendation(
            rank=rank,
            style_id=sid,
            style_name=meta.get("style_name", sid),
            score=round(similarity, 4),
            face_shapes=[shape for shape in meta.get("face_shapes", "").split(",") if shape],
            description=meta.get("description", ""),
            metadata={
                "length": meta.get("length"),
                "mood": [m for m in meta.get("mood", "").split(",") if m],
                "hair_types": [h for h in meta.get("hair_types", "").split(",") if h],
                "maintenance": meta.get("maintenance"),
                "popularity_score": meta.get("popularity_score"),
                "freshness_score": meta.get("freshness_score"),
                "face_shape_detected": face_shape,
                "golden_ratio_score": round(g_score, 4),
                "sd_positive": meta.get("sd_positive", ""),
                "sd_negative": meta.get("sd_negative", ""),
                "sd_guidance": meta.get("sd_guidance", 8.5),
                "trend_name": meta.get("trend_name", ""),
                "hairstyle_text": meta.get("hairstyle_text", meta.get("style_name", sid)),
                "color_text": meta.get("color_text", ""),
                "source": meta.get("source", ""),
                "year": meta.get("year", ""),
            },
        ))

    return recommendations


def recommend_to_dict(recommendations: List[StyleRecommendation]) -> List[Dict[str, Any]]:
    """추천 결과를 JSON-serializable dict 리스트로 변환."""
    return [
        {
            "rank": r.rank,
            "style_id": r.style_id,
            "style_name": r.style_name,
            "score": r.score,
            "face_shapes": r.face_shapes,
            "description": r.description,
            **r.metadata,
        }
        for r in recommendations
    ]


# ---------------------------------------------------------------------------
# CLI 테스트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트: 가상의 얼굴 비율 (타원형에 가까운)
    test_ratios = {
        "cheekbone_to_height": 0.72,
        "jaw_to_height": 0.60,
        "temple_to_height": 0.70,
        "jaw_to_cheekbone": 0.83,
    }

    print("=== 구조화된 취향 테스트 ===")
    results = recommend_top_k(
        face_ratios=test_ratios,
        preference={
            "length": "medium",
            "mood": ["trendy", "natural"],
            "hair_type": "wavy",
            "color_temp": "warm",
            "budget": "medium",
        },
    )
    for r in results:
        print(f"  #{r.rank} {r.style_name} (score={r.score:.4f}) faces={r.face_shapes}")

    print("\n=== 자연어 취향 테스트 ===")
    results = recommend_top_k(
        face_ratios=test_ratios,
        preference_text="자연스러운 웨이브 미디엄 길이, 따뜻한 톤",
        age=28,
    )
    for r in results:
        print(f"  #{r.rank} {r.style_name} (score={r.score:.4f}) faces={r.face_shapes}")

    print("\n=== 텍스트 없이 나이만 ===")
    results = recommend_top_k(
        face_ratios=test_ratios,
        age=45,
    )
    for r in results:
        print(f"  #{r.rank} {r.style_name} (score={r.score:.4f}) faces={r.face_shapes}")
