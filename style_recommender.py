"""
헤어스타일 추천 엔진 (Style Recommender)

3개 벡터 가중 결합 → ChromaDB 코사인 유사도 → Top-5 추천

┌───────────────────┬────────────────────────────┬────────┐
│ 입력 벡터          │ 구성 요소                    │ 가중치  │
├───────────────────┼────────────────────────────┼────────┤
│ 얼굴 비율 벡터     │ 비율 측정값 + 얼굴형 분류 결과 │ 40%    │
│ 황금비율 근접도     │ 황금비 편차 점수              │ 20%    │
│ user_preference   │ 길이·분위기·모발·컬러·예산     │ 40%    │
└───────────────────┴────────────────────────────┴────────┘

→ 3개 벡터를 가중 결합 → ChromaDB 코사인 유사도 비교 → Top-5 추천
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent / "data"
_HAIRSTYLES_PATH = _DATA_DIR / "trend_hairstyles.json"
_CHROMA_STYLE_DIR = _DATA_DIR / "rag" / "stores" / "chromadb_styles"

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
    trend_hairstyles.json의 스타일 1개 → 23차원 피처 벡터.
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


# ---------------------------------------------------------------------------
# ChromaDB 스타일 컬렉션
# ---------------------------------------------------------------------------
_collection_cache = None


def _get_style_collection():
    """ChromaDB 스타일 컬렉션 로드 (없으면 빌드)."""
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache

    import chromadb

    client = chromadb.PersistentClient(path=str(_CHROMA_STYLE_DIR))

    try:
        collection = client.get_collection(
            name="hairstyle_features",
        )
        if collection.count() > 0:
            _collection_cache = collection
            logger.info("Loaded existing style collection (%d items)", collection.count())
            return collection
    except Exception:
        pass

    # 컬렉션이 없거나 비어있으면 빌드
    collection = build_style_collection(client)
    _collection_cache = collection
    return collection


def build_style_collection(client=None):
    """trend_hairstyles.json → ChromaDB 컬렉션 빌드."""
    import chromadb

    if client is None:
        client = chromadb.PersistentClient(path=str(_CHROMA_STYLE_DIR))

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection("hairstyle_features")
    except Exception:
        pass

    collection = client.create_collection(
        name="hairstyle_features",
        metadata={
            "description": "Hairstyle feature vectors for recommendation",
            "hnsw:space": "cosine",
        },
    )

    styles = _load_hairstyles()
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for style in styles:
        vec = encode_style_vector(style)
        # 스타일 벡터에도 가중치 스케일 적용 (유저 벡터와 동일 공간)
        scaled = _apply_weight_scaling(vec)

        ids.append(style["id"])
        embeddings.append(scaled.tolist())
        metadatas.append({
            "style_name": style["style_name"],
            "description": style["description"],
            "face_shapes": ",".join(style.get("face_shapes", [])),
            "length": style.get("length", "medium"),
            "mood": ",".join(style.get("mood", [])),
            "hair_types": ",".join(style.get("hair_types", [])),
            "maintenance": style.get("maintenance", "medium"),
            "popularity_score": style.get("popularity_score", 0.5),
            "freshness_score": style.get("freshness_score", 0.5),
        })
        documents.append(
            f"{style['style_name']}: {style['description']} "
            f"Keywords: {', '.join(style.get('keywords', []))}"
        )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    logger.info("Built style collection with %d styles", len(styles))
    return collection


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


def _load_hairstyles() -> List[Dict[str, Any]]:
    with open(_HAIRSTYLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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

    # 5. ChromaDB 쿼리
    collection = _get_style_collection()
    results = collection.query(
        query_embeddings=[user_vec.tolist()],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "documents", "distances"],
    )

    # 6. 결과 매핑
    recommendations = []
    for rank, (sid, dist, meta, doc) in enumerate(zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
        results["documents"][0],
    )):
        # ChromaDB cosine distance → similarity (1 - distance)
        similarity = 1.0 - dist

        recommendations.append(StyleRecommendation(
            rank=rank,
            style_id=sid,
            style_name=meta.get("style_name", sid),
            score=round(similarity, 4),
            face_shapes=meta.get("face_shapes", "").split(","),
            description=meta.get("description", ""),
            metadata={
                "length": meta.get("length"),
                "mood": meta.get("mood", "").split(","),
                "hair_types": meta.get("hair_types", "").split(","),
                "maintenance": meta.get("maintenance"),
                "popularity_score": meta.get("popularity_score"),
                "freshness_score": meta.get("freshness_score"),
                "face_shape_detected": face_shape,
                "golden_ratio_score": round(g_score, 4),
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
