from __future__ import annotations

from typing import Dict, Optional, Tuple


GOLDEN_RATIO = 1.618


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def classify_face_shape(ratios: Dict[str, Optional[float]]) -> Tuple[str, Dict[str, float]]:
    c2h = ratios.get("cheekbone_to_height") or 0.72
    j2h = ratios.get("jaw_to_height") or 0.62
    t2h = ratios.get("temple_to_height") or 0.68
    j2c = ratios.get("jaw_to_cheekbone") or 0.85

    scores: Dict[str, float] = {}
    scores["oval"] = (
        _clamp(1.0 - abs(c2h - 0.72) * 5.0) * 0.4
        + _clamp(1.0 - abs(j2c - 0.82) * 4.0) * 0.3
        + _clamp(1.0 - abs(t2h - c2h) * 8.0) * 0.3
    )
    scores["round"] = (
        _clamp(1.0 - abs(c2h - 0.82) * 4.0) * 0.4
        + _clamp(1.0 - abs(j2c - 0.90) * 4.0) * 0.3
        + _clamp(c2h - 0.75) * 3.0 * 0.3
    )
    scores["square"] = (
        _clamp(1.0 - abs(j2c - 0.95) * 5.0) * 0.4
        + _clamp(1.0 - abs(t2h - c2h) * 6.0) * 0.3
        + _clamp(j2c - 0.88) * 4.0 * 0.3
    )
    scores["heart"] = (
        _clamp(1.0 - j2c) * 2.0 * 0.4
        + _clamp(t2h - j2h) * 3.0 * 0.3
        + _clamp(1.0 - abs(c2h - 0.73) * 5.0) * 0.3
    )
    scores["oblong"] = (
        _clamp(1.0 - c2h) * 2.5 * 0.4
        + _clamp(1.0 - abs(j2c - 0.83) * 4.0) * 0.3
        + _clamp(0.70 - c2h) * 5.0 * 0.3
    )

    total = sum(scores.values())
    if total > 1e-6:
        scores = {k: v / total for k, v in scores.items()}

    best = max(scores, key=scores.get)
    return best, scores


def golden_ratio_score(ratios: Dict[str, Optional[float]]) -> float:
    c2h = ratios.get("cheekbone_to_height")
    j2c = ratios.get("jaw_to_cheekbone")

    if c2h is None or c2h < 1e-6:
        return 0.5

    h_to_w = 1.0 / c2h
    dev_main = abs(h_to_w - GOLDEN_RATIO) / GOLDEN_RATIO
    score_main = _clamp(1.0 - dev_main * 2.0)

    score_sub = 0.5
    if j2c is not None and j2c > 1e-6:
        c_to_j = 1.0 / j2c
        dev_sub = abs(c_to_j - GOLDEN_RATIO) / GOLDEN_RATIO
        score_sub = _clamp(1.0 - dev_sub * 2.0)

    return score_main * 0.7 + score_sub * 0.3
