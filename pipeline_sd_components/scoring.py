"""
MirrAI SD Inpainting — CLIP 점수 계산 & top-k 랭킹
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
현재 CLIP 랭킹은 비활성화(use_clip_ranking=False)이며, 향후 확장용 stub.
결과는 seed 순서(rank)로 정렬됨.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _compute_clip_score(
    self,
    image_rgb: np.ndarray,
    prompt: str,
) -> float:
    """
    단일 이미지에 대한 CLIP 유사도 점수 계산.
    현재는 stub — use_clip_ranking=True 일 때 구현 예정.
    """
    return 0.0


def _rank_results_by_clip(
    self,
    results: List[Any],
    prompt: str,
) -> List[Any]:
    """
    CLIP 점수 기준으로 결과를 재정렬.
    use_clip_ranking=False(기본)이면 입력 순서(rank 순)를 그대로 반환.
    """
    if not self.config.use_clip_ranking:
        return results

    logger.info("[Scoring] CLIP 랭킹 계산 중...")
    for result in results:
        img_rgb = result.image_pil
        if img_rgb is not None:
            import numpy as np
            img_arr = np.array(img_rgb)
            result.clip_score = self._compute_clip_score(img_arr, prompt)

    results_sorted = sorted(results, key=lambda r: r.clip_score, reverse=True)
    for i, r in enumerate(results_sorted):
        r.rank = i
    return results_sorted


def bind_scoring_methods_to_pipeline(cls) -> None:
    """CLIP 점수/랭킹 메서드를 MirrAISDPipeline에 바인딩."""
    cls._compute_clip_score = _compute_clip_score
    cls._rank_results_by_clip = _rank_results_by_clip
