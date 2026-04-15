"""
MirrAI SD Inpainting — 마스크 빌더 (_build_*_mask 계열)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
postprocess.py에서 분리된 _build_*_mask() 계열 함수 전체.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import (
    CLOTH_CLASS_IDX,
    FACE_CLASS_IDXS,
    HAIR_CLASS_IDX,
    SD_SIZE,
    _SHORT_HAIR_KEYWORDS,
    _MEDIUM_HAIR_KEYWORDS,
)

logger = logging.getLogger(__name__)

def _build_post_cloth_refine_mask(
    self,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
    artifact_cleanup_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = removal_mask.shape[:2]
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255
    overlap_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
    if int((overlap_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    overlap_u8 = cv2.dilate(
        overlap_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 15) if hair_length == "short" else (11, 19),
        ),
        iterations=1,
    )

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * (0.05 if hair_length == "short" else 0.05)))
    bottom = min(H, int(cutoff_y + face_h * (0.54 if hair_length == "short" else 1.52)))
    left = max(0, int(x1 - face_w * (0.72 if hair_length == "short" else 1.18)))
    right = min(W, int(x2 + face_w * (0.72 if hair_length == "short" else 1.18)))
    if top < bottom and left < right:
        corridor_u8[top:bottom, left:right] = 255

    mask_u8 = cv2.bitwise_and(overlap_u8, corridor_u8)
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 11) if hair_length == "short" else (9, 13),
        ),
    )

    artifact_bonus_u8 = np.zeros((H, W), dtype=np.uint8)
    if artifact_cleanup_mask is not None and artifact_cleanup_mask.shape == (H, W):
        artifact_bonus_u8 = (
            (np.clip(artifact_cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255
        )
        artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, corridor_u8)
        artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, cloth_u8)
        artifact_bonus_u8 = cv2.dilate(
            artifact_bonus_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
            iterations=1,
        )

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(protect_u8))
        artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, cv2.bitwise_not(protect_u8))

    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
            iterations=1,
        )
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(final_hair_u8))
        artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, cv2.bitwise_not(final_hair_u8))

    if hair_length == "short":
        short_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        gate_half_w = max(14, int(face_w * 0.22))
        gate_left = max(0, cx - gate_half_w)
        gate_right = min(W, cx + gate_half_w)
        gate_top = max(top, int(cutoff_y + face_h * 0.04))
        gate_bottom = min(H, int(cutoff_y + face_h * 0.76))
        if gate_top < gate_bottom and gate_left < gate_right:
            short_gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255
        if int((short_gate_u8 > 0).sum()) > 0:
            mask_u8 = cv2.bitwise_and(mask_u8, short_gate_u8)
            artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, short_gate_u8)

        max_bottom = min(H, int(cutoff_y + face_h * 0.74))
        if max_bottom < H:
            mask_u8[max_bottom:, :] = 0
            artifact_bonus_u8[max_bottom:, :] = 0

        max_component_area = max(96, int(face_w * face_h * 0.08))
        max_component_width = max(24, int(face_w * 0.34))
        min_component_height = max(12, int(face_h * 0.10))
        center_allow = max(14, int(face_w * 0.18))
        filtered_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            comp_u8 = (labels == idx).astype(np.uint8) * 255
            comp_cx = x + (w * 0.5)
            artifact_overlap = int((cv2.bitwise_and(comp_u8, artifact_bonus_u8) > 0).sum())
            if area < 14 or area > max_component_area:
                continue
            if w > max_component_width or h < min_component_height:
                continue
            if (y + h) > max_bottom:
                continue
            if abs(comp_cx - cx) > center_allow:
                continue
            if artifact_overlap < max(4, int(area * 0.04)):
                continue
            if w > max(20, int(face_w * 0.26)) and h < max(20, int(face_h * 0.22)):
                continue
            filtered_u8 = cv2.bitwise_or(filtered_u8, comp_u8)
        mask_u8 = filtered_u8

    if int((artifact_bonus_u8 > 0).sum()) > 0:
        if hair_length == "short":
            artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, mask_u8)
        mask_u8 = cv2.bitwise_or(mask_u8, artifact_bonus_u8)
        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        )
        if hair_length == "short":
            max_total_px = max(72, int(face_w * face_h * 0.05))
            current_px = int((mask_u8 > 0).sum())
            if current_px > max_total_px:
                shrink_u8 = np.zeros((H, W), dtype=np.uint8)
                shrink_half_w = max(12, int(face_w * 0.18))
                shrink_left = max(0, cx - shrink_half_w)
                shrink_right = min(W, cx + shrink_half_w)
                shrink_top = max(top, int(cutoff_y + face_h * 0.06))
                shrink_bottom = min(H, int(cutoff_y + face_h * 0.72))
                if shrink_top < shrink_bottom and shrink_left < shrink_right:
                    shrink_u8[shrink_top:shrink_bottom, shrink_left:shrink_right] = 255
                mask_u8 = cv2.bitwise_and(mask_u8, shrink_u8)

    if int((mask_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)
    return (mask_u8 > 0).astype(np.float32)

def _build_generation_protect_mask(
    self,
    protect_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    hair_length: str,
    subject_gender: Optional[str] = None,
) -> np.ndarray:
    """
    SD 생성/합성에 사용할 얼굴 보호 마스크.
    short/medium 헤어에서는 목까지 네모나게 막히면 bob 라인이 끊겨 보여서,
    턱 아래는 빠르게 감쇠시키고 중앙 목 부분만 좁게 남긴다.
    """
    subject_profile = self._resolve_subject_pipeline_profile(subject_gender)
    mask = np.clip(protect_mask.astype(np.float32), 0.0, 1.0).copy()
    if hair_length not in ("short", "medium"):
        return mask

    H, W = mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    fade_start = max(0, int(y2 - face_h * 0.04))
    fade_len = max(10, int(face_h * (0.22 if hair_length == "short" else 0.28)))
    fade_end = min(H, fade_start + fade_len)
    if fade_end > fade_start:
        ramp = np.linspace(1.0, 0.0, fade_end - fade_start, dtype=np.float32)
        mask[fade_start:fade_end, :] *= ramp[:, np.newaxis]
    if fade_end < H:
        mask[fade_end:, :] = 0.0

    neck_half = max(18, int(face_w * (0.32 if hair_length == "short" else 0.38)))
    neck_x1 = max(0, cx - neck_half)
    neck_x2 = min(W, cx + neck_half)
    neck_y1 = max(0, int(y2 - face_h * 0.02))
    neck_y2 = min(H, int(y2 + face_h * (0.18 if hair_length == "short" else 0.26)))
    if neck_x1 < neck_x2 and neck_y1 < neck_y2:
        neck_guard_u8 = np.zeros((H, W), dtype=np.uint8)
        guard_center = (
            cx,
            int(y2 + face_h * (0.05 if hair_length == "short" else 0.08)),
        )
        guard_axes = (
            max(10, int(face_w * (0.18 if hair_length == "short" else 0.24))),
            max(8, int(face_h * (0.09 if hair_length == "short" else 0.13))),
        )
        cv2.ellipse(neck_guard_u8, guard_center, guard_axes, 0, 0, 360, 255, -1)
        neck_guard = cv2.GaussianBlur(
            neck_guard_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.5,
            sigmaY=4.5,
        )
        neck_guard[:neck_y1, :] = 0.0
        if neck_y2 > neck_y1:
            ramp = np.linspace(1.0, 0.0, neck_y2 - neck_y1, dtype=np.float32)
            neck_guard[neck_y1:neck_y2, :] *= ramp[:, np.newaxis]
        if neck_y2 < H:
            neck_guard[neck_y2:, :] = 0.0
        mask = np.maximum(
            mask,
            neck_guard
            * (0.58 if hair_length == "short" else 0.50)
            * float(subject_profile.generation_protect_neck_guard_scale),
        )

    side_release_strength = float(subject_profile.generation_protect_side_release_strength)
    if side_release_strength > 0.0:
        release_u8 = np.zeros((H, W), dtype=np.uint8)
        release_y = int(y1 + face_h * (0.62 if hair_length == "short" else 0.64))
        release_axes = (
            max(8, int(face_w * (0.12 if hair_length == "short" else 0.14))),
            max(14, int(face_h * (0.22 if hair_length == "short" else 0.26))),
        )
        left_center = (
            max(0, min(W - 1, int(x1 + face_w * 0.06))),
            max(0, min(H - 1, release_y)),
        )
        right_center = (
            max(0, min(W - 1, int(x2 - face_w * 0.06))),
            max(0, min(H - 1, release_y)),
        )
        cv2.ellipse(release_u8, left_center, release_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(release_u8, right_center, release_axes, 0, 0, 360, 255, -1)
        release_mask = cv2.GaussianBlur(
            release_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.8,
            sigmaY=5.6,
        )
        band_top = max(0, int(y1 + face_h * 0.14))
        band_bottom = min(H, int(y2 + face_h * (0.22 if hair_length == "short" else 0.28)))
        if band_top > 0:
            release_mask[:band_top, :] = 0.0
        if band_bottom < H:
            release_mask[band_bottom:, :] = 0.0
        mask = np.clip(mask - release_mask * side_release_strength, 0.0, 1.0)

    if hair_length == "short":
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, erode_k, iterations=1)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)

def _build_removal_protect_mask(
    self,
    protect_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    hair_length: str,
    subject_gender: Optional[str] = None,
) -> np.ndarray:
    """
    긴 머리 제거(pre-clean) 단계에서 사용할 얼굴 보호 마스크.
    생성 단계보다 목 중앙 보호를 훨씬 약하게 두어, 목 앞쪽으로 내려온 머리 가닥은
    제거 대상으로 남기고 얼굴/턱 주변만 보수적으로 보호한다.
    """
    subject_profile = self._resolve_subject_pipeline_profile(subject_gender)
    mask = np.clip(protect_mask.astype(np.float32), 0.0, 1.0).copy()
    if hair_length not in ("short", "medium"):
        return mask

    H, W = mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    face_core_u8 = np.zeros((H, W), dtype=np.uint8)
    face_core_center = (
        cx,
        int(y1 + face_h * (0.44 if hair_length == "short" else 0.47)),
    )
    face_core_axes = (
        max(16, int(face_w * (0.44 if hair_length == "short" else 0.48))),
        max(18, int(face_h * (0.56 if hair_length == "short" else 0.60))),
    )
    cv2.ellipse(face_core_u8, face_core_center, face_core_axes, 0, 0, 360, 255, -1)
    face_core = cv2.GaussianBlur(
        face_core_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.0,
        sigmaY=4.0,
    )
    mask = np.minimum(
        mask,
        np.clip(face_core * (1.15 * float(subject_profile.removal_protect_face_scale)), 0.0, 1.0),
    )

    fade_start = max(0, int(y2 - face_h * 0.10))
    fade_end = min(H, int(y2 + face_h * (0.005 if hair_length == "short" else 0.02)))
    if fade_end > fade_start:
        ramp = np.linspace(1.0, 0.0, fade_end - fade_start, dtype=np.float32)
        mask[fade_start:fade_end, :] *= ramp[:, np.newaxis]
    cutoff_y = min(H, int(y2 + face_h * (0.015 if hair_length == "short" else 0.03)))
    if cutoff_y < H:
        mask[cutoff_y:, :] = 0.0

    neck_cut_u8 = np.zeros((H, W), dtype=np.uint8)
    neck_cut_center = (
        cx,
        int(y2 + face_h * (0.04 if hair_length == "short" else 0.06)),
    )
    neck_cut_axes = (
        max(12, int(face_w * (0.18 if hair_length == "short" else 0.22))),
        max(10, int(face_h * (0.12 if hair_length == "short" else 0.16))),
    )
    cv2.ellipse(neck_cut_u8, neck_cut_center, neck_cut_axes, 0, 0, 360, 255, -1)
    neck_cut = cv2.GaussianBlur(
        neck_cut_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.6,
        sigmaY=3.6,
    )
    mask = np.clip(
        mask - neck_cut * float(subject_profile.removal_protect_neck_cut_scale),
        0.0,
        1.0,
    )

    side_release_strength = float(subject_profile.removal_protect_side_release_strength)
    if side_release_strength > 0.0:
        release_u8 = np.zeros((H, W), dtype=np.uint8)
        release_y = int(y1 + face_h * (0.64 if hair_length == "short" else 0.66))
        release_axes = (
            max(8, int(face_w * (0.13 if hair_length == "short" else 0.15))),
            max(15, int(face_h * (0.24 if hair_length == "short" else 0.28))),
        )
        left_center = (
            max(0, min(W - 1, int(x1 + face_w * 0.05))),
            max(0, min(H - 1, release_y)),
        )
        right_center = (
            max(0, min(W - 1, int(x2 - face_w * 0.05))),
            max(0, min(H - 1, release_y)),
        )
        cv2.ellipse(release_u8, left_center, release_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(release_u8, right_center, release_axes, 0, 0, 360, 255, -1)
        release_mask = cv2.GaussianBlur(
            release_u8.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=4.6,
            sigmaY=5.4,
        )
        band_top = max(0, int(y1 + face_h * 0.16))
        band_bottom = min(H, int(y2 + face_h * (0.24 if hair_length == "short" else 0.30)))
        if band_top > 0:
            release_mask[:band_top, :] = 0.0
        if band_bottom < H:
            release_mask[band_bottom:, :] = 0.0
        mask = np.clip(mask - release_mask * side_release_strength, 0.0, 1.0)

    erode_k = (5, 5) if hair_length == "short" else (7, 7)
    mask = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erode_k),
        iterations=1,
    )

    return np.clip(mask, 0.0, 1.0).astype(np.float32)

def _build_neckline_preserve_mask(
    self,
    face_bbox: Tuple[int, int, int, int],
    cloth_mask: Optional[np.ndarray],
    hair_length: str,
) -> np.ndarray:
    """
    목 중앙과 상의 neckline을 보존하기 위한 소프트 마스크.
    short hair에서 턱 아래 피부/옷 경계가 네모나게 끊기는 현상을 줄인다.
    """
    if cloth_mask is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = cloth_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    mask = np.zeros((H, W), dtype=np.uint8)
    neck_half = max(16, int(face_w * (0.19 if hair_length == "short" else 0.25)))
    x_left = max(0, cx - neck_half)
    x_right = min(W, cx + neck_half)
    y_top = max(0, int(y2 - face_h * 0.01))
    y_mid = min(H, int(y2 + face_h * (0.05 if hair_length == "short" else 0.08)))
    y_bottom = min(H, int(y2 + face_h * (0.08 if hair_length == "short" else 0.12)))

    if x_left < x_right and y_top < y_bottom:
        neck_poly = np.array(
            [
                [cx - max(8, int(neck_half * 0.48)), y_top],
                [cx + max(8, int(neck_half * 0.48)), y_top],
                [cx + max(10, int(neck_half * 0.88)), y_mid],
                [cx + max(8, int(neck_half * 0.72)), y_bottom],
                [cx - max(8, int(neck_half * 0.72)), y_bottom],
                [cx - max(10, int(neck_half * 0.88)), y_mid],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, neck_poly, 255)
        ellipse_center = (
            cx,
            int(y2 + face_h * (0.04 if hair_length == "short" else 0.06)),
        )
        ellipse_axes = (
            max(10, int(neck_half * 0.85)),
            max(8, int(face_h * (0.06 if hair_length == "short" else 0.08))),
        )
        cv2.ellipse(mask, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)

    if cloth_mask.shape == (H, W):
        cloth_u8 = (np.clip(cloth_mask, 0.0, 1.0) > 0.22).astype(np.uint8) * 255
        cloth_u8 = cv2.dilate(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(cloth_u8))

    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=7.0, sigmaY=7.0)
    if hair_length == "short":
        mask_f = np.clip(mask_f * 0.74, 0.0, 1.0)
    return np.clip(mask_f, 0.0, 1.0).astype(np.float32)

def _build_short_lateral_neck_preserve_mask(
    self,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    cloth_mask: Optional[np.ndarray],
    hair_length: str,
) -> np.ndarray:
    if hair_length != "short" or cloth_mask is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = cloth_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    mask_u8 = np.zeros((H, W), dtype=np.uint8)
    centers = (
        (
            int(cx - face_w * 0.44),
            int(y2 + face_h * 0.11),
        ),
        (
            int(cx + face_w * 0.44),
            int(y2 + face_h * 0.11),
        ),
    )
    axes = (
        max(16, int(face_w * 0.26)),
        max(14, int(face_h * 0.16)),
    )
    for center in centers:
        cv2.ellipse(mask_u8, center, axes, 0, 0, 360, 255, -1)

    band_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 - face_h * 0.03))
    bottom = min(H, int(cutoff_y + face_h * 0.26))
    left = max(0, int(x1 - face_w * 0.36))
    right = min(W, int(x2 + face_w * 0.36))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    band_u8[top:bottom, left:right] = 255
    mask_u8 = cv2.bitwise_and(mask_u8, band_u8)

    center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    center_half = max(14, int(face_w * 0.18))
    center_keepout_u8[
        top:bottom,
        max(0, cx - center_half):min(W, cx + center_half),
    ] = 255
    mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(center_keepout_u8))

    if int((mask_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    mask_f = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=5.0, sigmaY=5.0)
    return np.clip(mask_f * 0.92, 0.0, 1.0).astype(np.float32)

def _build_shoulder_protect_mask(
    self,
    cloth_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
) -> np.ndarray:
    """
    어깨선(옷 상단 경계) 보호 마스크 생성.
    short/medium 후처리에서 어깨 라인 훼손을 줄이기 위해 사용한다.
    """
    H, W = cloth_mask.shape[:2]
    if cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    cloth_u8 = (cloth_mask > 0.35).astype(np.uint8) * 255
    if int((cloth_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    y_start = max(0, int(cutoff_y - face_h * 0.08))
    y_end = min(H, int(cutoff_y + face_h * 1.05))
    band = np.zeros((H, W), dtype=np.uint8)
    band[y_start:y_end, :] = cloth_u8[y_start:y_end, :]
    if int((band > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(band, cv2.MORPH_GRADIENT, edge_k)
    edge = cv2.dilate(
        edge,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )

    x_min = max(0, int(x1 - face_w * 1.30))
    x_max = min(W, int(x2 + face_w * 1.30))
    if x_min >= x_max:
        return np.zeros((H, W), dtype=np.float32)

    corridor = np.zeros((H, W), dtype=np.uint8)
    corridor[:, x_min:x_max] = 255
    edge = cv2.bitwise_and(edge, corridor)

    center_half = max(18, int(face_w * 0.45))
    center_zone = np.zeros((H, W), dtype=np.uint8)
    center_zone[:, max(0, cx - center_half):min(W, cx + center_half)] = 255
    side_edge = cv2.bitwise_and(edge, cv2.bitwise_not(center_zone))
    if int((side_edge > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    side_edge = cv2.GaussianBlur(
        side_edge.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
    )
    return np.clip(side_edge, 0.0, 1.0).astype(np.float32)

def _build_torso_cloth_preserve_mask(
    self,
    face_bbox: Tuple[int, int, int, int],
    cloth_mask: Optional[np.ndarray],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if cloth_mask is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = cloth_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255
    if int((cloth_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    preserve_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * (0.06 if hair_length == "short" else 0.10)))
    bottom = min(H, int(cutoff_y + face_h * (0.96 if hair_length == "short" else 1.35)))
    half_w = max(18, int(face_w * (0.58 if hair_length == "short" else 0.68)))
    if top < bottom:
        preserve_u8[top:bottom, max(0, cx - half_w):min(W, cx + half_w)] = 255

    ellipse_center = (
        cx,
        min(H - 1, int(y2 + face_h * (0.24 if hair_length == "short" else 0.32))),
    )
    ellipse_axes = (
        max(14, int(face_w * (0.44 if hair_length == "short" else 0.52))),
        max(12, int(face_h * (0.22 if hair_length == "short" else 0.32))),
    )
    cv2.ellipse(preserve_u8, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)

    preserve_u8 = cv2.bitwise_and(preserve_u8, cloth_u8)
    preserve_u8 = cv2.morphologyEx(
        preserve_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    preserve_u8 = cv2.dilate(
        preserve_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    if hair_length == "short":
        center_release_u8 = np.zeros((H, W), dtype=np.uint8)
        release_top = max(0, int(cutoff_y + face_h * 0.10))
        release_bottom = min(H, int(cutoff_y + face_h * 1.34))
        release_half = max(10, int(face_w * 0.13))
        if release_top < release_bottom:
            center_release_u8[
                release_top:release_bottom,
                max(0, cx - release_half):min(W, cx + release_half)
            ] = 255
            center_release_f = cv2.GaussianBlur(
                center_release_u8.astype(np.float32) / 255.0,
                (0, 0),
                sigmaX=4.0,
                sigmaY=6.0,
            )
            preserve_u8 = cv2.bitwise_and(
                preserve_u8,
                cv2.bitwise_not((center_release_f > 0.12).astype(np.uint8) * 255),
            )

    preserve = cv2.GaussianBlur(
        preserve_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=6.0,
        sigmaY=6.0,
    )
    return np.clip(preserve * (0.92 if hair_length == "short" else 0.68), 0.0, 1.0).astype(np.float32)

def _build_short_below_bob_torso_mask(
    self,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
    torso_candidate_mask: Optional[np.ndarray] = None,
    completed_torso_fill_mask: Optional[np.ndarray] = None,
    shoulder_anchor_mask: Optional[np.ndarray] = None,
    debug_info: Optional[Dict[str, Any]] = None,
    debug_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    base_shape = None
    for mask in (
        cloth_mask,
        torso_candidate_mask,
        completed_torso_fill_mask,
        shoulder_anchor_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)
    if hair_length != "short":
        return np.zeros(base_shape, dtype=np.float32)

    H, W = base_shape
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    def _component_debug(mask_u8: np.ndarray, *, limit: int = 4) -> Dict[str, Any]:
        mask_bin = (mask_u8 > 0).astype(np.uint8)
        if int(mask_bin.sum()) == 0:
            return {
                "bbox": None,
                "component_count": 0,
                "largest_component_area": 0,
                "components": [],
            }
        ys, xs = np.where(mask_bin > 0)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bin, 8)
        components: List[Dict[str, int]] = []
        largest_component_area = 0
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            largest_component_area = max(largest_component_area, area)
            components.append({
                "bbox": [x, y, x + w, y + h],
                "area": area,
            })
        components.sort(key=lambda item: int(item["area"]), reverse=True)
        return {
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "component_count": max(0, num_labels - 1),
            "largest_component_area": int(largest_component_area),
            "components": components[:limit],
        }

    amodal_support_u8 = np.zeros((H, W), dtype=np.uint8)
    for mask, kernel in (
        (torso_candidate_mask, (15, 21)),
        (completed_torso_fill_mask, (17, 23)),
        (shoulder_anchor_mask, (17, 21)),
    ):
        if mask is None or mask.shape != (H, W):
            continue
        support_part_u8 = cv2.dilate(
            (np.clip(mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel),
            iterations=1,
        )
        amodal_support_u8 = cv2.bitwise_or(amodal_support_u8, support_part_u8)

    bob_floor = max(0, int(max(y2 + face_h * 0.10, cutoff_y + face_h * 0.08)))
    bottom = min(H, int(cutoff_y + face_h * 1.74))
    left = max(0, int(x1 - face_w * 1.18))
    right = min(W, int(x2 + face_w * 1.18))
    if bob_floor >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)

    torso_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_u8[bob_floor:bottom, left:right] = 255
    chest_center = (cx, min(H - 1, int(y2 + face_h * 0.66)))
    chest_axes = (
        max(24, int(face_w * 0.58)),
        max(28, int(face_h * 0.60)),
    )
    cv2.ellipse(torso_u8, chest_center, chest_axes, 0, 0, 360, 255, -1)

    center_top = min(bottom, int(cutoff_y + face_h * 0.28))
    center_half = max(34, int(face_w * 0.42))
    if center_top < bottom:
        torso_u8[center_top:bottom, max(0, cx - center_half):min(W, cx + center_half)] = 255

    use_amodal_support = int((amodal_support_u8 > 0).sum()) >= 40
    if use_amodal_support:
        torso_u8 = cv2.bitwise_or(torso_u8, amodal_support_u8)

    if int((torso_u8 > 0).sum()) < 80:
        if debug_info is not None:
            debug_info.update({
                "amodal_support_px": int((amodal_support_u8 > 0).sum()),
                "seed_pretrim_px": int((torso_u8 > 0).sum()),
                "seed_posttrim_px": 0,
                "trim_removed_px": 0,
                "use_amodal_support": bool(use_amodal_support),
                "cutoff_y": int(cutoff_y),
                "bob_floor": int(bob_floor),
                "bottom": int(bottom),
                "center_half": int(center_half),
                "reason": "seed_pretrim_too_small",
            })
        if debug_masks is not None:
            debug_masks["amodal_support"] = amodal_support_u8.astype(np.float32) / 255.0
            debug_masks["seed_pretrim"] = torso_u8.astype(np.float32) / 255.0
        return np.zeros((H, W), dtype=np.float32)

    seed_pretrim_u8 = torso_u8.copy()
    torso_u8 = cv2.morphologyEx(
        torso_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
    )
    torso_u8 = cv2.dilate(
        torso_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
        iterations=1,
    )
    torso_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=torso_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=80,
    )
    if int((torso_u8 > 0).sum()) < 80:
        # v110: fallback_half를 최소 face_w * 0.80 이상으로 보장하여
        # 좁은 수직 strip seed가 생성되지 않도록 한다.
        fallback_half = max(
            int(face_w * 0.80),
            int(face_w * float(getattr(self.config, "overwrite_core_fallback_half_ratio", 0.28))),
        )
        fallback_top = max(
            bob_floor,
            int(cutoff_y + face_h * float(getattr(self.config, "overwrite_core_fallback_top_ratio", 0.22))),
        )
        fallback_bottom = min(
            bottom,
            int(cutoff_y + face_h * float(getattr(self.config, "overwrite_core_fallback_bottom_ratio", 0.98))),
        )
        fallback_window_u8 = np.zeros((H, W), dtype=np.uint8)
        if fallback_top < fallback_bottom:
            fallback_window_u8[
                fallback_top:fallback_bottom,
                max(0, cx - fallback_half):min(W, cx + fallback_half),
            ] = 255
            # 중앙 타원도 fallback_half 비율에 맞게 확장
            fallback_center = (cx, min(H - 1, int(y2 + face_h * 0.64)))
            fallback_axes = (
                max(int(face_w * 0.72), int(face_w * 0.26)),
                max(20, int(face_h * 0.48)),
            )
            cv2.ellipse(fallback_window_u8, fallback_center, fallback_axes, 0, 0, 360, 255, -1)
        fallback_source_u8 = cv2.bitwise_and(seed_pretrim_u8, fallback_window_u8)
        if use_amodal_support:
            supported_fallback_u8 = cv2.bitwise_and(amodal_support_u8, fallback_window_u8)
            if int((supported_fallback_u8 > 0).sum()) >= 80:
                fallback_source_u8 = cv2.bitwise_or(fallback_source_u8, supported_fallback_u8)
        # fallback이 비었으면 fallback_window 자체를 seed로 사용 (가슴 전체를 overwrite 영역으로)
        if int((fallback_source_u8 > 0).sum()) < 80:
            fallback_source_u8 = fallback_window_u8.copy()
        fallback_source_u8 = cv2.morphologyEx(
            fallback_source_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
        )
        fallback_source_u8 = cv2.morphologyEx(
            fallback_source_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 19)),
        )
        # v110: 수평 방향 dilation을 추가하여 수직 strip 형태를 방지
        fallback_source_u8 = cv2.dilate(
            fallback_source_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(11, int(face_w * 0.14)), 5)),
            iterations=1,
        )
        fallback_source_u8 = cv2.bitwise_and(fallback_source_u8, fallback_window_u8)
        fallback_source_u8 = cv2.erode(
            fallback_source_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
            iterations=1,
        )
        if int((fallback_source_u8 > 0).sum()) >= 80:
            torso_u8 = fallback_source_u8
            if debug_info is not None:
                debug_info["reason"] = "fallback_center_anchor"
                debug_info["fallback_center_anchor_px"] = int((fallback_source_u8 > 0).sum())
        elif debug_info is not None:
            debug_info["fallback_center_anchor_px"] = int((fallback_source_u8 > 0).sum())
    if debug_info is not None:
        seed_pretrim_px = int((seed_pretrim_u8 > 0).sum())
        seed_posttrim_px = int((torso_u8 > 0).sum())
        seed_pretrim_debug = _component_debug(seed_pretrim_u8)
        seed_posttrim_debug = _component_debug(torso_u8)
        debug_info.update({
            "amodal_support_px": int((amodal_support_u8 > 0).sum()),
            "seed_pretrim_px": seed_pretrim_px,
            "seed_posttrim_px": seed_posttrim_px,
            "trim_removed_px": max(0, seed_pretrim_px - seed_posttrim_px),
            "use_amodal_support": bool(use_amodal_support),
            "cutoff_y": int(cutoff_y),
            "bob_floor": int(bob_floor),
            "bottom": int(bottom),
            "center_half": int(center_half),
            "reason": debug_info.get("reason") or ("active" if seed_posttrim_px >= 80 else "trimmed_to_empty"),
            "seed_pretrim_bbox": seed_pretrim_debug["bbox"],
            "seed_pretrim_component_count": seed_pretrim_debug["component_count"],
            "seed_pretrim_largest_component_area": seed_pretrim_debug["largest_component_area"],
            "seed_pretrim_components": seed_pretrim_debug["components"],
            "seed_posttrim_bbox": seed_posttrim_debug["bbox"],
            "seed_posttrim_component_count": seed_posttrim_debug["component_count"],
            "seed_posttrim_largest_component_area": seed_posttrim_debug["largest_component_area"],
            "seed_posttrim_components": seed_posttrim_debug["components"],
        })
        if "fallback_source_u8" in locals():
            fallback_debug = _component_debug(fallback_source_u8)
            debug_info.update({
                "fallback_center_anchor_bbox": fallback_debug["bbox"],
                "fallback_center_anchor_component_count": fallback_debug["component_count"],
                "fallback_center_anchor_largest_component_area": fallback_debug["largest_component_area"],
                "fallback_center_anchor_components": fallback_debug["components"],
            })
    if debug_masks is not None:
        debug_masks["amodal_support"] = amodal_support_u8.astype(np.float32) / 255.0
        debug_masks["seed_pretrim"] = seed_pretrim_u8.astype(np.float32) / 255.0
        debug_masks["seed_posttrim"] = torso_u8.astype(np.float32) / 255.0
        if "fallback_source_u8" in locals():
            debug_masks["fallback_center_anchor"] = fallback_source_u8.astype(np.float32) / 255.0
    if int((torso_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        torso_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=12.0,
        sigmaY=16.0,
    ).astype(np.float32)

def _build_short_torso_garment_repaint_mask(
    self,
    *,
    cloth_mask: Optional[np.ndarray],
    torso_mask: Optional[np.ndarray],
    torso_anchor_mask: Optional[np.ndarray],
    torso_candidate_mask: Optional[np.ndarray],
    shoulder_bridge_mask: Optional[np.ndarray],
    sam2_hair_mask: Optional[np.ndarray],
    face_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length != "short":
        base_shape = None
        for mask in (
            cloth_mask,
            torso_mask,
            torso_anchor_mask,
            torso_candidate_mask,
            shoulder_bridge_mask,
            sam2_hair_mask,
            face_mask,
        ):
            if isinstance(mask, np.ndarray):
                base_shape = mask.shape[:2]
                break
        return np.zeros(base_shape or (1, 1), dtype=np.float32)

    base_shape = None
    for mask in (
        cloth_mask,
        torso_mask,
        torso_anchor_mask,
        torso_candidate_mask,
        shoulder_bridge_mask,
        sam2_hair_mask,
        face_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if cloth_mask is not None and cloth_mask.shape != (H, W):
        cloth_mask = None
    if torso_mask is not None and torso_mask.shape != (H, W):
        torso_mask = None
    if torso_anchor_mask is not None and torso_anchor_mask.shape != (H, W):
        torso_anchor_mask = None
    if torso_candidate_mask is not None and torso_candidate_mask.shape != (H, W):
        torso_candidate_mask = None
    if shoulder_bridge_mask is not None and shoulder_bridge_mask.shape != (H, W):
        shoulder_bridge_mask = None
    if sam2_hair_mask is not None and sam2_hair_mask.shape != (H, W):
        sam2_hair_mask = None
    if face_mask is not None and face_mask.shape != (H, W):
        face_mask = None
    if final_hair_mask is not None and final_hair_mask.shape != (H, W):
        final_hair_mask = None
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(max(y2 + face_h * 0.12, cutoff_y + face_h * 0.08)))
    bottom = min(H, int(cutoff_y + face_h * 1.46))
    left = max(0, int(x1 - face_w * 1.14))
    right = min(W, int(x2 + face_w * 1.14))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    upper_body_window_u8 = np.zeros((H, W), dtype=np.uint8)
    upper_top = max(top, int(y2 + face_h * 0.02))
    upper_bottom = min(bottom, int(cutoff_y + face_h * 1.22))
    upper_left = max(0, int(x1 - face_w * 1.06))
    upper_right = min(W, int(x2 + face_w * 1.06))
    if upper_top < upper_bottom and upper_left < upper_right:
        upper_body_window_u8[upper_top:upper_bottom, upper_left:upper_right] = 255
        chest_center = (cx, min(H - 1, int(y2 + face_h * 0.60)))
        chest_axes = (
            max(24, int(face_w * 0.62)),
            max(28, int(face_h * 0.74)),
        )
        cv2.ellipse(upper_body_window_u8, chest_center, chest_axes, 0, 0, 360, 255, -1)
    upper_body_window_u8 = cv2.bitwise_and(upper_body_window_u8, corridor_u8)

    support_u8 = np.zeros((H, W), dtype=np.uint8)
    cloth_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_candidate_u8 = np.zeros((H, W), dtype=np.uint8)
    bridge_u8 = np.zeros((H, W), dtype=np.uint8)
    sam2_torso_u8 = np.zeros((H, W), dtype=np.uint8)

    if cloth_mask is not None:
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        )
    if torso_mask is not None:
        support_u8 = cv2.bitwise_or(
            support_u8,
            cv2.dilate(
                (np.clip(torso_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 25)),
                iterations=1,
            ),
        )
    if torso_anchor_mask is not None:
        torso_anchor_u8 = cv2.dilate(
            (np.clip(torso_anchor_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 21)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_or(support_u8, torso_anchor_u8)
    if torso_candidate_mask is not None:
        torso_candidate_u8 = cv2.dilate(
            (np.clip(torso_candidate_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 21)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_or(support_u8, torso_candidate_u8)
    if shoulder_bridge_mask is not None:
        bridge_u8 = cv2.dilate(
            (np.clip(shoulder_bridge_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_or(support_u8, bridge_u8)

    support_u8 = cv2.bitwise_and(support_u8, corridor_u8)
    if int((support_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    amodal_support_u8 = support_u8.copy()
    if int((cloth_u8 > 0).sum()) > 0 and int((amodal_support_u8 > 0).sum()) > 0:
        cloth_hint_u8 = cv2.bitwise_and(
            cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                iterations=1,
            ),
            cv2.dilate(
                amodal_support_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 41)),
                iterations=1,
            ),
        )
        support_u8 = cv2.bitwise_or(amodal_support_u8, cloth_hint_u8)
    else:
        support_u8 = amodal_support_u8.copy()
    support_u8 = cv2.bitwise_and(support_u8, corridor_u8)
    if int((support_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    upper_body_support_u8 = cv2.bitwise_and(support_u8, upper_body_window_u8)
    if int((upper_body_support_u8 > 0).sum()) < 100:
        upper_body_support_u8 = support_u8.copy()

    if sam2_hair_mask is not None:
        sam2_torso_u8 = cv2.dilate(
            (np.clip(sam2_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
            iterations=1,
        )
        sam2_torso_u8 = cv2.bitwise_and(sam2_torso_u8, corridor_u8)
        candidate_u8 = cv2.bitwise_and(
            sam2_torso_u8,
            cv2.dilate(
                support_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
                iterations=1,
            ),
        )
        if int((torso_candidate_u8 > 0).sum()) > 0:
            underhair_torso_u8 = cv2.bitwise_and(
                torso_candidate_u8,
                cv2.dilate(
                    sam2_torso_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 41)),
                    iterations=1,
                ),
            )
            candidate_u8 = cv2.bitwise_or(candidate_u8, underhair_torso_u8)
        candidate_u8 = cv2.bitwise_or(candidate_u8, upper_body_support_u8)
    else:
        candidate_u8 = upper_body_support_u8.copy()
    if int((candidate_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    face_guard_u8 = np.zeros((H, W), dtype=np.uint8)
    if face_mask is not None:
        face_guard_u8 = cv2.dilate(
            (np.clip(face_mask.astype(np.float32), 0.0, 1.0) > 0.10).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
    else:
        guard_left = max(0, int(x1 - face_w * 0.10))
        guard_right = min(W, int(x2 + face_w * 0.10))
        guard_top = max(0, int(y1 - face_h * 0.08))
        guard_bottom = min(H, int(y2 + face_h * 0.18))
        if guard_top < guard_bottom and guard_left < guard_right:
            face_guard_u8[guard_top:guard_bottom, guard_left:guard_right] = 255
    candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(face_guard_u8))

    neck_guard_u8 = np.zeros((H, W), dtype=np.uint8)
    neck_top = max(0, int(y2 - face_h * 0.02))
    neck_bottom = min(H, int(cutoff_y + face_h * 0.22))
    neck_half = max(18, int(face_w * 0.20))
    if neck_top < neck_bottom:
        neck_guard_u8[
            neck_top:neck_bottom,
            max(0, cx - neck_half):min(W, cx + neck_half)
        ] = 255
        candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(neck_guard_u8))

    if protect_mask is not None:
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.10).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(protect_u8))

    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23)),
    )
    candidate_u8 = cv2.dilate(
        candidate_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
    candidate_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=candidate_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=16,
    )
    if int((candidate_u8 > 0).sum()) < 16:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        candidate_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.6,
        sigmaY=6.8,
    ).astype(np.float32)

def _build_bright_cloth_preserve_mask(
    self,
    img_rgb: np.ndarray,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    H, W = removal_mask.shape[:2]
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255
    zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
    if int((zone_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.04))
    bottom = min(H, int(cutoff_y + face_h * (1.22 if hair_length == "short" else 1.46)))
    left = max(0, int(x1 - face_w * (1.14 if hair_length == "short" else 1.08)))
    right = min(W, int(x2 + face_w * (1.14 if hair_length == "short" else 1.08)))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sat = hsv[:, :, 1]
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
    bright_u8 = (
        (gray > (176 if hair_length == "short" else 170))
        & (blur > (182 if hair_length == "short" else 176))
        & (sat < (58 if hair_length == "short" else 64))
    ).astype(np.uint8) * 255
    preserve_u8 = cv2.bitwise_and(bright_u8, zone_u8)
    if int((preserve_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    preserve_u8 = cv2.morphologyEx(
        preserve_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    preserve_u8 = cv2.morphologyEx(
        preserve_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
    )
    preserve_u8 = cv2.dilate(
        preserve_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(preserve_u8, 8)
    min_area = max(40, int(face_w * face_h * 0.006))
    max_area = max(2400, int(face_w * face_h * 0.22))
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        comp_cx = float(centroids[idx][0])
        if area < min_area or area > max_area:
            continue
        if (y + h) < int(cutoff_y + face_h * 0.10):
            continue
        if abs(comp_cx - 0.5 * (x1 + x2)) > max(14, int(face_w * 0.14)):
            filtered_u8[labels == idx] = 255

    if int((filtered_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    preserve = cv2.GaussianBlur(
        filtered_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )
    return np.clip(preserve, 0.0, 1.0).astype(np.float32)

def _build_micro_cloth_artifact_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    zone_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y - face_h * 0.04))
    bottom = min(H, int(cutoff_y + face_h * (0.82 if hair_length == "short" else 0.92)))
    left = max(0, int(x1 - face_w * (1.04 if hair_length == "short" else 1.12)))
    right = min(W, int(x2 + face_w * (1.04 if hair_length == "short" else 1.12)))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    zone_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, cloth_u8)
    if int((zone_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 15) if hair_length == "short" else (7, 13)),
    )
    candidate_u8 = (
        (
            ((gray < (172.0 if hair_length == "short" else 166.0)) & ((blur - gray) > (2.5 if hair_length == "short" else 2.8)))
            | (blackhat > (8 if hair_length == "short" else 9))
        ).astype(np.uint8)
        * 255
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(final_hair_u8))
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    if int((candidate_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    max_area = max(160, int(face_w * face_h * 0.018))
    max_width = max(24, int(face_w * 0.20))
    max_height = max(72, int(face_h * 0.62))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        if area < 6 or area > max_area:
            continue
        if w > max_width or h > max_height:
            continue
        if bottom_y > int(cutoff_y + face_h * 0.88):
            continue
        fill_ratio = float(area) / float(max(w * h, 1))
        if fill_ratio > 0.72 and area > max(42, int(face_w * face_h * 0.006)):
            continue
        if abs(comp_cx - cx) > max(44, int(face_w * 0.56)) and area > max(28, int(face_w * face_h * 0.004)):
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
    return (keep_u8 > 0).astype(np.float32)
def _build_shoulder_cloth_restore_mask(
    self,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(removal_mask.shape[:2], dtype=np.float32)

    H, W = removal_mask.shape[:2]
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    base_u8 = (
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255
    )
    cloth_u8 = (
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255
    )
    base_u8 = cv2.bitwise_and(base_u8, cloth_u8)
    if int((base_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y - face_h * 0.02))
    bottom = min(H, int(cutoff_y + face_h * (1.10 if hair_length == "short" else 1.22)))
    left = max(0, int(x1 - face_w * 1.18))
    right = min(W, int(x2 + face_w * 1.18))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    base_u8 = cv2.bitwise_and(base_u8, corridor_u8)

    center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    keepout_half = max(14, int(face_w * 0.20))
    keepout_top = max(0, int(cutoff_y - face_h * 0.02))
    keepout_bottom = min(H, int(cutoff_y + face_h * 1.18))
    if keepout_top < keepout_bottom:
        center_keepout_u8[
            keepout_top:keepout_bottom,
            max(0, cx - keepout_half):min(W, cx + keepout_half),
        ] = 255
        base_u8 = cv2.bitwise_and(base_u8, cv2.bitwise_not(center_keepout_u8))

    base_u8 = cv2.morphologyEx(
        base_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    base_u8 = cv2.morphologyEx(
        base_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
    )
    if int((base_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_u8, 8)
    min_area = max(60, int(face_w * face_h * 0.008))
    max_area = max(1800, int(face_w * face_h * 0.42))
    max_width = max(88, int(face_w * 1.12))
    min_height = max(36, int(face_h * 0.18))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_cx = float(centroids[idx][0])
        if area < min_area or area > max_area:
            continue
        if w > max_width or h < min_height:
            continue
        if abs(comp_cx - cx) < max(12, int(face_w * 0.14)):
            continue
        if (y + h) < int(cutoff_y + face_h * 0.08):
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23)),
    )
    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    keep_u8 = cv2.bitwise_and(keep_u8, cloth_u8)
    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(protect_u8))
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(final_hair_u8))
    if int((keep_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)
    return (keep_u8 > 0).astype(np.float32)
def _build_final_source_cloth_rescue_mask(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(current_rgb.shape[:2], dtype=np.float32)

    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * (0.02 if hair_length == "short" else -0.04)))
    bottom = min(H, int(cutoff_y + face_h * (1.38 if hair_length == "short" else 1.52)))
    left = max(0, int(x1 - face_w * (1.26 if hair_length == "short" else 1.34)))
    right = min(W, int(x2 + face_w * (1.26 if hair_length == "short" else 1.34)))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    cloth_u8 = (
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255
    )
    cloth_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
    if int((cloth_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (23, 33) if hair_length == "short" else (23, 33),
        ),
        iterations=1,
    )
    removal_u8 = cv2.bitwise_and(removal_u8, corridor_u8)
    if int((removal_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    diff_rgb = np.abs(
        current_rgb.astype(np.float32) - source_rgb.astype(np.float32)
    ).mean(axis=2)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_sat = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    gray_delta = np.abs(current_gray - source_gray)
    strong_diff_u8 = (
        (
            (diff_rgb > (26.0 if hair_length == "short" else 22.0))
            | (gray_delta > (24.0 if hair_length == "short" else 20.0))
        ).astype(np.uint8)
        * 255
    )
    strong_diff_u8 = cv2.bitwise_and(strong_diff_u8, cloth_u8)
    strong_diff_u8 = cv2.bitwise_and(strong_diff_u8, removal_u8)

    deep_zone_u8 = np.zeros((H, W), dtype=np.uint8)
    if hair_length == "short":
        deep_start = min(H, int(cutoff_y + face_h * 0.22))
        if deep_start < bottom:
            deep_zone_u8[deep_start:bottom, left:right] = 255
        deep_zone_u8 = cv2.bitwise_and(deep_zone_u8, cloth_u8)

    keep_u8 = cv2.morphologyEx(
        strong_diff_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 17) if hair_length == "short" else (13, 21),
        ),
    )
    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 15) if hair_length == "short" else (11, 17),
        ),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, cloth_u8)

    visible_cloth_u8 = cv2.bitwise_and(
        cloth_u8,
        cv2.bitwise_not(
            cv2.dilate(
                removal_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (27, 35) if hair_length == "short" else (31, 41),
                ),
                iterations=1,
            )
        ),
    )
    cloth_gray_median = float(np.median(source_gray[cloth_u8 > 0])) if int((cloth_u8 > 0).sum()) >= 80 else 168.0
    cloth_sat_median = float(np.median(source_sat[cloth_u8 > 0])) if int((cloth_u8 > 0).sum()) >= 80 else 96.0
    dark_strand_u8 = (
        (
            (current_gray < min(cloth_gray_median - (18.0 if hair_length == "short" else 14.0), 172.0))
            & (current_sat < cloth_sat_median + (44.0 if hair_length == "short" else 36.0))
            & ((diff_rgb > 10.0) | (gray_delta > 10.0))
        ).astype(np.uint8)
        * 255
    )
    dark_strand_u8 = cv2.bitwise_and(dark_strand_u8, cloth_u8)
    dark_strand_u8 = cv2.bitwise_and(dark_strand_u8, removal_u8)
    if hair_length == "short":
        dark_strand_u8 = cv2.bitwise_and(dark_strand_u8, deep_zone_u8)
    keep_u8 = cv2.bitwise_or(keep_u8, dark_strand_u8)
    if int((visible_cloth_u8 > 0).sum()) >= 120:
        visible_rgb = source_rgb[visible_cloth_u8 > 0].astype(np.float32)
        visible_gray = source_gray[visible_cloth_u8 > 0]
        visible_sat = source_sat[visible_cloth_u8 > 0]
        cloth_rgb_median = np.median(visible_rgb, axis=0)
        cloth_gray_median = float(np.median(visible_gray))
        cloth_sat_median = float(np.median(visible_sat))
        color_delta = np.sqrt(
            np.sum(
                (source_rgb.astype(np.float32) - cloth_rgb_median.reshape(1, 1, 3)) ** 2,
                axis=2,
            )
        )
        tone_match_u8 = (
            (
                (np.abs(source_gray - cloth_gray_median) <= (34.0 if hair_length == "short" else 42.0))
                & (np.abs(source_sat - cloth_sat_median) <= (42.0 if hair_length == "short" else 54.0))
                & (color_delta <= (62.0 if hair_length == "short" else 78.0))
            ).astype(np.uint8)
            * 255
        )
        if hair_length == "short":
            tone_match_u8 = cv2.bitwise_or(
                tone_match_u8,
                cv2.bitwise_and(dark_strand_u8, deep_zone_u8),
            )
        keep_u8 = cv2.bitwise_and(keep_u8, tone_match_u8)

    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(protect_u8))
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (17, 25) if hair_length == "short" else (15, 23),
            ),
            iterations=1,
        )
        if hair_length == "short":
            upper_hair_guard_u8 = np.zeros((H, W), dtype=np.uint8)
            guard_bottom = min(H, int(cutoff_y + face_h * 0.34))
            guard_left = max(0, int(x1 - face_w * 0.92))
            guard_right = min(W, int(x2 + face_w * 0.92))
            if top < guard_bottom and guard_left < guard_right:
                upper_hair_guard_u8[top:guard_bottom, guard_left:guard_right] = 255
                final_hair_u8 = cv2.bitwise_and(final_hair_u8, upper_hair_guard_u8)
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(final_hair_u8))

    if int((keep_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(keep_u8, 8)
    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    min_area = max(56, int(face_w * face_h * 0.008))
    max_area = max(3600, int(face_w * face_h * 0.44))
    max_width = max(120, int(face_w * 1.26))
    min_height = max(24, int(face_h * 0.10))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if w > max_width or h < min_height:
            continue
        if (y + h) < int(cutoff_y + face_h * 0.06):
            continue
        filtered_u8[labels == idx] = 255

    if int((filtered_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 23) if hair_length == "short" else (13, 21),
        ),
        iterations=1,
    )
    filtered_u8 = cv2.bitwise_and(filtered_u8, cloth_u8)
    if hair_length == "short":
        filtered_u8 = self._trim_blocky_short_restore_mask_u8(
            mask_u8=filtered_u8,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            min_keep_px=80,
        )
    if int((filtered_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)
    return (filtered_u8 > 0).astype(np.float32)

def _build_short_subject_cloth_cleanup_mask(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    torso_mask: Optional[np.ndarray],
    face_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length != "short":
        return np.zeros(current_rgb.shape[:2], dtype=np.float32)

    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if torso_mask is not None and torso_mask.shape != (H, W):
        torso_mask = None
    if face_mask is not None and face_mask.shape != (H, W):
        face_mask = None
    if final_hair_mask is not None and final_hair_mask.shape != (H, W):
        final_hair_mask = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    if int((cloth_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.02))
    bottom = min(H, int(cutoff_y + face_h * 1.60))
    left = max(0, int(x1 - face_w * 1.28))
    right = min(W, int(x2 + face_w * 1.28))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    candidate_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
    if torso_mask is not None:
        torso_u8 = cv2.dilate(
            (np.clip(torso_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, torso_u8)
    if int((candidate_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    face_guard_u8 = np.zeros((H, W), dtype=np.uint8)
    if face_mask is not None:
        face_guard_u8 = cv2.dilate(
            (np.clip(face_mask.astype(np.float32), 0.0, 1.0) > 0.10).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
    else:
        guard_left = max(0, int(x1 - face_w * 0.12))
        guard_right = min(W, int(x2 + face_w * 0.12))
        guard_top = max(0, int(y1 - face_h * 0.10))
        guard_bottom = min(H, int(y2 + face_h * 0.18))
        if guard_top < guard_bottom and guard_left < guard_right:
            face_guard_u8[guard_top:guard_bottom, guard_left:guard_right] = 255
    candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(face_guard_u8))
    if int((candidate_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    diff_rgb = np.abs(current_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=2)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_sat = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    gray_delta = np.abs(current_gray - source_gray)

    strong_diff_u8 = (
        (
            (diff_rgb > 10.0)
            | (gray_delta > 10.0)
        ).astype(np.uint8)
        * 255
    )
    dark_residual_u8 = (
        (
            (
                (current_gray + 8.0 < source_gray)
                | (diff_rgb > 12.0)
                | (gray_delta > 12.0)
            )
            & (current_sat < source_sat + 64.0)
        ).astype(np.uint8)
        * 255
    )
    bright_smear_u8 = (
        (
            (
                (current_gray > source_gray + 12.0)
                | (gray_delta > 12.0)
            )
            & (diff_rgb > 10.0)
            & (current_sat < source_sat + 54.0)
        ).astype(np.uint8)
        * 255
    )

    keep_u8 = cv2.bitwise_or(strong_diff_u8, dark_residual_u8)
    keep_u8 = cv2.bitwise_or(keep_u8, bright_smear_u8)
    keep_u8 = cv2.bitwise_and(keep_u8, candidate_u8)

    protected_hair_u8 = np.zeros((H, W), dtype=np.uint8)
    if final_hair_mask is not None:
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 25)),
            iterations=1,
        )
        hair_guard_u8 = np.zeros((H, W), dtype=np.uint8)
        hair_guard_bottom = min(H, int(cutoff_y + face_h * 0.34))
        hair_guard_left = max(0, int(x1 - face_w * 0.96))
        hair_guard_right = min(W, int(x2 + face_w * 0.96))
        if top < hair_guard_bottom and hair_guard_left < hair_guard_right:
            hair_guard_u8[top:hair_guard_bottom, hair_guard_left:hair_guard_right] = 255
        protected_hair_u8 = cv2.bitwise_and(final_hair_u8, hair_guard_u8)
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(protected_hair_u8))

        lower_final_hair_u8 = cv2.bitwise_and(final_hair_u8, candidate_u8)
        lower_final_hair_top = max(0, int(max(y2 + face_h * 0.14, cutoff_y + face_h * 0.10)))
        if lower_final_hair_top > 0:
            lower_final_hair_u8[:lower_final_hair_top, :] = 0
        lower_final_hair_u8 = cv2.morphologyEx(
            lower_final_hair_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
        )
        keep_u8 = cv2.bitwise_or(keep_u8, lower_final_hair_u8)

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
    )
    if int((keep_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(keep_u8, 8)
    min_area = max(72, int(face_w * face_h * 0.004))
    max_area = max(24000, int(face_w * face_h * 0.90))
    min_height = max(20, int(face_h * 0.08))
    max_width = max(340, int(face_w * 1.72))
    max_offset = max(260, int(face_w * 1.34))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_cx = float(centroids[idx][0])
        bottom_y = y + h
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if bottom_y < int(cutoff_y + face_h * 0.06):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        filtered_u8[labels == idx] = 255

    if int((filtered_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
        iterations=1,
    )
    filtered_u8 = cv2.bitwise_and(
        filtered_u8,
        cv2.dilate(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        ),
    )
    filtered_u8 = cv2.bitwise_and(filtered_u8, corridor_u8)
    filtered_u8 = cv2.bitwise_and(filtered_u8, cv2.bitwise_not(face_guard_u8))
    if int((protected_hair_u8 > 0).sum()) > 0:
        filtered_u8 = cv2.bitwise_and(filtered_u8, cv2.bitwise_not(protected_hair_u8))
    filtered_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=filtered_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=100,
    )
    if int((filtered_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        filtered_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.2,
        sigmaY=6.8,
    ).astype(np.float32)

def _build_short_lower_garment_cleanup_mask(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length != "short":
        return np.zeros(current_rgb.shape[:2], dtype=np.float32)

    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if final_hair_mask is not None and final_hair_mask.shape != (H, W):
        final_hair_mask = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 27)),
        iterations=1,
    )
    top = max(0, int(cutoff_y + face_h * 0.06))
    bottom = min(H, int(cutoff_y + face_h * 1.34))
    left = max(0, int(x1 - face_w * 1.26))
    right = min(W, int(x2 + face_w * 1.26))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(removal_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.2, sigmaY=5.2)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    diff_rgb = np.abs(current_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=2)
    gray_delta = np.abs(gray - source_gray)

    low_texture_u8 = (
        ((np.abs(gray - blur) <= 7.0) | (lap < 9.0)).astype(np.uint8) * 255
    )
    low_texture_u8 = cv2.bitwise_and(
        low_texture_u8,
        (((diff_rgb > 14.0) | (gray_delta > 14.0)).astype(np.uint8) * 255),
    )

    deep_zone_u8 = np.zeros((H, W), dtype=np.uint8)
    deep_start = min(H, int(cutoff_y + face_h * 0.42))
    if deep_start < bottom:
        deep_zone_u8[deep_start:bottom, left:right] = 255
    deep_zone_u8 = cv2.bitwise_and(deep_zone_u8, zone_u8)

    dark_tail_u8 = (
        ((gray < 156.0) & ((blur - gray) > 2.2)).astype(np.uint8) * 255
    )
    dark_tail_u8 = cv2.bitwise_and(dark_tail_u8, zone_u8)

    keep_u8 = cv2.bitwise_or(low_texture_u8, dark_tail_u8)
    keep_u8 = cv2.bitwise_or(keep_u8, deep_zone_u8)
    keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
    if int((keep_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        cloth_support_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
        keep_u8 = cv2.bitwise_or(keep_u8, cv2.bitwise_and(deep_zone_u8, cloth_support_u8))

    upper_guard_u8 = np.zeros((H, W), dtype=np.uint8)
    upper_guard_bottom = min(H, int(cutoff_y + face_h * 0.32))
    upper_guard_left = max(0, int(cx - face_w * 0.74))
    upper_guard_right = min(W, int(cx + face_w * 0.74))
    if top < upper_guard_bottom and upper_guard_left < upper_guard_right:
        upper_guard_u8[top:upper_guard_bottom, upper_guard_left:upper_guard_right] = 255
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(upper_guard_u8))

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(protect_u8))

    if final_hair_mask is not None:
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 25)),
            iterations=1,
        )
        hair_guard_u8 = cv2.dilate(
            upper_guard_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        hair_protect_u8 = cv2.bitwise_and(final_hair_u8, hair_guard_u8)
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(hair_protect_u8))

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
    )
    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if int((keep_u8 > 0).sum()) < 120:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(keep_u8, 8)
    min_area = max(72, int(face_w * face_h * 0.003))
    max_area = max(120000, int(face_w * face_h * 1.85))
    min_height = max(22, int(face_h * 0.12))
    max_width = max(360, int(face_w * 1.88))
    max_offset = max(220, int(face_w * 1.16))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if bottom_y < int(cutoff_y + face_h * 0.18):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        filtered_u8[labels == idx] = 255

    if int((filtered_u8 > 0).sum()) < 120:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
        iterations=1,
    )
    filtered_u8 = cv2.bitwise_and(filtered_u8, corridor_u8)
    filtered_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=filtered_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=120,
    )
    return cv2.GaussianBlur(
        filtered_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.2,
        sigmaY=5.6,
    ).astype(np.float32)

def _build_short_lower_cloth_hard_override_mask(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length != "short":
        return np.zeros(current_rgb.shape[:2], dtype=np.float32)

    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 35)),
        iterations=1,
    )
    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
    if int((zone_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    top = max(0, int(cutoff_y + face_h * 0.12))
    bottom = min(H, int(cutoff_y + face_h * 1.52))
    left = max(0, int(x1 - face_w * 1.30))
    right = min(W, int(x2 + face_w * 1.30))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_sat = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    diff_rgb = np.abs(current_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=2)
    gray_delta = np.abs(current_gray - source_gray)

    shallow_zone_u8 = np.zeros((H, W), dtype=np.uint8)
    shallow_bottom = min(bottom, int(cutoff_y + face_h * 0.52))
    if top < shallow_bottom:
        shallow_zone_u8[top:shallow_bottom, left:right] = 255
    shallow_zone_u8 = cv2.bitwise_and(shallow_zone_u8, zone_u8)

    deep_zone_u8 = np.zeros((H, W), dtype=np.uint8)
    deep_top = min(bottom, int(cutoff_y + face_h * 0.48))
    if deep_top < bottom:
        deep_zone_u8[deep_top:bottom, left:right] = 255
    deep_zone_u8 = cv2.bitwise_and(deep_zone_u8, zone_u8)

    upper_residual_u8 = (
        (
            (
                (current_gray + 10.0 < source_gray)
                | (diff_rgb > 12.0)
                | (gray_delta > 12.0)
            )
            & (current_sat < 152.0)
        ).astype(np.uint8)
        * 255
    )
    upper_residual_u8 = cv2.bitwise_and(upper_residual_u8, shallow_zone_u8)

    deep_residual_u8 = (
        (
            (
                (current_gray + 12.0 < source_gray)
                | (diff_rgb > 14.0)
                | (gray_delta > 14.0)
            )
            & (current_sat < 164.0)
        ).astype(np.uint8)
        * 255
    )
    deep_residual_u8 = cv2.bitwise_and(deep_residual_u8, deep_zone_u8)

    keep_u8 = cv2.bitwise_or(deep_residual_u8, upper_residual_u8)
    keep_u8 = cv2.bitwise_and(keep_u8, cloth_u8)
    if int((keep_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    shoulder_guard_u8 = np.zeros((H, W), dtype=np.uint8)
    shoulder_guard_bottom = min(bottom, int(cutoff_y + face_h * 0.16))
    shoulder_inner_half = max(60, int(face_w * 0.54))
    shoulder_inner_left = max(left, int(cx - shoulder_inner_half))
    shoulder_inner_right = min(right, int(cx + shoulder_inner_half))
    if top < shoulder_guard_bottom:
        if left < shoulder_inner_left:
            shoulder_guard_u8[top:shoulder_guard_bottom, left:shoulder_inner_left] = 255
        if shoulder_inner_right < right:
            shoulder_guard_u8[top:shoulder_guard_bottom, shoulder_inner_right:right] = 255
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(shoulder_guard_u8))

    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23)),
            iterations=1,
        )
        lower_final_hair_u8 = cv2.bitwise_and(
            final_hair_u8,
            cv2.bitwise_or(cloth_u8, removal_u8),
        )
        lower_final_hair_u8 = cv2.bitwise_and(lower_final_hair_u8, corridor_u8)
        lower_final_hair_top = max(0, int(max(y2 + face_h * 0.16, cutoff_y + face_h * 0.14)))
        if lower_final_hair_top < H:
            lower_final_hair_u8[:lower_final_hair_top, :] = 0
        lower_final_hair_u8 = cv2.morphologyEx(
            lower_final_hair_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
        )
        keep_u8 = cv2.bitwise_or(keep_u8, lower_final_hair_u8)
        upper_hair_guard_u8 = np.zeros((H, W), dtype=np.uint8)
        hair_guard_bottom = min(H, int(cutoff_y + face_h * 0.34))
        hair_guard_left = max(0, int(x1 - face_w * 0.90))
        hair_guard_right = min(W, int(x2 + face_w * 0.90))
        if top < hair_guard_bottom and hair_guard_left < hair_guard_right:
            upper_hair_guard_u8[top:hair_guard_bottom, hair_guard_left:hair_guard_right] = 255
            final_hair_u8 = cv2.bitwise_and(final_hair_u8, upper_hair_guard_u8)
            keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(final_hair_u8))

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
    )
    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    keep_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=keep_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=80,
    )
    if int((keep_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.4,
        sigmaY=6.0,
    ).astype(np.float32)

def _build_side_column_cloth_restore_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    candidate_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
    debug_info: Optional[Dict[str, Any]] = None,
    debug_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    if cloth_mask is None or candidate_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask.shape != (H, W) or candidate_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    candidate_u8 = (np.clip(candidate_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    zone_u8 = cv2.bitwise_and(candidate_u8, cloth_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * (0.10 if hair_length == "short" else 0.20)))
    bottom = min(H, int(cutoff_y + face_h * (1.54 if hair_length == "short" else 1.34)))
    left = max(0, int(x1 - face_w * (1.18 if hair_length == "short" else 1.12)))
    right = min(W, int(x2 + face_w * (1.18 if hair_length == "short" else 1.12)))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)

    if int((zone_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    short_outer_strip_u8 = corridor_u8
    center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    neckline_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    if hair_length == "short":
        short_outer_strip_u8 = np.zeros((H, W), dtype=np.uint8)
        inner_keepout_half = max(
            40,
            int(face_w * float(getattr(self.config, "short_side_column_inner_keepout_ratio", 0.46))),
        )
        neckline_keepout_half = max(inner_keepout_half + 18, int(face_w * 0.66))
        keepout_top = max(top, int(cutoff_y + face_h * 0.10))
        keepout_bottom = min(bottom, int(cutoff_y + face_h * 1.22))
        if keepout_top < keepout_bottom:
            center_keepout_u8[
                keepout_top:keepout_bottom,
                max(left, int(cx - inner_keepout_half)):min(right, int(cx + inner_keepout_half)),
            ] = 255
            zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(center_keepout_u8))
        neckline_keepout_bottom = min(
            bottom,
            int(cutoff_y + face_h * float(getattr(self.config, "short_side_column_neckline_keepout_ratio", 0.38))),
        )
        if top < neckline_keepout_bottom:
            neckline_keepout_u8[
                top:neckline_keepout_bottom,
                max(left, int(cx - neckline_keepout_half)):min(right, int(cx + neckline_keepout_half)),
            ] = 255
            zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(neckline_keepout_u8))
        outer_gap_half = max(
            inner_keepout_half,
            int(face_w * float(getattr(self.config, "short_side_column_outer_strip_gap_ratio", 0.52))),
        )
        outer_strip_width = max(
            20,
            int(face_w * float(getattr(self.config, "short_side_column_outer_strip_width_scale", 0.96))),
        )
        strip_top = max(top, int(cutoff_y + face_h * 0.16))
        if strip_top < bottom:
            left_outer = max(left, int(x1 - face_w * 0.92))
            left_inner = min(right, left_outer + outer_strip_width)
            left_gap = max(left_inner, int(cx - outer_gap_half))
            right_outer = min(right, int(x2 + face_w * 0.92))
            right_inner = max(left, right_outer - outer_strip_width)
            right_gap = min(right_inner, int(cx + outer_gap_half))
            if left_outer < left_inner and left_outer < left_gap:
                short_outer_strip_u8[strip_top:bottom, left_outer:min(left_inner, left_gap)] = 255
            if right_gap < right_outer and right_inner < right_outer:
                short_outer_strip_u8[strip_top:bottom, max(right_inner, right_gap):right_outer] = 255
            zone_u8 = cv2.bitwise_and(zone_u8, short_outer_strip_u8)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    bright_smooth_u8 = (
        (gray > (194.0 if hair_length == "short" else 188.0))
        & (blur > (198.0 if hair_length == "short" else 192.0))
        & (sat < (92.0 if hair_length == "short" else 96.0))
        & (lap < (24.0 if hair_length == "short" else 26.0))
    ).astype(np.uint8) * 255
    dark_smooth_u8 = (
        (gray > (58.0 if hair_length == "short" else 72.0))
        & (gray < (188.0 if hair_length == "short" else 172.0))
        & (sat < (100.0 if hair_length == "short" else 100.0))
        & (lap < (24.0 if hair_length == "short" else 20.0))
    ).astype(np.uint8) * 255
    smooth_u8 = dark_smooth_u8 if hair_length == "short" else cv2.bitwise_or(bright_smooth_u8, dark_smooth_u8)
    zone_u8 = cv2.bitwise_and(zone_u8, smooth_u8)

    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 11) if hair_length == "short" else (11, 17),
            ),
            iterations=1,
        )
        if hair_length == "short":
            final_hair_u8[min(H, int(cutoff_y + face_h * 0.80)):, :] = 0
        zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(final_hair_u8))
    if int((zone_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
    )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    min_area = max(120, int(face_w * face_h * 0.014))
    max_area = max(4200, int(face_w * face_h * (0.22 if hair_length == "short" else 0.34)))
    min_height = max(36, int(face_h * 0.12))
    max_width = max(120, int(face_w * (0.78 if hair_length == "short" else 0.54)))
    max_offset = max(120, int(face_w * (0.98 if hair_length == "short" else 0.60)))
    short_min_offset = max(42, int(face_w * 0.42))
    short_side_width = max(92, int(face_w * 0.50))
    short_side_area = max(2800, int(face_w * face_h * 0.14))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        bottom_y = y + h
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if y < int(cutoff_y + face_h * (0.08 if hair_length == "short" else 0.22)):
            continue
        if bottom_y < int(cutoff_y + face_h * (0.34 if hair_length == "short" else 0.44)):
            continue
        if offset > max_offset:
            continue
        if hair_length == "short":
            if offset < short_min_offset:
                continue
            if w > short_side_width or area > short_side_area:
                continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (19, 31) if hair_length == "short" else (9, 17),
        ),
        iterations=1,
    )
    loose_cloth_u8 = cv2.dilate(
        cloth_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    keep_gate_u8 = cv2.bitwise_or(loose_cloth_u8, candidate_u8)
    keep_u8 = cv2.bitwise_and(keep_u8, keep_gate_u8)
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if hair_length == "short" and int((short_outer_strip_u8 > 0).sum()) > 0:
        keep_u8 = cv2.bitwise_and(keep_u8, short_outer_strip_u8)
    if hair_length == "short":
        keep_u8 = self._trim_blocky_short_restore_mask_u8(
            mask_u8=keep_u8,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            min_keep_px=60,
        )
        if int((short_outer_strip_u8 > 0).sum()) > 0:
            keep_u8 = cv2.bitwise_and(keep_u8, short_outer_strip_u8)
    if int((keep_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    if debug_info is not None:
        debug_info.update({
            "zone_px": int((zone_u8 > 0).sum()),
            "keep_px": int((keep_u8 > 0).sum()),
            "corridor_top": int(top),
            "corridor_bottom": int(bottom),
            "corridor_left": int(left),
            "corridor_right": int(right),
            "center_keepout_px": int((center_keepout_u8 > 0).sum()),
            "neckline_keepout_px": int((neckline_keepout_u8 > 0).sum()),
            "outer_strip_px": int((short_outer_strip_u8 > 0).sum()),
            "gaussian_sigma_x": 4.0,
            "gaussian_sigma_y": 6.0,
        })
    if debug_masks is not None:
        debug_masks["zone"] = zone_u8.astype(np.float32) / 255.0
        debug_masks["center_keepout"] = center_keepout_u8.astype(np.float32) / 255.0
        debug_masks["neckline_keepout"] = neckline_keepout_u8.astype(np.float32) / 255.0
        debug_masks["outer_strip"] = short_outer_strip_u8.astype(np.float32) / 255.0
        debug_masks["keep"] = keep_u8.astype(np.float32) / 255.0

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.0,
        sigmaY=6.0,
    ).astype(np.float32)

def _build_direct_short_column_restore_mask(
    self,
    removal_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if removal_mask is not None:
        base_shape = removal_mask.shape[:2]
    elif cloth_mask is not None:
        base_shape = cloth_mask.shape[:2]
    else:
        base_shape = (1, 1)
    if hair_length != "short":
        return np.zeros(base_shape, dtype=np.float32)

    if removal_mask is None or cloth_mask is None:
        return np.zeros(base_shape, dtype=np.float32)

    H, W = removal_mask.shape[:2]
    if cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.04))
    bottom = min(H, int(cutoff_y + face_h * 1.56))
    left = max(0, int(x1 - face_w * 1.20))
    right = min(W, int(x2 + face_w * 1.20))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    neckline_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    outer_strip_u8 = np.zeros((H, W), dtype=np.uint8)
    inner_keepout_half = max(
        40,
        int(face_w * float(getattr(self.config, "short_side_column_inner_keepout_ratio", 0.46))),
    )
    neckline_keepout_half = max(inner_keepout_half + 18, int(face_w * 0.66))
    keepout_top = max(top, int(cutoff_y + face_h * 0.10))
    keepout_bottom = min(bottom, int(cutoff_y + face_h * 1.24))
    if keepout_top < keepout_bottom:
        center_keepout_u8[
            keepout_top:keepout_bottom,
            max(left, int(cx - inner_keepout_half)):min(right, int(cx + inner_keepout_half)),
        ] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(center_keepout_u8))
    neckline_keepout_bottom = min(
        bottom,
        int(cutoff_y + face_h * float(getattr(self.config, "short_side_column_neckline_keepout_ratio", 0.38))),
    )
    if top < neckline_keepout_bottom:
        neckline_keepout_u8[
            top:neckline_keepout_bottom,
            max(left, int(cx - neckline_keepout_half)):min(right, int(cx + neckline_keepout_half)),
        ] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(neckline_keepout_u8))
    outer_gap_half = max(
        inner_keepout_half,
        int(face_w * float(getattr(self.config, "short_side_column_outer_strip_gap_ratio", 0.52))),
    )
    outer_strip_width = max(
        20,
        int(face_w * float(getattr(self.config, "short_side_column_outer_strip_width_scale", 0.96))),
    )
    strip_top = max(top, int(cutoff_y + face_h * 0.16))
    if strip_top < bottom:
        left_outer = max(left, int(x1 - face_w * 0.92))
        left_inner = min(right, left_outer + outer_strip_width)
        left_gap = max(left_inner, int(cx - outer_gap_half))
        right_outer = min(right, int(x2 + face_w * 0.92))
        right_inner = max(left, right_outer - outer_strip_width)
        right_gap = min(right_inner, int(cx + outer_gap_half))
        if left_outer < left_inner and left_outer < left_gap:
            outer_strip_u8[strip_top:bottom, left_outer:min(left_inner, left_gap)] = 255
        if right_gap < right_outer and right_inner < right_outer:
            outer_strip_u8[strip_top:bottom, max(right_inner, right_gap):right_outer] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, outer_strip_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
    )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    min_area = max(120, int(face_w * face_h * 0.010))
    max_area = max(9200, int(face_w * face_h * 0.34))
    min_height = max(72, int(face_h * 0.28))
    max_width = max(128, int(face_w * 0.78))
    max_offset = max(180, int(face_w * 0.98))
    min_side_offset = max(42, int(face_w * 0.42))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if bottom_y < int(cutoff_y + face_h * 0.24):
            continue
        if bottom_y > int(cutoff_y + face_h * 1.56):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        if abs(comp_cx - cx) < min_side_offset:
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 33)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(
        keep_u8,
        cv2.dilate(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        ),
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if int((outer_strip_u8 > 0).sum()) > 0:
        keep_u8 = cv2.bitwise_and(keep_u8, outer_strip_u8)
    keep_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=keep_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=80,
    )
    if int((outer_strip_u8 > 0).sum()) > 0:
        keep_u8 = cv2.bitwise_and(keep_u8, outer_strip_u8)
    if int((keep_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.2,
        sigmaY=7.4,
    ).astype(np.float32)

def _build_short_below_bob_cloth_restore_mask(
    self,
    removal_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    support_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if removal_mask is not None:
        base_shape = removal_mask.shape[:2]
    elif cloth_mask is not None:
        base_shape = cloth_mask.shape[:2]
    else:
        base_shape = (1, 1)
    if hair_length != "short":
        return np.zeros(base_shape, dtype=np.float32)
    if removal_mask is None or cloth_mask is None:
        return np.zeros(base_shape, dtype=np.float32)

    H, W = removal_mask.shape[:2]
    if cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)

    if support_mask is not None and support_mask.shape == (H, W):
        support_u8 = cv2.dilate(
            (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_or(zone_u8, cv2.bitwise_and(support_u8, cloth_u8))

    if int((zone_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    bob_floor = max(
        0,
        min(
            int(y2 + face_h * 0.10),
            int(cutoff_y + face_h * 0.08),
        ),
    )
    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    left = max(0, int(x1 - face_w * 1.26))
    right = min(W, int(x2 + face_w * 1.26))
    bottom = min(H, int(cutoff_y + face_h * 1.78))
    if bob_floor >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[bob_floor:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    lane_u8 = np.zeros((H, W), dtype=np.uint8)
    side_inner_gap = max(14, int(face_w * 0.08))
    left_lane_right = max(left + 1, int(cx - side_inner_gap))
    right_lane_left = min(right - 1, int(cx + side_inner_gap))
    lane_u8[bob_floor:bottom, left:left_lane_right] = 255
    lane_u8[bob_floor:bottom, right_lane_left:right] = 255
    center_lane_top = min(bottom, int(cutoff_y + face_h * 0.22))
    center_half = max(24, int(face_w * 0.28))
    center_x1 = max(left, int(cx - center_half))
    center_x2 = min(right, int(cx + center_half))
    if center_lane_top < bottom and center_x1 < center_x2:
        lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255
    deep_lane_top = min(bottom, int(cutoff_y + face_h * 0.30))
    deep_center_half = max(34, int(face_w * 0.40))
    deep_x1 = max(left, int(cx - deep_center_half))
    deep_x2 = min(right, int(cx + deep_center_half))
    if deep_lane_top < bottom and deep_x1 < deep_x2:
        lane_u8[deep_lane_top:bottom, deep_x1:deep_x2] = 255

    zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
    if int((zone_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
    )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    min_area = max(36, int(face_w * face_h * 0.002))
    max_area = max(12000, int(face_w * face_h * 0.28))
    min_height = max(18, int(face_h * 0.10))
    max_width = max(152, int(face_w * 0.98))
    center_keepout = max(16, int(face_w * 0.16))
    deep_center_bottom = int(y2 + face_h * 0.52)
    max_offset = max(170, int(face_w * 0.95))
    center_reject_offset = max(26, int(face_w * 0.30))

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if bottom_y < int(bob_floor + face_h * 0.10):
            continue
        if offset > max_offset:
            continue
        if offset < center_keepout and bottom_y < deep_center_bottom:
            continue
        if (
            offset <= center_reject_offset
            and w > max(84, int(face_w * 0.42))
            and area > max(1800, int(face_w * face_h * 0.07))
        ):
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 27)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, lane_u8)
    keep_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=keep_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=56,
    )
    if int((keep_u8 > 0).sum()) < 56:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.0,
        sigmaY=5.4,
    ).astype(np.float32)

def _build_short_below_bob_generation_block_mask(
    self,
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    support_mask: Optional[np.ndarray] = None,
    force_keep_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if removal_mask is None:
        return np.zeros((1, 1), dtype=np.float32)
    H, W = removal_mask.shape[:2]
    if hair_length != "short":
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23)),
        iterations=1,
    )
    force_keep_u8 = np.zeros((H, W), dtype=np.uint8)
    if force_keep_mask is not None and force_keep_mask.shape == (H, W):
        force_keep_u8 = cv2.dilate(
            (np.clip(force_keep_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 25)),
            iterations=1,
        )
    zone_u8 = removal_u8.copy()
    if support_mask is not None and support_mask.shape == (H, W):
        support_u8 = cv2.dilate(
            (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 27)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_or(zone_u8, support_u8)

    bob_floor = max(
        0,
        min(
            int(y2 + face_h * 0.00),
            int(cutoff_y + face_h * 0.01),
        ),
    )
    bottom = min(H, int(cutoff_y + face_h * 1.78))
    left = max(0, int(x1 - face_w * 1.36))
    right = min(W, int(x2 + face_w * 1.36))
    if bob_floor >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    corridor_u8[bob_floor:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    lane_u8 = np.zeros((H, W), dtype=np.uint8)
    side_inner_gap = max(10, int(face_w * 0.03))
    left_lane_right = max(left + 1, int(cx - side_inner_gap))
    right_lane_left = min(right - 1, int(cx + side_inner_gap))
    lane_u8[bob_floor:bottom, left:left_lane_right] = 255
    lane_u8[bob_floor:bottom, right_lane_left:right] = 255
    center_lane_top = min(bottom, int(cutoff_y + face_h * 0.16))
    center_half = max(24, int(face_w * 0.34))
    center_x1 = max(left, int(cx - center_half))
    center_x2 = min(right, int(cx + center_half))
    if center_lane_top < bottom and center_x1 < center_x2:
        lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
    if int((force_keep_u8 > 0).sum()) > 0:
        zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(force_keep_u8))
    if int((zone_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 35)),
    )
    zone_u8 = cv2.dilate(
        zone_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
        iterations=1,
    )
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    min_area = max(36, int(face_w * face_h * 0.0018))
    max_area = max(32000, int(face_w * face_h * 0.56))
    min_height = max(20, int(face_h * 0.10))
    max_width = max(248, int(face_w * 1.62))
    center_keepout = max(16, int(face_w * 0.12))
    deep_center_bottom = int(y2 + face_h * 0.20)

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if bottom_y < int(bob_floor + face_h * 0.10):
            continue
        if offset < center_keepout and bottom_y < deep_center_bottom:
            continue
        keep_u8[labels == idx] = 255

    if int((force_keep_u8 > 0).sum()) > 0:
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(force_keep_u8))

    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 41)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, lane_u8)
    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.2,
        sigmaY=5.2,
    ).astype(np.float32)

def _build_preclean_side_column_cleanup_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    base_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    if cloth_mask is None or base_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask.shape != (H, W) or base_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    base_u8 = (np.clip(base_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    zone_u8 = cv2.bitwise_and(base_u8, cloth_u8)
    if int((zone_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.04))
    bottom = min(H, int(cutoff_y + face_h * (1.18 if hair_length == "short" else 1.10)))
    left = max(0, int(x1 - face_w * 1.34))
    right = min(W, int(x2 + face_w * 1.34))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 21) if hair_length == "short" else (7, 17),
        ),
    )
    candidate_u8 = (
        (
            (
                (gray < (176.0 if hair_length == "short" else 170.0))
                & (blur < (182.0 if hair_length == "short" else 176.0))
                & (sat < (118.0 if hair_length == "short" else 112.0))
                & (lap < (28.0 if hair_length == "short" else 26.0))
            )
            | (blackhat > (9 if hair_length == "short" else 10))
        ).astype(np.uint8)
        * 255
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 13) if hair_length == "short" else (5, 11),
        ),
    )
    if int((candidate_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    max_area = max(2600, int(face_w * face_h * 0.12))
    max_width = max(44, int(face_w * 0.28))
    min_height = max(34, int(face_h * 0.18))
    max_offset = max(260, int(face_w * 1.14))
    center_offset = max(16, int(face_w * 0.14))
    center_max_width = max(58, int(face_w * 0.30))
    center_max_area = max(3400, int(face_w * face_h * 0.14))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        if area < 12 or area > max_area:
            continue
        if w > max_width or h < min_height:
            continue
        if y < int(cutoff_y + face_h * 0.02):
            continue
        if bottom_y < int(cutoff_y + face_h * 0.34):
            continue
        if bottom_y > int(cutoff_y + face_h * 1.02):
            continue
        if offset > max_offset:
            continue
        fill_ratio = float(area) / float(max(w * h, 1))
        if fill_ratio > 0.92 and area > max(80, int(face_w * face_h * 0.008)):
            continue
        if offset <= center_offset and (w > center_max_width or area > center_max_area):
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 15) if hair_length == "short" else (5, 11),
        ),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
    if int((keep_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=1.8,
        sigmaY=3.2,
    ).astype(np.float32)

def _build_preclean_cloth_hair_cleanup_mask(
    self,
    img_rgb: np.ndarray,
    removal_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if removal_mask is None or cloth_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if removal_mask.shape != (H, W) or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 13) if hair_length == "short" else (11, 11),
        ),
        iterations=1,
    )
    zone_u8 = cv2.bitwise_and(removal_u8, cloth_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.02))
    bottom = min(H, int(cutoff_y + face_h * (1.76 if hair_length == "short" else 1.08)))
    left = max(0, int(x1 - face_w * (1.46 if hair_length == "short" else 1.32)))
    right = min(W, int(x2 + face_w * (1.46 if hair_length == "short" else 1.32)))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 21) if hair_length == "short" else (7, 17),
        ),
    )
    darkness_u8 = (
        (
            (
                (gray < (208.0 if hair_length == "short" else 198.0))
                & (blur < (214.0 if hair_length == "short" else 206.0))
                & (sat < (142.0 if hair_length == "short" else 132.0))
            )
            | (blackhat > (7 if hair_length == "short" else 8))
        ).astype(np.uint8)
        * 255
    )
    guided_zone_u8 = cv2.bitwise_and(zone_u8, darkness_u8)
    if hair_length != "short" or int((guided_zone_u8 > 0).sum()) >= 120:
        zone_u8 = guided_zone_u8
    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 11) if hair_length == "short" else (5, 9),
        ),
    )
    if int((zone_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    min_area = max(80, int(face_w * face_h * 0.006))
    max_area = max(
        22000 if hair_length == "short" else 14000,
        int(face_w * face_h * (0.62 if hair_length == "short" else 0.26)),
    )
    min_height = max(44, int(face_h * (0.16 if hair_length == "short" else 0.16)))
    max_width = max(
        188 if hair_length == "short" else 136,
        int(face_w * (1.06 if hair_length == "short" else 0.62)),
    )
    max_offset = max(
        340 if hair_length == "short" else 280,
        int(face_w * (1.52 if hair_length == "short" else 1.22)),
    )
    center_guard_offset = max(18, int(face_w * 0.18))
    center_guard_width = max(108, int(face_w * 0.76))
    center_guard_area = max(5200, int(face_w * face_h * 0.34))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if y < int(cutoff_y + face_h * 0.02):
            continue
        if bottom_y < int(cutoff_y + face_h * 0.30):
            continue
        if bottom_y > int(cutoff_y + face_h * (1.72 if hair_length == "short" else 1.16)):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        fill_ratio = float(area) / float(max(w * h, 1))
        if (
            hair_length == "short"
            and abs(comp_cx - cx) <= center_guard_offset
            and (w > center_guard_width or area > center_guard_area)
        ):
            continue
        if fill_ratio > (0.96 if hair_length == "short" else 0.86) and area > max(180, int(face_w * face_h * 0.012)):
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 31) if hair_length == "short" else (5, 11),
        ),
        iterations=1,
    )
    keep_gate_u8 = cv2.bitwise_or(
        cloth_u8,
        cv2.dilate(
            removal_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (11, 19) if hair_length == "short" else (7, 13),
            ),
            iterations=1,
        ),
    )
    keep_u8 = cv2.bitwise_and(keep_u8, keep_gate_u8)
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.6 if hair_length == "short" else 2.0,
        sigmaY=6.2 if hair_length == "short" else 3.4,
    ).astype(np.float32)


def _build_short_generation_conditioning_cleanup_mask(
    self,
    *,
    source_garment_prepass_mask: Optional[np.ndarray],
    upper_body_repaint_seed_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    source_torso_hair_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    base_shape = None
    for mask in (
        cloth_mask,
        source_garment_prepass_mask,
        upper_body_repaint_seed_mask,
        removal_mask,
        source_torso_hair_mask,
        protect_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)
    if hair_length != "short":
        return np.zeros(base_shape, dtype=np.float32)

    H, W = base_shape
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if source_garment_prepass_mask is not None and source_garment_prepass_mask.shape != (H, W):
        source_garment_prepass_mask = None
    if upper_body_repaint_seed_mask is not None and upper_body_repaint_seed_mask.shape != (H, W):
        upper_body_repaint_seed_mask = None
    if removal_mask is not None and removal_mask.shape != (H, W):
        removal_mask = None
    if source_torso_hair_mask is not None and source_torso_hair_mask.shape != (H, W):
        source_torso_hair_mask = None
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(max(y2 + face_h * 0.04, cutoff_y - face_h * 0.02)))
    bottom = min(H, int(cutoff_y + face_h * 1.28))
    left = max(0, int(x1 - face_w * 1.08))
    right = min(W, int(x2 + face_w * 1.08))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    cloth_u8 = self._mask_to_u8(cloth_mask, threshold=0.04)
    if int((cloth_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)
    cloth_near_u8 = cv2.dilate(
        cloth_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(17, int(face_w * 0.42)), max(19, int(face_h * 0.30))),
        ),
        iterations=1,
    )
    cloth_core_u8 = cv2.dilate(
        cloth_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(9, int(face_w * 0.20)), max(11, int(face_h * 0.16))),
        ),
        iterations=1,
    )

    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    candidate_u8 = np.zeros((H, W), dtype=np.uint8)
    for mask, threshold, kernel in (
        (source_garment_prepass_mask, 0.08, (11, 17)),
        (upper_body_repaint_seed_mask, 0.08, (9, 13)),
    ):
        if mask is None:
            continue
        part_u8 = self._mask_to_u8(mask, threshold=threshold)
        if int((part_u8 > 0).sum()) == 0:
            continue
        part_u8 = cv2.dilate(
            part_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel),
            iterations=1,
        )
        part_u8 = cv2.bitwise_and(part_u8, corridor_u8)
        anchor_u8 = cv2.bitwise_or(anchor_u8, part_u8)
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(part_u8, cloth_core_u8))

    if removal_mask is not None:
        removal_u8 = self._mask_to_u8(removal_mask, threshold=0.08)
        if int((removal_u8 > 0).sum()) > 0:
            removal_u8 = cv2.bitwise_and(removal_u8, cloth_near_u8)
            removal_u8 = cv2.bitwise_and(removal_u8, corridor_u8)
            removal_u8 = cv2.morphologyEx(
                removal_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
            )
            anchor_u8 = cv2.bitwise_or(anchor_u8, removal_u8)
            candidate_u8 = cv2.bitwise_or(candidate_u8, removal_u8)

    if source_torso_hair_mask is not None:
        torso_hair_u8 = self._mask_to_u8(source_torso_hair_mask, threshold=0.08)
        if int((torso_hair_u8 > 0).sum()) > 0:
            torso_hair_u8 = cv2.bitwise_and(torso_hair_u8, cloth_near_u8)
            torso_hair_u8 = cv2.bitwise_and(torso_hair_u8, corridor_u8)
            if int((anchor_u8 > 0).sum()) > 0:
                torso_gate_u8 = cv2.dilate(
                    anchor_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 27)),
                    iterations=1,
                )
                torso_hair_u8 = cv2.bitwise_and(torso_hair_u8, torso_gate_u8)
            candidate_u8 = cv2.bitwise_or(candidate_u8, torso_hair_u8)

    if int((anchor_u8 > 0).sum()) > 0:
        fill_support_u8 = cv2.bitwise_and(
            cloth_core_u8,
            cv2.dilate(
                anchor_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 25)),
                iterations=1,
            ),
        )
        candidate_u8 = cv2.bitwise_or(candidate_u8, fill_support_u8)

        anchor_ys, anchor_xs = np.where(anchor_u8 > 0)
        if anchor_xs.size > 0 and anchor_ys.size > 0:
            anchor_gate_u8 = np.zeros((H, W), dtype=np.uint8)
            anchor_left = max(0, int(np.percentile(anchor_xs, 2)) - max(14, int(face_w * 0.24)))
            anchor_right = min(W, int(np.percentile(anchor_xs, 98)) + max(14, int(face_w * 0.24)))
            anchor_top = max(top, int(anchor_ys.min()) - max(8, int(face_h * 0.10)))
            anchor_bottom = min(bottom, int(anchor_ys.max()) + max(12, int(face_h * 0.18)))
            if anchor_top < anchor_bottom and anchor_left < anchor_right:
                anchor_gate_u8[anchor_top:anchor_bottom, anchor_left:anchor_right] = 255
                candidate_u8 = cv2.bitwise_and(candidate_u8, anchor_gate_u8)

    candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
    candidate_u8 = cv2.bitwise_and(candidate_u8, cloth_near_u8)

    if protect_mask is not None:
        protect_u8 = self._mask_to_u8(protect_mask, threshold=0.10)
        if int((protect_u8 > 0).sum()) > 0:
            protect_u8 = cv2.dilate(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                iterations=1,
            )
            candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(protect_u8))

    if int((candidate_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 21)),
    )
    candidate_u8 = cv2.dilate(
        candidate_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, cloth_near_u8)

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    min_area = max(48, int(face_w * face_h * 0.0012))
    min_height = max(14, int(face_h * 0.12))
    max_width = max(96, int(face_w * 1.26))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate_u8 > 0).astype(np.uint8),
        8,
    )
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or height < min_height or width > max_width:
            continue
        filtered_u8[labels == label] = 255

    if int((filtered_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        filtered_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.0,
        sigmaY=5.2,
    ).astype(np.float32)

def _build_residual_strand_cleanup_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if cloth_mask is None or removal_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    zone_u8 = cv2.bitwise_and(cloth_u8, removal_u8)
    if int((zone_u8 > 0).sum()) < 30:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * (0.16 if hair_length == "short" else 0.12)))
    bottom = min(H, int(cutoff_y + face_h * (1.18 if hair_length == "short" else 1.06)))
    left = max(0, int(x1 - face_w * 1.18))
    right = min(W, int(x2 + face_w * 1.18))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 30:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=4.2, sigmaY=4.2)
    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13) if hair_length == "short" else (5, 11)),
    )
    candidate_u8 = (
        (
            ((gray < (192.0 if hair_length == "short" else 188.0)) & ((blur - gray) > (1.4 if hair_length == "short" else 1.7)))
            | (blackhat > (5 if hair_length == "short" else 6))
        ).astype(np.uint8)
        * 255
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(final_hair_u8))
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    if int((candidate_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    max_area = max(900, int(face_w * face_h * 0.060))
    max_width = max(24, int(face_w * 0.18))
    max_height = max(96, int(face_h * 0.42))
    max_offset = max(160, int(face_w * 0.80))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        if area < 8 or area > max_area:
            continue
        if w > max_width or h > max_height:
            continue
        if bottom_y < int(cutoff_y + face_h * 0.30):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        fill_ratio = float(area) / float(max(w * h, 1))
        if fill_ratio > 0.74 and area > 36:
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
    if int((keep_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=1.8,
        sigmaY=2.8,
    ).astype(np.float32)

def _build_final_hair_lane_cleanup_mask(
    self,
    final_hair_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    debug_info: Optional[Dict[str, Any]] = None,
    debug_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
        return np.zeros(shape, dtype=np.float32)
    if final_hair_mask is None or cloth_mask is None or removal_mask is None:
        shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
        return np.zeros(shape, dtype=np.float32)

    H, W = final_hair_mask.shape[:2]
    if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    hair_u8 = cv2.dilate(
        (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.22).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        iterations=1,
    )
    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    zone_u8 = cv2.bitwise_and(hair_u8, cloth_u8)
    zone_u8 = cv2.bitwise_and(zone_u8, removal_u8)
    if int((zone_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.10))
    bottom = min(H, int(cutoff_y + face_h * 1.42))
    left = max(0, int(x1 - face_w * 1.18))
    right = min(W, int(x2 + face_w * 1.18))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    neckline_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    outer_strip_u8 = corridor_u8.copy()
    if hair_length == "short":
        center_keepout_half = max(
            28,
            int(face_w * float(getattr(self.config, "final_hair_lane_center_keepout_ratio", 0.20))),
        )
        neckline_keepout_half = max(center_keepout_half + 12, int(face_w * 0.42))
        center_keepout_top = max(top, int(cutoff_y + face_h * 0.18))
        center_keepout_bottom = min(bottom, int(cutoff_y + face_h * 1.08))
        if center_keepout_top < center_keepout_bottom:
            center_keepout_u8[
                center_keepout_top:center_keepout_bottom,
                max(left, int(cx - center_keepout_half)):min(right, int(cx + center_keepout_half)),
            ] = 255
            zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(center_keepout_u8))
        neckline_keepout_bottom = min(
            bottom,
            int(cutoff_y + face_h * float(getattr(self.config, "final_hair_lane_neckline_keepout_ratio", 0.30))),
        )
        if top < neckline_keepout_bottom:
            neckline_keepout_u8[
                top:neckline_keepout_bottom,
                max(left, int(cx - neckline_keepout_half)):min(right, int(cx + neckline_keepout_half)),
            ] = 255
            zone_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_not(neckline_keepout_u8))
        outer_gap_half = max(
            center_keepout_half + 8,
            int(face_w * float(getattr(self.config, "final_hair_lane_outer_strip_gap_ratio", 0.30))),
        )
        outer_strip_u8 = np.zeros((H, W), dtype=np.uint8)
        strip_top = max(top, int(cutoff_y + face_h * 0.22))
        if strip_top < bottom:
            left_outer = max(left, int(x1 - face_w * 0.86))
            left_inner = min(right, int(cx - outer_gap_half))
            right_inner = max(left, int(cx + outer_gap_half))
            right_outer = min(right, int(x2 + face_w * 0.86))
            if left_outer < left_inner:
                outer_strip_u8[strip_top:bottom, left_outer:left_inner] = 255
            if right_inner < right_outer:
                outer_strip_u8[strip_top:bottom, right_inner:right_outer] = 255
            zone_u8 = cv2.bitwise_and(zone_u8, outer_strip_u8)
    if int((zone_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)),
    )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    max_area = max(3200, int(face_w * face_h * 0.18))
    max_width = max(74, int(face_w * 0.44))
    min_height = max(22, int(face_h * 0.10))
    max_offset = max(260, int(face_w * 1.16))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        if area < 20 or area > max_area:
            continue
        if w > max_width or h < min_height:
            continue
        if bottom_y < int(cutoff_y + face_h * 0.20):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if hair_length == "short" and int((outer_strip_u8 > 0).sum()) > 0:
        keep_u8 = cv2.bitwise_and(keep_u8, outer_strip_u8)
    if debug_info is not None:
        debug_info.update({
            "zone_px": int((zone_u8 > 0).sum()),
            "keep_px": int((keep_u8 > 0).sum()),
            "corridor_top": int(top),
            "corridor_bottom": int(bottom),
            "corridor_left": int(left),
            "corridor_right": int(right),
            "center_keepout_px": int((center_keepout_u8 > 0).sum()),
            "neckline_keepout_px": int((neckline_keepout_u8 > 0).sum()),
            "outer_strip_px": int((outer_strip_u8 > 0).sum()),
            "gaussian_sigma_x": 2.2,
            "gaussian_sigma_y": 3.4,
        })
    if debug_masks is not None:
        debug_masks["zone"] = zone_u8.astype(np.float32) / 255.0
        debug_masks["center_keepout"] = center_keepout_u8.astype(np.float32) / 255.0
        debug_masks["neckline_keepout"] = neckline_keepout_u8.astype(np.float32) / 255.0
        debug_masks["outer_strip"] = outer_strip_u8.astype(np.float32) / 255.0
        debug_masks["keep"] = keep_u8.astype(np.float32) / 255.0
    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.2,
        sigmaY=3.4,
    ).astype(np.float32)

def _build_short_bob_tail_suppress_mask(
    self,
    img_rgb: np.ndarray,
    final_hair_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length != "short":
        shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
        return np.zeros(shape, dtype=np.float32)
    if final_hair_mask is None:
        shape = final_hair_mask.shape[:2] if isinstance(final_hair_mask, np.ndarray) else (0, 0)
        return np.zeros(shape, dtype=np.float32)

    H, W = final_hair_mask.shape[:2]
    if img_rgb.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if removal_mask is not None and removal_mask.shape != (H, W):
        removal_mask = None
    if cloth_mask is not None and cloth_mask.shape != (H, W):
        cloth_mask = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    hair_u8 = cv2.dilate(
        (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.14).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
        iterations=1,
    )
    removal_u8 = None
    if removal_mask is not None:
        removal_u8 = cv2.dilate(
            (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)),
            iterations=1,
        )
    cloth_u8 = None
    if cloth_mask is not None:
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        )

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(max(y2 + face_h * 0.04, cutoff_y + face_h * 0.04)))
    bottom = min(H, int(cutoff_y + face_h * 1.56))
    left = max(0, int(x1 - face_w * 1.46))
    right = min(W, int(x2 + face_w * 1.46))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    lane_u8 = np.zeros((H, W), dtype=np.uint8)
    left_lane_left = max(left, int(x1 - face_w * 0.82))
    left_lane_right = min(right, int(x1 + face_w * 0.20))
    right_lane_left = max(left, int(x2 - face_w * 0.20))
    right_lane_right = min(right, int(x2 + face_w * 0.82))
    if left_lane_left < left_lane_right:
        lane_u8[top:bottom, left_lane_left:left_lane_right] = 255
    if right_lane_left < right_lane_right:
        lane_u8[top:bottom, right_lane_left:right_lane_right] = 255
    center_lane_top = min(bottom, int(y2 + face_h * 0.16))
    center_half = max(28, int(face_w * 0.28))
    center_x1 = max(left, int(cx - center_half))
    center_x2 = min(right, int(cx + center_half))
    if center_lane_top < bottom and center_x1 < center_x2:
        lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=4.8, sigmaY=6.4)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
    )
    dark_tail_u8 = (
        (
            ((gray < 172.0) & (blur < 180.0) & (sat < 132.0))
            | (blackhat > 7)
        ).astype(np.uint8)
        * 255
    )

    candidate_u8 = cv2.bitwise_and(hair_u8, dark_tail_u8)
    if removal_u8 is not None:
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(dark_tail_u8, removal_u8))
    else:
        candidate_u8 = cv2.bitwise_or(candidate_u8, dark_tail_u8)
    hair_tail_u8 = cv2.bitwise_and(hair_u8, corridor_u8)
    if cloth_u8 is not None:
        cloth_support_u8 = cv2.dilate(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)),
            iterations=1,
        )
        hair_tail_u8 = cv2.bitwise_and(hair_tail_u8, cloth_support_u8)
        deep_lane_u8 = lane_u8.copy()
        deep_lane_u8[:max(0, int(y2 + face_h * 0.08)), :] = 0
        candidate_u8 = cv2.bitwise_and(
            candidate_u8,
            cv2.bitwise_or(cloth_support_u8, deep_lane_u8),
        )
    hair_tail_u8 = cv2.bitwise_and(hair_tail_u8, lane_u8)
    if removal_u8 is not None:
        hair_tail_u8 = cv2.bitwise_or(
            hair_tail_u8,
            cv2.bitwise_and(
                cv2.dilate(
                    removal_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 47)),
                    iterations=1,
                ),
                cv2.bitwise_and(hair_u8, corridor_u8),
            ),
        )
    lower_keepout = min(H, int(y2 + face_h * 0.06))
    if lower_keepout < H:
        hair_tail_u8[:lower_keepout, :] = 0
    hair_tail_u8 = cv2.morphologyEx(
        hair_tail_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
    )
    candidate_u8 = cv2.bitwise_or(candidate_u8, hair_tail_u8)
    candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
    candidate_u8 = cv2.bitwise_and(candidate_u8, lane_u8)
    if int((candidate_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    lower_start = min(H, int(y2 + face_h * 0.12))
    if lower_start < H:
        candidate_u8[:lower_start, :] = 0
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
    )
    if int((candidate_u8 > 0).sum()) < 36:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    center_keepout = max(12, int(face_w * 0.12))
    min_height = max(16, int(face_h * 0.08))
    min_area = max(24, int(face_w * face_h * 0.0011))
    max_area = max(24000, int(face_w * face_h * 0.48))
    max_width = max(172, int(face_w * 1.14))
    deep_bottom = int(y2 + face_h * 0.18)
    deepest_bottom = int(y2 + face_h * 0.28)
    max_offset = max(float(face_w * 1.30), 1.0)

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        if area < min_area or h < min_height:
            continue
        if area > max_area or w > max_width:
            continue
        if bottom_y < deep_bottom:
            continue

        side_ratio = float(np.clip((offset - center_keepout) / max(max_offset - center_keepout, 1.0), 0.0, 1.0))
        truncate_y = int(y2 + face_h * (0.06 + 0.10 * side_ratio))
        truncate_y = max(truncate_y, lower_start)
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        if truncate_y > 0:
            comp_u8[:truncate_y, :] = 0
        lane_overlap = int((cv2.bitwise_and(comp_u8, lane_u8) > 0).sum())
        if cloth_u8 is not None:
            cloth_overlap = int((cv2.bitwise_and(comp_u8, cloth_u8) > 0).sum())
            if cloth_overlap < 8 and lane_overlap < max(16, int(area * 0.08)) and bottom_y < int(cutoff_y + face_h * 0.72):
                continue
            if offset < center_keepout and bottom_y < deepest_bottom and cloth_overlap < 8:
                continue
            if cloth_overlap > 0:
                comp_u8 = cv2.dilate(
                    comp_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
                    iterations=1,
                )
        elif offset < center_keepout and bottom_y < deepest_bottom:
            continue
        keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.0,
        sigmaY=6.2,
    ).astype(np.float32)

def _build_short_final_side_lane_refine_mask(
    self,
    img_rgb: np.ndarray,
    final_hair_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length != "short":
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if removal_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if final_hair_mask is not None and final_hair_mask.shape != (H, W):
        final_hair_mask = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(max(float(y2) + face_h * 0.04, cutoff_y + face_h * 0.08)))
    bottom = min(H, int(y2 + face_h * 1.26))
    left = max(0, int(x1 - face_w * 1.34))
    right = min(W, int(x2 + face_w * 1.34))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    hair_u8 = np.zeros((H, W), dtype=np.uint8)
    if final_hair_mask is not None:
        hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
            iterations=1,
        )
    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    )
    candidate_u8 = cv2.bitwise_and(hair_u8, corridor_u8)
    candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(hair_u8, removal_u8))

    side_tail_u8 = (
        self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        ) > 0.08
    ).astype(np.uint8) * 255
    candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(side_tail_u8, corridor_u8))

    short_bob_tail_u8 = np.zeros((H, W), dtype=np.uint8)
    if final_hair_mask is not None:
        short_bob_tail_u8 = (
            self._build_short_bob_tail_suppress_mask(
                img_rgb=img_rgb,
                final_hair_mask=final_hair_mask,
                cloth_mask=cloth_mask,
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            ) > 0.08
        ).astype(np.uint8) * 255
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(short_bob_tail_u8, corridor_u8))

    dark_tail_u8 = (
        self._build_dark_tail_residual_mask(
            img_rgb=img_rgb,
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        ) > 0.08
    ).astype(np.uint8) * 255
    if int((dark_tail_u8 > 0).sum()) > 0:
        dark_tail_u8 = cv2.dilate(
            dark_tail_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_or(
            candidate_u8,
            cv2.bitwise_and(cv2.bitwise_and(dark_tail_u8, removal_u8), corridor_u8),
        )

    cloth_u8 = np.zeros((H, W), dtype=np.uint8)
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        lower_tail_u8 = (
            self._build_short_lower_tail_cleanup_mask(
                img_rgb=img_rgb,
                cloth_mask=cloth_mask,
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
                final_hair_mask=final_hair_mask,
            ) > 0.08
        ).astype(np.uint8) * 255
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(lower_tail_u8, corridor_u8))

    if int((candidate_u8 > 0).sum()) > 0 and int((cloth_u8 > 0).sum()) > 0:
        cloth_support_u8 = cv2.bitwise_and(
            cv2.dilate(
                candidate_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 25)),
                iterations=1,
            ),
            cloth_u8,
        )
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(cloth_support_u8, corridor_u8))

    lower_keepout = min(H, int(y2 + face_h * 0.10))
    if lower_keepout < H:
        candidate_u8[:lower_keepout, :] = 0

    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
    if int((candidate_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    min_area = max(28, int(face_w * face_h * 0.0014))
    max_area = max(18000, int(face_w * face_h * 0.36))
    min_height = max(18, int(face_h * 0.08))
    max_width = max(156, int(face_w * 0.94))
    center_keepout = max(14, int(face_w * 0.12))
    deep_start = int(y2 + face_h * 0.16)
    deep_center_start = int(y2 + face_h * 0.42)

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if y < top or bottom_y < deep_start:
            continue
        cloth_overlap = int((cv2.bitwise_and((labels == idx).astype(np.uint8) * 255, cloth_u8) > 0).sum())
        if cloth_overlap < max(12, int(area * 0.06)) and bottom_y < int(cutoff_y + face_h * 0.74):
            continue
        if offset < center_keepout and bottom_y < deep_center_start and cloth_overlap < 10:
            continue
        if offset < center_keepout and cloth_overlap < 24:
            continue
        if offset < center_keepout and w > max(76, int(face_w * 0.42)) and cloth_overlap < 24:
            continue

        comp_u8 = (labels == idx).astype(np.uint8) * 255
        if int((cloth_u8 > 0).sum()) > 0 and cloth_overlap > 0:
            comp_u8 = cv2.dilate(
                comp_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
                iterations=1,
            )
        keep_u8 = cv2.bitwise_or(keep_u8, cv2.bitwise_and(comp_u8, corridor_u8))

    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.4,
        sigmaY=5.8,
    ).astype(np.float32)

def _build_short_lower_tail_cleanup_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length != "short":
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if cloth_mask is None or removal_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if final_hair_mask is not None and final_hair_mask.shape != (H, W):
        final_hair_mask = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    hair_u8 = np.zeros((H, W), dtype=np.uint8)
    if final_hair_mask is not None:
        hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.14).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 27)),
            iterations=1,
        )
    zone_u8 = cv2.bitwise_and(cloth_u8, cv2.bitwise_or(removal_u8, hair_u8))
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, min(int(cutoff_y - face_h * 0.02), int(y2 - face_h * 0.04)))
    bottom = min(H, int(cutoff_y + face_h * 1.82))
    left = max(0, int(x1 - face_w * 1.52))
    right = min(W, int(x2 + face_w * 1.52))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    lane_u8 = np.zeros((H, W), dtype=np.uint8)
    left_lane_left = max(left, int(x1 - face_w * 0.78))
    left_lane_right = min(right, int(x1 + face_w * 0.16))
    right_lane_left = max(left, int(x2 - face_w * 0.16))
    right_lane_right = min(right, int(x2 + face_w * 0.78))
    if left_lane_left < left_lane_right:
        lane_u8[top:bottom, left_lane_left:left_lane_right] = 255
    if right_lane_left < right_lane_right:
        lane_u8[top:bottom, right_lane_left:right_lane_right] = 255
    center_lane_top = min(bottom, int(y2 + face_h * 0.42))
    center_half = max(14, int(face_w * 0.10))
    center_x1 = max(left, int(cx - center_half))
    center_x2 = min(right, int(cx + center_half))
    if center_lane_top < bottom and center_x1 < center_x2:
        lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255
    lane_zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
    if int((lane_zone_u8 > 0).sum()) >= 60:
        zone_u8 = lane_zone_u8

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 23)),
    )

    ring_u8 = cv2.subtract(
        cv2.dilate(zone_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39)), iterations=1),
        cv2.dilate(zone_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1),
    )
    ring_u8 = cv2.bitwise_and(ring_u8, cloth_u8)
    ring_u8 = cv2.bitwise_and(ring_u8, cv2.bitwise_not(cv2.bitwise_or(removal_u8, hair_u8)))
    ring_u8 = cv2.bitwise_and(ring_u8, corridor_u8)
    if int((ring_u8 > 0).sum()) < 60:
        ring_u8 = cv2.bitwise_and(cv2.bitwise_and(cloth_u8, corridor_u8), cv2.bitwise_not(zone_u8))
    if int((ring_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    ref_gray = float(np.median(gray[ring_u8 > 0]))
    ref_sat = float(np.median(sat[ring_u8 > 0]))
    if ref_gray < 142.0:
        fallback_ring_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
        fallback_ring_u8 = cv2.bitwise_and(fallback_ring_u8, cv2.bitwise_not(zone_u8))
        if int((fallback_ring_u8 > 0).sum()) >= 60:
            ref_gray = max(ref_gray, float(np.median(gray[fallback_ring_u8 > 0])))
            ref_sat = min(ref_sat, float(np.median(sat[fallback_ring_u8 > 0])))
    if ref_gray < 120.0 and int((hair_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    tone_ceiling = max(108.0, ref_gray - 16.0)
    blur_ceiling = max(118.0, ref_gray - 8.0)
    sat_ceiling = max(118.0, ref_sat + 24.0)
    candidate_u8 = (
        (
            (
                (gray < tone_ceiling)
                & (blur < blur_ceiling)
                & (sat < sat_ceiling)
                & (lap < 52.0)
            )
            | (blackhat > 9)
        ).astype(np.uint8)
        * 255
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
    if int((hair_u8 > 0).sum()) > 0:
        hair_dark_seed_u8 = cv2.bitwise_and(
            hair_u8,
            cv2.bitwise_and(
                zone_u8,
                cv2.bitwise_or(
                    ((gray < (tone_ceiling + 10.0)).astype(np.uint8) * 255),
                    (blackhat > 8).astype(np.uint8) * 255,
                ),
            ),
        )
        candidate_u8 = cv2.bitwise_or(candidate_u8, hair_dark_seed_u8)
        candidate_u8 = cv2.bitwise_or(
            candidate_u8,
            cv2.bitwise_and((blackhat > 8).astype(np.uint8) * 255, hair_dark_seed_u8),
        )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 23)),
    )
    if int((candidate_u8 > 0).sum()) < 50:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    min_area = max(80, int(face_w * face_h * 0.004))
    max_area = max(26000, int(face_w * face_h * 0.62))
    min_height = max(40, int(face_h * 0.16))
    max_width = max(138, int(face_w * 0.82))
    max_offset = max(360, int(face_w * 1.64))
    center_keepout = max(16, int(face_w * 0.16))
    center_max_width = max(96, int(face_w * 0.50))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if y < top:
            continue
        if bottom_y < int(cutoff_y + face_h * 0.28):
            continue
        if offset > max_offset:
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        cloth_overlap = int((cv2.bitwise_and(comp_u8, cloth_u8) > 0).sum())
        hair_overlap = int((cv2.bitwise_and(comp_u8, hair_u8) > 0).sum())
        lane_overlap = int((cv2.bitwise_and(comp_u8, lane_u8) > 0).sum())
        if offset <= center_keepout and w > center_max_width and hair_overlap < 20:
            continue
        if lane_overlap < max(20, int(area * 0.12)) and hair_overlap < 20 and bottom_y < int(cutoff_y + face_h * 0.82):
            continue
        if cloth_overlap < 10 and hair_overlap < 10:
            continue
        if cloth_overlap > 0 or hair_overlap > 0:
            comp_u8 = cv2.dilate(
                comp_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21)),
                iterations=1,
            )
        keep_u8 = cv2.bitwise_or(keep_u8, cv2.bitwise_and(comp_u8, corridor_u8))

    if int((keep_u8 > 0).sum()) < 50:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 25)),
        iterations=1,
    )
    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if int((keep_u8 > 0).sum()) < 50:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.2,
        sigmaY=6.6,
    ).astype(np.float32)

def _build_dark_lane_cleanup_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if cloth_mask is None or removal_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    zone_u8 = cv2.bitwise_and(cloth_u8, removal_u8)
    if int((zone_u8 > 0).sum()) < 30:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.18))
    bottom = min(H, int(cutoff_y + face_h * 1.24))
    left = max(0, int(x1 - face_w * 1.04))
    right = min(W, int(x2 + face_w * 1.04))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 30:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    candidate_u8 = (
        (gray < (170.0 if hair_length == "short" else 164.0))
        & (sat < (95.0 if hair_length == "short" else 100.0))
        & (lap < (20.0 if hair_length == "short" else 22.0))
    ).astype(np.uint8) * 255
    candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    if int((candidate_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    max_area = max(3000, int(face_w * face_h * 0.12))
    max_width = max(82, int(face_w * 0.42))
    min_height = max(26, int(face_h * 0.12))
    max_offset = max(260, int(face_w * 1.08))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        if area < 24 or area > max_area:
            continue
        if w > max_width or h < min_height:
            continue
        if bottom_y < int(cutoff_y + face_h * 0.30):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 15)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=1.8,
        sigmaY=3.0,
    ).astype(np.float32)

def _build_side_tail_cleanup_mask(
    self,
    removal_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str = "short",
) -> np.ndarray:
    """
    bob 아래로 남는 side-tail blob만 분리한 cleanup 힌트 마스크.
    넓은 가로 band는 제외하고, 옆으로 내려오는 잔머리 성분만 유지한다.
    """
    H, W = removal_mask.shape[:2]
    if removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    base = np.clip(removal_mask.astype(np.float32), 0.0, 1.0).copy()
    top_keep = max(0, int(cutoff_y - face_h * 0.04))
    base[:top_keep, :] = 0.0

    base_thresh = 0.42 if hair_length == "short" else 0.48
    base_u8 = (base > base_thresh).astype(np.uint8) * 255
    if int((base_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    corridor_ratio = 1.45 if hair_length == "short" else 1.35
    x_min = max(0, int(x1 - face_w * corridor_ratio))
    x_max = min(W, int(x2 + face_w * corridor_ratio))
    if x_min >= x_max:
        return np.zeros((H, W), dtype=np.float32)

    corridor = np.zeros((H, W), dtype=np.uint8)
    corridor[:, x_min:x_max] = 255
    base_u8 = cv2.bitwise_and(base_u8, corridor)
    if int((base_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_u8, 8)
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    width_cap = max(42, int(face_w * (1.55 if hair_length == "short" else 1.75)))
    min_height = max(18, int(face_h * (0.16 if hair_length == "short" else 0.20)))
    min_area = max(120, int(face_w * face_h * (0.003 if hair_length == "short" else 0.004)))
    center_keepout_half = max(18, int(face_w * 0.26))
    deep_start = min(H, int(cutoff_y + face_h * (0.14 if hair_length == "short" else 0.18)))
    deep_grace = max(20, int(face_h * 0.18))

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area or h < min_height or w > width_cap:
            continue

        comp_cx = float(centroids[idx][0])
        y_max = int(y + h)
        is_side_component = abs(comp_cx - cx) >= center_keepout_half
        is_deep_component = y_max >= (deep_start + deep_grace)
        if not is_side_component and not is_deep_component:
            continue

        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 13) if hair_length == "short" else (7, 11),
    )
    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 17) if hair_length == "short" else (9, 13),
    )
    keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_CLOSE, close_k)
    keep_u8 = cv2.dilate(keep_u8, dilate_k, iterations=1)
    keep_u8[:top_keep, :] = 0

    return (keep_u8 > 0).astype(np.float32)

def _build_short_tail_core_mask(
    self,
    removal_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str = "short",
) -> np.ndarray:
    """
    side-tail cleanup 영역에서 중앙 폭만 남긴 좁은 core mask.
    넓은 어깨/옷 영역 재생성을 피하면서 하단 잔머리 꼬리만 다시 칠하도록 쓴다.
    """
    side_tail = self._build_side_tail_cleanup_mask(
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length=hair_length,
    )
    H, W = side_tail.shape[:2]
    if removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    side_u8 = (side_tail > 0.08).astype(np.uint8) * 255
    if int((side_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    deep_start = min(
        H,
        int(cutoff_y + face_h * (0.04 if hair_length == "short" else 0.10)),
    )
    min_height = max(16, int(face_h * (0.10 if hair_length == "short" else 0.14)))
    min_area = max(60, int(face_w * face_h * (0.0012 if hair_length == "short" else 0.0018)))
    core_u8 = np.zeros((H, W), dtype=np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(side_u8, 8)
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area or h < min_height:
            continue

        comp_y1 = max(int(y), deep_start)
        comp_y2 = min(H, int(y + h))
        if comp_y2 - comp_y1 < min_height:
            continue

        comp_cx = float(centroids[idx][0])
        core_half = max(
            10,
            min(
                int(round(w * (0.20 if hair_length == "short" else 0.26))),
                int(face_w * (0.16 if hair_length == "short" else 0.22)),
            ),
        )
        rect_x1 = max(0, int(round(comp_cx - core_half)))
        rect_x2 = min(W, int(round(comp_cx + core_half)))
        if rect_x2 <= rect_x1:
            continue

        comp_u8 = (labels == idx).astype(np.uint8) * 255
        rect_u8 = np.zeros((H, W), dtype=np.uint8)
        rect_u8[comp_y1:comp_y2, rect_x1:rect_x2] = 255
        comp_core_u8 = cv2.bitwise_and(comp_u8, rect_u8)
        if int((comp_core_u8 > 0).sum()) < 24:
            continue

        comp_core_u8 = cv2.dilate(
            comp_core_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 17) if hair_length == "short" else (7, 13),
            ),
            iterations=1,
        )
        core_u8 = cv2.bitwise_or(core_u8, comp_core_u8)

    return (core_u8 > 0).astype(np.float32)

def _build_front_strand_cleanup_mask(
    self,
    removal_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str = "short",
    anchor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    셔츠 앞쪽 중앙으로 떨어진 얇은 front strand를 위한 cleanup 마스크.
    side-tail 로직과 분리해서, 얼굴 중앙 아래의 가는 세로 성분만 보수적으로 남긴다.
    """
    H, W = removal_mask.shape[:2]
    if removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))
    target_cx = cx
    anchor_half_override = 0
    anchor_bottom = None
    lower_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        anchor_u8[:max(0, int(cutoff_y - face_h * 0.06)), :] = 0
        if int((anchor_u8 > 0).sum()) > 0:
            lower_anchor_u8 = anchor_u8.copy()
            lower_anchor_u8[:min(H, int(cutoff_y + face_h * (0.18 if hair_length == "short" else 0.12))), :] = 0
            focus_u8 = lower_anchor_u8 if int((lower_anchor_u8 > 0).sum()) >= 10 else anchor_u8
            ys, xs = np.where(focus_u8 > 0)
            if ys.size and xs.size:
                target_cx = int(np.clip(round(float(xs.mean())), 0, max(W - 1, 0)))
                anchor_bottom = int(ys.max()) + 1
                anchor_half_override = max(
                    14,
                    min(
                        max(28, int(face_w * 0.34)),
                        int(max(xs.max() - xs.min() + 1, 12) * 0.85),
                    ),
                )

    zone = np.clip(removal_mask.astype(np.float32), 0.0, 1.0)
    zone[:max(0, int(cutoff_y - face_h * (0.10 if hair_length == "short" else 0.06))), :] = 0.0
    zone_u8 = (zone > 0.08).astype(np.uint8) * 255
    if int((zone_u8 > 0).sum()) < 30:
        return np.zeros((H, W), dtype=np.float32)

    front_half = max(14, int(face_w * (0.24 if hair_length == "short" else 0.18)))
    if anchor_half_override > 0:
        front_half = max(front_half, anchor_half_override)
    front_x1 = max(0, target_cx - front_half)
    front_x2 = min(W, target_cx + front_half)
    front_y1 = max(0, int(cutoff_y - face_h * (0.10 if hair_length == "short" else 0.06)))
    front_y2 = min(H, int(cutoff_y + face_h * (1.02 if hair_length == "short" else 0.82)))
    probe_y2 = min(H, int(cutoff_y + face_h * (1.34 if hair_length == "short" else 1.02)))
    if anchor_bottom is not None:
        front_y2 = min(H, max(front_y2, int(anchor_bottom + face_h * 0.10)))
    if front_x1 < front_x2 and front_y1 < probe_y2:
        front_probe_u8 = np.zeros((H, W), dtype=np.uint8)
        front_probe_u8[front_y1:probe_y2, front_x1:front_x2] = 255
        front_tail_u8 = cv2.bitwise_and(zone_u8, front_probe_u8)
        ys = np.where(front_tail_u8 > 0)[0]
        if ys.size:
            detected_bottom = int(ys.max()) + 1
            front_y2 = min(H, max(front_y2, int(detected_bottom + face_h * 0.06)))
    if front_x1 >= front_x2 or front_y1 >= front_y2:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    corridor_u8[front_y1:front_y2, front_x1:front_x2] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
    if int((zone_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)
    if hair_length == "short":
        filtered_u8 = np.zeros((H, W), dtype=np.uint8)
        anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
        if int((lower_anchor_u8 > 0).sum()) > 0:
            anchor_lane_u8 = cv2.dilate(
                lower_anchor_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21)),
                iterations=1,
            )
            anchor_lane_u8 = cv2.bitwise_and(anchor_lane_u8, corridor_u8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
        max_comp_width = max(22, int(face_w * 0.24))
        max_comp_area = max(160, int(face_w * face_h * 0.030))
        min_comp_height = max(18, int(face_h * 0.10))
        center_half = max(16, int(face_w * 0.22))
        if anchor_half_override > 0:
            center_half = max(center_half, anchor_half_override + 6)
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            comp_cx = float(centroids[idx][0])
            if area < 10 or area > max_comp_area:
                continue
            if w > max_comp_width or h < min_comp_height:
                continue
            if abs(comp_cx - target_cx) > center_half:
                continue
            if (y + h) > front_y2:
                continue
            filtered_u8[labels == idx] = 255
        if int((filtered_u8 > 0).sum()) < 12:
            for idx in range(1, num_labels):
                w = int(stats[idx, cv2.CC_STAT_WIDTH])
                h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                area = int(stats[idx, cv2.CC_STAT_AREA])
                comp_cx = float(centroids[idx][0])
                if area < 12 or area > max(220, int(face_w * face_h * 0.040)):
                    continue
                if w > max(28, int(face_w * 0.30)) or h < max(20, int(face_h * 0.10)):
                    continue
                if abs(comp_cx - target_cx) > max(18, max(anchor_half_override + 6, int(face_w * 0.26))):
                    continue
                filtered_u8[labels == idx] = 255
        if int((anchor_lane_u8 > 0).sum()) > 0:
            filtered_u8 = cv2.bitwise_or(filtered_u8, anchor_lane_u8)
        zone_u8 = filtered_u8
        if int((zone_u8 > 0).sum()) < 12:
            return np.zeros((H, W), dtype=np.float32)
    keep_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15) if hair_length == "short" else (5, 9)),
    )
    keep_u8 = cv2.erode(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) if hair_length == "short" else (3, 5)),
        iterations=1,
    )
    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 23) if hair_length == "short" else (7, 13)),
        iterations=1,
    )
    return (keep_u8 > 0).astype(np.float32)

def _build_short_regen_tail_mask(
    self,
    img_rgb: np.ndarray,
    removal_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str = "short",
) -> np.ndarray:
    """
    short hair에서 cleanup 후 남는 하단 잔머리만 좁게 다시 생성하도록 하는 mask.
    side-tail core를 기본으로 하고, dark residual은 core 주변으로만 허용한다.
    """
    H, W = img_rgb.shape[:2]
    if removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    _, y1, _, y2 = face_bbox
    face_h = max(int(y2 - y1), 1)

    side_mask = self._build_side_tail_cleanup_mask(
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length=hair_length,
    )
    side_u8 = (side_mask > 0.08).astype(np.uint8) * 255

    core_mask = self._build_short_tail_core_mask(
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length=hair_length,
    )
    core_u8 = (core_mask > 0.08).astype(np.uint8) * 255

    dark_mask = self._build_dark_tail_residual_mask(
        img_rgb=img_rgb,
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length=hair_length,
    )
    dark_u8 = (dark_mask > 0.08).astype(np.uint8) * 255

    shallow_band = np.zeros((H, W), dtype=np.uint8)
    shallow_top = max(0, int(cutoff_y - face_h * 0.02))
    shallow_bottom = min(H, int(cutoff_y + face_h * 0.92))
    if shallow_top < shallow_bottom:
        shallow_band[shallow_top:shallow_bottom, :] = 255
    side_u8 = cv2.bitwise_and(side_u8, shallow_band)
    core_u8 = cv2.bitwise_and(core_u8, shallow_band)
    dark_u8 = cv2.bitwise_and(dark_u8, shallow_band)

    side_u8 = cv2.morphologyEx(
        side_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
    )

    regen_u8 = cv2.bitwise_or(side_u8, core_u8)
    regen_u8 = cv2.bitwise_or(regen_u8, dark_u8)
    if int((regen_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    regen_u8 = cv2.morphologyEx(
        regen_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
    )
    return (regen_u8 > 0).astype(np.float32)

def _build_lower_hair_tail_support_mask(
    self,
    *,
    img_rgb: np.ndarray,
    hair_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    hair_length: str,
) -> np.ndarray:
    """
    밝은 옷 위로 내려온 얇은 앞머리 가닥이 SegFace/SAM2에서 빠질 때
    기존 hair mask 하단에 붙은 어두운 세로 성분만 보수적으로 다시 포함한다.
    """
    if hair_length not in ("short", "medium"):
        return np.zeros_like(hair_mask, dtype=np.float32)

    H, W = hair_mask.shape[:2]
    if img_rgb.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    x_min = max(0, int(x1 - face_w * (1.28 if hair_length == "short" else 1.10)))
    x_max = min(W, int(x2 + face_w * (1.28 if hair_length == "short" else 1.10)))
    y_min = max(0, int(y2 - face_h * 0.03))
    y_max = min(H, int(y2 + face_h * (1.72 if hair_length == "short" else 1.24)))
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[y_min:y_max, x_min:x_max] = 255

    hair_u8 = (np.clip(hair_mask.astype(np.float32), 0.0, 1.0) > 0.35).astype(np.uint8) * 255
    anchor_u8 = cv2.dilate(
        hair_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (41, 41) if hair_length == "short" else (29, 29),
        ),
        iterations=1,
    )
    anchor_u8[:max(0, int(y2 - face_h * 0.18)), :] = 0
    anchor_u8 = cv2.bitwise_and(anchor_u8, corridor_u8)
    if int((anchor_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)
    anchor_support_u8 = cv2.dilate(
        anchor_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (35, 61) if hair_length == "short" else (27, 45),
        ),
        iterations=1,
    )
    anchor_support_u8 = cv2.bitwise_and(anchor_support_u8, corridor_u8)
    front_strand_zone_u8 = np.zeros((H, W), dtype=np.uint8)
    front_half = max(12, int(face_w * (0.22 if hair_length == "short" else 0.18)))
    front_x1 = max(0, int(0.5 * (x1 + x2)) - front_half)
    front_x2 = min(W, int(0.5 * (x1 + x2)) + front_half)
    front_y1 = max(0, int(y2 - face_h * 0.02))
    front_y2 = min(H, int(y2 + face_h * (0.82 if hair_length == "short" else 0.58)))
    if front_x1 < front_x2 and front_y1 < front_y2:
        front_strand_zone_u8[front_y1:front_y2, front_x1:front_x2] = 255

    support_zone_u8 = corridor_u8.copy()
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_hint_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        support_zone_u8 = cv2.bitwise_and(
            support_zone_u8,
            cv2.bitwise_or(cloth_hint_u8, anchor_support_u8),
        )
        if int((support_zone_u8 > 0).sum()) < 60:
            support_zone_u8 = corridor_u8.copy()
        front_strand_zone_u8 = cv2.bitwise_and(front_strand_zone_u8, cloth_hint_u8)

    candidate_zone_u8 = cv2.bitwise_or(anchor_support_u8, front_strand_zone_u8)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0, sigmaY=7.0)
    bright_bg = blur > (136.0 if hair_length == "short" else 132.0)
    dark_thresh = 140.0 if hair_length == "short" else 136.0
    contrast_thresh = 7.0 if hair_length == "short" else 6.0

    dark_u8 = (
        (gray < dark_thresh)
        & ((blur - gray) > contrast_thresh)
        & bright_bg
    ).astype(np.uint8) * 255
    dark_u8 = cv2.bitwise_and(dark_u8, support_zone_u8)
    dark_u8 = cv2.bitwise_and(dark_u8, candidate_zone_u8)
    if int((dark_u8 > 0).sum()) < 18:
        return np.zeros((H, W), dtype=np.float32)

    main_seed_u8 = cv2.dilate(
        hair_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (19, 25) if hair_length == "short" else (15, 21),
        ),
        iterations=1,
    )
    main_seed_u8[:max(0, int(y2 - face_h * 0.08)), :] = 0
    main_seed_u8 = cv2.bitwise_and(main_seed_u8, corridor_u8)
    main_seed_u8 = cv2.bitwise_or(main_seed_u8, anchor_u8)

    dark_u8 = cv2.morphologyEx(
        dark_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    dark_u8 = cv2.morphologyEx(
        dark_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 29) if hair_length == "short" else (5, 21),
        ),
    )
    dark_u8 = cv2.dilate(
        dark_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 17) if hair_length == "short" else (9, 13),
        ),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_u8, 8)
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    max_area = max(
        280,
        min(
            int(H * W * 0.020),
            int(face_w * face_h * (0.42 if hair_length == "short" else 0.28)),
        ),
    )
    min_tail_bottom = int(y2 + face_h * (0.10 if hair_length == "short" else 0.08))
    min_height = max(14, int(face_h * (0.10 if hair_length == "short" else 0.08)))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 20 or area > max_area:
            continue
        if h < min_height:
            continue
        if (y + h) < min_tail_bottom:
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        dark_overlap = int((cv2.bitwise_and(comp_u8, dark_u8) > 0).sum())
        if dark_overlap < 8:
            continue
        seed_overlap = int((cv2.bitwise_and(comp_u8, main_seed_u8) > 0).sum())
        anchor_overlap = int((cv2.bitwise_and(comp_u8, anchor_support_u8) > 0).sum())
        front_overlap = int((cv2.bitwise_and(comp_u8, front_strand_zone_u8) > 0).sum())
        is_front_strand = (
            front_overlap >= 10
            and w <= max(26, int(face_w * 0.24))
            and h >= max(26, int(face_h * 0.16))
        )
        if seed_overlap < 12 and anchor_overlap < 10 and not is_front_strand:
            continue
        keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

    fallback_seed_u8 = cv2.bitwise_and(hair_u8, corridor_u8)
    fallback_seed_u8[:max(0, int(y2 + face_h * 0.02)), :] = 0
    if int((fallback_seed_u8 > 0).sum()) >= 40:
        fallback_keep_u8 = np.zeros((H, W), dtype=np.uint8)
        fallback_labels, fallback_cc, fallback_stats, fallback_centroids = cv2.connectedComponentsWithStats(
            fallback_seed_u8,
            8,
        )
        fallback_min_area = max(36, int(face_w * face_h * 0.0024))
        fallback_min_height = max(min_height, int(face_h * 0.22))
        fallback_max_area = max(max_area * 4, int(face_w * face_h * 0.88))
        face_cx = float(0.5 * (x1 + x2))
        for idx in range(1, fallback_labels):
            x = int(fallback_stats[idx, cv2.CC_STAT_LEFT])
            y = int(fallback_stats[idx, cv2.CC_STAT_TOP])
            w = int(fallback_stats[idx, cv2.CC_STAT_WIDTH])
            h = int(fallback_stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(fallback_stats[idx, cv2.CC_STAT_AREA])
            if area < fallback_min_area or area > fallback_max_area:
                continue
            if h < fallback_min_height:
                continue
            if (y + h) < min_tail_bottom:
                continue
            comp_cx = float(fallback_centroids[idx][0])
            is_side_component = abs(comp_cx - face_cx) > max(14, int(face_w * 0.16))
            is_front_strand = (
                abs(comp_cx - face_cx) <= max(18, int(face_w * 0.18))
                and w <= max(34, int(face_w * 0.30))
                and h >= max(28, int(face_h * 0.18))
            )
            if not is_side_component and not is_front_strand:
                continue
            if w > max(56, int(face_w * 0.46)) and not is_front_strand:
                continue
            comp_u8 = (fallback_cc == idx).astype(np.uint8) * 255
            fallback_keep_u8 = cv2.bitwise_or(fallback_keep_u8, comp_u8)
        if int((fallback_keep_u8 > 0).sum()) >= 20:
            keep_u8 = cv2.bitwise_or(keep_u8, fallback_keep_u8)

    if int((keep_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    bridge_seed_u8 = cv2.dilate(
        main_seed_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 31) if hair_length == "short" else (7, 25),
        ),
        iterations=1,
    )
    bridge_keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 23) if hair_length == "short" else (5, 17),
        ),
        iterations=1,
    )
    bridge_u8 = cv2.bitwise_and(bridge_seed_u8, bridge_keep_u8)
    bridge_u8 = cv2.bitwise_and(bridge_u8, corridor_u8)
    keep_u8 = cv2.bitwise_or(keep_u8, bridge_u8)
    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 41) if hair_length == "short" else (7, 31),
        ),
    )

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 23) if hair_length == "short" else (9, 13),
        ),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    return (keep_u8 > 0).astype(np.float32)

def _build_center_chest_strand_support_mask(
    self,
    *,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    support_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length not in ("short", "medium", "long"):
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    lane_u8 = np.zeros((H, W), dtype=np.uint8)
    # v111: 가슴 중앙 감지 대역폭 확장 (sideways strands 포착 목적)
    # v111_ablation: Revert lane_half to v110 levels (sideways strands 포착 목적 축소)
    if hair_length == "long":
        lane_half = max(32, int(face_w * 0.38))
        lane_y_extent = 1.62
    elif hair_length == "medium":
        lane_half = max(24, int(face_w * 0.28))
        lane_y_extent = 1.42
    else: # short
        lane_half = max(20, int(face_w * 0.24))
        lane_y_extent = 2.05

    lane_x1 = max(0, cx - lane_half)
    lane_x2 = min(W, cx + lane_half)
    lane_y1 = max(0, int(cutoff_y - face_h * 0.08))
    lane_y2 = min(H, int(cutoff_y + face_h * lane_y_extent))
    if lane_x1 >= lane_x2 or lane_y1 >= lane_y2:
        return np.zeros((H, W), dtype=np.float32)
    lane_u8[lane_y1:lane_y2, lane_x1:lane_x2] = 255

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    zone_u8 = cv2.bitwise_and(lane_u8, cloth_u8)

    support_hint_u8 = np.zeros((H, W), dtype=np.uint8)
    if support_mask is not None and support_mask.shape == (H, W):
        support_hint_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255
        if int((support_hint_u8 > 0).sum()) > 0:
            support_hint_u8 = cv2.dilate(
                support_hint_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (11, 31) if hair_length == "short" else (9, 25),
                ),
                iterations=1,
            )
            support_hint_u8 = cv2.bitwise_and(support_hint_u8, lane_u8)
            support_hint_u8 = cv2.morphologyEx(
                support_hint_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (7, 31) if hair_length == "short" else (5, 23),
                ),
            )
            if hair_length == "short":
                zone_u8 = cv2.bitwise_or(zone_u8, support_hint_u8)

    if int((zone_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=9.0, sigmaY=9.0)
    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 25) if hair_length == "short" else (9, 21),
        ),
    )
    blackhat_u8 = (
        blackhat > (9 if hair_length == "short" else 10)
    ).astype(np.uint8) * 255
    # v111: 배경이 밝은 흰색/아이보리일 때 머리카락 감지를 위해 임계값(154->178) 완화
    # v111_ablation: Revert gray_threshold to v110 levels (154.0 / 150.0 / 148.0)
    if hair_length == "long":
        gray_threshold = 154.0
        diff_threshold = 1.8
        blur_base_threshold = 96.0
    elif hair_length == "medium":
        gray_threshold = 150.0
        diff_threshold = 2.0
        blur_base_threshold = 100.0
    else: # short
        gray_threshold = 148.0
        diff_threshold = 2.2
        blur_base_threshold = 110.0

    dark_u8 = (
        (gray < gray_threshold)
        & ((blur - gray) > diff_threshold)
        & (blur > blur_base_threshold)
    ).astype(np.uint8) * 255
    dark_u8 = cv2.bitwise_or(dark_u8, blackhat_u8)
    dark_u8 = cv2.bitwise_and(dark_u8, zone_u8)
    if int((support_hint_u8 > 0).sum()) >= 8:
        dark_u8 = cv2.bitwise_and(dark_u8, support_hint_u8)
    if int((dark_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    dark_u8 = cv2.morphologyEx(
        dark_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    dark_u8 = cv2.morphologyEx(
        dark_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)),
    )
    if int((support_hint_u8 > 0).sum()) >= 8:
        dark_u8 = cv2.bitwise_or(dark_u8, support_hint_u8)
        dark_u8 = cv2.morphologyEx(
            dark_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 35) if hair_length == "short" else (5, 27),
            ),
        )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_u8, 8)
    max_area = max(640, int(face_w * face_h * 0.12))
    max_width = max(26, int(face_w * 0.30))
    min_height = max(22, int(face_h * 0.12))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_cx = float(centroids[idx][0])
        if area < 8 or area > max_area:
            continue
        if w > max_width or h < min_height:
            continue
        # v111_ablation: Revert cx_offset_max to v110 levels
        cx_offset_max = max(24, int(face_w * (0.34 if hair_length == "long" else 0.22)))
        if abs(comp_cx - cx) > cx_offset_max:
            continue
        if (y + h) < int(cutoff_y + face_h * 0.14):
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        if int((support_hint_u8 > 0).sum()) >= 8:
            support_overlap = int((cv2.bitwise_and(comp_u8, support_hint_u8) > 0).sum())
            dark_overlap = int((cv2.bitwise_and(comp_u8, blackhat_u8) > 0).sum())
            if support_overlap < 10 or dark_overlap < 8:
                continue
        keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

    if int((keep_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 23) if hair_length == "short" else (7, 17),
        ),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
    return (keep_u8 > 0).astype(np.float32)

def _build_lower_tail_post_support_mask(
    self,
    *,
    support_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    torso_hair_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    debug_outputs: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros_like(support_mask, dtype=np.float32)

    H, W = support_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    support_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((support_u8 > 0).sum()) < 12:
        support_u8 = np.zeros((H, W), dtype=np.uint8)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    x_min = max(0, int(x1 - face_w * (1.08 if hair_length == "short" else 0.92)))
    x_max = min(W, int(x2 + face_w * (1.08 if hair_length == "short" else 0.92)))
    y_min = max(0, int(cutoff_y + face_h * (0.02 if hair_length == "short" else 0.00)))
    y_max = min(H, int(cutoff_y + face_h * (1.46 if hair_length == "short" else 0.76)))
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[y_min:y_max, x_min:x_max] = 255
    support_u8 = cv2.bitwise_and(support_u8, corridor_u8)

    # 중앙 세로 가닥은 별도 center-support 경로로 다루고,
    # lower-tail post support는 좌우 하단 tail lane만 남긴다.
    center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    center_half = max(16, int(face_w * (0.18 if hair_length == "short" else 0.16)))
    keepout_top = max(0, int(cutoff_y - face_h * 0.02))
    keepout_bottom = min(H, int(cutoff_y + face_h * (1.18 if hair_length == "short" else 0.80)))
    if keepout_top < keepout_bottom:
        center_keepout_u8[
            keepout_top:keepout_bottom,
            max(0, cx - center_half):min(W, cx + center_half),
        ] = 255
        support_u8 = cv2.bitwise_and(support_u8, cv2.bitwise_not(center_keepout_u8))

    support_raw_u8 = support_u8.copy()
    cloth_near_u8 = np.zeros((H, W), dtype=np.uint8)
    side_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_side_support_raw_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_side_support_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_side_support_cloth_gated_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_side_rescue_u8 = np.zeros((H, W), dtype=np.uint8)
    if hair_length == "short" and torso_hair_mask is not None and torso_hair_mask.shape == (H, W):
        torso_u8 = (np.clip(torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        torso_u8 = cv2.bitwise_and(torso_u8, corridor_u8)
        torso_u8[:max(0, int(cutoff_y + face_h * 0.06)), :] = 0
        lane_top = max(0, int(cutoff_y + face_h * 0.08))
        lane_bottom = min(H, int(cutoff_y + face_h * 1.52))
        left_outer = max(0, int(x1 - face_w * 0.30))
        left_inner = min(W, int(x1 + face_w * 0.02))
        right_inner = max(0, int(x2 - face_w * 0.02))
        right_outer = min(W, int(x2 + face_w * 0.30))
        if lane_top < lane_bottom:
            if left_outer < left_inner:
                side_lane_u8[lane_top:lane_bottom, left_outer:left_inner] = 255
            if right_inner < right_outer:
                side_lane_u8[lane_top:lane_bottom, right_inner:right_outer] = 255
        torso_side_support_raw_u8 = cv2.bitwise_and(torso_u8, side_lane_u8)
        torso_side_support_raw_u8 = cv2.morphologyEx(
            torso_side_support_raw_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 19)),
        )
        torso_side_support_raw_u8 = cv2.dilate(
            torso_side_support_raw_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13)),
            iterations=1,
        )
        torso_side_support_raw_u8 = cv2.bitwise_and(torso_side_support_raw_u8, corridor_u8)
        torso_side_support_u8 = torso_side_support_raw_u8.copy()

    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_near_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_and(support_u8, cloth_near_u8)
        if int((torso_side_support_u8 > 0).sum()) > 0:
            torso_side_support_u8 = cv2.bitwise_and(
                torso_side_support_u8,
                cv2.bitwise_or(
                    cloth_near_u8,
                    cv2.dilate(
                        support_raw_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
                        iterations=1,
                    ),
                ),
            )
            torso_side_support_cloth_gated_u8 = torso_side_support_u8.copy()
            if hair_length == "short" and int((torso_side_support_raw_u8 > 0).sum()) >= 80:
                rescue_top = max(0, int(cutoff_y + face_h * 0.18))
                rescue_bottom = min(H, int(cutoff_y + face_h * 1.52))
                rescue_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
                rescue_left_outer = max(0, int(x1 - face_w * 0.54))
                rescue_left_inner = max(rescue_left_outer + 1, int(cx - face_w * 0.14))
                rescue_right_inner = min(W - 1, int(cx + face_w * 0.14))
                rescue_right_outer = min(W, int(x2 + face_w * 0.54))
                if rescue_top < rescue_bottom:
                    if rescue_left_outer < rescue_left_inner:
                        rescue_corridor_u8[rescue_top:rescue_bottom, rescue_left_outer:rescue_left_inner] = 255
                    if rescue_right_inner < rescue_right_outer:
                        rescue_corridor_u8[rescue_top:rescue_bottom, rescue_right_inner:rescue_right_outer] = 255

                num_torso_labels, torso_labels, torso_stats, torso_centroids = cv2.connectedComponentsWithStats(
                    (torso_side_support_raw_u8 > 0).astype(np.uint8),
                    8,
                )
                for torso_idx in range(1, num_torso_labels):
                    comp_area = int(torso_stats[torso_idx, cv2.CC_STAT_AREA])
                    if comp_area < 80:
                        continue
                    comp_width = int(torso_stats[torso_idx, cv2.CC_STAT_WIDTH])
                    comp_height = int(torso_stats[torso_idx, cv2.CC_STAT_HEIGHT])
                    comp_bottom = int(torso_stats[torso_idx, cv2.CC_STAT_TOP] + comp_height)
                    comp_cx = float(torso_centroids[torso_idx][0])
                    if comp_width > max(52, int(face_w * 0.46)):
                        continue
                    if comp_height < max(26, int(face_h * 0.18)):
                        continue
                    if comp_bottom < int(cutoff_y + face_h * 0.28):
                        continue
                    if abs(comp_cx - cx) < max(16, int(face_w * 0.16)):
                        continue

                    comp_u8 = np.zeros((H, W), dtype=np.uint8)
                    comp_u8[torso_labels == torso_idx] = 255
                    comp_u8 = cv2.bitwise_and(comp_u8, rescue_corridor_u8)
                    if int((comp_u8 > 0).sum()) < 40:
                        continue

                    filtered_overlap = int((cv2.bitwise_and(comp_u8, torso_side_support_u8) > 0).sum())
                    if filtered_overlap >= max(18, int(comp_area * 0.18)):
                        continue

                    comp_u8 = cv2.morphologyEx(
                        comp_u8,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 17)),
                    )
                    torso_side_rescue_u8 = cv2.bitwise_or(torso_side_rescue_u8, comp_u8)

                if int((torso_side_rescue_u8 > 0).sum()) > 0:
                    torso_side_support_u8 = cv2.bitwise_or(
                        torso_side_support_u8,
                        torso_side_rescue_u8,
                    )
        if hair_length == "short":
            raw_px = int((support_raw_u8 > 0).sum())
            cloth_px = int((support_u8 > 0).sum())
            if raw_px >= 48 and cloth_px < max(12, int(raw_px * 0.22)):
                fallback_u8 = support_raw_u8.copy()
                fallback_u8[:max(0, int(cutoff_y + face_h * 0.22)), :] = 0
                side_deep_u8 = np.zeros((H, W), dtype=np.uint8)
                deep_top = max(0, int(cutoff_y + face_h * 0.18))
                deep_bottom = min(H, int(cutoff_y + face_h * 0.98))
                side_inner_gap = max(18, int(face_w * 0.18))
                left_outer = max(0, int(x1 - face_w * 0.68))
                left_inner = max(left_outer + 1, cx - side_inner_gap)
                right_inner = min(W - 1, cx + side_inner_gap)
                right_outer = min(W, int(x2 + face_w * 0.68))
                if deep_top < deep_bottom:
                    if left_outer < left_inner:
                        side_deep_u8[deep_top:deep_bottom, left_outer:left_inner] = 255
                    if right_inner < right_outer:
                        side_deep_u8[deep_top:deep_bottom, right_inner:right_outer] = 255
                fallback_u8 = cv2.bitwise_and(fallback_u8, side_deep_u8)
                support_u8 = cv2.bitwise_or(support_u8, fallback_u8)
    if hair_length == "short" and int((torso_side_support_u8 > 0).sum()) > 0:
        support_u8 = cv2.bitwise_or(support_u8, torso_side_support_u8)

    if int((support_u8 > 0).sum()) < 12:
        return np.zeros((H, W), dtype=np.float32)

    support_u8 = cv2.morphologyEx(
        support_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    support_u8 = cv2.dilate(
        support_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 11) if hair_length == "short" else (5, 9),
        ),
        iterations=1,
    )
    if int((center_keepout_u8 > 0).sum()) > 0:
        support_u8 = cv2.bitwise_and(support_u8, cv2.bitwise_not(center_keepout_u8))

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support_u8, 8)
    max_area = max(640, int(face_w * face_h * (0.44 if hair_length == "short" else 0.14)))
    max_width = max(34, int(face_w * (0.34 if hair_length == "short" else 0.20)))
    rescue_max_width = max_width
    if hair_length == "short" and int((torso_side_rescue_u8 > 0).sum()) > 0:
        rescue_max_width = max(max_width, max(44, int(face_w * 0.42)))
    min_height = max(24, int(face_h * (0.22 if hair_length == "short" else 0.14)))
    min_bottom = int(cutoff_y + face_h * (0.10 if hair_length == "short" else 0.08))
    max_offset = max(34, int(face_w * 1.04))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_mask = labels == idx
        comp_xs = np.where(comp_mask)[1]
        comp_cx = float(comp_xs.mean()) if comp_xs.size else float(x + (w * 0.5))
        allowed_max_width = max_width
        if rescue_max_width > max_width:
            rescue_overlap = int((torso_side_rescue_u8[comp_mask] > 0).sum())
            if rescue_overlap >= max(24, int(area * 0.08)):
                allowed_max_width = rescue_max_width
        if area < 8 or area > max_area:
            continue
        if w > allowed_max_width or h < min_height:
            continue
        if (y + h) < min_bottom:
            continue
        if abs(comp_cx - cx) < center_half or abs(comp_cx - cx) > max_offset:
            continue
        filtered_u8[comp_mask] = 255

    if int((filtered_u8 > 0).sum()) < 12:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 17) if hair_length == "short" else (7, 13),
        ),
        iterations=1,
    )
    filtered_u8 = cv2.bitwise_and(filtered_u8, corridor_u8)
    if debug_outputs is not None:
        debug_outputs["lower_tail_post_support_corridor_mask"] = corridor_u8.astype(np.float32) / 255.0
        debug_outputs["lower_tail_post_support_center_keepout_mask"] = center_keepout_u8.astype(np.float32) / 255.0
        debug_outputs["lower_tail_post_support_support_raw_mask"] = support_raw_u8.astype(np.float32) / 255.0
        debug_outputs["lower_tail_post_support_side_lane_mask"] = side_lane_u8.astype(np.float32) / 255.0
        debug_outputs["lower_tail_post_support_cloth_near_mask"] = cloth_near_u8.astype(np.float32) / 255.0
        debug_outputs["lower_tail_post_support_torso_side_raw_mask"] = (
            torso_side_support_raw_u8.astype(np.float32) / 255.0
        )
        debug_outputs["lower_tail_post_support_torso_side_cloth_gated_mask"] = (
            torso_side_support_cloth_gated_u8.astype(np.float32) / 255.0
        )
        debug_outputs["lower_tail_post_support_torso_side_rescue_mask"] = (
            torso_side_rescue_u8.astype(np.float32) / 255.0
        )
        debug_outputs["lower_tail_post_support_filtered_mask"] = filtered_u8.astype(np.float32) / 255.0
    return (filtered_u8 > 0).astype(np.float32)

def _build_lower_tail_removal_extension_mask(
    self,
    *,
    support_mask: np.ndarray,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros_like(support_mask, dtype=np.float32)

    H, W = support_mask.shape[:2]
    if removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    support_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((support_u8 > 0).sum()) < 12:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    x_min = max(0, int(x1 - face_w * (0.96 if hair_length == "short" else 0.84)))
    x_max = min(W, int(x2 + face_w * (0.96 if hair_length == "short" else 0.84)))
    y_min = max(0, int(cutoff_y - face_h * 0.03))
    y_max = min(H, int(cutoff_y + face_h * (1.48 if hair_length == "short" else 1.08)))
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[y_min:y_max, x_min:x_max] = 255
    support_u8 = cv2.bitwise_and(support_u8, corridor_u8)

    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_hint_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_and(support_u8, cloth_hint_u8)

    base_hint_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255
    base_hint_u8[:cutoff_y, :] = 0
    if int((base_hint_u8 > 0).sum()) > 0:
        base_hint_u8 = cv2.dilate(
            base_hint_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (17, 23) if hair_length == "short" else (13, 17),
            ),
            iterations=1,
        )
        support_u8 = cv2.bitwise_and(support_u8, cv2.bitwise_or(base_hint_u8, corridor_u8))

    front_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    center_half = max(16, int(face_w * (0.34 if hair_length == "short" else 0.20)))
    lane_x1 = max(0, int(0.5 * (x1 + x2)) - center_half)
    lane_x2 = min(W, int(0.5 * (x1 + x2)) + center_half)
    lane_y1 = max(0, int(cutoff_y - face_h * 0.04))
    lane_y2 = min(H, int(cutoff_y + face_h * (1.78 if hair_length == "short" else 0.96)))
    if lane_x1 < lane_x2 and lane_y1 < lane_y2:
        front_lane_u8[lane_y1:lane_y2, lane_x1:lane_x2] = 255

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(support_u8, 8)
    max_area = max(360, int(face_w * face_h * (0.42 if hair_length == "short" else 0.14)))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 8 or area > max_area:
            continue
        if (y + h) < int(cutoff_y + face_h * 0.02):
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        base_overlap = int((cv2.bitwise_and(comp_u8, base_hint_u8) > 0).sum())
        front_overlap = int((cv2.bitwise_and(comp_u8, front_lane_u8) > 0).sum())
        comp_cx = float(centroids[idx][0])
        is_side_component = abs(comp_cx - cx) > max(18, int(face_w * 0.22))
        is_front_strand = (
            front_overlap >= 10
            and w <= max(22, int(face_w * 0.26))
            and h >= max(24, int(face_h * 0.18))
        )
        anchored_side_component = (
            hair_length == "short"
            and is_side_component
            and h >= max(28, int(face_h * 0.24))
            and w <= max(48, int(face_w * 0.44))
            and area <= max(680, int(face_w * face_h * 0.16))
        )
        if base_overlap < 8 and not is_front_strand and not anchored_side_component:
            continue
        if hair_length == "short":
            if w > max(48, int(face_w * 0.44)) and not is_front_strand:
                continue
            if is_side_component:
                if base_overlap < max(12, int(area * 0.08)) and front_overlap < 12 and not anchored_side_component:
                    continue
                if area > max(680, int(face_w * face_h * 0.16)):
                    continue
        filtered_u8 = cv2.bitwise_or(filtered_u8, comp_u8)

    if int((filtered_u8 > 0).sum()) < 12:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 21) if hair_length == "short" else (5, 11),
        ),
        iterations=1,
    )
    filtered_u8 = cv2.bitwise_and(filtered_u8, corridor_u8)
    return (filtered_u8 > 0).astype(np.float32)

def _build_dark_tail_residual_mask(
    self,
    img_rgb: np.ndarray,
    removal_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str = "short",
) -> np.ndarray:
    """
    SegFace가 놓친 회색/검은 잔여 tail blob을 밝기 기반으로 추가 검출한다.
    cleanup 이후 흐리게 남는 하단 머리 덩어리 제거용이다.
    """
    H, W = img_rgb.shape[:2]
    if removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    zone = np.clip(removal_mask.astype(np.float32), 0.0, 1.0).copy()
    zone[:cutoff_y, :] = 0.0
    tail_hint = self._build_side_tail_cleanup_mask(
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length=hair_length,
    )
    if float(tail_hint.sum()) > 20.0:
        tail_hint_u8 = (np.clip(tail_hint.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
        if hair_length == "short":
            tail_hint_u8 = cv2.dilate(
                tail_hint_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 33)),
                iterations=1,
            )
        zone = np.maximum(zone, (tail_hint_u8.astype(np.float32) / 255.0) * (1.36 if hair_length == "short" else 1.20))
    if hair_length == "short":
        tail_core_hint = self._build_short_tail_core_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        if float(tail_core_hint.sum()) > 12.0:
            tail_core_u8 = cv2.dilate(
                (np.clip(tail_core_hint.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 31)),
                iterations=1,
            )
            zone = np.maximum(zone, (tail_core_u8.astype(np.float32) / 255.0) * 1.42)
    front_hint = np.zeros((H, W), dtype=np.float32)
    front_half = max(12, int(face_w * (0.24 if hair_length == "short" else 0.20)))
    front_x1 = max(0, int(0.5 * (x1 + x2)) - front_half)
    front_x2 = min(W, int(0.5 * (x1 + x2)) + front_half)
    front_y1 = max(0, int(cutoff_y - face_h * 0.04))
    front_y2 = min(H, int(cutoff_y + face_h * (0.90 if hair_length == "short" else 0.64)))
    if front_x1 < front_x2 and front_y1 < front_y2:
        front_hint[front_y1:front_y2, front_x1:front_x2] = 1.0
        front_hint = np.clip(front_hint * np.clip(removal_mask.astype(np.float32), 0.0, 1.0), 0.0, 1.0)
        if float(front_hint.sum()) > 20.0:
            zone = np.maximum(zone, front_hint * 1.10)

    zone_thresh = 0.12 if hair_length == "short" else 0.34
    zone_u8 = (zone > zone_thresh).astype(np.uint8) * 255
    deep_start = min(H, int(cutoff_y + face_h * (0.00 if hair_length == "short" else 0.10)))
    x_min = max(0, int(x1 - face_w * (1.35 if hair_length == "short" else 1.20)))
    x_max = min(W, int(x2 + face_w * (1.35 if hair_length == "short" else 1.20)))
    corridor = np.zeros((H, W), dtype=np.uint8)
    if x_min < x_max and deep_start < H:
        corridor[deep_start:, x_min:x_max] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor)
    if hair_length == "short" and int((zone_u8 > 0).sum()) > 0:
        zone_u8 = cv2.dilate(
            zone_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 27)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_and(zone_u8, corridor)
    if int((zone_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=9.0, sigmaY=9.0)
    zone_vals = gray[zone_u8 > 0]
    dark_thresh = 150.0 if hair_length == "short" else 148.0
    if zone_vals.size >= 80:
        dark_thresh = float(
            np.clip(
                np.percentile(zone_vals, 80.0) + (8.0 if hair_length == "short" else 14.0),
                110.0 if hair_length == "short" else 96.0,
                172.0 if hair_length == "short" else 164.0,
            )
        )
    contrast_thresh = 3.0 if hair_length == "short" else 6.5
    dark_u8 = (
        (gray < dark_thresh)
        & ((blur - gray) > contrast_thresh)
    ).astype(np.uint8) * 255
    dark_u8 = cv2.bitwise_and(dark_u8, zone_u8)
    if int((dark_u8 > 0).sum()) < 30:
        return np.zeros((H, W), dtype=np.float32)

    dark_u8 = cv2.morphologyEx(
        dark_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    dark_u8 = cv2.dilate(
        dark_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21) if hair_length == "short" else (7, 7)),
        iterations=1,
    )
    return (dark_u8 > 0).astype(np.float32)


def _maybe_standardize_input_portrait(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    *,
    target_hair_length: str = "long",
) -> Dict[str, Any]:
    """
    얼굴이 너무 작게 잡히는 입력을 미용실 상담용 포트레이트 프레임으로 정규화한다.
    작은 얼굴/넓은 배경/과도한 상반신 정보가 포함된 이미지만 보수적으로 크롭한다.
    """
    meta: Dict[str, Any] = {"applied": False}
    if not getattr(self.config, "enable_input_standardization", True):
        return meta

    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(x2 - x1, 1)
    face_h = max(y2 - y1, 1)
    face_w_ratio = face_w / max(float(W), 1.0)
    face_h_ratio = face_h / max(float(H), 1.0)
    top_gap_ratio = y1 / max(float(H), 1.0)
    is_landscape = W > H
    enable_reframe = bool(getattr(self.config, "enable_portrait_reframe", True))
    needs_reframe = (
        enable_reframe
        and not is_landscape
        and (
            face_h_ratio > float(getattr(self.config, "portrait_reframe_face_height_ratio_max", 0.40))
            or top_gap_ratio < float(getattr(self.config, "portrait_reframe_top_gap_ratio_min", 0.06))
        )
    )

    if (
        face_h_ratio >= float(self.config.standardize_face_height_ratio_min)
        and face_w_ratio >= float(self.config.standardize_face_width_ratio_min)
        and not is_landscape
        and not needs_reframe
    ):
        return meta

    if needs_reframe:
        target_w = int(W)
        target_h = int(H)
    else:
        target_w = int(getattr(self.config, "standardized_width", 768))
        target_h = int(getattr(self.config, "standardized_height", 1024))
        crop_top_ratio = float(getattr(self.config, "standardize_crop_top_face_ratio", 0.95))
        adaptive_framing_enabled = bool(
            getattr(self.config, "adaptive_input_framing_by_target_length", True)
        )
        if adaptive_framing_enabled:
            length_key = str(target_hair_length or "long").strip().lower()
            if length_key == "short":
                crop_bottom_ratio = float(
                    getattr(self.config, "standardize_crop_bottom_face_ratio_short", 1.42)
                )
            elif length_key == "medium":
                crop_bottom_ratio = float(
                    getattr(self.config, "standardize_crop_bottom_face_ratio_medium", 1.86)
                )
            else:
                crop_bottom_ratio = float(
                    getattr(self.config, "standardize_crop_bottom_face_ratio_long", 2.30)
                )
        else:
            crop_bottom_ratio = 2.15
        crop_top = int(round(y1 - face_h * crop_top_ratio))
        crop_bottom = int(round(y2 + face_h * crop_bottom_ratio))
    target_aspect = target_w / max(float(target_h), 1.0)

    cx = 0.5 * (x1 + x2)
    if needs_reframe:
        scale_candidates = [1.0]
        max_face_ratio = float(getattr(self.config, "portrait_reframe_face_height_ratio_max", 0.40))
        min_top_gap = float(getattr(self.config, "portrait_reframe_top_gap_ratio_min", 0.06))
        if max_face_ratio > 0.0:
            scale_candidates.append(face_h_ratio / max_face_ratio)
        if min_top_gap > 0.0 and top_gap_ratio > 1e-6:
            scale_candidates.append(min_top_gap / top_gap_ratio)
        reframe_scale = float(np.clip(max(scale_candidates) * 1.04, 1.02, 1.18))
        crop_h = max(int(round(H * reframe_scale)), face_h + 1)
        crop_w = max(int(round(crop_h * target_aspect)), W + 1)
        extra_h = max(0, crop_h - H)
        crop_top = int(round(-extra_h * 0.58))
        crop_bottom = crop_top + crop_h
    else:
        crop_h = max(crop_bottom - crop_top, face_h + 1)
        crop_w = max(int(round(crop_h * target_aspect)), face_w + 1)
    crop_left = int(round(cx - crop_w * 0.5))
    crop_right = crop_left + crop_w
    crop_rgb = self._crop_with_soft_padding(
        img_rgb,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_right=crop_right,
        crop_bottom=crop_bottom,
    )
    if crop_rgb.size == 0:
        return meta

    interp = cv2.INTER_AREA
    if crop_rgb.shape[0] < target_h or crop_rgb.shape[1] < target_w:
        interp = cv2.INTER_CUBIC
    standardized_rgb = cv2.resize(crop_rgb, (target_w, target_h), interpolation=interp)

    meta.update({
        "applied": True,
        "reason": {
            "face_h_ratio": round(face_h_ratio, 4),
            "face_w_ratio": round(face_w_ratio, 4),
            "top_gap_ratio": round(top_gap_ratio, 4),
            "is_landscape": bool(is_landscape),
            "reframe_applied": bool(needs_reframe),
            "target_hair_length": str(target_hair_length or "long"),
        },
        "crop_box": [int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)],
        "original_shape": [int(H), int(W)],
        "standardized_shape": [int(target_h), int(target_w)],
        "image_rgb": standardized_rgb,
    })
    return meta


@staticmethod
def _crop_with_soft_padding(
    img_rgb: np.ndarray,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    pad_left = max(0, -int(crop_left))
    pad_top = max(0, -int(crop_top))
    pad_right = max(0, int(crop_right) - W)
    pad_bottom = max(0, int(crop_bottom) - H)

    padded = img_rgb
    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            img_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        pad_mask = np.zeros(padded.shape[:2], dtype=np.float32)
        if pad_top:
            pad_mask[:pad_top, :] = 1.0
        if pad_bottom:
            pad_mask[-pad_bottom:, :] = 1.0
        if pad_left:
            pad_mask[:, :pad_left] = 1.0
        if pad_right:
            pad_mask[:, -pad_right:] = 1.0
        pad_mask = cv2.GaussianBlur(pad_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
        blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=18.0, sigmaY=18.0)
        padded = (
            padded.astype(np.float32) * (1.0 - pad_mask[..., np.newaxis])
            + blurred.astype(np.float32) * pad_mask[..., np.newaxis]
        )
        padded = np.clip(padded, 0, 255).astype(np.uint8)

    x1 = int(crop_left) + pad_left
    y1 = int(crop_top) + pad_top
    x2 = int(crop_right) + pad_left
    y2 = int(crop_bottom) + pad_top
    return padded[y1:y2, x1:x2]


@staticmethod
def _mask_bbox(
    mask: Optional[np.ndarray],
    *,
    threshold: float = 0.5,
) -> Optional[Tuple[int, int, int, int]]:
    if mask is None or mask.ndim != 2:
        return None
    ys, xs = np.where(mask > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _analyze_source_cloth_preclean_need(
    self,
    *,
    source_hair_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    subject_gender: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source_hair_length": "unknown",
        "needs_preclean": True,
        "skip_preclean": False,
        "skip_reason": "insufficient_mask_data",
        "hair_bottom_ratio": 0.0,
        "torso_hair_ratio": 0.0,
        "cloth_overlap_ratio": 0.0,
        "torso_hair_mask": None,
        "cloth_overlap_mask": None,
    }

    normalized_gender = self._normalize_subject_gender(subject_gender)
    if normalized_gender == "male":
        result["skip_preclean"] = True
        result["skip_reason"] = "male_subject"

    if source_hair_mask is None or cloth_mask is None:
        return result
    if source_hair_mask.ndim != 2 or cloth_mask.ndim != 2:
        return result

    H, W = source_hair_mask.shape[:2]
    if cloth_mask.shape != (H, W):
        return result

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    face_area = max(face_w * face_h, 1)

    hair_u8 = (np.clip(source_hair_mask.astype(np.float32), 0.0, 1.0) > 0.18).astype(np.uint8) * 255
    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    if int((hair_u8 > 0).sum()) < 40:
        if normalized_gender != "male":
            result["needs_preclean"] = False
            result["skip_preclean"] = True
            result["skip_reason"] = "source_hair_too_small"
        return result

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.02))
    bottom = min(H, int(y2 + face_h * 1.62))
    left = max(0, int(x1 - face_w * 1.28))
    right = min(W, int(x2 + face_w * 1.28))
    if top >= bottom or left >= right:
        return result
    corridor_u8[top:bottom, left:right] = 255

    torso_hair_u8 = cv2.bitwise_and(hair_u8, corridor_u8)
    cloth_overlap_u8 = cv2.bitwise_and(torso_hair_u8, cloth_u8)

    hair_bbox = self._mask_bbox(source_hair_mask, threshold=0.18)
    hair_bottom = int(hair_bbox[3] - 1) if hair_bbox is not None else int(y2)
    hair_bottom_ratio = max(0.0, float(hair_bottom - y2) / float(face_h))
    torso_hair_px = int((torso_hair_u8 > 0).sum())
    cloth_overlap_px = int((cloth_overlap_u8 > 0).sum())
    torso_hair_ratio = float(torso_hair_px) / float(face_area)
    cloth_overlap_ratio = float(cloth_overlap_px) / float(face_area)

    if hair_bottom_ratio <= 0.36:
        source_hair_length = "short"
    elif hair_bottom_ratio <= 0.92:
        source_hair_length = "medium"
    else:
        source_hair_length = "long"

    needs_preclean = source_hair_length != "short" and bool(
        cloth_overlap_px >= max(140, int(face_area * 0.012))
        or torso_hair_px >= max(220, int(face_area * 0.16))
        or hair_bottom_ratio >= 0.44
    )

    if normalized_gender == "male":
        skip_preclean = True
        skip_reason = "male_subject"
    elif not needs_preclean:
        skip_preclean = True
        skip_reason = "low_source_garment_occlusion"
    else:
        skip_preclean = False
        skip_reason = "source_garment_occlusion_detected"

    result.update({
        "source_hair_length": source_hair_length,
        "needs_preclean": needs_preclean,
        "skip_preclean": skip_preclean,
        "skip_reason": skip_reason,
        "hair_bottom_ratio": round(hair_bottom_ratio, 4),
        "torso_hair_ratio": round(torso_hair_ratio, 4),
        "cloth_overlap_ratio": round(cloth_overlap_ratio, 4),
        "torso_hair_mask": torso_hair_u8.astype(np.float32) / 255.0,
        "cloth_overlap_mask": cloth_overlap_u8.astype(np.float32) / 255.0,
    })
    return result


def _build_upper_clothes_overwrite_mask(
    self,
    *,
    cloth_mask: Optional[np.ndarray],
    amodal_torso_mask: Optional[np.ndarray],
    shoulder_anchor_mask: Optional[np.ndarray],
    torso_candidate_mask: Optional[np.ndarray],
    completed_torso_fill_mask: Optional[np.ndarray],
    source_torso_hair_mask: Optional[np.ndarray],
    source_cloth_overlap_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    base_shape = None
    for mask in (
        amodal_torso_mask,
        shoulder_anchor_mask,
        cloth_mask,
        torso_candidate_mask,
        completed_torso_fill_mask,
        source_torso_hair_mask,
        source_cloth_overlap_mask,
        protect_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if amodal_torso_mask is not None and amodal_torso_mask.shape != (H, W):
        amodal_torso_mask = None
    if shoulder_anchor_mask is not None and shoulder_anchor_mask.shape != (H, W):
        shoulder_anchor_mask = None
    if cloth_mask is not None and cloth_mask.shape != (H, W):
        cloth_mask = None
    if torso_candidate_mask is not None and torso_candidate_mask.shape != (H, W):
        torso_candidate_mask = None
    if completed_torso_fill_mask is not None and completed_torso_fill_mask.shape != (H, W):
        completed_torso_fill_mask = None
    if source_torso_hair_mask is not None and source_torso_hair_mask.shape != (H, W):
        source_torso_hair_mask = None
    if source_cloth_overlap_mask is not None and source_cloth_overlap_mask.shape != (H, W):
        source_cloth_overlap_mask = None
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.02))
    bottom = min(H, int(y2 + face_h * 1.65))
    left = max(0, int(x1 - face_w * 1.18))
    right = min(W, int(x2 + face_w * 1.18))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    cloth_u8 = np.zeros((H, W), dtype=np.uint8)
    if cloth_mask is not None:
        cloth_dilated = self._dilate_mask_with_px(
            np.clip(cloth_mask.astype(np.float32), 0.0, 1.0),
            max(4, int(self.config.upper_clothes_expand_px)),
        )
        cloth_u8 = (cloth_dilated > 0.04).astype(np.uint8) * 255
        cloth_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)

    amodal_base_u8 = np.zeros((H, W), dtype=np.uint8)
    for mask, kernel in (
        (amodal_torso_mask, (19, 27)),
        (completed_torso_fill_mask, (17, 23)),
        (torso_candidate_mask, (15, 21)),
        (shoulder_anchor_mask, (17, 21)),
    ):
        if mask is None:
            continue
        part_u8 = cv2.dilate(
            (np.clip(mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel),
            iterations=1,
        )
        amodal_base_u8 = cv2.bitwise_or(amodal_base_u8, part_u8)
    amodal_base_u8 = cv2.bitwise_and(amodal_base_u8, corridor_u8)
    if int((amodal_base_u8 > 0).sum()) > 0:
        amodal_base_u8 = cv2.morphologyEx(
            amodal_base_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 27)),
        )
        amodal_base_u8 = cv2.dilate(
            amodal_base_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
            iterations=1,
        )

    torso_u8 = np.zeros((H, W), dtype=np.uint8)
    for mask in (torso_candidate_mask, completed_torso_fill_mask):
        if mask is None:
            continue
        torso_part_u8 = (
            np.clip(mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        torso_u8 = cv2.bitwise_or(torso_u8, torso_part_u8)
    torso_u8 = cv2.bitwise_and(torso_u8, corridor_u8)
    if int((torso_u8 > 0).sum()) > 0:
        torso_u8 = cv2.morphologyEx(
            torso_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 23)),
        )

    support_gate_u8 = cv2.bitwise_or(amodal_base_u8, torso_u8)
    if int((support_gate_u8 > 0).sum()) < 80 and shoulder_anchor_mask is not None:
        anchor_support_u8 = (
            np.clip(shoulder_anchor_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        support_gate_u8 = cv2.bitwise_or(
            support_gate_u8,
            cv2.bitwise_and(anchor_support_u8, corridor_u8),
        )

    cloth_hint_u8 = np.zeros((H, W), dtype=np.uint8)
    if int((cloth_u8 > 0).sum()) > 0 and int((support_gate_u8 > 0).sum()) > 0:
        cloth_hint_u8 = cv2.bitwise_and(
            cloth_u8,
            cv2.dilate(
                support_gate_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 31)),
                iterations=1,
            ),
        )

    overlap_seed_u8 = np.zeros((H, W), dtype=np.uint8)
    if source_cloth_overlap_mask is not None:
        overlap_part_u8 = (
            np.clip(source_cloth_overlap_mask.astype(np.float32), 0.0, 1.0) > 0.04
        ).astype(np.uint8) * 255
        overlap_seed_u8 = cv2.bitwise_or(overlap_seed_u8, overlap_part_u8)
    if source_torso_hair_mask is not None:
        torso_hair_u8 = (
            np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        torso_hair_u8 = cv2.bitwise_and(torso_hair_u8, corridor_u8)
        if int((support_gate_u8 > 0).sum()) > 0:
            cloth_overlap_gate_u8 = cv2.dilate(
                support_gate_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        max(5, int(self.config.hair_overlap_expand_px)),
                        max(5, int(self.config.hair_overlap_expand_px)),
                    ),
                ),
                iterations=1,
            )
            overlap_seed_u8 = cv2.bitwise_or(
                overlap_seed_u8,
                cv2.bitwise_and(torso_hair_u8, cloth_overlap_gate_u8),
            )
        overlap_seed_u8 = cv2.bitwise_or(overlap_seed_u8, cv2.bitwise_and(torso_hair_u8, torso_u8))
    overlap_seed_u8 = cv2.bitwise_and(overlap_seed_u8, corridor_u8)

    front_window_u8 = np.zeros((H, W), dtype=np.uint8)
    front_seed_u8 = cv2.bitwise_or(support_gate_u8, overlap_seed_u8)
    front_seed_u8 = cv2.bitwise_or(front_seed_u8, cloth_hint_u8)
    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    if shoulder_anchor_mask is not None:
        anchor_u8 = (
            np.clip(shoulder_anchor_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        anchor_u8 = cv2.bitwise_and(anchor_u8, corridor_u8)
        front_seed_u8 = cv2.bitwise_or(front_seed_u8, anchor_u8)

    anchor_ys, anchor_xs = np.where(anchor_u8 > 0)
    front_ys, front_xs = np.where(front_seed_u8 > 0)
    if anchor_xs.size > 0 and anchor_ys.size > 0:
        support_y1 = int(anchor_ys.min())
        support_y2 = int(anchor_ys.max())
        support_x1 = int(np.percentile(anchor_xs, 4))
        support_x2 = int(np.percentile(anchor_xs, 96))
        center_x = int(round(0.5 * (support_x1 + support_x2)))
        anchor_half = max(int((support_x2 - support_x1 + 1) * 0.56), int(face_w * 0.58))
    elif front_xs.size > 0 and front_ys.size > 0:
        support_y1 = int(front_ys.min())
        support_y2 = int(front_ys.max())
        support_x1 = int(np.percentile(front_xs, 4))
        support_x2 = int(np.percentile(front_xs, 96))
        center_x = int(round(0.5 * (support_x1 + support_x2)))
        anchor_half = max(int((support_x2 - support_x1 + 1) * 0.50), int(face_w * 0.62))
    else:
        support_y1 = top
        support_y2 = min(H - 1, int(y2 + face_h * 1.10))
        center_x = int(round(0.5 * (x1 + x2)))
        anchor_half = int(face_w * 0.76)

    front_top = max(top, min(support_y1, int(y2 + face_h * 0.04)))
    front_bottom = min(
        H,
        min(
            int(y2 + face_h * 1.18),
            max(
                int(y2 + face_h * 0.92),
                min(H - 1, support_y2) + int(face_h * 0.08),
            ),
        ),
    )
    shoulder_half = max(anchor_half, int(face_w * 0.76))
    lower_half = max(int(face_w * 0.58), int(round(shoulder_half * 0.68)))
    if front_top < front_bottom:
        for y in range(front_top, front_bottom):
            progress = (
                0.0
                if front_bottom <= front_top + 1
                else float(y - front_top) / float(front_bottom - front_top - 1)
            )
            half_width = int(round(shoulder_half * (1.0 - progress) + lower_half * progress))
            left_x = max(0, center_x - half_width)
            right_x = min(W, center_x + half_width)
            if right_x > left_x:
                front_window_u8[y, left_x:right_x] = 255
        front_window_u8 = cv2.morphologyEx(
            front_window_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
        )
        front_window_u8 = cv2.bitwise_and(front_window_u8, corridor_u8)

    front_fill_gate_u8 = cv2.bitwise_or(support_gate_u8, anchor_u8)
    if int((front_fill_gate_u8 > 0).sum()) < 80:
        front_fill_gate_u8 = front_seed_u8.copy()
    overwrite_u8 = cv2.bitwise_or(front_fill_gate_u8, overlap_seed_u8)
    overwrite_u8 = cv2.bitwise_or(overwrite_u8, cloth_hint_u8)
    overwrite_u8 = cv2.bitwise_and(overwrite_u8, corridor_u8)
    if int((front_window_u8 > 0).sum()) > 0:
        overwrite_u8 = cv2.bitwise_or(
            overwrite_u8,
            cv2.bitwise_and(
                front_window_u8,
                cv2.dilate(
                    front_fill_gate_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 49)),
                    iterations=1,
                ),
            ),
        )
    overwrite_u8 = cv2.morphologyEx(
        overwrite_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 27)),
    )
    overwrite_u8 = cv2.dilate(
        overwrite_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    if int((front_window_u8 > 0).sum()) > 0:
        overwrite_u8 = cv2.bitwise_and(overwrite_u8, front_window_u8)
        overwrite_u8 = cv2.bitwise_or(overwrite_u8, cv2.bitwise_and(overlap_seed_u8, front_window_u8))
        overwrite_u8 = cv2.bitwise_or(overwrite_u8, cv2.bitwise_and(front_fill_gate_u8, front_window_u8))
        overwrite_u8 = cv2.morphologyEx(
            overwrite_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
        )

    if protect_mask is not None:
        protect_u8 = (
            np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.10
        ).astype(np.uint8) * 255
        if int((protect_u8 > 0).sum()) > 0:
            protect_u8 = cv2.dilate(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                iterations=1,
            )
            overwrite_u8 = cv2.bitwise_and(overwrite_u8, cv2.bitwise_not(protect_u8))

    if int((overwrite_u8 > 0).sum()) < 180:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        overwrite_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=4.8,
        sigmaY=6.2,
    ).astype(np.float32)


def _build_source_garment_prepass_mask(
    self,
    *,
    hair_length: Optional[str],
    source_torso_hair_mask: Optional[np.ndarray],
    source_cloth_overlap_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    torso_candidate_mask: Optional[np.ndarray] = None,
    completed_torso_fill_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    base_shape = None
    for mask in (
        completed_torso_fill_mask,
        torso_candidate_mask,
        source_torso_hair_mask,
        source_cloth_overlap_mask,
        cloth_mask,
        protect_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if completed_torso_fill_mask is not None and completed_torso_fill_mask.shape != (H, W):
        completed_torso_fill_mask = None
    if torso_candidate_mask is not None and torso_candidate_mask.shape != (H, W):
        torso_candidate_mask = None
    if source_torso_hair_mask is not None and source_torso_hair_mask.shape != (H, W):
        source_torso_hair_mask = None
    if source_cloth_overlap_mask is not None and source_cloth_overlap_mask.shape != (H, W):
        source_cloth_overlap_mask = None
    if cloth_mask is not None and cloth_mask.shape != (H, W):
        cloth_mask = None
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    def _odd_size(value: float, minimum: int = 3) -> int:
        size = max(int(minimum), int(round(float(value))))
        return size if size % 2 == 1 else size + 1

    def _filter_components_by_area(
        mask_u8: np.ndarray,
        *,
        min_area: int,
        max_area: Optional[int] = None,
    ) -> np.ndarray:
        if int((mask_u8 > 0).sum()) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        filtered_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask_u8 > 0).astype(np.uint8),
            8,
        )
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            if max_area is not None and area > int(max_area):
                continue
            filtered_u8[labels == label] = 255
        return filtered_u8

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.02))
    bottom = min(H, int(y2 + face_h * 1.80))
    left = max(0, int(x1 - face_w * 1.36))
    right = min(W, int(x2 + face_w * 1.36))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    garment_u8 = np.zeros((H, W), dtype=np.uint8)
    candidate_u8 = np.zeros((H, W), dtype=np.uint8)
    if torso_candidate_mask is not None:
        candidate_u8 = (
            np.clip(torso_candidate_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
        candidate_u8 = _filter_components_by_area(
            candidate_u8,
            min_area=max(24, int(face_w * face_h * 0.0012)),
        )

    bbox_source_u8 = candidate_u8.copy()
    if int((bbox_source_u8 > 0).sum()) == 0 and completed_torso_fill_mask is not None:
        bbox_source_u8 = (
            np.clip(completed_torso_fill_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        bbox_source_u8 = cv2.bitwise_and(bbox_source_u8, corridor_u8)
    if int((bbox_source_u8 > 0).sum()) == 0 and source_torso_hair_mask is not None:
        bbox_source_u8 = (
            np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        bbox_source_u8 = cv2.bitwise_and(bbox_source_u8, corridor_u8)

    bbox_ys, bbox_xs = np.where(bbox_source_u8 > 0)
    if bbox_xs.size > 0 and bbox_ys.size > 0:
        support_y1 = int(bbox_ys.min())
        support_y2 = int(bbox_ys.max())
    else:
        support_y1 = top
        support_y2 = bottom - 1
    support_h = max(support_y2 - support_y1 + 1, 1)

    completed_u8 = np.zeros((H, W), dtype=np.uint8)
    if completed_torso_fill_mask is not None:
        completed_u8 = (
            np.clip(completed_torso_fill_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        completed_u8 = cv2.bitwise_and(completed_u8, corridor_u8)

    overlap_u8 = np.zeros((H, W), dtype=np.uint8)
    if source_cloth_overlap_mask is not None:
        overlap_u8 = (
            np.clip(source_cloth_overlap_mask.astype(np.float32), 0.0, 1.0) > 0.04
        ).astype(np.uint8) * 255
        overlap_u8 = cv2.bitwise_and(overlap_u8, corridor_u8)

    torso_hair_u8 = np.zeros((H, W), dtype=np.uint8)
    if source_torso_hair_mask is not None:
        torso_hair_u8 = (
            np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        torso_hair_u8 = cv2.bitwise_and(torso_hair_u8, corridor_u8)

    extra_u8 = np.zeros((H, W), dtype=np.uint8)
    if int((completed_u8 > 0).sum()) > 0:
        extra_u8 = cv2.subtract(completed_u8, candidate_u8)
        extra_u8 = cv2.bitwise_and(extra_u8, corridor_u8)

    cloth_u8 = np.zeros((H, W), dtype=np.uint8)
    if cloth_mask is not None:
        cloth_u8 = (
            np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04
        ).astype(np.uint8) * 255
        cloth_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)

    torso_hair_support_u8 = np.zeros((H, W), dtype=np.uint8)
    if hair_length == "short" and int((torso_hair_u8 > 0).sum()) > 0:
        torso_support_gate_u8 = cv2.bitwise_or(candidate_u8, completed_u8)
        if int((torso_support_gate_u8 > 0).sum()) > 0:
            torso_support_gate_u8 = cv2.dilate(
                torso_support_gate_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        _odd_size(
                            face_w
                            * float(
                                self.config.source_garment_prepass_torso_support_expand_x_ratio
                            ),
                            15,
                        ),
                        _odd_size(
                            face_h
                            * float(
                                self.config.source_garment_prepass_torso_support_expand_y_ratio
                            ),
                            11,
                        ),
                    ),
                ),
                iterations=1,
            )

        cloth_support_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        if int((cloth_u8 > 0).sum()) > 0:
            cloth_support_gate_u8 = cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        _odd_size(
                            face_w
                            * float(
                                self.config.source_garment_prepass_cloth_expand_x_ratio
                            ),
                            17,
                        ),
                        _odd_size(
                            face_h
                            * float(
                                self.config.source_garment_prepass_cloth_expand_y_ratio
                            ),
                            13,
                        ),
                    ),
                ),
                iterations=1,
            )

        overlap_support_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        if int((overlap_u8 > 0).sum()) > 0:
            overlap_support_gate_u8 = cv2.dilate(
                overlap_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (_odd_size(face_w * 0.08, 11), _odd_size(face_h * 0.10, 11)),
                ),
                iterations=1,
            )

        torso_hair_support_gate_u8 = cv2.bitwise_or(
            torso_support_gate_u8,
            cloth_support_gate_u8,
        )
        torso_hair_support_gate_u8 = cv2.bitwise_or(
            torso_hair_support_gate_u8,
            overlap_support_gate_u8,
        )
        if int((torso_hair_support_gate_u8 > 0).sum()) > 0:
            torso_hair_seed_u8 = cv2.bitwise_and(
                torso_hair_u8,
                torso_hair_support_gate_u8,
            )
            filtered_torso_hair_support_u8 = np.zeros((H, W), dtype=np.uint8)
            min_torso_hair_area = max(
                18,
                int(
                    face_w
                    * face_h
                    * float(self.config.source_garment_prepass_torso_hair_min_area_ratio)
                ),
            )
            min_torso_hair_bottom = support_y1 + int(support_h * 0.12)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                (torso_hair_seed_u8 > 0).astype(np.uint8),
                8,
            )
            for label in range(1, num_labels):
                y = int(stats[label, cv2.CC_STAT_TOP])
                h = int(stats[label, cv2.CC_STAT_HEIGHT])
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < min_torso_hair_area:
                    continue

                component_u8 = np.zeros((H, W), dtype=np.uint8)
                component_u8[labels == label] = 255
                bottom = y + h - 1
                if bottom < min_torso_hair_bottom:
                    continue

                support_pixels = int(
                    np.logical_and(component_u8 > 0, torso_hair_support_gate_u8 > 0).sum()
                )
                cloth_pixels = int(
                    np.logical_and(component_u8 > 0, cloth_support_gate_u8 > 0).sum()
                )
                if support_pixels <= 0:
                    continue
                if int((cloth_support_gate_u8 > 0).sum()) > 0 and cloth_pixels <= 0:
                    continue
                filtered_torso_hair_support_u8 = cv2.bitwise_or(
                    filtered_torso_hair_support_u8,
                    component_u8,
                )

            if int((filtered_torso_hair_support_u8 > 0).sum()) > 0:
                torso_hair_support_u8 = cv2.morphologyEx(
                    filtered_torso_hair_support_u8,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (_odd_size(face_w * 0.020, 5), _odd_size(face_h * 0.024, 5)),
                    ),
                )

    extra_filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    if int((extra_u8 > 0).sum()) > 0:
        overlap_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        if int((overlap_u8 > 0).sum()) > 0:
            overlap_gate_u8 = cv2.dilate(
                overlap_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (_odd_size(face_w * 0.10, 15), _odd_size(face_h * 0.12, 15)),
                ),
                iterations=1,
            )
        min_extra_area = max(12, int(face_w * face_h * 0.00015))
        max_extra_bottom = support_y1 + int(support_h * 0.45)
        thin_extra_width = max(12, int(face_w * 0.10))
        tall_extra_height = max(42, int(face_h * 0.34))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (extra_u8 > 0).astype(np.uint8),
            8,
        )
        for label in range(1, num_labels):
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_extra_area:
                continue
            if y > max_extra_bottom:
                continue

            component_u8 = np.zeros((H, W), dtype=np.uint8)
            component_u8[labels == label] = 255
            overlap_pixels = int(
                np.logical_and(component_u8 > 0, overlap_gate_u8 > 0).sum()
            )
            hair_pixels = int(
                np.logical_and(component_u8 > 0, torso_hair_u8 > 0).sum()
            )
            if overlap_pixels <= 0 and hair_pixels < int(area * 0.40):
                continue
            if w <= thin_extra_width and h >= tall_extra_height:
                continue
            extra_filtered_u8 = cv2.bitwise_or(extra_filtered_u8, component_u8)

    overlap_seed_u8 = np.zeros((H, W), dtype=np.uint8)
    torso_support_u8 = cv2.bitwise_or(candidate_u8, completed_u8)
    if int((overlap_u8 > 0).sum()) > 0 and int((torso_support_u8 > 0).sum()) > 0:
        inner_support_u8 = cv2.erode(
            torso_support_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_size(face_w * 0.02, 5), _odd_size(face_h * 0.03, 5)),
            ),
            iterations=1,
        )
        overlap_seed_u8 = cv2.dilate(
            overlap_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_size(face_w * 0.025, 7), _odd_size(face_h * 0.035, 7)),
            ),
            iterations=1,
        )
        overlap_seed_u8 = cv2.bitwise_and(overlap_seed_u8, inner_support_u8)
        upper_seed_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        upper_seed_top = max(top, support_y1 - 2)
        upper_seed_bottom = min(H, support_y1 + int(support_h * 0.55))
        if upper_seed_top < upper_seed_bottom:
            upper_seed_gate_u8[upper_seed_top:upper_seed_bottom, :] = 255
            overlap_seed_u8 = cv2.bitwise_and(overlap_seed_u8, upper_seed_gate_u8)
        if int((cloth_u8 > 0).sum()) > 0:
            cloth_local_support_u8 = cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (_odd_size(face_w * 0.12, 13), _odd_size(face_h * 0.10, 13)),
                ),
                iterations=1,
            )
            overlap_seed_u8 = cv2.bitwise_and(overlap_seed_u8, cloth_local_support_u8)

        filtered_overlap_seed_u8 = np.zeros((H, W), dtype=np.uint8)
        min_overlap_area = max(16, int(face_w * face_h * 0.00012))
        max_overlap_bottom = support_y1 + int(support_h * 0.55)
        thin_overlap_width = max(14, int(face_w * 0.09))
        tall_overlap_height = max(52, int(face_h * 0.32))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (overlap_seed_u8 > 0).astype(np.uint8),
            8,
        )
        for label in range(1, num_labels):
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_overlap_area:
                continue
            if y > max_overlap_bottom:
                continue
            if w <= thin_overlap_width and h >= tall_overlap_height:
                continue

            component_u8 = np.zeros((H, W), dtype=np.uint8)
            component_u8[labels == label] = 255
            overlap_pixels = int(
                np.logical_and(component_u8 > 0, overlap_u8 > 0).sum()
            )
            support_pixels = int(
                np.logical_and(
                    cv2.dilate(
                        component_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                        iterations=1,
                    ) > 0,
                    inner_support_u8 > 0,
                ).sum()
            )
            if overlap_pixels <= 0:
                continue
            if support_pixels < max(12, int(area * 0.10)):
                continue
            filtered_overlap_seed_u8 = cv2.bitwise_or(
                filtered_overlap_seed_u8,
                component_u8,
            )
        overlap_seed_u8 = filtered_overlap_seed_u8

    garment_u8 = cv2.bitwise_or(extra_filtered_u8, overlap_seed_u8)
    garment_u8 = cv2.bitwise_or(garment_u8, torso_hair_support_u8)

    if int((garment_u8 > 0).sum()) > 0:
        garment_u8 = cv2.morphologyEx(
            garment_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_size(face_w * 0.016, 5), _odd_size(face_h * 0.020, 5)),
            ),
        )

    if cloth_mask is not None and int((garment_u8 > 0).sum()) > 0:
        if int((cloth_u8 > 0).sum()) > 0:
            cloth_support_u8 = cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (_odd_size(face_w * 0.14, 15), _odd_size(face_h * 0.12, 15)),
                ),
                iterations=1,
            )
            supported_u8 = cv2.bitwise_and(garment_u8, cloth_support_u8)
            if int((supported_u8 > 0).sum()) > 0:
                garment_u8 = supported_u8

    garment_u8 = cv2.bitwise_and(garment_u8, corridor_u8)
    filtered_garment_u8 = np.zeros((H, W), dtype=np.uint8)
    thin_final_width = max(14, int(face_w * 0.09))
    tall_final_height = max(52, int(face_h * 0.32))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (garment_u8 > 0).astype(np.uint8),
        8,
    )
    for label in range(1, num_labels):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(18, int(face_w * face_h * 0.0002)):
            continue
        if w <= thin_final_width and h >= tall_final_height:
            continue

        component_u8 = np.zeros((H, W), dtype=np.uint8)
        component_u8[labels == label] = 255
        overlap_pixels = int(
            np.logical_and(component_u8 > 0, overlap_u8 > 0).sum()
        )
        extra_pixels = int(
            np.logical_and(component_u8 > 0, extra_filtered_u8 > 0).sum()
        )
        torso_hair_support_pixels = int(
            np.logical_and(component_u8 > 0, torso_hair_support_u8 > 0).sum()
        )
        if (
            overlap_pixels <= 0
            and extra_pixels <= 0
            and torso_hair_support_pixels <= 0
        ):
            continue
        filtered_garment_u8 = cv2.bitwise_or(filtered_garment_u8, component_u8)
    garment_u8 = filtered_garment_u8

    if protect_mask is not None:
        protect_u8 = (
            np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        if int((protect_u8 > 0).sum()) > 0:
            protect_u8 = cv2.dilate(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
                iterations=1,
            )
            garment_u8 = cv2.bitwise_and(garment_u8, cv2.bitwise_not(protect_u8))

    garment_u8 = cv2.morphologyEx(
        garment_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return garment_u8.astype(np.float32) / 255.0


def _build_completed_subject_torso_fill_mask(
    self,
    *,
    torso_candidate_mask: Optional[np.ndarray],
    source_torso_hair_mask: Optional[np.ndarray],
    shoulder_bridge_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    base_shape = None
    for mask in (
        torso_candidate_mask,
        source_torso_hair_mask,
        shoulder_bridge_mask,
        protect_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if torso_candidate_mask is not None and torso_candidate_mask.shape != (H, W):
        torso_candidate_mask = None
    if source_torso_hair_mask is not None and source_torso_hair_mask.shape != (H, W):
        source_torso_hair_mask = None
    if shoulder_bridge_mask is not None and shoulder_bridge_mask.shape != (H, W):
        shoulder_bridge_mask = None
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    def _odd_size(value: float, minimum: int = 3) -> int:
        size = max(int(minimum), int(round(float(value))))
        return size if size % 2 == 1 else size + 1

    def _filter_components_by_area(
        mask_u8: np.ndarray,
        *,
        min_area: int,
        max_area: Optional[int] = None,
    ) -> np.ndarray:
        if int((mask_u8 > 0).sum()) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        filtered_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask_u8 > 0).astype(np.uint8),
            8,
        )
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            if max_area is not None and area > int(max_area):
                continue
            filtered_u8[labels == label] = 255
        return filtered_u8

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.02))
    bottom = min(H, int(y2 + face_h * 1.80))
    left = max(0, int(x1 - face_w * 1.36))
    right = min(W, int(x2 + face_w * 1.36))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    candidate_u8 = np.zeros((H, W), dtype=np.uint8)
    if torso_candidate_mask is not None:
        candidate_u8 = (
            np.clip(torso_candidate_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)

    if int((candidate_u8 > 0).sum()) == 0 and source_torso_hair_mask is not None:
        candidate_u8 = (
            np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
        candidate_u8 = cv2.dilate(
            candidate_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
            iterations=1,
        )

    if int((candidate_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.float32)

    cleaned_candidate_u8 = _filter_components_by_area(
        candidate_u8,
        min_area=max(24, int(face_w * face_h * 0.0012)),
    )
    if int((cleaned_candidate_u8 > 0).sum()) > 0:
        candidate_u8 = cleaned_candidate_u8

    candidate_ys, candidate_xs = np.where(candidate_u8 > 0)
    if candidate_xs.size == 0 or candidate_ys.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    candidate_x1 = int(candidate_xs.min())
    candidate_y1 = int(candidate_ys.min())
    candidate_x2 = int(candidate_xs.max())
    candidate_y2 = int(candidate_ys.max())
    candidate_w = max(candidate_x2 - candidate_x1 + 1, 1)
    candidate_h = max(candidate_y2 - candidate_y1 + 1, 1)

    hair_u8 = np.zeros((H, W), dtype=np.uint8)
    if source_torso_hair_mask is not None:
        hair_u8 = (
            np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        hair_u8 = cv2.bitwise_and(hair_u8, corridor_u8)

    hair_fill_gate_u8 = np.zeros((H, W), dtype=np.uint8)
    if int((hair_u8 > 0).sum()) > 0:
        hair_fill_gate_u8 = cv2.dilate(
            hair_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_size(face_w * 0.035, 5), _odd_size(face_h * 0.05, 5)),
            ),
            iterations=1,
        )
        hair_fill_gate_u8 = cv2.bitwise_and(hair_fill_gate_u8, corridor_u8)

    upper_band_u8 = np.zeros((H, W), dtype=np.uint8)
    upper_top = max(candidate_y1 - 2, int(y2 + face_h * 0.02))
    upper_bottom = min(
        candidate_y2 + 1,
        max(candidate_y1 + 1, int(y2 + face_h * 0.48)),
    )
    if upper_top < upper_bottom:
        upper_band_u8[upper_top:upper_bottom, :] = 255

    candidate_upper_support_u8 = cv2.bitwise_and(candidate_u8, upper_band_u8)
    upper_support_u8 = candidate_upper_support_u8.copy()
    bridge_u8 = np.zeros((H, W), dtype=np.uint8)
    bridge_anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    if shoulder_bridge_mask is not None:
        bridge_u8 = (
            np.clip(shoulder_bridge_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        bridge_u8 = cv2.bitwise_and(bridge_u8, upper_band_u8)
        bridge_u8 = cv2.erode(
            bridge_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 9)),
            iterations=1,
        )
        if int((candidate_upper_support_u8 > 0).sum()) > 0:
            bridge_anchor_gate_u8 = cv2.dilate(
                candidate_upper_support_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (_odd_size(face_w * 0.16, 21), _odd_size(face_h * 0.10, 17)),
                ),
                iterations=1,
            )
            bridge_anchor_u8 = cv2.bitwise_and(bridge_u8, bridge_anchor_gate_u8)
            bridge_anchor_u8 = cv2.morphologyEx(
                bridge_anchor_u8,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            bridge_anchor_u8 = _filter_components_by_area(
                bridge_anchor_u8,
                min_area=max(16, int(face_w * face_h * 0.00012)),
                max_area=max(1800, int(face_w * face_h * 0.0045)),
            )
            upper_support_u8 = cv2.bitwise_or(upper_support_u8, bridge_anchor_u8)

    top_profile = np.full(W, -1, dtype=np.int32)
    support_x_left = max(left, candidate_x1 - int(candidate_w * 0.02))
    support_x_right = min(right - 1, candidate_x2 + int(candidate_w * 0.02))
    profile_source_u8 = candidate_upper_support_u8
    if int((profile_source_u8 > 0).sum()) < max(28, int(face_w * 0.08)):
        profile_source_u8 = upper_support_u8
    for x in range(max(0, support_x_left), min(W, support_x_right + 1)):
        ys = np.where(profile_source_u8[:, x] > 0)[0]
        if ys.size > 0:
            top_profile[x] = int(ys.min())

    valid_x = np.where(top_profile >= 0)[0]
    top_line_u8 = np.zeros((H, W), dtype=np.uint8)
    if valid_x.size >= 2:
        span_x = np.arange(int(valid_x.min()), int(valid_x.max()) + 1, dtype=np.int32)
        span_y = np.interp(
            span_x,
            valid_x.astype(np.float32),
            top_profile[valid_x].astype(np.float32),
        )
        sigma_x = max(3.0, face_w * 0.045)
        span_y = cv2.GaussianBlur(
            span_y.reshape(1, -1).astype(np.float32),
            (0, 0),
            sigmaX=max(2.0, min(sigma_x, face_w * 0.03)),
        ).reshape(-1)
        span_y = np.clip(
            np.rint(span_y).astype(np.int32),
            upper_top,
            min(upper_bottom - 1, candidate_y1 + max(6, int(candidate_h * 0.18))),
        )
        pts = np.stack([span_x, span_y], axis=1).reshape(-1, 1, 2)
        cv2.polylines(
            top_line_u8,
            [pts],
            isClosed=False,
            color=255,
            thickness=max(5, int(face_h * 0.02)),
            lineType=cv2.LINE_AA,
        )
        seam_guard_u8 = cv2.bitwise_or(
            bridge_anchor_u8,
            candidate_upper_support_u8,
        )
        seam_guard_u8 = cv2.dilate(
            seam_guard_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_size(face_w * 0.10, 5), _odd_size(face_h * 0.05, 5)),
            ),
            iterations=1,
        )
        support_x_mask_u8 = np.zeros((H, W), dtype=np.uint8)
        support_x_mask_u8[:, max(0, support_x_left) : min(W, support_x_right + 1)] = 255
        top_line_u8 = cv2.bitwise_and(top_line_u8, seam_guard_u8)
        top_line_u8 = cv2.bitwise_and(top_line_u8, support_x_mask_u8)

    base_fill_u8 = cv2.bitwise_or(candidate_u8, top_line_u8)
    local_fill_u8 = np.zeros((H, W), dtype=np.uint8)
    if int((hair_fill_gate_u8 > 0).sum()) > 0:
        closed_fill_u8 = cv2.morphologyEx(
            base_fill_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd_size(face_w * 0.07, 5), _odd_size(face_h * 0.10, 7)),
            ),
        )
        local_fill_u8 = cv2.bitwise_and(
            closed_fill_u8,
            cv2.bitwise_not(base_fill_u8),
        )
        local_fill_u8 = cv2.bitwise_and(local_fill_u8, hair_fill_gate_u8)

        filtered_local_fill_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (local_fill_u8 > 0).astype(np.uint8),
            8,
        )
        min_fill_area = max(12, int(face_w * face_h * 0.00012))
        max_fill_area = max(600, int(face_w * face_h * 0.01))
        max_fill_bottom = candidate_y1 + int(candidate_h * 0.55)
        max_fill_height = max(28, int(candidate_h * 0.18))
        lateral_margin = max(4, int(candidate_w * 0.03))
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_fill_area or area > max_fill_area:
                continue
            if y >= max_fill_bottom or h > max_fill_height:
                continue
            if x <= candidate_x1 + lateral_margin or x + w >= candidate_x2 - lateral_margin:
                continue

            component_u8 = np.zeros((H, W), dtype=np.uint8)
            component_u8[labels == label] = 255
            contact_u8 = cv2.dilate(
                component_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            contact_pixels = int(
                np.logical_and(contact_u8 > 0, candidate_u8 > 0).sum()
            )
            contact_ratio = float(contact_pixels) / float(max(area, 1))
            if contact_ratio < 1.8:
                continue
            filtered_local_fill_u8 = cv2.bitwise_or(
                filtered_local_fill_u8,
                component_u8,
            )
        local_fill_u8 = filtered_local_fill_u8

    filled_u8 = cv2.bitwise_or(base_fill_u8, local_fill_u8)
    filled_u8 = cv2.bitwise_and(filled_u8, corridor_u8)

    if protect_mask is not None:
        protect_u8 = (
            np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        protect_u8 = cv2.dilate(
            protect_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
            iterations=1,
        )
        filled_u8 = cv2.bitwise_and(filled_u8, cv2.bitwise_not(protect_u8))

    filled_u8 = np.where(filled_u8 > 0, 255, 0).astype(np.uint8)
    filled_u8 = cv2.morphologyEx(
        filled_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return filled_u8.astype(np.float32) / 255.0


def _build_source_shoulder_contour_anchor_mask(
    self,
    *,
    source_torso_hair_mask: Optional[np.ndarray],
    source_cloth_overlap_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    shoulder_bridge_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    base_shape = None
    for mask in (
        cloth_mask,
        source_torso_hair_mask,
        source_cloth_overlap_mask,
        shoulder_bridge_mask,
        protect_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if cloth_mask is not None and cloth_mask.shape != (H, W):
        cloth_mask = None
    if source_torso_hair_mask is not None and source_torso_hair_mask.shape != (H, W):
        source_torso_hair_mask = None
    if source_cloth_overlap_mask is not None and source_cloth_overlap_mask.shape != (H, W):
        source_cloth_overlap_mask = None
    if shoulder_bridge_mask is not None and shoulder_bridge_mask.shape != (H, W):
        shoulder_bridge_mask = None
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    shoulder_band_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.00))
    bottom = min(H, int(y2 + face_h * 0.68))
    left = max(0, int(x1 - face_w * 1.34))
    right = min(W, int(x2 + face_w * 1.34))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    shoulder_band_u8[top:bottom, left:right] = 255

    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    if cloth_mask is not None:
        cloth_u8 = (
            np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04
        ).astype(np.uint8) * 255
        cloth_outer_u8 = cv2.dilate(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        )
        cloth_inner_u8 = cv2.erode(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        cloth_edge_u8 = cv2.subtract(cloth_outer_u8, cloth_inner_u8)
        anchor_u8 = cv2.bitwise_and(cloth_edge_u8, shoulder_band_u8)

    support_u8 = np.zeros((H, W), dtype=np.uint8)
    if source_torso_hair_mask is not None:
        torso_u8 = (
            np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        torso_u8 = cv2.dilate(
            torso_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 23)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_or(support_u8, torso_u8)
    if source_cloth_overlap_mask is not None:
        overlap_u8 = (
            np.clip(source_cloth_overlap_mask.astype(np.float32), 0.0, 1.0) > 0.04
        ).astype(np.uint8) * 255
        overlap_u8 = cv2.dilate(
            overlap_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_or(support_u8, overlap_u8)
    if shoulder_bridge_mask is not None:
        bridge_u8 = (
            np.clip(shoulder_bridge_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        bridge_u8 = cv2.bitwise_and(bridge_u8, shoulder_band_u8)
        bridge_u8 = cv2.dilate(
            bridge_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        support_u8 = cv2.bitwise_or(support_u8, bridge_u8)

    if int((support_u8 > 0).sum()) >= 80:
        focused_u8 = cv2.bitwise_and(anchor_u8, support_u8)
        if int((focused_u8 > 0).sum()) >= 60:
            anchor_u8 = focused_u8

    anchor_u8 = cv2.morphologyEx(
        anchor_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
    )
    anchor_u8 = cv2.dilate(
        anchor_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
        iterations=1,
    )

    if protect_mask is not None:
        protect_u8 = (
            np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        protect_u8 = cv2.dilate(
            protect_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        anchor_u8 = cv2.bitwise_and(anchor_u8, cv2.bitwise_not(protect_u8))

    anchor_u8 = cv2.bitwise_and(anchor_u8, shoulder_band_u8)
    anchor_u8 = cv2.morphologyEx(
        anchor_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return anchor_u8.astype(np.float32) / 255.0


def _build_short_shoulder_cross_bridge_mask(
    self,
    *,
    source_shoulder_contour_anchor_mask: Optional[np.ndarray],
    source_torso_hair_mask: Optional[np.ndarray],
    cloth_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    support_mask: Optional[np.ndarray] = None,
    center_support_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    base_shape = None
    for mask in (
        source_shoulder_contour_anchor_mask,
        source_torso_hair_mask,
        cloth_mask,
        protect_mask,
        support_mask,
        center_support_mask,
    ):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if (
        source_shoulder_contour_anchor_mask is None
        or source_shoulder_contour_anchor_mask.shape != (H, W)
    ):
        return np.zeros((H, W), dtype=np.float32)
    if source_torso_hair_mask is not None and source_torso_hair_mask.shape != (H, W):
        source_torso_hair_mask = None
    if cloth_mask is not None and cloth_mask.shape != (H, W):
        cloth_mask = None
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None
    if support_mask is not None and support_mask.shape != (H, W):
        support_mask = None
    if center_support_mask is not None and center_support_mask.shape != (H, W):
        center_support_mask = None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    face_cx = int(0.5 * (x1 + x2))

    band_u8 = np.zeros((H, W), dtype=np.uint8)
    band_top = max(0, int(y2 + face_h * 0.04))
    band_bottom = min(H, int(y2 + face_h * 0.72))
    band_left = max(0, int(x1 - face_w * 1.18))
    band_right = min(W, int(x2 + face_w * 1.18))
    if band_top >= band_bottom or band_left >= band_right:
        return np.zeros((H, W), dtype=np.float32)
    band_u8[band_top:band_bottom, band_left:band_right] = 255

    anchor_u8 = (
        np.clip(source_shoulder_contour_anchor_mask.astype(np.float32), 0.0, 1.0) > 0.08
    ).astype(np.uint8) * 255
    anchor_u8 = cv2.dilate(
        anchor_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    anchor_u8 = cv2.bitwise_and(anchor_u8, band_u8)
    if int((anchor_u8 > 0).sum()) < max(20, int(face_w * face_h * 0.00015)):
        return np.zeros((H, W), dtype=np.float32)

    left_anchor_u8 = anchor_u8.copy()
    left_anchor_u8[:, face_cx:] = 0
    right_anchor_u8 = anchor_u8.copy()
    right_anchor_u8[:, :face_cx] = 0
    if int((left_anchor_u8 > 0).sum()) < 10 or int((right_anchor_u8 > 0).sum()) < 10:
        return np.zeros((H, W), dtype=np.float32)

    left_ys, left_xs = np.where(left_anchor_u8 > 0)
    right_ys, right_xs = np.where(right_anchor_u8 > 0)
    left_inner_x = int(left_xs.max())
    right_inner_x = int(right_xs.min())
    if left_inner_x >= right_inner_x:
        return np.zeros((H, W), dtype=np.float32)

    left_strip = left_ys[left_xs >= max(band_left, left_inner_x - max(4, int(face_w * 0.03)))]
    right_strip = right_ys[right_xs <= min(band_right - 1, right_inner_x + max(4, int(face_w * 0.03)))]
    if left_strip.size == 0 or right_strip.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    left_y = int(np.percentile(left_strip, 35))
    right_y = int(np.percentile(right_strip, 35))
    left_pt = (left_inner_x, int(np.clip(left_y, band_top, band_bottom - 1)))
    right_pt = (right_inner_x, int(np.clip(right_y, band_top, band_bottom - 1)))

    bridge_u8 = np.zeros((H, W), dtype=np.uint8)
    bridge_thickness = max(12, int(face_h * 0.16))
    cv2.line(bridge_u8, left_pt, right_pt, 255, thickness=bridge_thickness)
    bridge_u8 = cv2.morphologyEx(
        bridge_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    support_u8 = np.zeros((H, W), dtype=np.uint8)
    if source_torso_hair_mask is not None:
        torso_u8 = (
            np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        support_u8 = cv2.bitwise_or(
            support_u8,
            cv2.dilate(
                torso_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 17)),
                iterations=1,
            ),
        )
    if support_mask is not None:
        support_u8 = cv2.bitwise_or(
            support_u8,
            cv2.dilate(
                (
                    np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08
                ).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            ),
        )
    if center_support_mask is not None:
        support_u8 = cv2.bitwise_or(
            support_u8,
            cv2.dilate(
                (
                    np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08
                ).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
                iterations=1,
            ),
        )
    if cloth_mask is not None:
        cloth_u8 = (
            np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04
        ).astype(np.uint8) * 255
        support_u8 = cv2.bitwise_or(
            support_u8,
            cv2.dilate(
                cloth_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            ),
        )

    if int((support_u8 > 0).sum()) > 0:
        bridge_overlap = cv2.bitwise_and(
            bridge_u8,
            cv2.dilate(
                support_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 11)),
                iterations=1,
            ),
        )
        if int((bridge_overlap > 0).sum()) >= max(40, int(face_w * 0.12)):
            bridge_u8 = cv2.bitwise_or(bridge_u8, bridge_overlap)

    bridge_u8 = cv2.bitwise_and(bridge_u8, band_u8)
    bridge_u8 = cv2.bitwise_or(bridge_u8, anchor_u8)

    if protect_mask is not None:
        protect_u8 = (
            np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        protect_u8 = cv2.dilate(
            protect_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        bridge_u8 = cv2.bitwise_and(bridge_u8, cv2.bitwise_not(protect_u8))

    bridge_u8 = cv2.morphologyEx(
        bridge_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return bridge_u8.astype(np.float32) / 255.0


def _build_residual_side_hair_lane_removal_mask(
    self,
    *,
    source_torso_hair_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    base_shape = None
    for mask in (source_torso_hair_mask, protect_mask):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if source_torso_hair_mask is None or source_torso_hair_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if protect_mask is not None and protect_mask.shape != (H, W):
        protect_mask = None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    face_cx = int(0.5 * (x1 + x2))

    hair_u8 = (
        np.clip(source_torso_hair_mask.astype(np.float32), 0.0, 1.0) > 0.08
    ).astype(np.uint8) * 255
    if int((hair_u8 > 0).sum()) < max(36, int(face_w * face_h * 0.00018)):
        return np.zeros((H, W), dtype=np.float32)

    lane_gate_u8 = np.zeros((H, W), dtype=np.uint8)
    gate_top = max(0, max(int(cutoff_y), int(y2 + face_h * 0.02)))
    gate_bottom = min(H, int(y2 + face_h * 1.34))
    left_outer = max(0, int(x1 - face_w * 0.42))
    left_inner = max(left_outer + 1, int(face_cx - face_w * 0.12))
    right_inner = min(W - 1, int(face_cx + face_w * 0.12))
    right_outer = min(W, int(x2 + face_w * 0.42))
    if gate_top >= gate_bottom or left_outer >= left_inner or right_inner >= right_outer:
        return np.zeros((H, W), dtype=np.float32)
    lane_gate_u8[gate_top:gate_bottom, left_outer:left_inner] = 255
    lane_gate_u8[gate_top:gate_bottom, right_inner:right_outer] = 255

    candidate_u8 = cv2.bitwise_and(hair_u8, lane_gate_u8)
    if int((candidate_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.float32)

    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
    )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    min_area = max(26, int(face_w * face_h * 0.00018))
    min_height = max(42, int(face_h * 0.26))
    max_width = max(112, int(face_w * 0.32))
    min_bottom = min(H, int(y2 + face_h * 0.34))
    max_top = min(H, int(y2 + face_h * 0.78))
    side_offset = max(18, int(face_w * 0.16))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (candidate_u8 > 0).astype(np.uint8),
        8,
    )
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        bottom = y + h
        comp_cx = float(centroids[label][0])
        if area < min_area or h < min_height or w > max_width:
            continue
        if bottom < min_bottom or y > max_top:
            continue
        if abs(comp_cx - float(face_cx)) < side_offset:
            continue

        component_u8 = np.zeros((H, W), dtype=np.uint8)
        component_u8[labels == label] = 255
        lane_pixels = int(np.logical_and(component_u8 > 0, lane_gate_u8 > 0).sum())
        if lane_pixels < max(14, int(area * 0.42)):
            continue
        keep_u8 = cv2.bitwise_or(keep_u8, component_u8)

    if int((keep_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, lane_gate_u8)

    if protect_mask is not None:
        protect_u8 = (
            np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.08
        ).astype(np.uint8) * 255
        protect_u8 = cv2.dilate(
            protect_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(protect_u8))

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return keep_u8.astype(np.float32) / 255.0


def _estimate_head_generation_box(
    self,
    *,
    image_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
    landmark_face_mask: Optional[np.ndarray],
    landmark_debug_data: Optional[Dict[str, Any]],
    hair_length: str,
) -> Tuple[int, int, int, int, int]:
    H, W = image_shape
    x1f, y1f, x2f, y2f = [int(v) for v in face_bbox]
    ref_x1, ref_y1, ref_x2, ref_y2 = x1f, y1f, x2f, y2f

    mask_bbox = self._mask_bbox(landmark_face_mask, threshold=0.20)
    if mask_bbox is not None:
        ref_x1, ref_y1, ref_x2, ref_y2 = mask_bbox

    keypoints = landmark_debug_data.get("keypoints", {}) if isinstance(landmark_debug_data, dict) else {}
    forehead_top = keypoints.get("forehead_top", {}).get("px")
    chin = keypoints.get("chin", {}).get("px")
    forehead_y = int(forehead_top[1]) if isinstance(forehead_top, list) and len(forehead_top) >= 2 else ref_y1
    chin_y = int(chin[1]) if isinstance(chin, list) and len(chin) >= 2 else ref_y2

    ref_w = max(int(ref_x2 - ref_x1), 1)
    ref_h = max(int(max(ref_y2, chin_y) - min(ref_y1, forehead_y)), 1)

    if hair_length == "short":
        margin_x = int(ref_w * 0.42)
        top_pad = int(ref_h * 0.72)
        cutoff_y = int(chin_y + ref_h * 0.10)
    else:
        margin_x = int(ref_w * 0.50)
        top_pad = int(ref_h * 0.64)
        cutoff_y = int(chin_y + ref_h * 0.46)

    head_x1 = max(0, ref_x1 - margin_x)
    head_x2 = min(W, ref_x2 + margin_x)
    head_y1 = max(0, min(ref_y1, forehead_y) - top_pad)
    cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
    head_y2 = max(head_y1 + 1, cutoff_y)
    return head_x1, head_y1, head_x2, head_y2, cutoff_y


@staticmethod
def _mask_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float((mask > 0.5).sum()) / float(mask.size)


@staticmethod
def _resize_mask_to_shape(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    H, W = shape
    if mask.shape[:2] == (H, W):
        return np.clip(mask.astype(np.float32), 0.0, 1.0)
    return cv2.resize(
        np.clip(mask.astype(np.float32), 0.0, 1.0),
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)


@staticmethod
def _mask_to_u8(mask: Optional[np.ndarray], threshold: float = 0.08) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        return np.zeros((0, 0), dtype=np.uint8)
    cutoff = float(np.clip(threshold, 0.0, 1.0))
    return (np.clip(mask.astype(np.float32), 0.0, 1.0) > cutoff).astype(np.uint8) * 255


@staticmethod
def _count_active_mask_px(mask: Optional[np.ndarray], threshold: float = 0.08) -> int:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return 0
    cutoff = float(np.clip(threshold, 0.0, 1.0))
    return int((np.clip(mask.astype(np.float32), 0.0, 1.0) > cutoff).sum())


def _get_cleanup_apply_min_px(
    self,
    stage_name: str,
    hair_length: str,
) -> int:
    defaults = {
        "short_generation_conditioning_cleanup": 180,
        "short_final_side_lane_refine": 140,
        "short_lower_tail_cleanup": 80,
        "dark_lane_cleanup": 40,
        "final_hair_lane_cleanup": 40,
        "short_bob_tail_suppress": 60,
        "short_lower_garment_cleanup": 140,
        "short_lower_cloth_hard_override": 96,
        "residual_strand_cleanup": 16,
        "final_source_cloth_rescue": 120,
        "short_subject_cloth_cleanup": 120,
        "controlnet_garment_repaint": 120,
    }
    config_fields = {
        "short_generation_conditioning_cleanup": "short_generation_conditioning_cleanup_min_px",
        "short_final_side_lane_refine": "short_final_side_lane_refine_min_px",
        "short_lower_tail_cleanup": "short_lower_tail_cleanup_min_px",
        "dark_lane_cleanup": "dark_lane_cleanup_min_px",
        "final_hair_lane_cleanup": "final_hair_lane_cleanup_min_px",
        "short_bob_tail_suppress": "short_bob_tail_suppress_min_px",
        "short_lower_garment_cleanup": "short_lower_garment_cleanup_min_px",
        "short_lower_cloth_hard_override": "short_lower_cloth_hard_override_min_px",
        "residual_strand_cleanup": "residual_strand_cleanup_min_px",
        "final_source_cloth_rescue": "final_source_cloth_rescue_min_px",
        "short_subject_cloth_cleanup": "short_subject_cloth_cleanup_min_px",
        "controlnet_garment_repaint": "controlnet_garment_repaint_min_px",
    }
    default_value = int(defaults.get(stage_name, 0))
    field_name = config_fields.get(stage_name)
    if field_name is None:
        return max(default_value, 0)
    value = getattr(self.config, field_name, default_value)
    return max(int(value), 0)


def _should_apply_cleanup_mask(
    self,
    stage_name: str,
    active_px: int,
    hair_length: str,
) -> bool:
    min_px = self._get_cleanup_apply_min_px(stage_name, hair_length)
    return int(active_px) >= int(min_px)


def _build_short_sam2_torso_bridge_mask(
    self,
    *,
    hair_mask_for_removal: Optional[np.ndarray],
    source_torso_hair_mask: Optional[np.ndarray],
) -> np.ndarray:
    base_shape = None
    for mask in (source_torso_hair_mask, hair_mask_for_removal):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return np.zeros((1, 1), dtype=np.float32)

    H, W = base_shape
    if source_torso_hair_mask is None or source_torso_hair_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if hair_mask_for_removal is not None and hair_mask_for_removal.shape != (H, W):
        hair_mask_for_removal = None

    sam2_u8 = self._mask_to_u8(hair_mask_for_removal, threshold=0.08)
    torso_u8 = self._mask_to_u8(source_torso_hair_mask, threshold=0.08)
    if int((torso_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.float32)

    bridge_u8 = np.zeros((H, W), dtype=np.uint8)
    min_gap_px = max(
        1,
        int(getattr(self.config, "source_garment_prepass_bridge_min_gap_px", 4)),
    )
    for col in range(W):
        sam2_ys = np.where(sam2_u8[:, col] > 0)[0]
        torso_ys = np.where(torso_u8[:, col] > 0)[0]
        if len(sam2_ys) == 0 or len(torso_ys) == 0:
            continue
        sam2_bottom = int(sam2_ys.max())
        torso_top = int(torso_ys.min())
        if torso_top > sam2_bottom + min_gap_px:
            bridge_u8[sam2_bottom:torso_top, col] = 255
    return (bridge_u8 > 0).astype(np.float32)


def _finalize_source_garment_prepass(
    self,
    *,
    source_garment_prepass_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    hair_length: str,
    skip_source_cloth_preclean: bool,
    hair_mask_for_removal: Optional[np.ndarray] = None,
    source_torso_hair_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    base_shape = None
    for mask in (source_garment_prepass_mask, source_torso_hair_mask, hair_mask_for_removal):
        if isinstance(mask, np.ndarray):
            base_shape = mask.shape[:2]
            break
    if base_shape is None:
        return {
            "mask": np.zeros((1, 1), dtype=np.float32),
            "px": 0,
            "enabled": False,
            "min_px": 0,
            "bridge_px": 0,
            "bridge_applied": False,
            "bridge_min_px": 0,
            "bridge_mask": np.zeros((1, 1), dtype=np.float32),
        }

    H, W = base_shape
    prepass_mask = np.zeros((H, W), dtype=np.float32)
    if isinstance(source_garment_prepass_mask, np.ndarray) and source_garment_prepass_mask.shape == (H, W):
        prepass_mask = np.clip(source_garment_prepass_mask.astype(np.float32), 0.0, 1.0)

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    min_px = max(
        int(getattr(self.config, "source_garment_prepass_min_px", 180)),
        int(
            face_w
            * face_h
            * float(getattr(self.config, "source_garment_prepass_min_face_area_ratio", 0.016))
        ),
    )
    prepass_px = self._count_active_mask_px(prepass_mask, threshold=0.08)
    enabled = (not skip_source_cloth_preclean) and prepass_px >= min_px

    bridge_mask = np.zeros((H, W), dtype=np.float32)
    bridge_px = 0
    bridge_applied = False
    bridge_min_px = max(
        1,
        int(getattr(self.config, "source_garment_prepass_bridge_min_px", 200)),
    )
    if hair_length == "short" and not skip_source_cloth_preclean:
        bridge_mask = self._build_short_sam2_torso_bridge_mask(
            hair_mask_for_removal=hair_mask_for_removal,
            source_torso_hair_mask=source_torso_hair_mask,
        )
        bridge_px = self._count_active_mask_px(bridge_mask, threshold=0.5)
        if bridge_px >= bridge_min_px:
            sigma = max(
                float(getattr(self.config, "source_garment_prepass_bridge_sigma", 5.0)),
                0.1,
            )
            alpha = float(
                np.clip(getattr(self.config, "source_garment_prepass_bridge_alpha", 0.95), 0.0, 1.0)
            )
            bridge_soft = cv2.GaussianBlur(
                np.clip(bridge_mask.astype(np.float32), 0.0, 1.0),
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            ).astype(np.float32)
            prepass_mask = np.maximum(
                prepass_mask,
                np.clip(bridge_soft * alpha, 0.0, 1.0),
            ).astype(np.float32)
            prepass_px = self._count_active_mask_px(prepass_mask, threshold=0.08)
            enabled = True
            bridge_applied = True
            logger.info(
                "[SDPipeline] sam2-torso bridge merged into garment_prepass_mask: bridge_px=%d total_px=%d",
                bridge_px,
                prepass_px,
            )

    return {
        "mask": prepass_mask.astype(np.float32),
        "px": int(prepass_px),
        "enabled": bool(enabled),
        "min_px": int(min_px),
        "bridge_px": int(bridge_px),
        "bridge_applied": bool(bridge_applied),
        "bridge_min_px": int(bridge_min_px),
        "bridge_mask": bridge_mask.astype(np.float32),
    }


def _sanitize_cloth_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    hair_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """Keep subject cloth local while preserving the torso area below the bob line."""
    H, W = cloth_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    bw = max(int(x2 - x1), 1)
    bh = max(int(y2 - y1), 1)

    raw_cloth_u8 = (np.clip(cloth_mask, 0.0, 1.0) > 0.5).astype(np.uint8) * 255
    corridor = np.zeros((H, W), dtype=np.uint8)
    x_min = max(0, int(x1 - bw * 1.20))
    x_max = min(W, int(x2 + bw * 1.20))
    y_min = max(0, int(y2 - bh * 0.05))
    y_max = min(H, int(y2 + bh * 0.95))
    if x_min < x_max and y_min < y_max:
        corridor[y_min:y_max, x_min:x_max] = 255

    cloth_u8 = cv2.bitwise_and(
        raw_cloth_u8,
        corridor,
    )
    cloth_u8 = cv2.morphologyEx(
        cloth_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    cloth_u8 = cv2.erode(
        cloth_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    original_cloth_u8 = cloth_u8.copy()
    subject_anchor_u8 = self._build_subject_cloth_anchor_mask(
        image_shape=(H, W),
        face_bbox=face_bbox,
    )
    subject_torso_anchor_u8 = self._build_subject_torso_anchor_mask(
        image_shape=(H, W),
        face_bbox=face_bbox,
    )
    filtered_subject_cloth_u8 = self._filter_cloth_mask_to_subject_anchor(
        cloth_mask_u8=cloth_u8,
        subject_anchor_u8=subject_anchor_u8,
        face_bbox=face_bbox,
    )
    torso_cloth_u8 = cv2.bitwise_and(
        raw_cloth_u8,
        cv2.dilate(
            subject_torso_anchor_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        ),
    )
    torso_cloth_u8 = cv2.morphologyEx(
        torso_cloth_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    torso_cloth_u8 = cv2.morphologyEx(
        torso_cloth_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
    )
    filtered_subject_torso_u8 = self._filter_cloth_mask_to_subject_anchor(
        cloth_mask_u8=torso_cloth_u8,
        subject_anchor_u8=subject_torso_anchor_u8,
        face_bbox=face_bbox,
        min_area_scale=0.004,
        max_dx_scale=1.18,
        min_bottom_scale=0.18,
        max_width_scale=1.72,
        max_area_scale=0.90,
    )
    shoulder_bridge_u8 = self._build_subject_shoulder_bridge_mask(
        image_shape=(H, W),
        face_bbox=face_bbox,
        base_cloth_mask_u8=raw_cloth_u8,
        torso_cloth_mask_u8=filtered_subject_torso_u8 if int((filtered_subject_torso_u8 > 0).sum()) >= 80 else torso_cloth_u8,
    )
    cloth_u8 = np.zeros((H, W), dtype=np.uint8)
    if int((filtered_subject_cloth_u8 > 0).sum()) >= 60:
        cloth_u8 = cv2.bitwise_or(cloth_u8, filtered_subject_cloth_u8)
    if int((filtered_subject_torso_u8 > 0).sum()) >= 80:
        cloth_u8 = cv2.bitwise_or(cloth_u8, filtered_subject_torso_u8)
    if isinstance(self._last_segface_mask_debug, dict):
        self._last_segface_mask_debug["subject_cloth_anchor_mask"] = (
            subject_anchor_u8 > 0
        ).astype(np.float32)
        self._last_segface_mask_debug["subject_torso_anchor_mask"] = (
            subject_torso_anchor_u8 > 0
        ).astype(np.float32)
        self._last_segface_mask_debug["subject_torso_candidate_mask"] = (
            torso_cloth_u8 > 0
        ).astype(np.float32)
        self._last_segface_mask_debug["subject_cloth_filtered_mask"] = (
            cloth_u8 > 0
        ).astype(np.float32)
        self._last_segface_mask_debug["subject_torso_filtered_mask"] = (
            filtered_subject_torso_u8 > 0
        ).astype(np.float32)
        self._last_segface_mask_debug["subject_shoulder_bridge_mask"] = (
            shoulder_bridge_u8 > 0
        ).astype(np.float32)

    sparse_thresh_px = max(900, int(bw * bh * 0.020))
    filtered_px = int((cloth_u8 > 0).sum())
    original_px = int((original_cloth_u8 > 0).sum())
    should_add_sparse_support = (
        img_rgb.shape[:2] == (H, W)
        and (
            filtered_px < sparse_thresh_px
            or (original_px >= 240 and filtered_px < max(90, int(original_px * 0.22)))
        )
    )
    if should_add_sparse_support:
        support_anchor_u8 = cv2.bitwise_or(subject_anchor_u8, subject_torso_anchor_u8)
        sparse_dark_support = self._build_sparse_dark_cloth_support_mask(
            img_rgb=img_rgb,
            base_cloth_mask_u8=cloth_u8,
            hair_mask=hair_mask,
            face_bbox=face_bbox,
            subject_anchor_u8=support_anchor_u8,
        )
        if int((sparse_dark_support > 0).sum()) >= 120:
            cloth_u8 = cv2.bitwise_or(cloth_u8, sparse_dark_support)
            if isinstance(self._last_segface_mask_debug, dict):
                self._last_segface_mask_debug["sparse_dark_cloth_support_mask"] = (
                    sparse_dark_support > 0
                ).astype(np.float32)

    cloth_f = (cloth_u8 > 0).astype(np.float32)
    cloth_ratio = self._mask_ratio(cloth_f)
    hair_area = float((hair_mask > 0.5).sum())
    overlap = float(((cloth_f > 0.5) & (hair_mask > 0.5)).sum())
    overlap_ratio = overlap / max(hair_area, 1.0)

    cloth_ratio_limit = 0.118
    overlap_ratio_limit = 0.34
    if getattr(self.config, "enable_upper_clothes_overwrite", False):
        cloth_ratio_limit = max(cloth_ratio_limit, 0.16)
        overlap_ratio_limit = max(
            overlap_ratio_limit,
            float(getattr(self.config, "overwrite_cloth_overlap_ratio_limit", 0.68)),
        )

    if cloth_ratio > cloth_ratio_limit or overlap_ratio > overlap_ratio_limit:
        logger.info(
            "[SDPipeline] cloth mask disabled: ratio=%.4f overlap_ratio=%.4f limits=(%.4f, %.4f)",
            cloth_ratio,
            overlap_ratio,
            cloth_ratio_limit,
            overlap_ratio_limit,
        )
        return np.zeros((H, W), dtype=np.float32)

    return cloth_f


def _build_subject_cloth_anchor_mask(
    self,
    *,
    image_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    H, W = image_shape
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 - face_h * 0.02))
    bottom = min(H, int(y2 + face_h * 1.18))
    left = max(0, int(cx - face_w * 0.66))
    right = min(W, int(cx + face_w * 0.66))
    if top < bottom and left < right:
        anchor_u8[top:bottom, left:right] = 255

    shoulder_centers = (
        (int(cx - face_w * 0.40), int(y2 + face_h * 0.18)),
        (int(cx + face_w * 0.40), int(y2 + face_h * 0.18)),
    )
    shoulder_axes = (
        max(18, int(face_w * 0.24)),
        max(16, int(face_h * 0.16)),
    )
    for center in shoulder_centers:
        cv2.ellipse(anchor_u8, center, shoulder_axes, 0, 0, 360, 255, -1)

    anchor_u8 = cv2.morphologyEx(
        anchor_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19)),
    )
    return anchor_u8


def _build_subject_torso_anchor_mask(
    self,
    *,
    image_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    H, W = image_shape
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.14))
    bottom = min(H, int(y2 + face_h * 1.92))
    left = max(0, int(cx - face_w * 0.82))
    right = min(W, int(cx + face_w * 0.82))
    if top < bottom and left < right:
        anchor_u8[top:bottom, left:right] = 255

    chest_center = (cx, min(H - 1, int(y2 + face_h * 0.66)))
    chest_axes = (
        max(22, int(face_w * 0.58)),
        max(26, int(face_h * 0.54)),
    )
    cv2.ellipse(anchor_u8, chest_center, chest_axes, 0, 0, 360, 255, -1)

    shoulder_centers = (
        (int(cx - face_w * 0.46), int(y2 + face_h * 0.32)),
        (int(cx + face_w * 0.46), int(y2 + face_h * 0.32)),
    )
    shoulder_axes = (
        max(24, int(face_w * 0.34)),
        max(18, int(face_h * 0.22)),
    )
    for center in shoulder_centers:
        cv2.ellipse(anchor_u8, center, shoulder_axes, 0, 0, 360, 255, -1)

    anchor_u8 = cv2.morphologyEx(
        anchor_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 27)),
    )
    return anchor_u8


def _build_subject_shoulder_bridge_mask(
    self,
    *,
    image_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
    base_cloth_mask_u8: np.ndarray,
    torso_cloth_mask_u8: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = image_shape
    if base_cloth_mask_u8.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.uint8)
    if torso_cloth_mask_u8 is not None and torso_cloth_mask_u8.shape[:2] != (H, W):
        torso_cloth_mask_u8 = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 + face_h * 0.12))
    bottom = min(H, int(y2 + face_h * 1.26))
    left = max(0, int(x1 - face_w * 1.16))
    right = min(W, int(x2 + face_w * 1.16))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.uint8)
    corridor_u8[top:bottom, left:right] = 255

    support_u8 = cv2.bitwise_and(base_cloth_mask_u8, corridor_u8)
    if torso_cloth_mask_u8 is not None:
        support_u8 = cv2.bitwise_or(
            support_u8,
            cv2.bitwise_and(torso_cloth_mask_u8, corridor_u8),
        )
    support_u8 = cv2.morphologyEx(
        support_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
    )
    if int((support_u8 > 0).sum()) < max(120, int(face_w * face_h * 0.010)):
        return np.zeros((H, W), dtype=np.uint8)

    bridge_u8 = np.zeros((H, W), dtype=np.uint8)
    main_thickness = max(18, int(face_w * 0.16))
    sub_thickness = max(12, int(face_w * 0.10))
    side_specs = (
        (
            (int(cx - face_w * 0.92), int(y2 + face_h * 0.30)),
            (int(cx - face_w * 0.28), int(y2 + face_h * 0.86)),
            (int(cx - face_w * 0.74), int(y2 + face_h * 0.24)),
            (int(cx - face_w * 0.20), int(y2 + face_h * 0.64)),
        ),
        (
            (int(cx + face_w * 0.92), int(y2 + face_h * 0.30)),
            (int(cx + face_w * 0.28), int(y2 + face_h * 0.86)),
            (int(cx + face_w * 0.74), int(y2 + face_h * 0.24)),
            (int(cx + face_w * 0.20), int(y2 + face_h * 0.64)),
        ),
    )
    for outer_pt, inner_pt, upper_pt, lower_pt in side_specs:
        cv2.line(bridge_u8, outer_pt, inner_pt, 255, thickness=main_thickness)
        cv2.line(bridge_u8, upper_pt, lower_pt, 255, thickness=sub_thickness)
        cv2.ellipse(
            bridge_u8,
            outer_pt,
            (max(14, int(face_w * 0.14)), max(10, int(face_h * 0.12))),
            0,
            0,
            360,
            255,
            -1,
        )

    bridge_u8 = cv2.bitwise_and(bridge_u8, corridor_u8)
    bridge_u8 = cv2.morphologyEx(
        bridge_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 17)),
    )
    bridge_u8 = cv2.dilate(
        bridge_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    return bridge_u8


def _filter_cloth_mask_to_subject_anchor(
    self,
    *,
    cloth_mask_u8: np.ndarray,
    subject_anchor_u8: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    min_area_scale: float = 0.002,
    max_dx_scale: float = 0.95,
    min_bottom_scale: float = 0.02,
    max_width_scale: float | None = None,
    max_area_scale: float | None = None,
) -> np.ndarray:
    H, W = cloth_mask_u8.shape[:2]
    if subject_anchor_u8.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.uint8)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (cloth_mask_u8 > 0).astype(np.uint8),
        8,
    )
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    min_area = max(40, int(face_w * face_h * min_area_scale))
    max_dx = max(72, int(face_w * max_dx_scale))
    min_bottom = int(y2 + face_h * min_bottom_scale)
    max_width = max(140, int(face_w * max_width_scale)) if max_width_scale is not None else None
    max_area = max(4800, int(face_w * face_h * max_area_scale)) if max_area_scale is not None else None

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        if area < min_area or bottom_y < min_bottom:
            continue
        if max_width is not None and w > max_width:
            continue
        if max_area is not None and area > max_area:
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        overlap_px = int((cv2.bitwise_and(comp_u8, subject_anchor_u8) > 0).sum())
        comp_cx = float(centroids[idx][0])
        if overlap_px <= 0 and abs(comp_cx - cx) > max_dx:
            continue
        keep_u8[labels == idx] = 255

    return keep_u8


def _trim_blocky_short_restore_mask_u8(
    self,
    *,
    mask_u8: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    min_keep_px: int = 60,
) -> np.ndarray:
    H, W = mask_u8.shape[:2]
    original_px = int((mask_u8 > 0).sum())
    if original_px < min_keep_px:
        return mask_u8

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8),
        8,
    )
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    min_area = max(16, int(face_w * face_h * 0.0012))
    reject_far_offset = max(180, int(face_w * 1.05))
    reject_center_offset = max(24, int(face_w * 0.22))
    reject_center_width = max(72, int(face_w * 0.38))
    reject_wide_width = max(110, int(face_w * 0.62))
    reject_center_area = max(280, int(face_w * face_h * 0.030))
    reject_wide_area = max(520, int(face_w * face_h * 0.045))
    reject_far_area = max(180, int(face_w * face_h * 0.015))
    min_bottom = int(cutoff_y + face_h * 0.06)

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bottom_y = y + h
        comp_cx = float(centroids[idx][0])
        offset = abs(comp_cx - cx)
        fill_ratio = float(area) / float(max(w * h, 1))

        reject_far_blob = (
            offset > reject_far_offset * 1.5
            and area >= reject_far_area * 1.8
            and (fill_ratio >= 0.45 or w >= max(84, int(face_w * 0.42)))
        )
        reject_center_block = (
            offset <= reject_center_offset * 0.8
            and bottom_y >= min_bottom
            and w >= reject_center_width * 1.6
            and area >= reject_center_area * 1.8
            and fill_ratio >= 0.52
        )
        reject_dense_wide = (
            bottom_y >= min_bottom
            and w >= reject_wide_width * 1.5
            and area >= reject_wide_area * 1.8
            and fill_ratio >= 0.62
        )
        reject_rect_patch = (
            bottom_y >= min_bottom
            and h >= max(96, int(face_h * 0.42))
            and w >= max(120, int(face_w * 0.52))
            and fill_ratio >= 0.72
        )
        if reject_far_blob or reject_center_block or reject_dense_wide or reject_rect_patch:
            continue
        keep_u8[labels == idx] = 255

    kept_px = int((keep_u8 > 0).sum())
    if kept_px == original_px:
        return mask_u8
    if kept_px < max(12, min_keep_px // 4):
        if original_px >= max(min_keep_px * 2, int(face_w * face_h * 0.05)):
            return np.zeros((H, W), dtype=np.uint8)
        return mask_u8
    return keep_u8


def _build_sparse_dark_cloth_support_mask(
    self,
    *,
    img_rgb: np.ndarray,
    base_cloth_mask_u8: np.ndarray,
    hair_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    subject_anchor_u8: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    if base_cloth_mask_u8.shape[:2] != (H, W) or hair_mask.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.uint8)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.2, sigmaY=5.2)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 - face_h * 0.02))
    bottom = min(H, int(y2 + face_h * 1.18))
    left = max(0, int(x1 - face_w * 1.20))
    right = min(W, int(x2 + face_w * 1.20))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.uint8)
    corridor_u8[top:bottom, left:right] = 255

    candidate_u8 = (
        (
            (gray < 132.0)
            & (sat < 154.0)
            & (
                (np.abs(gray - blur) < 11.5)
                | (lap < 11.0)
            )
        ).astype(np.uint8)
        * 255
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)

    hair_u8 = cv2.dilate(
        (np.clip(hair_mask.astype(np.float32), 0.0, 1.0) > 0.45).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
        iterations=1,
    )
    upper_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
    upper_bottom = min(H, int(y2 + face_h * 0.30))
    upper_left = max(0, int(cx - face_w * 0.46))
    upper_right = min(W, int(cx + face_w * 0.46))
    if top < upper_bottom and upper_left < upper_right:
        upper_keepout_u8[top:upper_bottom, upper_left:upper_right] = 255
        candidate_u8 = cv2.bitwise_and(
            candidate_u8,
            cv2.bitwise_not(cv2.bitwise_and(hair_u8, upper_keepout_u8)),
        )

    anchor_u8 = cv2.dilate(
        base_cloth_mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    if subject_anchor_u8 is not None and subject_anchor_u8.shape[:2] == (H, W):
        anchor_u8 = cv2.bitwise_or(
            anchor_u8,
            cv2.bitwise_and(subject_anchor_u8, corridor_u8),
        )
    side_seed_u8 = np.zeros((H, W), dtype=np.uint8)
    side_centers = (
        (int(cx - face_w * 0.54), int(y2 + face_h * 0.14)),
        (int(cx + face_w * 0.54), int(y2 + face_h * 0.14)),
    )
    side_axes = (
        max(18, int(face_w * 0.28)),
        max(16, int(face_h * 0.18)),
    )
    for center in side_centers:
        cv2.ellipse(side_seed_u8, center, side_axes, 0, 0, 360, 255, -1)
    anchor_u8 = cv2.bitwise_or(anchor_u8, cv2.bitwise_and(side_seed_u8, corridor_u8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, 8)
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    min_area = max(90, int(face_w * face_h * 0.003))
    max_area = max(42000, int(face_w * face_h * 0.80))
    min_height = max(18, int(face_h * 0.10))
    max_width = max(280, int(face_w * 1.72))
    min_bottom = int(y2 + face_h * 0.10)
    deep_bottom = int(y2 + face_h * 0.42)
    thin_width = max(18, int(face_w * 0.14))
    tall_height = max(60, int(face_h * 0.52))

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bottom_y = y + h
        if area < min_area or area > max_area:
            continue
        if h < min_height or w > max_width:
            continue
        if w <= thin_width and h >= tall_height:
            continue
        if bottom_y < min_bottom:
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        anchor_overlap = int(cv2.bitwise_and(comp_u8, anchor_u8).sum() > 0)
        if anchor_overlap <= 0 and not (bottom_y >= deep_bottom and w >= max(26, int(face_w * 0.18))):
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 120:
        return np.zeros((H, W), dtype=np.uint8)

    keep_u8 = cv2.morphologyEx(
        keep_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
    )
    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    return cv2.bitwise_and(keep_u8, corridor_u8)


def _build_bangs_recovery_mask(
    self,
    hair_mask: np.ndarray,
    protect_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    *,
    landmark_debug_data: Optional[Dict[str, Any]] = None,
    hair_length: str = "short",
) -> np.ndarray:
    """
    얼굴 보호 마스크에 의해 같이 깎인 앞머리만 제한적으로 복원한다.
    중앙 이마 밴드에서 원래 hair mask가 잡고 있던 성분만 되살린다.
    """
    H, W = hair_mask.shape[:2]
    if protect_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    hair_f = np.clip(hair_mask.astype(np.float32), 0.0, 1.0)
    protect_f = np.clip(protect_mask.astype(np.float32), 0.0, 1.0)
    overlap = np.clip(hair_f * (protect_f > 0.10).astype(np.float32), 0.0, 1.0)
    if float(overlap.sum()) < 12.0:
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    keypoints = landmark_debug_data.get("keypoints", {}) if isinstance(landmark_debug_data, dict) else {}
    forehead_top = keypoints.get("forehead_top", {}).get("px")
    forehead_y = int(forehead_top[1]) if isinstance(forehead_top, list) and len(forehead_top) >= 2 else int(y1)

    band_top = max(0, int(min(y1, forehead_y) - face_h * 0.16))
    # band_bottom: 앞머리 복원 허용 하한. 이마 영역(눈썹 위)에만 머물도록 제한.
    # 기존 0.48/0.42/0.44는 눈·코 부근까지 침범해 얼굴 찌그러짐을 유발.
    band_bottom = min(
        H,
        int(forehead_y + face_h * (0.22 if hair_length == "short" else 0.18 if hair_length == "medium" else 0.20)),
    )
    center_half = max(
        18,
        int(face_w * (0.50 if hair_length == "short" else 0.50 if hair_length == "medium" else 0.54)),
    )
    band_x1 = max(0, cx - center_half)
    band_x2 = min(W, cx + center_half)
    if band_top >= band_bottom or band_x1 >= band_x2:
        return np.zeros((H, W), dtype=np.float32)

    corridor = np.zeros((H, W), dtype=np.uint8)
    corridor[band_top:band_bottom, band_x1:band_x2] = 255
    recover_u8 = cv2.bitwise_and((overlap > 0.12).astype(np.uint8) * 255, corridor)
    if int((recover_u8 > 0).sum()) < 18:
        return np.zeros((H, W), dtype=np.float32)

    support_top = max(0, int(band_top - face_h * 0.22))
    support_bottom = min(H, int(forehead_y + face_h * (0.14 if hair_length == "short" else 0.11)))
    support_x1 = max(0, cx - max(24, int(face_w * (0.52 if hair_length == "short" else 0.48))))
    support_x2 = min(W, cx + max(24, int(face_w * (0.52 if hair_length == "short" else 0.48))))
    support_u8 = np.zeros((H, W), dtype=np.uint8)
    if support_top < support_bottom and support_x1 < support_x2:
        support_u8[support_top:support_bottom, support_x1:support_x2] = 255
        support_u8 = cv2.bitwise_and(
            support_u8,
            (hair_f > 0.20).astype(np.uint8) * 255,
        )

    recover_u8 = cv2.morphologyEx(
        recover_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(recover_u8, 8)
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    max_component_area = max(160, int(face_w * face_h * (0.20 if hair_length == "short" else 0.16)))
    max_component_width = max(34, int(face_w * (1.02 if hair_length == "short" else 0.92)))
    min_component_height = max(8, int(face_h * 0.08))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 14 or area > max_component_area:
            continue
        if w > max_component_width or h < min_component_height:
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        support_overlap = int((cv2.bitwise_and(comp_u8, support_u8) > 0).sum())
        if support_overlap < 6 and y > int(forehead_y + face_h * 0.08):
            continue
        keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

    if int((keep_u8 > 0).sum()) < 14:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),  # 세로 팽창 축소: band가 좁아진 만큼 하단 침범 방지
        ),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor)
    return (keep_u8 > 0).astype(np.float32)


def _normalize_front_mask_axis_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canonicalize_front_mask_axis_value(axis_key: str, value: Any) -> str:
    norm_key = _normalize_front_mask_axis_key(axis_key)
    normalized = _normalize_front_mask_axis_key(value)
    lowered = _flatten_front_mask_text(value).lower()

    if norm_key in {"front_styling", "front_style", "front"}:
        if normalized in {
            "lifted",
            "up",
            "up_style",
            "updo",
            "slick_back",
            "slicked_back",
            "back",
            "open_forehead",
            "forehead_open",
            "flexible",
        }:
            return "lifted"
        if normalized in {"down", "down_style", "down_perm", "fringe", "bang", "bangs"}:
            return normalized
        if any(
            token in lowered
            for token in (
                "front=flexible",
                "front_styling=flexible",
                "front_style=flexible",
                "front=up",
                "front_styling=up",
                "front_style=up",
                "앞머리 올림",
                "앞머리 올려",
                "올린 앞머리",
                "이마 보이게",
                "open forehead",
                "exposed forehead",
                "lifted front",
            )
        ):
            return "lifted"
    if norm_key in {"parting", "part"}:
        if normalized in {"non_parted", "nonparted", "no_part"}:
            return "non_parted"
        if normalized in {"side_part", "sidepart", "parted", "either", "flexible", "any"}:
            return "side_part"
        if normalized in {"middle_part", "middlepart", "center_part", "centerpart"}:
            return "center_part"
    return normalized


def _flatten_front_mask_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, dict):
        return " ".join(
            text
            for text in (_flatten_front_mask_text(item) for item in value.values())
            if text
        ).strip()
    if isinstance(value, (list, tuple, set)):
        return " ".join(
            text
            for text in (_flatten_front_mask_text(item) for item in value)
            if text
        ).strip()
    return _flatten_front_mask_text(str(value))


def _resolve_front_mask_axis_value(style_axes: Any, *keys: str) -> str:
    if not isinstance(style_axes, dict):
        return ""
    normalized_axes = {
        _normalize_front_mask_axis_key(key): value
        for key, value in style_axes.items()
        if _normalize_front_mask_axis_key(key)
    }
    for key in keys:
        norm_key = _normalize_front_mask_axis_key(key)
        if norm_key not in normalized_axes:
            continue
        value = normalized_axes.get(norm_key)
        if isinstance(value, dict):
            for candidate in ("value", "label", "name", "slug", "id"):
                if candidate in value:
                    normalized = _canonicalize_front_mask_axis_value(norm_key, value.get(candidate))
                    if normalized:
                        return normalized
            normalized = _canonicalize_front_mask_axis_value(norm_key, _flatten_front_mask_text(value))
            if normalized:
                return normalized
            continue
        normalized = _canonicalize_front_mask_axis_value(norm_key, _flatten_front_mask_text(value))
        if normalized:
            return normalized
    return ""


def _build_requested_front_coverage_mask(
    image_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
    prompt_context: Optional[Dict[str, Any]] = None,
    *,
    hair_length: str = "short",
    subject_gender: str = "unknown",
    fringe_requested: bool = False,
) -> np.ndarray:
    H, W = image_shape
    if H <= 0 or W <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    if hair_length not in ("short", "medium") or not fringe_requested:
        return np.zeros((H, W), dtype=np.float32)

    context = prompt_context if isinstance(prompt_context, dict) else {}
    style_axes = context.get("style_axes") if isinstance(context.get("style_axes"), dict) else {}
    combined_text = " ".join(
        text
        for text in (
            _flatten_front_mask_text(style_axes),
            _flatten_front_mask_text(context.get("question_answers")),
            _flatten_front_mask_text(context.get("derived_preferences")),
            _flatten_front_mask_text(context.get("legacy_fields")),
        )
        if text
    ).lower()

    normalized_gender = _normalize_front_mask_axis_key(subject_gender)
    if normalized_gender not in {"male", "female"}:
        branch = _normalize_front_mask_axis_key(context.get("gender_branch"))
        normalized_gender = branch if branch in {"male", "female"} else "unknown"

    front_styling = _resolve_front_mask_axis_value(style_axes, "front_styling", "front_style", "front")
    parting = _resolve_front_mask_axis_value(style_axes, "parting", "part")
    non_parted = parting in {"non_parted", "nonparted", "no_part"}
    parted = parting in {"parted", "side_part", "middle_part", "center_part"}
    down_requested = front_styling in {"down", "down_style", "down_perm", "fringe", "bang", "bangs"}
    if not down_requested:
        down_requested = any(
            token in combined_text
            for token in (
                "bang",
                "bangs",
                "fringe",
                "앞머리",
                "시스루",
                "내리는 스타일",
                "다운펌",
                "down style",
                "down perm",
            )
        )
    if not (down_requested or non_parted or fringe_requested):
        return np.zeros((H, W), dtype=np.float32)

    curly_requested = any(
        token in combined_text
        for token in ("curly", "wavy", "wave", "컬", "웨이브", "텍스처", "texture")
    )
    full_front = bool(down_requested or non_parted)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    coverage_u8 = np.zeros((H, W), dtype=np.uint8)
    band_top = max(0, int(y1 - face_h * (0.12 if normalized_gender == "male" else 0.08)))
    band_bottom = min(
        H,
        int(
            y1
            + face_h
            * (
                0.56 if normalized_gender == "male" and hair_length == "short" and full_front
                else 0.50 if normalized_gender == "male" and full_front
                else 0.46 if full_front
                else 0.38
            )
        ),
    )
    half_w = max(
        22,
        int(
            face_w
            * (
                0.70 if normalized_gender == "male" and full_front
                else 0.62 if full_front
                else 0.52
            )
        ),
    )
    band_x1 = max(0, cx - half_w)
    band_x2 = min(W, cx + half_w)
    if band_top >= band_bottom or band_x1 >= band_x2:
        return np.zeros((H, W), dtype=np.float32)
    coverage_u8[band_top:band_bottom, band_x1:band_x2] = 255

    arc_center = (
        cx,
        max(0, min(H - 1, int(y1 + face_h * (0.16 if normalized_gender == "male" else 0.18)))),
    )
    arc_axes = (
        max(
            18,
            int(
                face_w
                * (
                    0.64 if normalized_gender == "male" and full_front
                    else 0.56 if full_front
                    else 0.46
                )
            ),
        ),
        max(
            14,
            int(
                face_h
                * (
                    0.30 if normalized_gender == "male" and full_front
                    else 0.26 if full_front
                    else 0.22
                )
            ),
        ),
    )
    cv2.ellipse(coverage_u8, arc_center, arc_axes, 0, 0, 360, 255, -1)

    if normalized_gender == "male" and full_front:
        block_top = max(0, int(y1 + face_h * 0.02))
        block_bottom = min(H, int(y1 + face_h * (0.34 if hair_length == "short" else 0.30)))
        if block_top < block_bottom:
            coverage_u8[block_top:block_bottom, band_x1:band_x2] = 255

    if parted and not non_parted:
        keepout_half = max(8, int(face_w * (0.07 if normalized_gender == "male" else 0.08)))
        keepout_top = max(band_top, int(y1 - face_h * 0.02))
        keepout_bottom = min(band_bottom, int(y1 + face_h * 0.18))
        if keepout_top < keepout_bottom:
            coverage_u8[
                keepout_top:keepout_bottom,
                max(0, cx - keepout_half):min(W, cx + keepout_half),
            ] = 0

    coverage_u8 = cv2.morphologyEx(
        coverage_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 11) if normalized_gender == "male" and full_front else (7, 9),
        ),
    )
    if curly_requested:
        coverage_u8 = cv2.dilate(
            coverage_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
            iterations=1,
        )

    alpha = cv2.GaussianBlur(
        coverage_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.4 if normalized_gender == "male" and full_front else 2.1,
        sigmaY=2.8 if normalized_gender == "male" and full_front else 2.4,
    )
    strength = (
        0.98 if normalized_gender == "male" and hair_length == "short" and full_front
        else 0.90 if normalized_gender == "male"
        else 0.82
    )
    return np.clip(alpha * strength, 0.0, 1.0).astype(np.float32)


def _build_soft_bangs_generation_mask(
    self,
    bangs_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    hair_length: str = "long",
    subject_gender: str = "unknown",
    fringe_requested: bool = False,
) -> np.ndarray:
    H, W = bangs_mask.shape[:2]
    preserve_fringe_detail = bool(
        subject_gender == "male"
        and fringe_requested
        and hair_length in ("short", "medium")
    )
    base = (np.clip(bangs_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255
    if int((base > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    base = cv2.morphologyEx(
        base,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
    )
    base = cv2.dilate(
        base,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
        iterations=1,
    )
    ys = np.where(base > 0)[0]
    if ys.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    top = int(ys.min())
    bottom = int(ys.max()) + 1
    cx = int(0.5 * (x1 + x2))
    side_keep = max(18, int(face_w * 0.46))
    x_min = max(0, cx - side_keep)
    x_max = min(W, cx + side_keep)
    band_top = max(0, int(top - face_h * 0.08))
    band_bottom = min(H, int(bottom + face_h * 0.08))
    if band_top >= band_bottom or x_min >= x_max:
        return np.zeros((H, W), dtype=np.float32)

    band_u8 = np.zeros((H, W), dtype=np.uint8)
    band_u8[band_top:band_bottom, x_min:x_max] = 255
    soft_u8 = cv2.bitwise_and(base, band_u8)
    if int((soft_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    soft_u8 = cv2.morphologyEx(
        soft_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
    )
    soft_u8 = cv2.dilate(
        soft_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 7) if preserve_fringe_detail else (5, 9),
        ),
        iterations=1,
    )
    if int((soft_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    if preserve_fringe_detail:
        _sx = 1.8 if hair_length == "short" else 2.0
        _sy = 2.1 if hair_length == "short" else 2.3
    else:
        _sx = 2.2 if hair_length == "short" else 2.4 if hair_length == "medium" else 2.6
        _sy = 2.6 if hair_length == "short" else 2.8 if hair_length == "medium" else 3.0
    alpha = cv2.GaussianBlur(
        soft_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=_sx,
        sigmaY=_sy,
    )
    alpha = np.clip((alpha - (0.01 if preserve_fringe_detail else 0.02)) / 0.94, 0.0, 1.0)

    fade = np.zeros((H,), dtype=np.float32)
    if band_bottom > band_top:
        inner_top = band_top
        inner_mid = min(band_bottom, band_top + max(10, int((band_bottom - band_top) * 0.34)))
        inner_low = min(band_bottom, band_top + max(16, int((band_bottom - band_top) * 0.72)))
        if inner_mid > inner_top:
            fade[inner_top:inner_mid] = np.linspace(
                0.22 if preserve_fringe_detail else 0.16,
                0.50 if preserve_fringe_detail else 0.42,
                inner_mid - inner_top,
                dtype=np.float32,
            )
        if inner_low > inner_mid:
            fade[inner_mid:inner_low] = np.linspace(
                0.50 if preserve_fringe_detail else 0.42,
                0.80 if preserve_fringe_detail else 0.72,
                inner_low - inner_mid,
                dtype=np.float32,
            )
        if band_bottom > inner_low:
            fade[inner_low:band_bottom] = np.linspace(
                0.80 if preserve_fringe_detail else 0.72,
                0.62 if preserve_fringe_detail else 0.52,
                band_bottom - inner_low,
                dtype=np.float32,
            )
    tail_end = min(H, band_bottom + max(6, int(face_h * 0.06)))
    if tail_end > band_bottom:
        fade[band_bottom:tail_end] = np.linspace(
            max(0.0, float(fade[band_bottom - 1])) if band_bottom > 0 else 0.44,
            0.0,
            tail_end - band_bottom,
            dtype=np.float32,
        )

    alpha = alpha * fade[:, np.newaxis]
    x_coords = np.arange(W, dtype=np.float32)
    side_scale = max(float(side_keep), 1.0)
    x_dist = np.abs(x_coords - float(cx)) / side_scale
    x_fade = np.clip(
        1.0 - (x_dist ** 1.55) * 0.52,
        0.52 if preserve_fringe_detail else 0.44,
        1.0,
    ).astype(np.float32)
    alpha = alpha * x_fade[np.newaxis, :]
    return np.clip(alpha, 0.0, 0.82 if preserve_fringe_detail else 0.68).astype(np.float32)


def _build_eye_region_restore_mask(
    self,
    landmark_debug_data: Optional[Dict[str, Any]],
    image_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
    hair_length: str = "long",
    final_hair_mask: Optional[np.ndarray] = None,
    subject_gender: str = "unknown",
    fringe_requested: bool = False,
) -> np.ndarray:
    H, W = image_shape
    preserve_fringe_detail = bool(
        subject_gender == "male"
        and fringe_requested
        and hair_length in ("short", "medium")
    )
    if not isinstance(landmark_debug_data, dict):
        return np.zeros((H, W), dtype=np.float32)

    landmarks_px = landmark_debug_data.get("landmarks_px")
    if not isinstance(landmarks_px, list) or len(landmarks_px) < 20:
        return np.zeros((H, W), dtype=np.float32)

    try:
        import mediapipe as mp

        left_eye_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_LEFT_EYE for i in edge})
        right_eye_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE for i in edge})
        left_brow_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW for i in edge})
        right_brow_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW for i in edge})
    except Exception:
        return np.zeros((H, W), dtype=np.float32)

    pts = np.asarray(landmarks_px, dtype=np.int32)
    n = int(len(pts))
    left_eye_pts = np.asarray([pts[i] for i in left_eye_indices if i < n], dtype=np.int32)
    right_eye_pts = np.asarray([pts[i] for i in right_eye_indices if i < n], dtype=np.int32)
    left_brow_pts = np.asarray([pts[i] for i in left_brow_indices if i < n], dtype=np.int32)
    right_brow_pts = np.asarray([pts[i] for i in right_brow_indices if i < n], dtype=np.int32)
    if len(left_eye_pts) < 3 or len(right_eye_pts) < 3:
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_h = max(int(y2 - y1), 1)
    face_w = max(int(x2 - x1), 1)
    dilate_px = max(8, int(face_h * (0.085 if preserve_fringe_detail else 0.10)))
    brow_dilate_px = max(6, int(face_h * (0.060 if preserve_fringe_detail else 0.07)))

    left_mask = self._build_landmark_hull_mask(left_eye_pts, (H, W), dilate_px=dilate_px)
    right_mask = self._build_landmark_hull_mask(right_eye_pts, (H, W), dilate_px=dilate_px)
    brow_mask = np.zeros((H, W), dtype=np.float32)
    if len(left_brow_pts) >= 3:
        brow_mask = np.maximum(
            brow_mask,
            self._build_landmark_hull_mask(left_brow_pts, (H, W), dilate_px=brow_dilate_px),
        ).astype(np.float32)
    if len(right_brow_pts) >= 3:
        brow_mask = np.maximum(
            brow_mask,
            self._build_landmark_hull_mask(right_brow_pts, (H, W), dilate_px=brow_dilate_px),
        ).astype(np.float32)
    eye_mask = np.maximum(left_mask, right_mask).astype(np.float32)
    if float(brow_mask.sum()) > 0.0:
        eye_mask = np.maximum(
            eye_mask,
            np.clip(brow_mask * 0.88, 0.0, 1.0),
        ).astype(np.float32)
    if float(eye_mask.sum()) < 10.0:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y1 + face_h * 0.18))
    bottom = min(H, int(y1 + face_h * 0.64))
    left = max(0, int(x1 - face_w * 0.04))
    right = min(W, int(x2 + face_w * 0.04))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    eye_u8 = cv2.bitwise_and(
        (np.clip(eye_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
        corridor_u8,
    )
    eye_u8 = cv2.dilate(
        eye_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 7) if preserve_fringe_detail else (7, 9),
        ),
        iterations=1,
    )
    band_pts = [left_eye_pts, right_eye_pts]
    if len(left_brow_pts) >= 3:
        band_pts.append(left_brow_pts)
    if len(right_brow_pts) >= 3:
        band_pts.append(right_brow_pts)
    try:
        band_pts_arr = np.concatenate(band_pts, axis=0)
    except Exception:
        band_pts_arr = np.zeros((0, 2), dtype=np.int32)
    if len(band_pts_arr) >= 6:
        band_x1 = max(0, int(np.min(band_pts_arr[:, 0]) - face_w * 0.04))
        band_x2 = min(W, int(np.max(band_pts_arr[:, 0]) + face_w * 0.04))
        band_top = max(0, int(np.min(band_pts_arr[:, 1]) - face_h * 0.04))
        band_bottom = min(H, int(max(left_eye_pts[:, 1].max(), right_eye_pts[:, 1].max()) + face_h * 0.06))
        if band_top < band_bottom and band_x1 < band_x2:
            band_u8 = np.zeros((H, W), dtype=np.uint8)
            band_u8[band_top:band_bottom, band_x1:band_x2] = 255
            band_u8 = cv2.bitwise_and(band_u8, corridor_u8)
            band_u8 = cv2.GaussianBlur(
                band_u8,
                (0, 0),
                sigmaX=1.8 if preserve_fringe_detail else 2.2,
                sigmaY=1.3 if preserve_fringe_detail else 1.6,
            )
            eye_u8 = cv2.bitwise_or(
                eye_u8,
                (band_u8 > (32 if preserve_fringe_detail else 24)).astype(np.uint8) * 255,
            )
    if (
        hair_length != "long"
        and final_hair_mask is not None
        and final_hair_mask.shape == (H, W)
    ):
        hair_u8 = cv2.dilate(
            (
                np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0)
                > (0.18 if preserve_fringe_detail else 0.28)
            ).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 9) if preserve_fringe_detail else (5, 7),
            ),
            iterations=1,
        )
        eye_u8 = cv2.bitwise_and(eye_u8, cv2.bitwise_not(hair_u8))
    if int((eye_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        eye_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=1.6 if preserve_fringe_detail else 2.0,
        sigmaY=1.9 if preserve_fringe_detail else 2.4,
    ).astype(np.float32)


def _build_face_eye_band_restore_mask(
    self,
    face_mask: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    H, W = image_shape
    if face_mask is None or face_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_h = max(int(y2 - y1), 1)
    face_w = max(int(x2 - x1), 1)

    face_u8 = cv2.dilate(
        (np.clip(face_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    band_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y1 + face_h * 0.20))
    bottom = min(H, int(y1 + face_h * 0.60))
    left = max(0, int(x1 - face_w * 0.04))
    right = min(W, int(x2 + face_w * 0.04))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    band_u8[top:bottom, left:right] = 255
    band_u8 = cv2.bitwise_and(band_u8, face_u8)
    if int((band_u8 > 0).sum()) < 36:
        return np.zeros((H, W), dtype=np.float32)
    return cv2.GaussianBlur(
        band_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.8,
        sigmaY=2.4,
    ).astype(np.float32)


@staticmethod
def _restore_reference_region(
    base_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    restore_mask: np.ndarray,
    *,
    strength: float = 0.94,
) -> np.ndarray:
    H, W = base_rgb.shape[:2]
    if reference_rgb.shape[:2] != (H, W) or restore_mask.shape != (H, W):
        return base_rgb
    alpha = cv2.GaussianBlur(
        np.clip(restore_mask.astype(np.float32), 0.0, 1.0),
        (0, 0),
        sigmaX=1.8,
        sigmaY=1.8,
    )[..., np.newaxis]
    alpha = np.clip(alpha * strength, 0.0, 1.0)
    out = reference_rgb.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _dilate_mask_with_px(self, mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate with an explicit pixel size without changing the global config."""
    if px <= 0:
        return np.clip(mask, 0.0, 1.0).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return np.clip(dilated, 0.0, 1.0).astype(np.float32)


def _dilate_hair_mask_for_length(
    self,
    mask: np.ndarray,
    hair_length: str,
) -> np.ndarray:
    """Use a smaller dilation for short/medium hair to avoid lower-mask overgrowth."""
    px = int(self.config.mask_dilate_px)
    if hair_length == "short":
        px = max(5, min(11, int(round(px * 0.35))))
    elif hair_length == "medium":
        px = max(7, min(15, int(round(px * 0.50))))
    return self._dilate_mask_with_px(mask, px)


def _resolve_mask_refine_mode(
    self,
    mask_refine_mode: Optional[str],
) -> str:
    value = str(
        mask_refine_mode
        or getattr(self.config, "mask_refine_mode", "")
        or os.environ.get("MASK_REFINE_MODE", "sam2")
    ).strip().lower()
    if value not in {"sam2", "segface_priority", "segface_only"}:
        logger.warning(
            "[SDPipeline] unsupported mask_refine_mode '%s', falling back to sam2",
            value,
        )
        return "sam2"
    return value


def _merge_segface_priority_mask(
    self,
    *,
    base_mask: np.ndarray,
    sam2_mask: np.ndarray,
    hair_length: str,
) -> np.ndarray:
    base_u8 = ((np.clip(base_mask, 0.0, 1.0) > 0.5).astype(np.uint8) * 255)
    sam2_u8 = ((np.clip(sam2_mask, 0.0, 1.0) > 0.5).astype(np.uint8) * 255)

    if hair_length == "short":
        core_kernel = (5, 5)
        growth_px = 7
        close_kernel = (7, 7)
    elif hair_length == "medium":
        core_kernel = (7, 7)
        growth_px = 9
        close_kernel = (9, 9)
    else:
        core_kernel = (9, 9)
        growth_px = 13
        close_kernel = (11, 11)

    base_core = cv2.erode(
        base_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, core_kernel),
        iterations=1,
    )
    allow_growth = (self._dilate_mask_with_px(base_u8.astype(np.float32) / 255.0, growth_px) > 0.5).astype(np.uint8) * 255
    sam2_local = cv2.bitwise_and(sam2_u8, allow_growth)
    merged_u8 = cv2.bitwise_or(base_core, sam2_local)

    base_px = int((base_u8 > 0).sum())
    merged_px = int((merged_u8 > 0).sum())
    if merged_px < max(60, int(base_px * 0.55)):
        logger.info(
            "[SDPipeline] segface_priority fallback to base mask: base_px=%s merged_px=%s",
            base_px,
            merged_px,
        )
        return (base_u8 > 0).astype(np.float32)

    merged_u8 = cv2.morphologyEx(
        merged_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel),
    )
    return (merged_u8 > 0).astype(np.float32)


def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
    """마스크 dilate (경계 확장)"""
    px = self.config.mask_dilate_px
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return np.clip(dilated, 0.0, 1.0).astype(np.float32)


def _crop_face(
    self,
    img_pil: Image.Image,
    face_bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """IP-Adapter용 얼굴 crop (얼굴만 — 머리카락 최소화)

    padding을 아래쪽은 넉넉히(턱/목 포함), 위쪽/옆은 최소화(머리카락 제외)
    IP-Adapter가 원본 헤어 스타일을 conditioning하면 숏컷 변환이 안 됨.
    """
    x1, y1, x2, y2 = face_bbox
    W, H = img_pil.size
    bw, bh = x2 - x1, y2 - y1
    # 위/옆은 패딩 최소화(0.05) → 머리카락 포함 억제
    # 아래는 패딩 넉넉히(0.2) → 턱/목 포함 → 얼굴 identity 안정화
    pad_side = int(bw * 0.05)
    pad_top  = int(bh * 0.05)
    pad_bot  = int(bh * 0.20)
    crop = img_pil.crop((
        max(0, x1 - pad_side),
        max(0, y1 - pad_top),
        min(W, x2 + pad_side),
        min(H, y2 + pad_bot),
    ))
    return crop.resize((224, 224), Image.LANCZOS)


def bind_mask_builder_methods_to_pipeline(cls) -> None:
    """마스크 빌더 메서드를 MirrAISDPipeline에 바인딩."""
    cls._build_post_cloth_refine_mask = _build_post_cloth_refine_mask
    cls._build_generation_protect_mask = _build_generation_protect_mask
    cls._build_removal_protect_mask = _build_removal_protect_mask
    cls._build_neckline_preserve_mask = _build_neckline_preserve_mask
    cls._build_short_lateral_neck_preserve_mask = _build_short_lateral_neck_preserve_mask
    cls._build_shoulder_protect_mask = _build_shoulder_protect_mask
    cls._build_torso_cloth_preserve_mask = _build_torso_cloth_preserve_mask
    cls._build_short_below_bob_torso_mask = _build_short_below_bob_torso_mask
    cls._build_bright_cloth_preserve_mask = _build_bright_cloth_preserve_mask
    cls._build_micro_cloth_artifact_mask = _build_micro_cloth_artifact_mask
    cls._build_shoulder_cloth_restore_mask = _build_shoulder_cloth_restore_mask
    cls._build_final_source_cloth_rescue_mask = _build_final_source_cloth_rescue_mask
    cls._build_short_subject_cloth_cleanup_mask = _build_short_subject_cloth_cleanup_mask
    cls._build_short_lower_garment_cleanup_mask = _build_short_lower_garment_cleanup_mask
    cls._build_short_lower_cloth_hard_override_mask = _build_short_lower_cloth_hard_override_mask
    cls._build_side_column_cloth_restore_mask = _build_side_column_cloth_restore_mask
    cls._build_direct_short_column_restore_mask = _build_direct_short_column_restore_mask
    cls._build_short_below_bob_cloth_restore_mask = _build_short_below_bob_cloth_restore_mask
    cls._build_short_below_bob_generation_block_mask = _build_short_below_bob_generation_block_mask
    cls._build_preclean_side_column_cleanup_mask = _build_preclean_side_column_cleanup_mask
    cls._build_preclean_cloth_hair_cleanup_mask = _build_preclean_cloth_hair_cleanup_mask
    cls._build_short_generation_conditioning_cleanup_mask = _build_short_generation_conditioning_cleanup_mask
    cls._build_residual_strand_cleanup_mask = _build_residual_strand_cleanup_mask
    cls._build_final_hair_lane_cleanup_mask = _build_final_hair_lane_cleanup_mask
    cls._build_short_bob_tail_suppress_mask = _build_short_bob_tail_suppress_mask
    cls._build_short_final_side_lane_refine_mask = _build_short_final_side_lane_refine_mask
    cls._build_short_lower_tail_cleanup_mask = _build_short_lower_tail_cleanup_mask
    cls._build_dark_lane_cleanup_mask = _build_dark_lane_cleanup_mask
    cls._build_side_tail_cleanup_mask = _build_side_tail_cleanup_mask
    cls._build_short_tail_core_mask = _build_short_tail_core_mask
    cls._build_front_strand_cleanup_mask = _build_front_strand_cleanup_mask
    cls._build_short_regen_tail_mask = _build_short_regen_tail_mask
    cls._build_lower_hair_tail_support_mask = _build_lower_hair_tail_support_mask
    cls._build_center_chest_strand_support_mask = _build_center_chest_strand_support_mask
    cls._build_lower_tail_post_support_mask = _build_lower_tail_post_support_mask
    cls._build_lower_tail_removal_extension_mask = _build_lower_tail_removal_extension_mask
    cls._build_dark_tail_residual_mask = _build_dark_tail_residual_mask
    cls._maybe_standardize_input_portrait = _maybe_standardize_input_portrait
    cls._crop_with_soft_padding = staticmethod(_crop_with_soft_padding)
    cls._mask_bbox = staticmethod(_mask_bbox)
    cls._analyze_source_cloth_preclean_need = _analyze_source_cloth_preclean_need
    cls._build_upper_clothes_overwrite_mask = _build_upper_clothes_overwrite_mask
    cls._build_source_garment_prepass_mask = _build_source_garment_prepass_mask
    cls._build_completed_subject_torso_fill_mask = _build_completed_subject_torso_fill_mask
    cls._build_source_shoulder_contour_anchor_mask = _build_source_shoulder_contour_anchor_mask
    cls._build_short_shoulder_cross_bridge_mask = _build_short_shoulder_cross_bridge_mask
    cls._build_residual_side_hair_lane_removal_mask = _build_residual_side_hair_lane_removal_mask
    cls._estimate_head_generation_box = _estimate_head_generation_box
    cls._mask_ratio = staticmethod(_mask_ratio)
    cls._resize_mask_to_shape = staticmethod(_resize_mask_to_shape)
    cls._mask_to_u8 = staticmethod(_mask_to_u8)
    cls._count_active_mask_px = staticmethod(_count_active_mask_px)
    cls._get_cleanup_apply_min_px = _get_cleanup_apply_min_px
    cls._should_apply_cleanup_mask = _should_apply_cleanup_mask
    cls._sanitize_cloth_mask = _sanitize_cloth_mask
    cls._build_short_sam2_torso_bridge_mask = _build_short_sam2_torso_bridge_mask
    cls._finalize_source_garment_prepass = _finalize_source_garment_prepass
    cls._build_subject_cloth_anchor_mask = _build_subject_cloth_anchor_mask
    cls._build_subject_torso_anchor_mask = _build_subject_torso_anchor_mask
    cls._build_subject_shoulder_bridge_mask = _build_subject_shoulder_bridge_mask
    cls._filter_cloth_mask_to_subject_anchor = _filter_cloth_mask_to_subject_anchor
    cls._trim_blocky_short_restore_mask_u8 = _trim_blocky_short_restore_mask_u8
    cls._build_sparse_dark_cloth_support_mask = _build_sparse_dark_cloth_support_mask
    cls._build_bangs_recovery_mask = _build_bangs_recovery_mask
    cls._build_requested_front_coverage_mask = staticmethod(_build_requested_front_coverage_mask)
    cls._build_soft_bangs_generation_mask = _build_soft_bangs_generation_mask
    cls._build_eye_region_restore_mask = _build_eye_region_restore_mask
    cls._build_face_eye_band_restore_mask = _build_face_eye_band_restore_mask
    cls._restore_reference_region = staticmethod(_restore_reference_region)
    cls._dilate_mask_with_px = _dilate_mask_with_px
    cls._dilate_hair_mask_for_length = _dilate_hair_mask_for_length
    cls._resolve_mask_refine_mode = _resolve_mask_refine_mode
    cls._merge_segface_priority_mask = _merge_segface_priority_mask
    cls._dilate_mask = _dilate_mask
    cls._crop_face = _crop_face
