"""
MirrAI SD Inpainting — output helpers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
최종 반환 이미지의 길이별 크롭 계산처럼 출력 포맷팅 도메인에 속하는
보조 함수만 모아둔 모듈.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import SDInpaintConfig


def _shift_box_within_bounds(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    box_w = max(int(right) - int(left), 1)
    box_h = max(int(bottom) - int(top), 1)
    left = int(left)
    top = int(top)
    right = left + box_w
    bottom = top + box_h

    if left < 0:
        right -= left
        left = 0
    if right > width:
        left -= right - width
        right = width
    if top < 0:
        bottom -= top
        top = 0
    if bottom > height:
        top -= bottom - height
        bottom = height

    left = max(0, left)
    top = max(0, top)
    right = min(width, max(left + 1, right))
    bottom = min(height, max(top + 1, bottom))
    return left, top, right, bottom


def build_length_aware_output_crop_box(
    config: SDInpaintConfig,
    image_shape: Tuple[int, int],
    face_bbox: Optional[Tuple[int, int, int, int]],
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if not getattr(config, "enable_output_crop_by_target_length", True):
        return None
    if face_bbox is None:
        return None

    height, width = [int(v) for v in image_shape[:2]]
    if height <= 1 or width <= 1:
        return None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(x2 - x1, 1)
    face_h = max(y2 - y1, 1)
    length_key = str(hair_length or "long").strip().lower()
    if length_key == "short":
        top_ratio = float(
            getattr(
                config,
                "output_crop_top_face_ratio_short",
                getattr(config, "output_crop_top_face_ratio", 0.85),
            )
        )
        bottom_ratio = float(getattr(config, "output_crop_bottom_face_ratio_short", 1.15))
        hair_side_ratio = 0.96
        hair_bottom_pad_ratio = 0.10
    elif length_key == "medium":
        top_ratio = float(getattr(config, "output_crop_top_face_ratio_medium", 0.48))
        bottom_ratio = float(getattr(config, "output_crop_bottom_face_ratio_medium", 1.85))
        hair_side_ratio = 1.22
        hair_bottom_pad_ratio = 0.18
    else:
        top_ratio = float(getattr(config, "output_crop_top_face_ratio_long", 0.34))
        bottom_ratio = float(getattr(config, "output_crop_bottom_face_ratio_long", 2.85))
        hair_side_ratio = 1.55
        hair_bottom_pad_ratio = 0.30

    face_crop_top = int(round(y1 - face_h * top_ratio))
    face_crop_bottom_cap = int(round(y2 + face_h * bottom_ratio))
    union_left = int(x1)
    union_top = int(face_crop_top)
    union_right = int(x2)
    union_bottom = int(face_crop_bottom_cap)

    hair_bbox: Optional[Tuple[int, int, int, int]] = None
    if (
        isinstance(final_hair_mask, np.ndarray)
        and final_hair_mask.ndim == 2
        and final_hair_mask.shape[:2] == (height, width)
    ):
        threshold = float(getattr(config, "output_crop_hair_mask_threshold", 0.18))
        hair_candidate = (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > threshold).astype(np.uint8)
        corridor_left = max(0, int(round(x1 - face_w * hair_side_ratio)))
        corridor_right = min(width, int(round(x2 + face_w * hair_side_ratio)))
        corridor_top = max(0, int(round(face_crop_top - face_h * 0.12)))
        corridor_bottom = min(height, int(round(face_crop_bottom_cap + face_h * 0.22)))
        corridor_u8 = np.zeros((height, width), dtype=np.uint8)
        if corridor_left < corridor_right and corridor_top < corridor_bottom:
            corridor_u8[corridor_top:corridor_bottom, corridor_left:corridor_right] = 1
            hair_candidate = cv2.bitwise_and(hair_candidate, corridor_u8)
        if int(hair_candidate.sum()) > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_candidate, 8)
            keep_u8 = np.zeros((height, width), dtype=np.uint8)
            min_area = max(24, int(face_w * face_h * 0.008))
            for label_idx in range(1, int(num_labels)):
                area = int(stats[label_idx, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                ly = int(stats[label_idx, cv2.CC_STAT_TOP])
                lw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
                lh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
                if lw <= 0 or lh <= 0:
                    continue
                if ly >= face_crop_bottom_cap:
                    continue
                keep_u8[labels == label_idx] = 1
            ys, xs = np.where(keep_u8 > 0)
            if len(xs) > 0 and len(ys) > 0:
                hair_x1 = int(xs.min())
                hair_y1 = int(ys.min())
                hair_x2 = int(xs.max()) + 1
                hair_y2 = int(ys.max()) + 1
                hair_bbox = (
                    max(0, int(round(hair_x1 - face_w * 0.10))),
                    max(0, int(round(hair_y1 - face_h * 0.10))),
                    min(width, int(round(hair_x2 + face_w * 0.10))),
                    min(height, int(round(hair_y2 + face_h * hair_bottom_pad_ratio))),
                )
    if hair_bbox is not None:
        hx1, hy1, hx2, hy2 = hair_bbox
        union_left = min(union_left, hx1)
        union_top = min(union_top, hy1)
        union_right = max(union_right, hx2)
        union_bottom = min(face_crop_bottom_cap, max(int(y2 + face_h * 0.10), hy2))

    union_w = max(union_right - union_left, face_w + 1)
    union_h = max(union_bottom - union_top, face_h + 1)
    target_aspect = width / max(float(height), 1.0)
    crop_h = max(union_h, int(round(union_w / max(target_aspect, 1e-6))), face_h + 1)
    crop_w = max(int(round(crop_h * target_aspect)), union_w, face_w + 1)
    cx = 0.5 * (union_left + union_right)
    cy = 0.5 * (union_top + union_bottom)
    crop_left = int(round(cx - crop_w * 0.5))
    crop_top = int(round(cy - crop_h * 0.5))
    crop_right = crop_left + crop_w
    crop_bottom = crop_top + crop_h
    crop_left, crop_top, crop_right, crop_bottom = _shift_box_within_bounds(
        crop_left,
        crop_top,
        crop_right,
        crop_bottom,
        width=width,
        height=height,
    )

    if crop_left <= 0 and crop_top <= 0 and crop_right >= width and crop_bottom >= height:
        return None
    return crop_left, crop_top, crop_right, crop_bottom
