from __future__ import annotations

import dataclasses
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline_sd_inpainting import (
    CLOTH_CLASS_IDX,
    CONTROLNET_MODEL_ID,
    DEFAULT_RUNTIME_LORA_HF_FILENAME,
    DEFAULT_RUNTIME_LORA_HF_REPO_ID,
    DEFAULT_SEGFACE_BASE_HF_FILENAME,
    DEFAULT_SEGFACE_BASE_HF_REPO_ID,
    DEFAULT_SEGFACE_BASE_HF_SUBFOLDER,
    DEFAULT_SEGFACE_HF_FILENAME,
    DEFAULT_SEGFACE_HF_REPO_ID,
    DEFAULT_SEGFACE_HF_SUBFOLDER,
    DEFAULT_SEGFACE_INPUT_RES,
    DEFAULT_SEGFACE_MODEL_VARIANT,
    EARRING_CLASS_IDX,
    FACE_CLASS_IDXS,
    GLASS_CLASS_IDX,
    HAIR_CLASS_IDX,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_WEIGHT,
    NECKLACE_CLASS_IDX,
    PROJECT_ROOT,
    SD_INPAINT_MODEL_ID,
    SD_SIZE,
    MirrAISDPipeline,
    _COMMON_STYLE_BLOCK_NEGATIVE,
    _FEMALE_STYLE_HINTS,
    _FEMALE_SUBJECT_HINTS,
    _HAIR_COLOR_TARGET_RGB,
    _MALE_STYLE_HINTS,
    _MALE_SUBJECT_HINTS,
    _MEDIUM_HAIR_KEYWORDS,
    _NEGATIVE_BASE,
    _NO_COLOR_HINTS,
    _SHORT_HAIR_KEYWORDS,
    logger,
)

# Extracted from pipeline_sd_inpainting.py to keep MirrAISDPipeline smaller.

def _classify_hair_length(hairstyle_text: str) -> str:
    """헤어스타일 텍스트 → 'short' | 'medium' | 'long'"""
    text = hairstyle_text.lower()
    for kw in _SHORT_HAIR_KEYWORDS:
        if kw in text:
            return "short"
    for kw in _MEDIUM_HAIR_KEYWORDS:
        if kw in text:
            return "medium"
    return "long"

def _normalize_color_text(color_text: str) -> str:
    text = str(color_text or "").strip()
    lowered = text.lower()
    if lowered in _NO_COLOR_HINTS:
        return ""
    return text

def _normalize_subject_gender(subject_gender: Optional[str]) -> str:
    lowered = str(subject_gender or "").strip().lower()
    if not lowered:
        return ""
    if lowered in {"m", "male", "man", "men", "boy", "masculine", "남자", "남성"}:
        return "male"
    if lowered in {"f", "female", "woman", "women", "girl", "feminine", "여자", "여성"}:
        return "female"
    return ""

def _infer_subject_gender(
    hairstyle_text: str,
    subject_gender: Optional[str] = None,
) -> str:
    explicit = MirrAISDPipeline._normalize_subject_gender(subject_gender)
    if explicit:
        return explicit

    lowered = " ".join(str(hairstyle_text or "").strip().lower().split())
    if not lowered:
        return "neutral"

    male_hits = sum(1 for token in _MALE_STYLE_HINTS if token in lowered)
    female_hits = sum(1 for token in _FEMALE_STYLE_HINTS if token in lowered)
    male_hits += sum(1 for token in _MALE_SUBJECT_HINTS if token in lowered)
    female_hits += sum(1 for token in _FEMALE_SUBJECT_HINTS if token in lowered)

    if male_hits >= max(1, female_hits + 1):
        return "male"
    if female_hits >= max(1, male_hits + 1):
        return "female"
    return "neutral"

def _normalize_male_short_hairstyle_prompt_text(hairstyle_text: str) -> str:
    raw = " ".join(str(hairstyle_text or "").strip().split())
    lowered = raw.lower()
    hints: List[str] = []
    is_afro_style = any(
        token in lowered
        for token in ("afro", "coily", "coils", "kinky", "tight curl", "tight curls")
    )

    if is_afro_style:
        base_style = "clean masculine short afro haircut with compact rounded silhouette"
        hints.extend([
            "defined tight coils",
            "dense coily top texture",
            "controlled rounded side shape",
            "clean low taper around the ears",
        ])
    elif any(token in lowered for token in ("mullet", "wolf cut", "soft mullet", "baby mullet", "mini mullet")):
        base_style = "modern masculine layered wolf cut with controlled soft mullet balance"
        hints.extend([
            "textured crown and top layers",
            "controlled nape length",
            "soft temple coverage",
        ])
    elif any(
        token in lowered
        for token in ("side part", "side-part", "dandy", "two block", "two-block", "comma", "comma hair")
    ):
        base_style = "clean masculine dandy haircut with neat side-part balance and compact side silhouette"
        hints.extend([
            "controlled crown volume close to the head",
            "neat top line with low volume",
            "smooth top flow without fluffy lift",
            "flat side-part transition without winged lift",
            "compact side panels close to the head",
            "tidy temple shape",
            "balanced forehead framing",
        ])
    elif any(token in lowered for token in ("swept-back", "swept back", "regent")):
        base_style = "clean masculine regent haircut with restrained swept-back top and tapered sides"
        hints.extend([
            "controlled top lift",
            "compact sides close to the head",
            "neat back sweep without airy volume",
            "balanced side silhouette",
        ])
    elif any(token in lowered for token in ("buzz", "crew", "fade", "taper", "undercut", "crop", "cropped", "short")):
        base_style = "clean masculine short crop haircut"
        hints.extend([
            "textured top",
            "clean tapered sides",
        ])
    else:
        base_style = "clean masculine short layered haircut"
        hints.extend([
            "balanced side shape",
            "natural top texture with light lift",
        ])

    if "bang" in lowered or "fringe" in lowered:
        hints.append("soft masculine fringe with natural forehead coverage")
    else:
        hints.append("natural masculine hairline with balanced forehead coverage")

    if is_afro_style:
        hints.append("coil definition from root to tip")
        hints.append("no loose straight flyaway strands")
        hints.append("no center part")
        hints.append("no straight side wings")
        hints.append("no parted curtain fringe")
    elif any(token in lowered for token in ("wave", "wavy", "curl", "curly", "perm")):
        hints.append("light natural texture")
    elif any(token in lowered for token in ("straight", "sleek")):
        hints.append("soft natural finish")

    hints.append("clean ear contour")
    hints.append("no feminine bob silhouette")
    hints.append("no dangling side locks")
    if any(token in lowered for token in ("side part", "side-part", "dandy", "comma", "comma hair")):
        hints.append("no oversized fluffy crown")
        hints.append("no airy side flare")
        hints.append("no winged side-part lift")

    parts = [base_style]
    for hint in hints:
        if hint not in parts:
            parts.append(hint)
    return ", ".join(parts)

def _is_compact_male_short_style(hairstyle_text: str) -> bool:
    lowered = " ".join(str(hairstyle_text or "").strip().split()).lower()
    return any(
        token in lowered
        for token in (
            "side part",
            "side-part",
            "dandy",
            "comma",
            "comma hair",
            "two block",
            "two-block",
        )
    )

def _normalize_male_medium_hairstyle_prompt_text(hairstyle_text: str) -> str:
    raw = " ".join(str(hairstyle_text or "").strip().split())
    lowered = raw.lower()
    hints: List[str] = []
    is_afro_style = any(
        token in lowered
        for token in ("afro", "coily", "coils", "kinky", "tight curl", "tight curls")
    )

    if is_afro_style:
        base_style = "masculine rounded afro hairstyle with controlled width and defined coils"
        hints.extend([
            "dense coily volume",
            "rounded silhouette around the crown",
            "controlled temple taper",
            "compact outline around the face",
        ])
    elif any(token in lowered for token in ("mullet", "wolf cut", "soft mullet", "baby mullet", "mini mullet")):
        base_style = "masculine medium layered wolf cut with restrained volume"
        hints.extend([
            "moderate crown height",
            "controlled nape length",
            "proportional silhouette around the face",
        ])
    elif any(
        token in lowered
        for token in ("swept-back", "swept back", "side part", "side-part", "dandy", "two block", "two-block", "comma", "regent")
    ):
        base_style = "masculine medium layered haircut with shorter back and sides"
        hints.extend([
            "moderate crown height",
            "restrained top lift",
            "compact temple volume",
            "soft front movement",
        ])
    else:
        base_style = "masculine medium layered haircut with proportional volume"
        hints.extend([
            "moderate top volume",
            "controlled side silhouette",
        ])

    if is_afro_style:
        hints.append("tight coil definition")
        hints.append("no straight dangling strands")
        hints.append("no center part")
        hints.append("no straight side wings")
        hints.append("no parted curtain fringe")
    elif any(token in lowered for token in ("wave", "wavy", "curl", "curly", "perm")):
        hints.append("light natural texture")
    elif any(token in lowered for token in ("straight", "sleek")):
        hints.append("soft natural finish")

    if "bang" in lowered or "fringe" in lowered:
        hints.append("soft masculine fringe with natural forehead coverage")
    else:
        hints.append("natural masculine hairline with balanced forehead coverage")

    hints.append("hairstyle proportional to face size")
    hints.append("no oversized fluffy crown")
    hints.append("no exaggerated side expansion")
    hints.append("clean ear contour")

    parts = [base_style]
    for hint in hints:
        if hint not in parts:
            parts.append(hint)
    return ", ".join(parts)

def _normalize_hairstyle_prompt_text(
    hairstyle_text: str,
    hair_length: str,
    subject_gender: Optional[str] = None,
) -> str:
    raw = " ".join(str(hairstyle_text or "").strip().split())
    if not raw:
        return ""
    gender_mode = MirrAISDPipeline._infer_subject_gender(raw, subject_gender)
    if gender_mode == "male":
        if hair_length == "short":
            return MirrAISDPipeline._normalize_male_short_hairstyle_prompt_text(raw)
        if hair_length == "medium":
            return MirrAISDPipeline._normalize_male_medium_hairstyle_prompt_text(raw)
    if hair_length != "short":
        return raw

    lowered = raw.lower()
    hints: List[str] = []
    if "hush" in lowered or "layer" in lowered:
        hints.append("soft internal bob layers above the jawline")
        hints.append("rounded jaw-length bob silhouette")
    if "blunt" in lowered:
        hints.append("clean blunt bob outline")
    if "bang" in lowered or "fringe" in lowered:
        hints.append("soft see-through bangs")
    if any(token in lowered for token in ("wave", "wavy", "curl", "curly")):
        hints.append("light natural texture")
    if any(token in lowered for token in ("straight", "sleek")):
        hints.append("sleek straight finish")
    if "tuck" in lowered:
        hints.append("tucked nape silhouette")
    else:
        hints.append("tucked inward ends at the jawline")
    if any(token in lowered for token in ("bob", "short", "chin")):
        hints.append("clear neckline and shoulders")
        hints.append("no lower side tails below the jawline")

    base_style = "strict short chin-length bob haircut with a compact side silhouette"
    if "pixie" in lowered or "buzz" in lowered:
        base_style = "strict short cropped haircut"

    parts = [base_style]
    for hint in hints:
        if hint not in parts:
            parts.append(hint)
    return ", ".join(parts)

def _resolve_target_hair_lab(color_text: str) -> Optional[np.ndarray]:
    query = str(color_text or "").strip().lower()
    if not query:
        return None
    for keyword, rgb in _HAIR_COLOR_TARGET_RGB:
        if keyword in query:
            rgb_np = np.array([[list(rgb)]], dtype=np.uint8)
            lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
            return lab
    return None

def _estimate_hair_color_distance(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    target_lab: np.ndarray,
) -> Optional[float]:
    hair_mask, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    hair_u8 = (hair_mask > 0.45).astype(np.uint8) * 255
    if int((hair_u8 > 0).sum()) < 80:
        return None

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hair_pixels = lab[hair_u8 > 0]
    if hair_pixels.shape[0] < 50:
        return None

    # 극단적인 shadow 영역 영향 완화
    if hair_pixels.shape[0] > 200:
        l_vals = hair_pixels[:, 0]
        keep = l_vals > np.percentile(l_vals, 15.0)
        if np.any(keep):
            hair_pixels = hair_pixels[keep]

    med = np.median(hair_pixels, axis=0)
    d_l = abs(float(med[0] - target_lab[0]))
    d_a = abs(float(med[1] - target_lab[1]))
    d_b = abs(float(med[2] - target_lab[2]))
    # 색조(a,b)를 더 강하게 반영
    return 0.25 * d_l + 0.85 * d_a + 0.85 * d_b

def _estimate_short_tail_penalty(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    removal_mask: np.ndarray,
) -> Optional[float]:
    H, W = img_rgb.shape[:2]
    if removal_mask.shape != (H, W):
        return None

    tail_hint = self._build_side_tail_cleanup_mask(
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length="short",
    )
    face_w = max(int(face_bbox[2] - face_bbox[0]), 1)
    face_h = max(int(face_bbox[3] - face_bbox[1]), 1)
    deep_start = min(H, int(cutoff_y + face_h * 0.08))
    zone = tail_hint.copy()
    zone[:deep_start, :] = 0.0
    broad_zone = np.zeros((H, W), dtype=np.float32)
    broad_left = max(0, int(face_bbox[0] - face_w * 0.92))
    broad_right = min(W, int(face_bbox[2] + face_w * 0.92))
    broad_bottom = min(H, int(cutoff_y + face_h * 1.14))
    if deep_start < broad_bottom and broad_left < broad_right:
        broad_zone[deep_start:broad_bottom, broad_left:broad_right] = 1.0
        zone = np.maximum(zone, broad_zone * 0.38)
    if float(zone.sum()) < 20.0:
        zone = removal_mask.copy().astype(np.float32)
        zone[:deep_start, :] = 0.0
        if float(broad_zone.sum()) > 0.0:
            zone = np.maximum(zone, broad_zone * 0.38)
    if float(zone.sum()) < 20.0:
        return None

    hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    hair_now[:deep_start, :] = 0.0

    zone_bool = zone > 0.08
    if int(zone_bool.sum()) < 20:
        return None

    hair_penalty = float(np.mean(hair_now[zone_bool]))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    darkness = np.clip((118.0 - gray) / 118.0, 0.0, 1.0)
    dark_penalty = float(np.mean(darkness[zone_bool]))
    deep_penalty = 0.0
    deep_zone_bool = broad_zone > 0.0
    if int(deep_zone_bool.sum()) >= 20:
        deep_penalty = float(np.mean(hair_now[deep_zone_bool]))
    return 0.60 * hair_penalty + 0.15 * dark_penalty + 0.25 * deep_penalty

def _estimate_accessory_penalty(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
) -> Optional[float]:
    H, W = img_rgb.shape[:2]
    _ = self._segface_hair_mask(img_rgb, face_bbox)
    segface_debug = self._last_segface_mask_debug or {}
    earring_mask = segface_debug.get("earring_mask")
    necklace_mask = segface_debug.get("necklace_mask")
    if not isinstance(earring_mask, np.ndarray) or not isinstance(necklace_mask, np.ndarray):
        return None
    if earring_mask.shape != (H, W) or necklace_mask.shape != (H, W):
        return None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y1 - face_h * 0.16))
    bottom = min(H, int(y2 + face_h * 0.92))
    left = max(0, int(x1 - face_w * 0.92))
    right = min(W, int(x2 + face_w * 0.92))
    if top >= bottom or left >= right:
        return None
    corridor_u8[top:bottom, left:right] = 255

    earring_u8 = cv2.dilate(
        (np.clip(earring_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    necklace_u8 = cv2.dilate(
        (np.clip(necklace_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    earring_u8 = cv2.bitwise_and(earring_u8, corridor_u8)
    necklace_u8 = cv2.bitwise_and(necklace_u8, corridor_u8)

    earring_px = int((earring_u8 > 0).sum())
    necklace_px = int((necklace_u8 > 0).sum())
    if earring_px < 2 and necklace_px < 6:
        return 0.0

    norm = float(max(face_w * face_h, 1))
    earring_penalty = min(float(earring_px) / norm * 36.0, 1.0)
    necklace_penalty = min(float(necklace_px) / norm * 20.0, 1.0)
    return 0.74 * earring_penalty + 0.26 * necklace_penalty

def _estimate_hair_shape_profile(
    self,
    hair_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    *,
    hair_length: str = "medium",
) -> Optional[Dict[str, float]]:
    if hair_mask is None:
        return None

    H, W = hair_mask.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    work = np.clip(hair_mask.astype(np.float32), 0.0, 1.0).copy()
    corridor_top = max(0, int(y1 - face_h * 0.78))
    corridor_bottom = min(H, int(y2 + face_h * (0.26 if hair_length == "medium" else 0.22)))
    corridor_left = max(0, int(x1 - face_w * 0.90))
    corridor_right = min(W, int(x2 + face_w * 0.90))
    if corridor_top >= corridor_bottom or corridor_left >= corridor_right:
        return None

    corridor = np.zeros((H, W), dtype=np.float32)
    corridor[corridor_top:corridor_bottom, corridor_left:corridor_right] = 1.0
    work *= corridor
    if float(work.sum()) < 20.0:
        return None

    bbox = self._mask_bbox(work, threshold=0.20)
    if bbox is None:
        return None
    hx1, hy1, hx2, hy2 = [int(v) for v in bbox]

    upper_top = max(0, int(y1 - face_h * 0.62))
    upper_bottom = min(H, int(y1 + face_h * 0.14))
    upper_left = max(0, int(x1 - face_w * 0.56))
    upper_right = min(W, int(x2 + face_w * 0.56))
    upper_band = work[upper_top:upper_bottom, upper_left:upper_right]
    upper_density = 0.0
    if upper_band.size > 0:
        upper_density = float(np.mean(upper_band > 0.20))

    crown_top = max(0, int(y1 - face_h * 0.62))
    crown_bottom = max(crown_top + 1, int(y1 - face_h * 0.04))
    crown_left = max(0, int(x1 + face_w * 0.10))
    crown_right = min(W, int(x2 - face_w * 0.10))
    crown_density = 0.0
    if crown_top < crown_bottom and crown_left < crown_right:
        crown_band = work[crown_top:crown_bottom, crown_left:crown_right]
        if crown_band.size > 0:
            crown_density = float(np.mean(crown_band > 0.20))

    face_area = float(max(face_w * face_h, 1))
    area_ratio = float(np.sum(work > 0.20)) / face_area

    return {
        "width_ratio": float(max(hx2 - hx1, 1)) / float(face_w),
        "top_lift": float(max(y1 - hy1, 0)) / float(face_h),
        "left_overhang": float(max(x1 - hx1, 0)) / float(face_w),
        "right_overhang": float(max(hx2 - x2, 0)) / float(face_w),
        "area_ratio": area_ratio,
        "upper_density": upper_density,
        "crown_density": crown_density,
        "center_offset": float(((0.5 * (hx1 + hx2)) - (0.5 * (x1 + x2))) / float(face_w)),
        "side_balance": float(abs(max(hx2 - x2, 0) - max(x1 - hx1, 0)) / float(face_w)),
        "mass_center_offset": float(self._estimate_mask_mass_center_offset(work, face_bbox)),
    }

def _estimate_male_medium_fit_penalty(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    source_profile: Optional[Dict[str, float]],
) -> Optional[float]:
    if source_profile is None:
        return None

    hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    candidate_profile = self._estimate_hair_shape_profile(
        hair_now,
        face_bbox,
        hair_length="medium",
    )
    if candidate_profile is None:
        return None

    def _oversize(metric: str, allowance: float, scale: float) -> float:
        base = float(source_profile.get(metric, 0.0))
        current = float(candidate_profile.get(metric, 0.0))
        excess = max(0.0, current - base - allowance)
        return float(np.clip(excess / max(scale, 1e-6), 0.0, 1.0))

    width_penalty = _oversize("width_ratio", allowance=0.08, scale=0.26)
    top_penalty = _oversize("top_lift", allowance=0.05, scale=0.18)
    left_penalty = _oversize("left_overhang", allowance=0.06, scale=0.18)
    right_penalty = _oversize("right_overhang", allowance=0.06, scale=0.18)
    area_penalty = _oversize("area_ratio", allowance=0.18, scale=0.44)
    upper_penalty = _oversize("upper_density", allowance=0.08, scale=0.28)
    crown_penalty = _oversize("crown_density", allowance=0.08, scale=0.28)
    side_balance_penalty = _oversize("side_balance", allowance=0.07, scale=0.18)

    base_center_bias = abs(float(source_profile.get("center_offset", 0.0)))
    current_center_bias = abs(float(candidate_profile.get("center_offset", 0.0)))
    center_offset_penalty = float(
        np.clip((current_center_bias - base_center_bias - 0.03) / 0.14, 0.0, 1.0)
    )

    base_mass_bias = abs(float(source_profile.get("mass_center_offset", 0.0)))
    current_mass_bias = abs(float(candidate_profile.get("mass_center_offset", 0.0)))
    mass_center_penalty = float(
        np.clip((current_mass_bias - base_mass_bias - 0.03) / 0.12, 0.0, 1.0)
    )

    current_side_bias = abs(float(candidate_profile.get("right_overhang", 0.0)) - float(candidate_profile.get("left_overhang", 0.0)))
    absolute_side_bias_penalty = float(np.clip((current_side_bias - 0.10) / 0.22, 0.0, 1.0))
    absolute_center_bias_penalty = float(np.clip((current_center_bias - 0.08) / 0.18, 0.0, 1.0))
    absolute_mass_bias_penalty = float(np.clip((current_mass_bias - 0.08) / 0.16, 0.0, 1.0))

    return float(
        0.18 * width_penalty
        + 0.14 * top_penalty
        + 0.09 * left_penalty
        + 0.09 * right_penalty
        + 0.10 * area_penalty
        + 0.05 * upper_penalty
        + 0.05 * crown_penalty
        + 0.11 * center_offset_penalty
        + 0.08 * side_balance_penalty
        + 0.07 * mass_center_penalty
        + 0.02 * absolute_side_bias_penalty
        + 0.01 * absolute_center_bias_penalty
        + 0.01 * absolute_mass_bias_penalty
    )

def _estimate_male_short_fit_penalty(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    source_profile: Optional[Dict[str, float]],
    hairstyle_text: str,
) -> Optional[float]:
    if source_profile is None:
        return None

    candidate_profile = self._estimate_hair_shape_profile(
        self._segface_hair_mask(img_rgb, face_bbox)[0],
        face_bbox,
        hair_length="short",
    )
    if candidate_profile is None:
        return None

    compact_style = self._is_compact_male_short_style(hairstyle_text)

    def _oversize(metric: str, allowance: float, scale: float) -> float:
        base = float(source_profile.get(metric, 0.0))
        current = float(candidate_profile.get(metric, 0.0))
        excess = max(0.0, current - base - allowance)
        return float(np.clip(excess / max(scale, 1e-6), 0.0, 1.0))

    if compact_style:
        width_penalty = _oversize("width_ratio", allowance=0.04, scale=0.18)
        top_penalty = _oversize("top_lift", allowance=0.02, scale=0.10)
        left_penalty = _oversize("left_overhang", allowance=0.04, scale=0.14)
        right_penalty = _oversize("right_overhang", allowance=0.04, scale=0.14)
        area_penalty = _oversize("area_ratio", allowance=0.10, scale=0.26)
        upper_penalty = _oversize("upper_density", allowance=0.04, scale=0.16)
        crown_penalty = _oversize("crown_density", allowance=0.04, scale=0.16)
        side_balance_penalty = _oversize("side_balance", allowance=0.05, scale=0.12)
        absolute_top_penalty = float(
            np.clip((float(candidate_profile.get("top_lift", 0.0)) - 0.28) / 0.12, 0.0, 1.0)
        )
        absolute_upper_penalty = float(
            np.clip((float(candidate_profile.get("upper_density", 0.0)) - 0.44) / 0.18, 0.0, 1.0)
        )
        absolute_crown_penalty = float(
            np.clip((float(candidate_profile.get("crown_density", 0.0)) - 0.42) / 0.18, 0.0, 1.0)
        )
    else:
        width_penalty = _oversize("width_ratio", allowance=0.06, scale=0.22)
        top_penalty = _oversize("top_lift", allowance=0.04, scale=0.14)
        left_penalty = _oversize("left_overhang", allowance=0.05, scale=0.16)
        right_penalty = _oversize("right_overhang", allowance=0.05, scale=0.16)
        area_penalty = _oversize("area_ratio", allowance=0.14, scale=0.32)
        upper_penalty = _oversize("upper_density", allowance=0.06, scale=0.20)
        crown_penalty = _oversize("crown_density", allowance=0.06, scale=0.20)
        side_balance_penalty = _oversize("side_balance", allowance=0.06, scale=0.16)
        absolute_top_penalty = float(
            np.clip((float(candidate_profile.get("top_lift", 0.0)) - 0.34) / 0.14, 0.0, 1.0)
        )
        absolute_upper_penalty = float(
            np.clip((float(candidate_profile.get("upper_density", 0.0)) - 0.50) / 0.20, 0.0, 1.0)
        )
        absolute_crown_penalty = float(
            np.clip((float(candidate_profile.get("crown_density", 0.0)) - 0.48) / 0.20, 0.0, 1.0)
        )

    base_center_bias = abs(float(source_profile.get("center_offset", 0.0)))
    current_center_bias = abs(float(candidate_profile.get("center_offset", 0.0)))
    center_offset_penalty = float(
        np.clip((current_center_bias - base_center_bias - 0.03) / 0.12, 0.0, 1.0)
    )

    base_mass_bias = abs(float(source_profile.get("mass_center_offset", 0.0)))
    current_mass_bias = abs(float(candidate_profile.get("mass_center_offset", 0.0)))
    mass_center_penalty = float(
        np.clip((current_mass_bias - base_mass_bias - 0.03) / 0.10, 0.0, 1.0)
    )

    return float(
        0.12 * width_penalty
        + 0.22 * top_penalty
        + 0.07 * left_penalty
        + 0.07 * right_penalty
        + 0.10 * area_penalty
        + 0.09 * upper_penalty
        + 0.12 * crown_penalty
        + 0.08 * center_offset_penalty
        + 0.05 * side_balance_penalty
        + 0.04 * mass_center_penalty
        + 0.02 * absolute_top_penalty
        + 0.01 * absolute_upper_penalty
        + 0.01 * absolute_crown_penalty
    )

def _estimate_mask_mass_center_offset(
    mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
) -> float:
    if mask is None:
        return 0.0

    work = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if work.ndim != 2:
        return 0.0

    x1, _, x2, _ = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    total = float(work.sum())
    if total <= 1e-6:
        return 0.0

    xs = np.arange(work.shape[1], dtype=np.float32)[np.newaxis, :]
    mass_center_x = float(np.sum(work * xs) / total)
    face_center_x = 0.5 * (x1 + x2)
    return float((mass_center_x - face_center_x) / float(face_w))

def _preserve_original_hair_tone(
    self,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    src_hair, _, _ = self._segface_hair_mask(source_rgb, face_bbox)
    tgt_hair, _, _ = self._segface_hair_mask(target_rgb, face_bbox)

    src_mask = (src_hair > 0.45)
    tgt_mask = (tgt_hair > 0.45)
    if int(src_mask.sum()) < 100 or int(tgt_mask.sum()) < 100:
        return target_rgb

    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    src_mean = src_lab[src_mask].mean(axis=0)
    tgt_mean = tgt_lab[tgt_mask].mean(axis=0)

    tuned_lab = tgt_lab.copy()
    vals = tuned_lab[tgt_mask]
    vals[:, 1] = np.clip(vals[:, 1] + (src_mean[1] - tgt_mean[1]) * 0.78, 0.0, 255.0)
    vals[:, 2] = np.clip(vals[:, 2] + (src_mean[2] - tgt_mean[2]) * 0.78, 0.0, 255.0)
    vals[:, 0] = np.clip(vals[:, 0] + (src_mean[0] - tgt_mean[0]) * 0.32, 0.0, 255.0)
    tuned_lab[tgt_mask] = vals

    tuned_rgb = cv2.cvtColor(tuned_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    alpha = cv2.GaussianBlur(tgt_hair.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
    alpha = np.clip(alpha * 0.70, 0.0, 1.0)[..., np.newaxis]
    out = tuned_rgb.astype(np.float32) * alpha + target_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _harmonize_short_bangs_tone(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    bangs_mask: np.ndarray,
    target_lab: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    if bangs_mask is None or bangs_mask.shape != (H, W):
        return img_rgb

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    final_hair, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    hair_u8 = (np.clip(final_hair.astype(np.float32), 0.0, 1.0) > 0.26).astype(np.uint8) * 255
    if int((hair_u8 > 0).sum()) < 120:
        return img_rgb

    bangs_u8 = (np.clip(bangs_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255
    if int((bangs_u8 > 0).sum()) < 20:
        return img_rgb

    band_top = max(0, int(y1 - face_h * 0.14))
    band_bottom = min(H, int(y1 + face_h * 0.42))
    band_left = max(0, int(cx - face_w * 0.64))
    band_right = min(W, int(cx + face_w * 0.64))
    if band_top >= band_bottom or band_left >= band_right:
        return img_rgb

    band_u8 = np.zeros((H, W), dtype=np.uint8)
    band_u8[band_top:band_bottom, band_left:band_right] = 255
    bangs_u8 = cv2.bitwise_and(bangs_u8, hair_u8)
    bangs_u8 = cv2.bitwise_and(bangs_u8, band_u8)
    bangs_u8 = cv2.morphologyEx(
        bangs_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
    )
    if int((bangs_u8 > 0).sum()) < 36:
        return img_rgb

    ref_top = max(0, int(y1 - face_h * 0.18))
    ref_bottom = min(H, int(y2 + face_h * 0.26))
    ref_left = max(0, int(x1 - face_w * 0.88))
    ref_right = min(W, int(x2 + face_w * 0.88))
    if ref_top >= ref_bottom or ref_left >= ref_right:
        return img_rgb

    ref_u8 = np.zeros((H, W), dtype=np.uint8)
    ref_u8[ref_top:ref_bottom, ref_left:ref_right] = 255
    ref_u8 = cv2.bitwise_and(ref_u8, hair_u8)
    ref_u8 = cv2.bitwise_and(
        ref_u8,
        cv2.bitwise_not(
            cv2.dilate(
                bangs_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
        ),
    )
    if int((ref_u8 > 0).sum()) < 80:
        return img_rgb

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bangs_vals = lab[bangs_u8 > 0]
    ref_vals = lab[ref_u8 > 0]
    if bangs_vals.size == 0 or ref_vals.size == 0:
        return img_rgb

    bangs_mean = bangs_vals.mean(axis=0)
    ref_mean = ref_vals.mean(axis=0)
    if target_lab is not None and np.asarray(target_lab).shape == (3,):
        ref_mean = ref_mean * 0.72 + np.asarray(target_lab, dtype=np.float32) * 0.28

    tuned_lab = lab.copy()
    tuned_vals = tuned_lab[bangs_u8 > 0]
    tuned_vals[:, 0] = np.clip(tuned_vals[:, 0] + (ref_mean[0] - bangs_mean[0]) * 0.52, 0.0, 255.0)
    tuned_vals[:, 1] = np.clip(tuned_vals[:, 1] + (ref_mean[1] - bangs_mean[1]) * 0.74, 0.0, 255.0)
    tuned_vals[:, 2] = np.clip(tuned_vals[:, 2] + (ref_mean[2] - bangs_mean[2]) * 0.74, 0.0, 255.0)
    tuned_lab[bangs_u8 > 0] = tuned_vals

    tuned_rgb = cv2.cvtColor(tuned_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    alpha = cv2.GaussianBlur(bangs_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=2.4, sigmaY=2.8)
    alpha = np.clip(alpha * 0.76, 0.0, 1.0)[..., np.newaxis]
    out = tuned_rgb.astype(np.float32) * alpha + img_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _build_prompt(
    hairstyle_text: str,
    color_text: str,
    hair_length: str = "long",
    subject_gender: Optional[str] = None,
    sd_prompt_data: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, float]:
    """
    Returns:
        positive_prompt, negative_prompt, guidance_scale

    sd_prompt_data가 제공되면 DB에 저장된 SD 프롬프트를 우선 사용.
    없으면 hairstyle_text 기반으로 폴백.
    """
    def _truncate_words(text: str, max_words: int) -> str:
        words = str(text).split()
        if len(words) <= max_words:
            return str(text).strip()
        return " ".join(words[:max_words]).strip(", ")

    def _compact_prompt_parts(parts: List[str], max_words: int = 34) -> str:
        compact: List[str] = []
        total_words = 0
        for part in parts:
            normalized_part = str(part).strip().strip(",")
            if not normalized_part:
                continue
            word_count = len(normalized_part.split())
            if compact and total_words + word_count > max_words:
                continue
            compact.append(normalized_part)
            total_words += word_count
        return ", ".join(compact)

    normalized_color = MirrAISDPipeline._normalize_color_text(color_text)
    gender_mode = MirrAISDPipeline._infer_subject_gender(
        hairstyle_text,
        subject_gender=subject_gender,
    )
    normalized_style = MirrAISDPipeline._normalize_hairstyle_prompt_text(
        hairstyle_text,
        hair_length,
        subject_gender=gender_mode,
    )

    # ── DB 프롬프트 데이터가 있으면 우선 사용 ─────────────────────────────
    if sd_prompt_data and sd_prompt_data.get("sd_positive"):
        style_part = _truncate_words(sd_prompt_data["sd_positive"], 18)
        sd_neg = sd_prompt_data.get("sd_negative", "")
        guidance = float(sd_prompt_data.get("sd_guidance", 8.5))

        color_pos_hint = ""
        color_neg_hint = ""
        lowered_color = normalized_color.lower()
        if normalized_color:
            style_part = f"{style_part}, {normalized_color.strip()} hair color"
            if "ash" in lowered_color:
                color_pos_hint = "cool-toned ash hair, no brassiness"
                color_neg_hint = "warm orange cast, yellow brassiness, copper tint, reddish tint, "
            else:
                color_pos_hint = "natural consistent hair color"

        positive_parts = [
            f"professional portrait photo of a person with {style_part}",
        ]
        if color_pos_hint:
            positive_parts.append(color_pos_hint)
        positive_parts.extend([
            "same outfit, clean neckline, preserved fabric folds",
            "photorealistic, natural lighting, sharp focus",
        ])
        positive = _compact_prompt_parts(positive_parts)
        negative_base = _NEGATIVE_BASE + ", " + _COMMON_STYLE_BLOCK_NEGATIVE
        negative = sd_neg + (", " if sd_neg else "") + color_neg_hint + negative_base

        return positive, negative, guidance

    parts = []
    if normalized_style:
        parts.append(normalized_style)
    if normalized_color:
        parts.append(f"{normalized_color.strip()} hair color")
    style = _truncate_words(", ".join(parts) if parts else "natural hairstyle", 18)
    subject_noun = "person"
    if gender_mode == "male":
        subject_noun = "man"
    elif gender_mode == "female":
        subject_noun = "woman"

    # 길이별 기본 보강 (직접 입력/DB 프롬프트 폴백 시 사용)
    if hair_length == "short" and gender_mode == "male":
        pos_suffix = (
            ", masculine short cut, balanced forehead, clean temple line, no side tails, no jewelry"
        )
        neg_prefix = (
            "feminine bob, chin-length bob, rounded bob, bixie, pixie bob, "
            "oversized exposed forehead, exaggerated high hairline, receding hairline, severe slicked-back hair, "
            "earring, earrings, hoop earrings, stud earrings, ear cuff, jewelry, necklace, makeup, "
        )
        guidance = 10.9
    elif hair_length == "short":
        pos_suffix = (
            ", short jaw-length bob, visible neck, above shoulders, no long tails"
        )
        neg_prefix = (
            "very long hair, medium hair, medium length hair, medium-length hair, shoulder-length hair, "
            "shoulder grazing hair, shoulder-grazing hair, collarbone-length hair, lob, "
            "flowing long hair, hair below shoulders, waist-length hair, side long locks over chest, "
            "long hush cut, long wolf cut, mullet tails, long layers below jawline, "
            "hair touching shoulders, hair covering collar, chest-length strands, neckline covered by hair, "
            "hair below jawline, hair below neckline, dangling lower tails, long side tails, nape tails, "
            "strands touching clothes, side locks on shoulders, hair covering blouse, "
            "overly voluminous hair, puffy hair, oversized bob, wide helmet shape, bulky side volume, "
            "blunt horizontal cut line, helmet hair, bowl-shaped edge, "
        )
        guidance = 11.2
    elif hair_length == "medium" and gender_mode == "male":
        pos_suffix = (
            ", masculine medium cut, balanced forehead, centered volume, no side sweep, no jewelry"
        )
        neg_prefix = (
            "feminine bob, rounded lob, dangling earrings, hoop earrings, necklace, jewelry, "
            "oversized exposed forehead, exaggerated high hairline, receding hairline, severe slicked-back hair, "
            "oversized fluffy crown, exaggerated pompadour, towering top volume, bulky side volume, oversized hair mass, "
            "hair pushed entirely to the right, hair pushed entirely to the left, heavy right sweep, heavy left sweep, "
            "off-center hair bulk, lopsided side volume, "
        )
        guidance = 8.9
    elif hair_length == "medium":
        pos_suffix = (
            ", medium length hair, shoulder-length hair, "
            "hair just above or at shoulder"
        )
        neg_prefix = "very long hair, very short hair, "
        guidance = 8.5
    else:
        pos_suffix = ", masculine hairline, balanced forehead, no jewelry" if gender_mode == "male" else ""
        neg_prefix = (
            "earring, earrings, hoop earrings, stud earrings, ear cuff, necklace, jewelry, "
            "oversized exposed forehead, exaggerated high hairline, receding hairline, "
            if gender_mode == "male"
            else ""
        )
        guidance = 7.5

    color_pos_hint = ""
    color_neg_hint = ""
    lowered_color = normalized_color.lower()
    if "ash" in lowered_color:
        color_pos_hint = "ash hair, no brassiness"
        color_neg_hint = "warm orange cast, yellow brassiness, copper tint, reddish tint, "
    elif normalized_color:
        color_pos_hint = "natural hair color"

    positive_parts = [
        f"professional portrait photo of a {subject_noun} with {style}{pos_suffix}",
    ]
    if color_pos_hint:
        positive_parts.append(color_pos_hint)
    positive_parts.extend([
        "balanced framing",
        "same outfit, clean neckline",
        "photorealistic portrait",
    ])
    positive = _compact_prompt_parts(positive_parts)
    negative_base = (
        _NEGATIVE_BASE
        + ", cropped head, cropped hair, cut off hair, top of head out of frame, tight close-up portrait, clipped hairstyle"
        + ", "
        + _COMMON_STYLE_BLOCK_NEGATIVE
    )
    negative = neg_prefix + color_neg_hint + negative_base

    return positive, negative, guidance

def bind_prompt_methods_to_pipeline(cls) -> None:
    cls._classify_hair_length = staticmethod(_classify_hair_length)
    cls._normalize_color_text = staticmethod(_normalize_color_text)
    cls._normalize_subject_gender = staticmethod(_normalize_subject_gender)
    cls._infer_subject_gender = staticmethod(_infer_subject_gender)
    cls._normalize_male_short_hairstyle_prompt_text = staticmethod(_normalize_male_short_hairstyle_prompt_text)
    cls._is_compact_male_short_style = staticmethod(_is_compact_male_short_style)
    cls._normalize_male_medium_hairstyle_prompt_text = staticmethod(_normalize_male_medium_hairstyle_prompt_text)
    cls._normalize_hairstyle_prompt_text = staticmethod(_normalize_hairstyle_prompt_text)
    cls._resolve_target_hair_lab = staticmethod(_resolve_target_hair_lab)
    cls._estimate_hair_color_distance = _estimate_hair_color_distance
    cls._estimate_short_tail_penalty = _estimate_short_tail_penalty
    cls._estimate_accessory_penalty = _estimate_accessory_penalty
    cls._estimate_hair_shape_profile = _estimate_hair_shape_profile
    cls._estimate_male_short_fit_penalty = _estimate_male_short_fit_penalty
    cls._estimate_male_medium_fit_penalty = _estimate_male_medium_fit_penalty
    cls._estimate_mask_mass_center_offset = staticmethod(_estimate_mask_mass_center_offset)
    cls._preserve_original_hair_tone = _preserve_original_hair_tone
    cls._harmonize_short_bangs_tone = _harmonize_short_bangs_tone
    cls._build_prompt = staticmethod(_build_prompt)
