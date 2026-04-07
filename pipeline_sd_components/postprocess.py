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

def _generate(
    self,
    img_512: Image.Image,
    mask_512: Image.Image,
    canny_512: Image.Image,
    face_crop_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    seeds: List[int],
    hair_length: str = "long",
) -> List[Image.Image]:
    """
    모든 seed를 단일 배치 forward pass로 생성 (순차 대비 ~절반 시간).

    diffusers는 generator를 리스트로 받으면 num_images_per_prompt 개의
    이미지를 각자 다른 seed로 한 번의 파이프라인 실행에 처리함.
    """
    # 숏컷/중단발 변환 시 IP-Adapter scale을 낮춤
    # → 원본 긴머리 identity가 생성에 과도하게 영향주는 것 방지
    if hair_length == "short":
        ip_scale = 0.01
        control_scale = min(self.config.controlnet_conditioning_scale, 0.08)
    elif hair_length == "medium":
        ip_scale = 0.18
        control_scale = min(self.config.controlnet_conditioning_scale, 0.20)
    else:
        ip_scale = self.config.ip_adapter_scale  # long은 기본값 유지
        control_scale = self.config.controlnet_conditioning_scale

    self._sd_pipe.set_ip_adapter_scale(ip_scale)
    logger.info(
        f"[SDPipeline] ip_adapter_scale={ip_scale}, "
        f"controlnet_scale={control_scale} (hair_length={hair_length})"
    )

    n = len(seeds)
    generators = [
        torch.Generator(device=self.device).manual_seed(s) for s in seeds
    ]
    logger.info(f"[SDPipeline] 배치 생성 시작 (n={n}, seeds={seeds})")

    with torch.inference_mode():
        out = self._sd_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_512,
            mask_image=mask_512,
            control_image=canny_512,
            ip_adapter_image=[face_crop_pil],
            height=SD_SIZE,
            width=SD_SIZE,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=control_scale,
            num_images_per_prompt=n,
            generator=generators,
            strength=1.0,
        )

    logger.info(f"[SDPipeline] 배치 생성 완료 → {len(out.images)}장")
    return out.images

def _cv2_refine_cloth_region(
    base_rgb: np.ndarray,
    cloth_refine_mask: np.ndarray,
    reference_rgb: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Clean residual shirt/blouse blur with a small deterministic inpaint pass.
    The mask is expected to already exclude hair and face-protect regions.
    """
    H, W = base_rgb.shape[:2]
    if cloth_refine_mask.shape != (H, W):
        return base_rgb

    ref_rgb = base_rgb
    if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
        ref_rgb = reference_rgb

    mask_u8 = (np.clip(cloth_refine_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((mask_u8 > 0).sum()) < 60:
        return base_rgb

    mask_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    ring_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )
    ring_u8 = cv2.subtract(
        ring_u8,
        cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ),
    )
    telea = cv2.inpaint(base_rgb, mask_u8, 5, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(base_rgb, mask_u8, 4, cv2.INPAINT_NS)
    refined = cv2.addWeighted(telea, 0.66, ns, 0.34, 0.0)
    mask_bool = mask_u8 > 0
    ring_bool = ring_u8 > 0
    if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
        ref_telea = cv2.inpaint(ref_rgb, mask_u8, 5, cv2.INPAINT_TELEA)
        ref_ns = cv2.inpaint(ref_rgb, mask_u8, 4, cv2.INPAINT_NS)
        ref_refined = cv2.addWeighted(ref_telea, 0.62, ref_ns, 0.38, 0.0)
        refined = cv2.addWeighted(refined, 0.42, ref_refined, 0.58, 0.0)
    if int(ring_bool.sum()) >= 80 and int(mask_bool.sum()) >= 60:
        refined_lab = cv2.cvtColor(refined, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ring_ref_bool = ring_bool.copy()
        if reference_mask is not None and reference_mask.shape == (H, W):
            ref_mask_u8 = cv2.dilate(
                (np.clip(reference_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            masked_ring = np.logical_and(ring_ref_bool, ref_mask_u8 > 0)
            if int(masked_ring.sum()) >= 40:
                ring_ref_bool = masked_ring
        ring_vals = ref_lab[ring_ref_bool]
        mask_vals = refined_lab[mask_bool]
        ring_mean = ring_vals.mean(axis=0)
        mask_mean = mask_vals.mean(axis=0)
        ring_std = ring_vals.std(axis=0)
        mask_std = np.maximum(mask_vals.std(axis=0), 1.0)
        tone_matched = mask_vals.copy()
        tone_matched[:, 0] = np.clip(
            (tone_matched[:, 0] - mask_mean[0]) * np.clip(ring_std[0] / mask_std[0], 0.82, 1.18)
            + mask_mean[0]
            + np.clip(ring_mean[0] - mask_mean[0], -16.0, 16.0) * 0.72,
            0.0,
            255.0,
        )
        tone_matched[:, 1] = np.clip(
            tone_matched[:, 1] + np.clip(ring_mean[1] - mask_mean[1], -5.0, 5.0) * 0.55,
            0.0,
            255.0,
        )
        tone_matched[:, 2] = np.clip(
            tone_matched[:, 2] + np.clip(ring_mean[2] - mask_mean[2], -5.0, 5.0) * 0.55,
            0.0,
            255.0,
        )
        refined_lab[mask_bool] = tone_matched
        refined = cv2.cvtColor(refined_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        detail_reference_rgb = ref_rgb
        lowpass = cv2.GaussianBlur(detail_reference_rgb, (0, 0), sigmaX=3.2, sigmaY=3.2)
        detail_src = np.clip(
            detail_reference_rgb.astype(np.float32) - lowpass.astype(np.float32) + 128.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        detail_telea = cv2.inpaint(detail_src, mask_u8, 3, cv2.INPAINT_TELEA)
        detail_ns = cv2.inpaint(detail_src, mask_u8, 3, cv2.INPAINT_NS)
        detail_fill = cv2.addWeighted(detail_telea, 0.70, detail_ns, 0.30, 0.0)
        detail_signed = detail_fill.astype(np.float32) - 128.0

        gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        ring_detail = float(np.mean(np.abs(lap[ring_bool]))) if int(ring_bool.sum()) > 0 else 0.0
        mask_detail = float(np.mean(np.abs(detail_signed[mask_bool]))) if int(mask_bool.sum()) > 0 else 0.0
        texture_gain = float(np.clip(ring_detail / max(mask_detail, 1.0), 0.65, 1.35))
        textured = np.clip(
            refined.astype(np.float32) + detail_signed * (0.46 * texture_gain),
            0.0,
            255.0,
        )
        refined = textured.astype(np.uint8)

    alpha = cv2.GaussianBlur(
        (mask_u8 > 0).astype(np.float32),
        (0, 0),
        sigmaX=2.4,
        sigmaY=2.4,
    )[..., np.newaxis]
    alpha = np.clip(alpha * 0.92, 0.0, 1.0)
    out = refined.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _overlay_reference_cloth_fill(
    base_rgb: np.ndarray,
    reference_rgb: Optional[np.ndarray],
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = base_rgb.shape[:2]
    if reference_rgb is None or reference_rgb.shape[:2] != (H, W) or fill_mask.shape != (H, W):
        return base_rgb

    mask_u8 = (np.clip(fill_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        mask_u8 = cv2.bitwise_and(mask_u8, cloth_u8)
    if int((mask_u8 > 0).sum()) < 40:
        return base_rgb

    base_gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    base_blur = cv2.GaussianBlur(base_gray, (0, 0), sigmaX=4.8, sigmaY=4.8)
    ref_blur = cv2.GaussianBlur(ref_gray, (0, 0), sigmaX=4.8, sigmaY=4.8)
    base_sat = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    blackhat = cv2.morphologyEx(
        base_gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21)),
    )

    residual_u8 = (
        (
            (
                (ref_blur - base_blur > 12.0)
                & (ref_gray - base_gray > 10.0)
                & (base_sat < 142.0)
            )
            | (
                (base_gray < 138.0)
                & (ref_gray > 160.0)
                & (blackhat > 8)
            )
        ).astype(np.uint8)
        * 255
    )
    residual_u8 = cv2.bitwise_and(residual_u8, mask_u8)
    if int((residual_u8 > 0).sum()) < 28:
        return base_rgb

    residual_u8 = cv2.morphologyEx(
        residual_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    residual_u8 = cv2.morphologyEx(
        residual_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
    )
    residual_u8 = cv2.dilate(
        residual_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
        iterations=1,
    )
    if int((residual_u8 > 0).sum()) < 28:
        return base_rgb

    alpha = cv2.GaussianBlur(
        (residual_u8 > 0).astype(np.float32),
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
    )[..., np.newaxis]
    delta = np.clip((ref_gray - base_gray) / 52.0, 0.0, 1.0)[..., np.newaxis]
    alpha = np.clip(alpha * (0.56 + 0.34 * delta), 0.0, 0.96)
    out = reference_rgb.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _blend_neighbor_cloth_tone(
    base_rgb: np.ndarray,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray] = None,
    reference_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = base_rgb.shape[:2]
    if fill_mask.shape != (H, W):
        return base_rgb
    ref_rgb = base_rgb
    if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
        ref_rgb = reference_rgb

    mask_u8 = (np.clip(fill_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((mask_u8 > 0).sum()) < 60:
        return base_rgb

    ring_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )
    ring_u8 = cv2.subtract(
        ring_u8,
        cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ),
    )
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        ring_u8 = cv2.bitwise_and(ring_u8, cloth_u8)
    if int((ring_u8 > 0).sum()) < 80:
        return base_rgb

    hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bright_u8 = ((gray > 152.0) & (sat < 118.0)).astype(np.uint8) * 255
    bright_ring_u8 = cv2.bitwise_and(ring_u8, bright_u8)
    sample_u8 = bright_ring_u8 if int((bright_ring_u8 > 0).sum()) >= 60 else ring_u8

    sample_pixels = ref_rgb[sample_u8 > 0]
    if sample_pixels.size == 0:
        return base_rgb
    fill_rgb = np.median(sample_pixels, axis=0).astype(np.float32)

    filled = base_rgb.astype(np.float32).copy()
    filled[mask_u8 > 0] = fill_rgb
    alpha = cv2.GaussianBlur(
        (mask_u8 > 0).astype(np.float32),
        (0, 0),
        sigmaX=4.2,
        sigmaY=4.2,
    )[..., np.newaxis]
    alpha = np.clip(alpha * 0.94, 0.0, 1.0)
    out = filled * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _resolve_background_fill_mode(
    self,
    requested_mode: str,
    *,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    hair_length: str,
    lower_tail_support_mask: Optional[np.ndarray] = None,
    center_support_mask: Optional[np.ndarray] = None,
) -> str:
    mode = str(requested_mode or "cv2").strip().lower() or "cv2"
    if mode == "sd":
        return "sd"
    if mode != "cv2":
        logger.warning(
            "[SDPipeline] unsupported bg_fill_mode '%s', falling back to cv2",
            requested_mode,
        )
        mode = "cv2"
    if hair_length not in ("short", "medium"):
        return mode

    H, W = removal_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    face_area = float(face_w * face_h)

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    removal_px = int((removal_u8 > 0).sum())
    if removal_px < max(120, int(face_area * 0.05)):
        return mode

    support_px = 0
    if lower_tail_support_mask is not None and lower_tail_support_mask.shape == (H, W):
        support_px += int((np.clip(lower_tail_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).sum())
    center_px = 0
    if center_support_mask is not None and center_support_mask.shape == (H, W):
        center_px = int((np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).sum())
        support_px += center_px

    cloth_overlap_px = 0
    cloth_ratio = 0.0
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255
        cloth_overlap_px = int((cv2.bitwise_and(removal_u8, cloth_u8) > 0).sum())
        cloth_ratio = float(cloth_overlap_px) / float(max(removal_px, 1))

    if hair_length == "short":
        promote = (
            cloth_overlap_px >= max(90, int(face_area * 0.12))
            or cloth_ratio >= 0.18
            or support_px >= max(84, int(face_area * 0.035))
            or center_px >= max(20, int(face_area * 0.008))
        )
    else:
        promote = (
            cloth_overlap_px >= max(120, int(face_area * 0.16))
            or cloth_ratio >= 0.22
            or support_px >= max(110, int(face_area * 0.045))
        )

    if promote:
        logger.info(
            "[SDPipeline] bg_fill_mode auto-promoted to sd: hair_length=%s removal_px=%s cloth_overlap_px=%s support_px=%s center_px=%s",
            hair_length,
            removal_px,
            cloth_overlap_px,
            support_px,
            center_px,
        )
        return "sd"

    return mode

def _describe_garment_color(sample_pixels: np.ndarray) -> str:
    if sample_pixels.size == 0:
        return "neutral-toned"

    pixels = np.asarray(sample_pixels, dtype=np.uint8).reshape(-1, 3)
    median_rgb = np.median(pixels, axis=0).astype(np.uint8)[np.newaxis, np.newaxis, :]
    hsv = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

    if v >= 244.0 and s <= 18.0:
        return "white"
    if v >= 222.0 and s <= 38.0:
        return "ivory"
    if v >= 196.0 and s <= 58.0:
        return "light gray"
    if s <= 24.0:
        if v <= 52.0:
            return "black"
        if v <= 96.0:
            return "charcoal gray"
        if v <= 168.0:
            return "gray"
        return "off-white"
    if 10.0 <= h < 26.0 and v >= 148.0 and s <= 120.0:
        return "beige"
    if 10.0 <= h < 26.0 and v < 148.0:
        return "brown"
    if h < 10.0 or h >= 170.0:
        return "red"
    if h < 22.0:
        return "rust"
    if h < 38.0:
        return "tan"
    if h < 52.0:
        return "mustard"
    if h < 86.0:
        return "olive green" if v < 138.0 else "green"
    if h < 114.0:
        return "teal"
    if h < 146.0:
        return "blue"
    if h < 170.0:
        return "purple"
    return "muted"

def _infer_visible_garment_prompt_hint(
    reference_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    *,
    fill_mask: Optional[np.ndarray] = None,
) -> Tuple[str, str]:
    H, W = reference_rgb.shape[:2]
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return "clean top", "scarf, ribbon, tie, armor, exposed chest"

    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255
    if int((cloth_u8 > 0).sum()) < 80:
        return "clean top", "scarf, ribbon, tie, armor, exposed chest"

    sample_u8 = cloth_u8.copy()
    if fill_mask is not None and fill_mask.shape == (H, W):
        exclusion_u8 = cv2.dilate(
            (np.clip(fill_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            iterations=1,
        )
        remaining_u8 = cv2.bitwise_and(sample_u8, cv2.bitwise_not(exclusion_u8))
        if int((remaining_u8 > 0).sum()) >= 120:
            sample_u8 = remaining_u8

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    focus_u8 = np.zeros((H, W), dtype=np.uint8)
    focus_top = max(0, int(y2 - face_h * 0.06))
    focus_bottom = min(H, int(y2 + face_h * 1.72))
    focus_half_w = max(28, int(face_w * 1.18))
    focus_left = max(0, cx - focus_half_w)
    focus_right = min(W, cx + focus_half_w)
    if focus_top < focus_bottom and focus_left < focus_right:
        focus_u8[focus_top:focus_bottom, focus_left:focus_right] = 255
        focused_u8 = cv2.bitwise_and(sample_u8, focus_u8)
        if int((focused_u8 > 0).sum()) >= 120:
            sample_u8 = focused_u8

    sample_pixels = reference_rgb[sample_u8 > 0]
    if sample_pixels.size == 0:
        return "clean top", "scarf, ribbon, tie, armor, exposed chest"

    color_name = _describe_garment_color(sample_pixels)
    sample_hsv = cv2.cvtColor(sample_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    median_sat = float(np.median(sample_hsv[:, 1])) if sample_hsv.size else 0.0
    median_val = float(np.median(sample_hsv[:, 2])) if sample_hsv.size else 0.0

    sample_area = max(int((sample_u8 > 0).sum()), 1)
    gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 72, 160)
    edge_density = float((cv2.bitwise_and(edges, sample_u8) > 0).sum()) / float(sample_area)

    neckline_u8 = np.zeros((H, W), dtype=np.uint8)
    neckline_top = max(0, int(y2 - face_h * 0.04))
    neckline_bottom = min(H, int(y2 + face_h * 0.24))
    side_half_w = max(18, int(face_w * 0.42))
    center_half_w = max(12, int(face_w * 0.16))
    if neckline_top < neckline_bottom:
        neckline_u8[neckline_top:neckline_bottom, max(0, cx - side_half_w):min(W, cx + side_half_w)] = 255
    center_u8 = np.zeros((H, W), dtype=np.uint8)
    if neckline_top < neckline_bottom:
        center_u8[neckline_top:neckline_bottom, max(0, cx - center_half_w):min(W, cx + center_half_w)] = 255
    side_u8 = cv2.bitwise_and(neckline_u8, cv2.bitwise_not(center_u8))
    center_coverage = float((cv2.bitwise_and(sample_u8, center_u8) > 0).sum()) / float(max((center_u8 > 0).sum(), 1))
    side_coverage = float((cv2.bitwise_and(sample_u8, side_u8) > 0).sum()) / float(max((side_u8 > 0).sum(), 1))
    open_collar = side_coverage >= max(0.18, center_coverage + 0.08)

    soft_texture = edge_density <= 0.075
    structured_texture = edge_density >= 0.135
    bright_plain = median_val >= 178.0 and median_sat <= 86.0
    dark_or_heavy = median_val <= 132.0

    if open_collar and bright_plain:
        garment_subject = f"{color_name} button-up shirt"
        garment_negative = "hoodie, turtleneck, scarf, ribbon, tie, heavy coat"
    elif open_collar and structured_texture:
        garment_subject = f"{color_name} collared jacket"
        garment_negative = "hoodie, sweater, scarf, ribbon, tie, robe"
    elif open_collar:
        garment_subject = f"{color_name} collared shirt"
        garment_negative = "hoodie, turtleneck, scarf, ribbon, tie"
    elif structured_texture and dark_or_heavy:
        garment_subject = f"{color_name} structured jacket"
        garment_negative = "hoodie, sweater, scarf, ribbon, tie, robe"
    elif soft_texture and dark_or_heavy and median_sat <= 104.0:
        garment_subject = f"{color_name} knit sweater"
        garment_negative = "hoodie, scarf, ribbon, tie, jacket lapels"
    elif bright_plain:
        garment_subject = f"{color_name} crew-neck top"
        garment_negative = "hoodie, scarf, ribbon, tie, deep v-neck"
    elif median_sat >= 108.0 and median_val >= 110.0:
        garment_subject = f"{color_name} blouse-like top"
        garment_negative = "hoodie, scarf, ribbon, tie, heavy outerwear"
    else:
        garment_subject = f"{color_name} clean top"
        garment_negative = "hoodie, scarf, ribbon, tie, armor"

    return garment_subject, garment_negative

def _sd_refine_removed_region(
    self,
    base_rgb: np.ndarray,          # H×W×3 RGB (cv2 inpaint 1차 결과)
    removal_mask: np.ndarray,      # H×W float32 (긴머리 제거 영역)
    face_bbox: Tuple[int, int, int, int],
    face_crop_pil: Image.Image,    # IP-Adapter conditioning face
    protect_mask: Optional[np.ndarray],  # H×W float32 (얼굴 보호)
    cloth_mask: Optional[np.ndarray],    # H×W float32 (의상 영역)
    hair_length: str,
    seed: int,
    reference_rgb: Optional[np.ndarray] = None,
    refine_mode: str = "generic",
    control_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    긴머리 제거 후 남는 어색한 영역(목/어깨/배경)을 SD로 한 번 더 정리.
    """
    H, W = base_rgb.shape[:2]
    removal_mask = self._resize_mask_to_shape(removal_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    if removal_mask.shape != (H, W):
        raise ValueError(f"removal_mask shape mismatch: {removal_mask.shape} vs {(H, W)}")

    # removal 영역 중심으로만 SD를 적용하기 위해 그대로 letterbox 변환
    fill_mask = (removal_mask > 0.5).astype(np.float32)
    img_512, mask_512, canny_512, scale, pad = self._prepare_sd_inputs(
        base_rgb,
        fill_mask,
        mask_edge_suppression=0.45,
    )
    control_512 = canny_512
    if control_rgb is not None and control_rgb.shape[:2] == (H, W):
        control_np = control_rgb
        if control_np.ndim == 2:
            control_np = cv2.cvtColor(np.clip(control_np, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif control_np.ndim == 3 and control_np.shape[2] == 3:
            control_np = np.clip(control_np, 0, 255).astype(np.uint8)
        else:
            control_np = None
        if control_np is not None:
            pad_l, pad_t = pad
            new_w = int(W * scale)
            new_h = int(H * scale)
            control_rs = cv2.resize(control_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            control_canvas = np.zeros((SD_SIZE, SD_SIZE, 3), dtype=np.uint8)
            control_canvas[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = control_rs
            if int((control_canvas > 0).sum()) >= 48:
                control_512 = Image.fromarray(control_canvas)
    prompt_reference_rgb = base_rgb
    if reference_rgb is not None and reference_rgb.shape[:2] == (H, W):
        prompt_reference_rgb = reference_rgb
    garment_subject, garment_negative = _infer_visible_garment_prompt_hint(
        prompt_reference_rgb,
        cloth_mask,
        face_bbox,
        fill_mask=fill_mask,
    )

    def _join_negative_terms(*parts: str) -> str:
        ordered: List[str] = []
        seen: set[str] = set()
        for part in parts:
            for token in str(part or "").split(","):
                cleaned = token.strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                ordered.append(cleaned)
        return ", ".join(ordered)

    def _under_jaw_negative(*, include_bare_skin_gap: bool = False) -> str:
        return _join_negative_terms(
            "hair strands, loose dangling hair, dangling side locks, side locks touching shoulders or clothing",
            "dark streaks, dark bib, black patch, black cloth shadow",
            "vertical black stripe, u-shaped dark notch under chin, deep shadow under chin",
            garment_negative,
            "smudged cloth, melted fabric, warped garment",
            "broken neckline, duplicate collar, extra folds, extra buttons",
            "exposed chest cutout",
            "bare skin gap" if include_bare_skin_gap else "",
            "deformed neck, artifacts, blurry, cartoon, painting",
            "earring, earrings, necklace, loose side tendrils touching clothing",
        )

    if refine_mode == "under_jaw_cloth":
        fill_prompt = (
            f"professional studio portrait photo, regenerate a similar {garment_subject} under the jaw, "
            "keep the same garment family, neckline and collar behavior, realistic fabric texture continuity, "
            "natural folds and seams, continuous clothing coverage under the jaw, clean neck and shoulders, "
            "no hair strands in masked region, no dark patch under the neck, no empty chest cutout, "
            "photorealistic clothing details"
        )
        fill_guidance = 6.9 if hair_length == "short" else 6.8
        fill_negative = _under_jaw_negative(include_bare_skin_gap=True)
    elif refine_mode == "short_cloth_crop" and hair_length == "short":
        fill_prompt = (
            f"professional studio portrait photo, regenerate only the short-hair under-jaw central {garment_subject} region, "
            "preserve the same neckline, central placket or fold direction, continuous cloth coverage directly below the chin, "
            "clean neck and shoulders, no hollow chest cutout, no dark patch, no hair strands in masked region, "
            "photorealistic clothing details"
        )
        fill_guidance = 7.4
        fill_negative = _under_jaw_negative()
    elif refine_mode == "short_cloth_insert" and hair_length == "short":
        fill_prompt = (
            f"professional studio portrait photo, freshly regenerate only the short-hair under-jaw {garment_subject} insert, "
            "keep the same garment family, neckline, collar opening, central placket or fold direction, "
            "clean continuous cloth directly below the chin, natural fabric folds, realistic seam continuity, "
            "no hair strands in masked region, no dark patch, no hollow chest cutout, photorealistic clothing details"
        )
        fill_guidance = 7.8
        fill_negative = _under_jaw_negative()
    elif refine_mode == "cloth_only_second_pass":
        fill_prompt = (
            f"professional studio portrait photo, regenerate only the central visible {garment_subject} below the chin, "
            "preserve the same neckline, central placket or fold direction, continuous cloth coverage, "
            "realistic fabric texture continuity, clean neck and shoulders, no hair strands in masked region, "
            "no dark patch, no hollow chest cutout, photorealistic clothing details"
        )
        fill_guidance = 7.0 if hair_length == "short" else 6.9
        fill_negative = _under_jaw_negative()
    elif refine_mode == "cloth":
        fill_prompt = (
            f"professional portrait photo, preserve a similar {garment_subject} shape, "
            "realistic clothing fabric texture continuity, coherent folds and seams, color continuity, "
            "clean neck and shoulders, no hair strands in masked region, photorealistic details"
        )
        fill_guidance = 6.8 if hair_length == "short" else 7.0
        fill_negative = (
            "hair strands, loose dangling hair, long hair, blur, blurry cloth, smudged cloth, "
            f"melted fabric, duplicate collar, broken neckline, extra folds, extra buttons, {garment_negative}, "
            "warped garment, deformed neck, artifacts, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )
    elif refine_mode == "short_tail" and hair_length == "short":
        fill_prompt = (
            "professional portrait photo, neat compact short jaw-length bob haircut, "
            "clean side silhouette above the shoulders, visible neck and shoulders, "
            f"preserve the same visible {garment_subject}, realistic clothing fabric texture continuity, "
            "clean neckline, no hair below jawline, no shoulder-length side hair, "
            "no dangling strands in masked region, photorealistic details"
        )
        fill_guidance = 8.2
        fill_negative = (
            "long hair, shoulder-length hair, medium hair, lob haircut, hair below jawline, "
            "hair touching shoulders, dangling side tails, loose strands, extra hair mass, "
            f"warped garment, melted fabric, {garment_negative}, deformed neck, artifacts, blurry, "
            "smudged texture, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )
    elif hair_length == "short":
        fill_prompt = (
            "professional portrait photo, clean natural neck and shoulders, "
            f"preserve the same visible {garment_subject}, realistic clothing fabric texture continuity, "
            "coherent neckline, collar and sleeve folds, coherent background, "
            "short-hair silhouette maintained, no long hair below jawline, "
            "no loose dangling strands in masked region, photorealistic details"
        )
        fill_guidance = 7.1
        fill_negative = (
            "long hair, hair below chin, hair below shoulders, loose hair strands, "
            "wavy hair, straight long hair, wig, ponytail, braid, bangs, side locks, "
            f"{garment_negative}, deformed neck, artifacts, blurry, smudged texture, melted details, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )
    else:
        fill_prompt = (
            "professional portrait photo, clean neck and shoulders, "
            "natural skin and clothing texture continuity, coherent background, "
            "no loose long hair strands in masked region, photorealistic details"
        )
        fill_guidance = 7.6
        fill_negative = (
            "long hair, hair below chin, hair below shoulders, loose hair strands, "
            "wavy hair, straight long hair, wig, ponytail, braid, bangs, side locks, "
            "deformed neck, artifacts, blurry, smudged texture, melted details, cartoon, painting, "
            f"{_COMMON_STYLE_BLOCK_NEGATIVE}"
        )

    # 배경 복원은 identity 영향이 과하면 긴머리가 다시 생길 수 있어 scale을 낮춘다.
    self._sd_pipe.set_ip_adapter_scale(0.0)
    generator = torch.Generator(device=self.device).manual_seed(int(seed))
    if refine_mode == "under_jaw_cloth":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.14), 0.10, 0.22))
        fill_steps = max(18, self.config.num_inference_steps - 10)
        fill_strength = 0.82
    elif refine_mode == "short_cloth_crop" and hair_length == "short":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.20), 0.16, 0.30))
        fill_steps = max(20, self.config.num_inference_steps - 8)
        fill_strength = 0.84
    elif refine_mode == "short_cloth_insert" and hair_length == "short":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.22), 0.18, 0.34))
        fill_steps = max(22, self.config.num_inference_steps - 6)
        fill_strength = 0.88
    elif refine_mode == "cloth_only_second_pass":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.12), 0.10, 0.18))
        fill_steps = max(16, self.config.num_inference_steps - 12)
        fill_strength = 0.78
    elif refine_mode == "cloth":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.16), 0.10, 0.24))
        fill_steps = max(20, self.config.num_inference_steps - 8)
        fill_strength = 0.84
    elif refine_mode == "short_tail" and hair_length == "short":
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.14), 0.10, 0.20))
        fill_steps = max(22, self.config.num_inference_steps - 6)
        fill_strength = 0.90
    else:
        fill_control = float(np.clip(max(self.config.controlnet_conditioning_scale, 0.18), 0.12, 0.30))
        fill_steps = max(24, self.config.num_inference_steps - 4)
        fill_strength = 0.88

    with torch.inference_mode():
        out = self._sd_pipe(
            prompt=fill_prompt,
            negative_prompt=fill_negative,
            image=img_512,
            mask_image=mask_512,
            control_image=control_512,
            ip_adapter_image=[face_crop_pil],
            height=SD_SIZE,
            width=SD_SIZE,
            num_inference_steps=fill_steps,
            guidance_scale=fill_guidance,
            controlnet_conditioning_scale=fill_control,
            num_images_per_prompt=1,
            generator=generator,
            strength=fill_strength,
        )

    gen_np = np.array(out.images[0])  # 512×512 RGB

    # letterbox 역변환
    pad_l, pad_t = pad
    new_w = int(W * scale)
    new_h = int(H * scale)
    gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
    gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 중앙 편향을 완화해 side 잔존 영역도 자연스럽게 복원한다.
    x1, y1, x2, y2 = face_bbox
    cx = 0.5 * (x1 + x2)
    face_w = max(float(x2 - x1), 1.0)
    sigma_x = max(face_w * 1.45, 44.0)
    xs = np.arange(W, dtype=np.float32)
    center_weight = np.exp(-0.5 * ((xs - cx) / sigma_x) ** 2)
    alpha = alpha * (0.65 + 0.35 * center_weight[np.newaxis, :])

    # 의상 영역은 과도한 hallucination을 줄이기 위해 SD 블렌딩 가중치를 낮춘다.
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
        alpha = alpha * (1.0 - 0.18 * cloth_w)

    # 얼굴은 기존 픽셀 고정
    if refine_mode in {"cloth", "under_jaw_cloth"}:
        alpha = cv2.GaussianBlur(fill_mask, (0, 0), sigmaX=6.0, sigmaY=6.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_w = np.clip(cloth_mask.astype(np.float32), 0.0, 1.0)
            cloth_gain = 0.22 if refine_mode == "under_jaw_cloth" else 0.18
            alpha = np.clip(alpha * (0.94 + cloth_gain * cloth_w), 0.0, 1.0)

    if protect_mask is not None:
        protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        protect = cv2.dilate(protect_mask.astype(np.float32), protect_k)
        alpha = alpha * (1.0 - np.clip(protect, 0.0, 1.0))

    alpha = alpha[..., np.newaxis]
    refined = (
        gen_orig.astype(np.float32) * alpha
        + base_rgb.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(refined, 0, 255).astype(np.uint8)

def _filter_short_center_cleanup_mask(
    self,
    mask_u8: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    *,
    anchor_u8: Optional[np.ndarray] = None,
    top_scale: float = 0.02,
    bottom_scale: float = 0.82,
    half_w_scale: float = 0.24,
    shrink_half_w_scale: float = 0.18,
    max_area_scale: float = 0.10,
    max_width_scale: float = 0.34,
    min_height_scale: float = 0.10,
    center_allow_scale: float = 0.18,
    max_total_scale: float = 0.05,
) -> np.ndarray:
    H, W = mask_u8.shape[:2]
    filtered_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    if int((filtered_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    gate_u8 = np.zeros((H, W), dtype=np.uint8)
    gate_half_w = max(14, int(face_w * half_w_scale))
    gate_left = max(0, cx - gate_half_w)
    gate_right = min(W, cx + gate_half_w)
    gate_top = max(0, int(cutoff_y + face_h * top_scale))
    gate_bottom = min(H, int(cutoff_y + face_h * bottom_scale))
    if gate_top >= gate_bottom or gate_left >= gate_right:
        return np.zeros((H, W), dtype=np.uint8)
    gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255
    filtered_u8 = cv2.bitwise_and(filtered_u8, gate_u8)
    if int((filtered_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    anchor_local_u8 = np.zeros((H, W), dtype=np.uint8)
    if anchor_u8 is not None and anchor_u8.shape == (H, W):
        anchor_local_u8 = cv2.bitwise_and((anchor_u8 > 0).astype(np.uint8) * 255, gate_u8)

    max_component_area = max(96, int(face_w * face_h * max_area_scale))
    max_component_width = max(24, int(face_w * max_width_scale))
    min_component_height = max(12, int(face_h * min_height_scale))
    center_allow = max(14, int(face_w * center_allow_scale))
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_u8, 8)
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        comp_cx = x + (w * 0.5)
        if area < 14 or area > max_component_area:
            continue
        if w > max_component_width or h < min_component_height:
            continue
        if (y + h) > gate_bottom:
            continue
        if abs(comp_cx - cx) > center_allow:
            continue
        if w > max(20, int(face_w * 0.28)) and h < max(20, int(face_h * 0.22)):
            continue
        if int((anchor_local_u8 > 0).sum()) > 0:
            anchor_overlap = int((cv2.bitwise_and(comp_u8, anchor_local_u8) > 0).sum())
            if anchor_overlap < max(4, int(area * 0.04)):
                continue
        keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

    if int((keep_u8 > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    max_total_px = max(72, int(face_w * face_h * max_total_scale))
    current_px = int((keep_u8 > 0).sum())
    if current_px > max_total_px:
        shrink_u8 = np.zeros((H, W), dtype=np.uint8)
        shrink_half_w = max(12, int(face_w * shrink_half_w_scale))
        shrink_left = max(0, cx - shrink_half_w)
        shrink_right = min(W, cx + shrink_half_w)
        shrink_top = max(gate_top, int(cutoff_y + face_h * max(top_scale, 0.04)))
        shrink_bottom = min(H, int(cutoff_y + face_h * max(bottom_scale - 0.06, 0.10)))
        if shrink_top < shrink_bottom and shrink_left < shrink_right:
            shrink_u8[shrink_top:shrink_bottom, shrink_left:shrink_right] = 255
            keep_u8 = cv2.bitwise_and(keep_u8, shrink_u8)

    return keep_u8

def _build_post_cloth_refine_mask(
    self,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    subject_gender_mode: str = "",
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
    artifact_cleanup_mask: Optional[np.ndarray] = None,
    exclusion_mask: Optional[np.ndarray] = None,
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

    exclusion_u8 = np.zeros((H, W), dtype=np.uint8)
    if exclusion_mask is not None and exclusion_mask.shape == (H, W):
        exclusion_u8 = (
            (np.clip(exclusion_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255
        )
        exclusion_u8 = cv2.dilate(
            exclusion_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (19, 31) if hair_length == "short" else (15, 25),
            ),
            iterations=1,
        )
        exclusion_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        exclusion_half_w = max(24, int(face_w * (0.34 if hair_length == "short" else 0.28)))
        exclusion_left = max(0, cx - exclusion_half_w)
        exclusion_right = min(W, cx + exclusion_half_w)
        exclusion_top = max(top, int(cutoff_y + face_h * 0.04))
        exclusion_bottom = min(H, int(cutoff_y + face_h * (0.98 if hair_length == "short" else 1.20)))
        if exclusion_top < exclusion_bottom and exclusion_left < exclusion_right:
            exclusion_gate_u8[exclusion_top:exclusion_bottom, exclusion_left:exclusion_right] = 255
            exclusion_u8 = cv2.bitwise_and(exclusion_u8, exclusion_gate_u8)

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

    gender_mode = str(subject_gender_mode or "").strip().lower()
    lateral_only_short_restore = hair_length == "short" and gender_mode != "male"
    if hair_length == "short":
        short_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        gate_top = max(top, int(cutoff_y + face_h * 0.02))
        gate_bottom = min(H, int(cutoff_y + face_h * (1.20 if lateral_only_short_restore else 1.04)))
        if lateral_only_short_restore:
            left_lane_left = max(0, int(x1 - face_w * 0.88))
            left_lane_right = min(W, int(x1 + face_w * 0.14))
            right_lane_left = max(0, int(x2 - face_w * 0.14))
            right_lane_right = min(W, int(x2 + face_w * 0.88))
            if gate_top < gate_bottom and left_lane_left < left_lane_right:
                short_gate_u8[gate_top:gate_bottom, left_lane_left:left_lane_right] = 255
            if gate_top < gate_bottom and right_lane_left < right_lane_right:
                short_gate_u8[gate_top:gate_bottom, right_lane_left:right_lane_right] = 255
            shoulder_band_top = max(gate_top, int(cutoff_y + face_h * 0.24))
            shoulder_band_bottom = min(gate_bottom, int(cutoff_y + face_h * 0.60))
            shoulder_left = max(0, int(x1 - face_w * 1.02))
            shoulder_right = min(W, int(x2 + face_w * 1.02))
            if shoulder_band_top < shoulder_band_bottom and shoulder_left < shoulder_right:
                short_gate_u8[shoulder_band_top:shoulder_band_bottom, shoulder_left:shoulder_right] = 255
        else:
            gate_half_w = max(32, int(face_w * 0.72))
            gate_left = max(0, cx - gate_half_w)
            gate_right = min(W, cx + gate_half_w)
            if gate_top < gate_bottom and gate_left < gate_right:
                short_gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255
        if int((short_gate_u8 > 0).sum()) > 0:
            mask_u8 = cv2.bitwise_and(mask_u8, short_gate_u8)
            artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, short_gate_u8)

        max_bottom = min(H, int(cutoff_y + face_h * (1.20 if lateral_only_short_restore else 1.02)))
        if max_bottom < H:
            mask_u8[max_bottom:, :] = 0
            artifact_bonus_u8[max_bottom:, :] = 0

    if int((exclusion_u8 > 0).sum()) > 0:
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(exclusion_u8))
        artifact_bonus_u8 = cv2.bitwise_and(artifact_bonus_u8, cv2.bitwise_not(exclusion_u8))

    if hair_length == "short":
        max_component_area = max(320, int(face_w * face_h * 0.26))
        max_component_width = max(84, int(face_w * 0.88))
        min_component_height = max(12, int(face_h * 0.10))
        center_allow = max(44, int(face_w * 0.72))
        artifact_bonus_present = int((artifact_bonus_u8 > 0).sum()) >= 24
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
            if artifact_bonus_present and artifact_overlap < max(4, int(area * 0.04)):
                if area < max(48, int(face_w * face_h * 0.018)):
                    continue
                if h < max(18, int(face_h * 0.16)):
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

def _build_under_jaw_cloth_refine_mask(
    self,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
    center_support_mask: Optional[np.ndarray] = None,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    removal_mask = self._resize_mask_to_shape(removal_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    final_hair_mask = self._resize_mask_to_shape(final_hair_mask, (H, W))
    candidate_mask = self._resize_mask_to_shape(candidate_mask, (H, W))
    center_support_mask = self._resize_mask_to_shape(center_support_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))
    short_center_gate_u8 = np.zeros((H, W), dtype=np.uint8)
    if hair_length == "short":
        gate_top = max(0, int(cutoff_y + face_h * 0.02))
        gate_bottom = min(H, int(cutoff_y + face_h * 0.88))
        gate_half_w = max(18, int(face_w * 0.28))
        gate_left = max(0, cx - gate_half_w)
        gate_right = min(W, cx + gate_half_w)
        if gate_top < gate_bottom and gate_left < gate_right:
            short_center_gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    if int((cloth_u8 > 0).sum()) < 100:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.06))
    bottom = min(
        H,
        int(cutoff_y + face_h * (0.98 if hair_length == "short" else 1.22 if hair_length == "medium" else 1.40)),
    )
    half_w = max(24, int(face_w * (0.38 if hair_length == "short" else 0.52 if hair_length == "medium" else 0.60)))
    left = max(0, cx - half_w)
    right = min(W, cx + half_w)
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    mask_u8 = cv2.bitwise_and(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
        cloth_u8,
    )
    mask_u8 = cv2.bitwise_and(mask_u8, corridor_u8)

    if candidate_mask is not None and candidate_mask.shape == (H, W):
        candidate_u8 = cv2.dilate(
            (np.clip(candidate_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19) if hair_length == "short" else (15, 25)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, cloth_u8)
        candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
        if hair_length == "short" and int((short_center_gate_u8 > 0).sum()) >= 40:
            narrowed_candidate_u8 = cv2.bitwise_and(candidate_u8, short_center_gate_u8)
            if int((narrowed_candidate_u8 > 0).sum()) >= 36:
                candidate_u8 = narrowed_candidate_u8
        mask_u8 = cv2.bitwise_or(mask_u8, candidate_u8)

    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_sat = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    cloth_bool = cloth_u8 > 0
    cloth_gray_ref = float(np.median(source_gray[cloth_bool])) if int(cloth_bool.sum()) >= 80 else 0.0
    cloth_sat_ref = float(np.median(source_sat[cloth_bool])) if int(cloth_bool.sum()) >= 80 else 0.0
    residual_bool = (
        ((source_gray - current_gray) > (10.0 if hair_length == "short" else 8.0))
        | (
            (current_gray < cloth_gray_ref - 12.0)
            & (current_sat < min(138.0, cloth_sat_ref + 20.0))
        )
    )
    residual_u8 = (residual_bool.astype(np.uint8) * 255)
    residual_u8 = cv2.bitwise_and(residual_u8, cloth_u8)
    residual_u8 = cv2.bitwise_and(residual_u8, corridor_u8)
    residual_u8 = cv2.morphologyEx(
        residual_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
    )
    residual_u8 = cv2.dilate(
        residual_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    if hair_length == "short" and int((short_center_gate_u8 > 0).sum()) >= 40:
        narrowed_residual_u8 = cv2.bitwise_and(residual_u8, short_center_gate_u8)
        if int((narrowed_residual_u8 > 0).sum()) >= 28:
            residual_u8 = narrowed_residual_u8
    mask_u8 = cv2.bitwise_or(mask_u8, residual_u8)

    if center_support_mask is not None and center_support_mask.shape == (H, W):
        support_u8 = cv2.dilate(
            (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 21) if hair_length == "short" else (13, 23)),
            iterations=1,
        )
        support_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        support_half_w = max(18, int(face_w * (0.18 if hair_length == "short" else 0.22)))
        support_left = max(0, cx - support_half_w)
        support_right = min(W, cx + support_half_w)
        support_top = max(top, int(cutoff_y + face_h * 0.08))
        support_bottom = min(bottom, int(cutoff_y + face_h * (0.92 if hair_length == "short" else 1.12)))
        if support_top < support_bottom and support_left < support_right:
            support_gate_u8[support_top:support_bottom, support_left:support_right] = 255
            support_u8 = cv2.bitwise_and(support_u8, support_gate_u8)
            support_u8 = cv2.bitwise_and(support_u8, cloth_u8)
            mask_u8 = cv2.bitwise_or(mask_u8, support_u8)

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19) if hair_length == "short" else (9, 17)),
    )
    mask_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13) if hair_length == "short" else (7, 11)),
        iterations=1,
    )
    mask_u8 = cv2.bitwise_and(mask_u8, corridor_u8)

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.14).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(protect_u8))
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 21)),
            iterations=1,
        )
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(final_hair_u8))
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        preserve_u8 = cv2.dilate(
            (np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11) if hair_length == "short" else (9, 9)),
            iterations=1,
        )
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(preserve_u8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    max_component_area = max(360, int(face_w * face_h * (0.48 if hair_length == "short" else 0.96)))
    max_component_width = max(72, int(face_w * (0.68 if hair_length == "short" else 1.10)))
    min_component_height = max(20, int(face_h * 0.18))
    center_allow = max(24, int(face_w * (0.28 if hair_length == "short" else 0.52)))
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_cx = float(centroids[idx][0])
        if area < 40 or area > max_component_area:
            continue
        if h < min_component_height or w > max_component_width:
            continue
        if abs(comp_cx - cx) > center_allow:
            continue
        filtered_u8[labels == idx] = 255
    if int((filtered_u8 > 0).sum()) >= 60:
        mask_u8 = filtered_u8

    if hair_length == "short" and int((short_center_gate_u8 > 0).sum()) >= 40:
        narrowed_mask_u8 = cv2.bitwise_and(mask_u8, short_center_gate_u8)
        if int((narrowed_mask_u8 > 0).sum()) >= 48:
            mask_u8 = narrowed_mask_u8
        mask_u8 = _tighten_short_under_jaw_mask_to_center(
            self,
            mask_u8=mask_u8,
            cloth_mask=cloth_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            neck_preserve_mask=neck_preserve_mask,
            min_pixels=42,
        )

    max_total_px = max(160, int(face_w * face_h * (0.26 if hair_length == "short" else 0.62)))
    current_px = int((mask_u8 > 0).sum())
    if current_px > max_total_px:
        shrink_u8 = np.zeros((H, W), dtype=np.uint8)
        shrink_half_w = max(16, int(face_w * (0.24 if hair_length == "short" else 0.46)))
        shrink_left = max(0, cx - shrink_half_w)
        shrink_right = min(W, cx + shrink_half_w)
        shrink_top = max(top, int(cutoff_y + face_h * 0.06))
        shrink_bottom = min(bottom, int(cutoff_y + face_h * (0.88 if hair_length == "short" else 1.18)))
        if shrink_top < shrink_bottom and shrink_left < shrink_right:
            shrink_u8[shrink_top:shrink_bottom, shrink_left:shrink_right] = 255
            mask_u8 = cv2.bitwise_and(mask_u8, shrink_u8)

    if int((mask_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    out = cv2.GaussianBlur(
        mask_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=3.6,
        sigmaY=5.4,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _build_short_cloth_only_second_pass_mask(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length != "short":
        return np.zeros(current_rgb.shape[:2], dtype=np.float32)

    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    final_hair_mask = self._resize_mask_to_shape(final_hair_mask, (H, W))
    seed_mask = self._resize_mask_to_shape(seed_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    gate_u8 = np.zeros((H, W), dtype=np.uint8)
    gate_top = max(0, int(cutoff_y + face_h * 0.04))
    gate_bottom = min(H, int(cutoff_y + face_h * 0.82))
    gate_half_w = max(16, int(face_w * 0.22))
    gate_left = max(0, cx - gate_half_w)
    gate_right = min(W, cx + gate_half_w)
    if gate_top >= gate_bottom or gate_left >= gate_right:
        return np.zeros((H, W), dtype=np.float32)
    gate_u8[gate_top:gate_bottom, gate_left:gate_right] = 255

    work_u8 = cv2.bitwise_and(cloth_u8, gate_u8)
    if seed_mask is not None and seed_mask.shape == (H, W):
        seed_u8 = cv2.dilate(
            (np.clip(seed_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
            iterations=1,
        )
        seed_u8 = cv2.bitwise_and(seed_u8, gate_u8)
        if int((seed_u8 > 0).sum()) >= 24:
            work_u8 = cv2.bitwise_and(work_u8, seed_u8)
    if int((work_u8 > 0).sum()) < 36:
        return np.zeros((H, W), dtype=np.float32)

    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_sat = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    diff_rgb = np.abs(current_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=2)
    blackhat = cv2.morphologyEx(
        current_gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
    ).astype(np.float32)

    candidate_u8 = (
        (
            (
                ((source_gray - current_gray) > 8.0)
                & (current_sat < np.minimum(source_sat + 28.0, 142.0))
            )
            | (
                ((source_gray - current_gray) > 4.0)
                & (blackhat > 6.0)
            )
            | (
                (diff_rgb > 18.0)
                & (current_gray < source_gray + 4.0)
            )
        ).astype(np.uint8)
        * 255
    )
    candidate_u8 = cv2.bitwise_and(candidate_u8, work_u8)
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
    )
    candidate_u8 = cv2.dilate(
        candidate_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
        iterations=1,
    )

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(protect_u8))
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.14).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 17)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(final_hair_u8))
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        preserve_u8 = cv2.dilate(
            (np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_and(candidate_u8, cv2.bitwise_not(preserve_u8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    max_component_area = max(120, int(face_w * face_h * 0.12))
    max_component_width = max(52, int(face_w * 0.38))
    min_component_height = max(18, int(face_h * 0.12))
    center_allow = max(18, int(face_w * 0.18))
    for idx in range(1, num_labels):
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_cx = float(centroids[idx][0])
        if area < 18 or area > max_component_area:
            continue
        if h < min_component_height or w > max_component_width:
            continue
        if abs(comp_cx - cx) > center_allow:
            continue
        filtered_u8[labels == idx] = 255
    if int((filtered_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)
    filtered_u8 = _tighten_short_under_jaw_mask_to_center(
        self,
        mask_u8=filtered_u8,
        cloth_mask=cloth_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        neck_preserve_mask=neck_preserve_mask,
        min_pixels=20,
    )

    out = cv2.GaussianBlur(
        filtered_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.8,
        sigmaY=4.4,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _build_short_cloth_generation_silhouette_mask(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    protect_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
    seed_mask: Optional[np.ndarray] = None,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    final_hair_mask = self._resize_mask_to_shape(final_hair_mask, (H, W))
    seed_mask = self._resize_mask_to_shape(seed_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    if int((cloth_u8 > 0).sum()) < 36:
        return np.zeros((H, W), dtype=np.float32)

    gate_u8 = np.zeros((H, W), dtype=np.uint8)
    upper_top = max(0, int(cutoff_y + face_h * 0.03))
    upper_bottom = min(H, int(cutoff_y + face_h * 0.38))
    upper_half_w = max(12, int(face_w * 0.15))
    lower_top = max(upper_bottom - max(6, int(face_h * 0.06)), int(cutoff_y + face_h * 0.16))
    lower_bottom = min(H, int(cutoff_y + face_h * 0.96))
    lower_half_w = max(18, int(face_w * 0.24))
    if upper_top < upper_bottom:
        gate_u8[upper_top:upper_bottom, max(0, cx - upper_half_w):min(W, cx + upper_half_w)] = 255
    if lower_top < lower_bottom:
        gate_u8[lower_top:lower_bottom, max(0, cx - lower_half_w):min(W, cx + lower_half_w)] = 255
    gate_u8 = cv2.bitwise_and(gate_u8, cloth_u8)
    if int((gate_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    spine_u8 = np.zeros((H, W), dtype=np.uint8)
    spine_half_w = max(10, int(face_w * 0.11))
    spine_top = max(0, int(cutoff_y + face_h * 0.04))
    spine_bottom = min(H, int(cutoff_y + face_h * 0.94))
    if spine_top < spine_bottom:
        spine_u8[spine_top:spine_bottom, max(0, cx - spine_half_w):min(W, cx + spine_half_w)] = 255
    work_u8 = cv2.bitwise_and(spine_u8, gate_u8)

    if seed_mask is not None and seed_mask.shape == (H, W):
        seed_u8 = cv2.dilate(
            (np.clip(seed_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)),
            iterations=1,
        )
        seed_u8 = cv2.bitwise_and(seed_u8, gate_u8)
        if int((seed_u8 > 0).sum()) >= 24:
            work_u8 = cv2.bitwise_or(work_u8, seed_u8)

    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_sat = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    diff_rgb = np.abs(current_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=2)
    residual_u8 = (
        (
            (
                ((source_gray - current_gray) > 6.0)
                & (current_sat < np.minimum(source_sat + 26.0, 142.0))
            )
            | (
                (diff_rgb > 16.0)
                & (current_gray < source_gray + 6.0)
            )
        ).astype(np.uint8)
        * 255
    )
    residual_u8 = cv2.bitwise_and(residual_u8, gate_u8)
    residual_u8 = cv2.morphologyEx(
        residual_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
    )
    residual_u8 = cv2.dilate(
        residual_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
        iterations=1,
    )
    if int((residual_u8 > 0).sum()) >= 24:
        work_u8 = cv2.bitwise_or(work_u8, residual_u8)

    work_u8 = cv2.morphologyEx(
        work_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
    )
    work_u8 = cv2.dilate(
        work_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
        iterations=1,
    )
    work_u8 = cv2.bitwise_and(work_u8, gate_u8)

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        work_u8 = cv2.bitwise_and(work_u8, cv2.bitwise_not(protect_u8))
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.14).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 17)),
            iterations=1,
        )
        work_u8 = cv2.bitwise_and(work_u8, cv2.bitwise_not(final_hair_u8))
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        preserve_u8 = cv2.dilate(
            (np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        work_u8 = cv2.bitwise_and(work_u8, cv2.bitwise_not(preserve_u8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(work_u8, 8)
    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    max_component_area = max(96, int(face_w * face_h * 0.14))
    max_component_width = max(48, int(face_w * 0.40))
    min_component_height = max(16, int(face_h * 0.14))
    center_allow = max(16, int(face_w * 0.18))
    for idx in range(1, num_labels):
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_cx = float(centroids[idx][0])
        if area < 18 or area > max_component_area:
            continue
        if h < min_component_height or w > max_component_width:
            continue
        if abs(comp_cx - cx) > center_allow:
            continue
        filtered_u8[labels == idx] = 255
    if int((filtered_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)
    filtered_u8 = _tighten_short_under_jaw_mask_to_center(
        self,
        mask_u8=filtered_u8,
        cloth_mask=cloth_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        neck_preserve_mask=neck_preserve_mask,
        min_pixels=20,
    )

    out = cv2.GaussianBlur(
        filtered_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.4,
        sigmaY=4.0,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _build_short_cloth_control_map(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    seed_mask: Optional[np.ndarray] = None,
    neck_preserve_mask: Optional[np.ndarray] = None,
    protect_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W):
        return None

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    seed_mask = self._resize_mask_to_shape(seed_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    if int((cloth_u8 > 0).sum()) < 48:
        return None

    gate_u8 = np.zeros((H, W), dtype=np.uint8)
    upper_top = max(0, int(cutoff_y + face_h * 0.02))
    upper_bottom = min(H, int(cutoff_y + face_h * 0.40))
    upper_half_w = max(14, int(face_w * 0.17))
    lower_top = max(upper_bottom - max(6, int(face_h * 0.08)), int(cutoff_y + face_h * 0.18))
    lower_bottom = min(H, int(cutoff_y + face_h * 1.00))
    lower_half_w = max(20, int(face_w * 0.26))
    if upper_top < upper_bottom:
        gate_u8[upper_top:upper_bottom, max(0, cx - upper_half_w):min(W, cx + upper_half_w)] = 255
    if lower_top < lower_bottom:
        gate_u8[lower_top:lower_bottom, max(0, cx - lower_half_w):min(W, cx + lower_half_w)] = 255
    gate_u8 = cv2.bitwise_and(gate_u8, cloth_u8)
    if int((gate_u8 > 0).sum()) < 32:
        return None

    focus_u8 = gate_u8.copy()
    if seed_mask is not None and seed_mask.shape == (H, W):
        seed_u8 = cv2.dilate(
            (np.clip(seed_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
            iterations=1,
        )
        seed_u8 = cv2.bitwise_and(seed_u8, gate_u8)
        if int((seed_u8 > 0).sum()) >= 24:
            focus_u8 = cv2.bitwise_and(
                gate_u8,
                cv2.dilate(seed_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)), iterations=1),
            )
            if int((focus_u8 > 0).sum()) < 24:
                focus_u8 = seed_u8

    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    source_edge_u8 = cv2.Canny(source_gray, 68, 156)
    current_edge_u8 = cv2.Canny(current_gray, 72, 164)
    edge_gate_u8 = cv2.dilate(
        focus_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    source_edge_u8 = cv2.bitwise_and(source_edge_u8, edge_gate_u8)
    current_edge_u8 = cv2.bitwise_and(current_edge_u8, edge_gate_u8)

    cloth_outline_u8 = cv2.morphologyEx(
        cloth_u8,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    cloth_outline_u8 = cv2.bitwise_and(cloth_outline_u8, gate_u8)

    preserve_outline_u8 = np.zeros((H, W), dtype=np.uint8)
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        preserve_u8 = cv2.dilate(
            (np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        preserve_outline_u8 = cv2.morphologyEx(
            preserve_u8,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        preserve_outline_u8 = cv2.bitwise_and(
            preserve_outline_u8,
            cv2.dilate(gate_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1),
        )

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        protect_u8 = cv2.bitwise_and(protect_u8, cv2.bitwise_not(preserve_outline_u8))
        source_edge_u8 = cv2.bitwise_and(source_edge_u8, cv2.bitwise_not(protect_u8))
        current_edge_u8 = cv2.bitwise_and(current_edge_u8, cv2.bitwise_not(protect_u8))
        cloth_outline_u8 = cv2.bitwise_and(cloth_outline_u8, cv2.bitwise_not(protect_u8))

    control_u8 = cv2.bitwise_or(source_edge_u8, cloth_outline_u8)
    if int((control_u8 > 0).sum()) < 40:
        control_u8 = cv2.bitwise_or(control_u8, current_edge_u8)
    control_u8 = cv2.bitwise_or(control_u8, preserve_outline_u8)

    control_u8 = cv2.morphologyEx(
        control_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    control_u8 = cv2.dilate(
        control_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    control_u8 = cv2.bitwise_and(
        control_u8,
        cv2.dilate(gate_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1),
    )
    if int((control_u8 > 0).sum()) < 24:
        return None

    return cv2.cvtColor(control_u8, cv2.COLOR_GRAY2RGB)

def _build_source_conditioned_cloth_base(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    hair_length: str = "",
    preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or fill_mask.shape != (H, W):
        return current_rgb

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    preserve_mask = self._resize_mask_to_shape(preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return current_rgb

    cloth_fill_mask = np.clip(
        fill_mask.astype(np.float32)
        * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    if preserve_mask is not None and preserve_mask.shape == (H, W):
        cloth_fill_mask = np.clip(
            cloth_fill_mask - np.clip(preserve_mask.astype(np.float32), 0.0, 1.0) * 0.98,
            0.0,
            1.0,
        )
    cloth_fill_u8 = (cloth_fill_mask > 0.08).astype(np.uint8) * 255
    if int((cloth_fill_u8 > 0).sum()) < 24:
        return current_rgb

    reference_fill_u8 = cv2.dilate(
        cloth_fill_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 17) if str(hair_length or "").strip().lower() == "short" else (15, 25),
        ),
        iterations=1,
    )
    wide_cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    reference_fill_u8 = cv2.bitwise_and(reference_fill_u8, wide_cloth_u8)
    if int((reference_fill_u8 > 0).sum()) < 32:
        return current_rgb

    reference_fill_mask = reference_fill_u8.astype(np.float32) / 255.0
    reference_fill_rgb = self._restore_cloth_overlap_from_source(
        source_rgb=source_rgb,
        current_rgb=source_rgb,
        restore_mask=reference_fill_mask,
        final_hair_mask=None,
        tone_reference_rgb=source_rgb,
        tone_reference_mask=cloth_mask,
    )

    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    visible_cloth_u8 = cv2.bitwise_and(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.bitwise_not(
            cv2.dilate(
                cloth_fill_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
                iterations=1,
            )
        ),
    )
    if int((visible_cloth_u8 > 0).sum()) >= 80:
        plain_gray = float(np.median(source_gray[visible_cloth_u8 > 0]))
        plain_sat = float(np.median(source_sat[visible_cloth_u8 > 0]))
        if plain_gray >= 168.0 and plain_sat <= 84.0:
            plain_fill_rgb = source_rgb.copy()
            plain_fill_color = np.median(source_rgb[visible_cloth_u8 > 0], axis=0).astype(np.uint8)
            plain_fill_rgb[reference_fill_u8 > 0] = plain_fill_color
            plain_fill_rgb = self._restore_reference_region(
                reference_fill_rgb,
                plain_fill_rgb,
                reference_fill_mask,
                strength=0.98,
            )
            reference_fill_rgb = self._blend_neighbor_cloth_tone(
                plain_fill_rgb,
                reference_fill_mask,
                cloth_mask=cloth_mask,
                reference_rgb=source_rgb,
            )
            reference_fill_rgb = self._cv2_refine_cloth_region(
                reference_fill_rgb,
                reference_fill_mask,
                reference_rgb=source_rgb,
                reference_mask=cloth_mask,
            )

    conditioned = self._restore_reference_region(
        current_rgb,
        reference_fill_rgb,
        cloth_fill_mask,
        strength=0.985 if str(hair_length or "").strip().lower() == "short" else 0.95,
    )
    conditioned = self._overlay_reference_cloth_fill(
        conditioned,
        reference_fill_rgb,
        cloth_fill_mask,
        cloth_mask=cloth_mask,
    )
    conditioned = self._blend_neighbor_cloth_tone(
        conditioned,
        cloth_fill_mask,
        cloth_mask=cloth_mask,
        reference_rgb=reference_fill_rgb,
    )
    conditioned = self._cv2_refine_cloth_region(
        conditioned,
        cloth_fill_mask,
        reference_rgb=reference_fill_rgb,
        reference_mask=cloth_mask,
    )
    if preserve_mask is not None and preserve_mask.shape == (H, W):
        conditioned = self._restore_reference_region(
            conditioned,
            source_rgb,
            np.clip(preserve_mask.astype(np.float32), 0.0, 1.0),
            strength=0.99 if str(hair_length or "").strip().lower() == "short" else 0.96,
        )
    return conditioned

def _apply_short_source_cloth_anchor_restore(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or fill_mask.shape != (H, W):
        return current_rgb

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return current_rgb

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    anchor_mask = np.clip(
        fill_mask.astype(np.float32)
        * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        anchor_mask = np.clip(
            anchor_mask - np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) * 0.98,
            0.0,
            1.0,
        )

    anchor_u8 = (anchor_mask > 0.08).astype(np.uint8) * 255
    if int((anchor_u8 > 0).sum()) < 32:
        return current_rgb

    gate_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.04))
    bottom = min(H, int(cutoff_y + face_h * 0.98))
    upper_half_w = max(12, int(face_w * 0.14))
    lower_half_w = max(18, int(face_w * 0.23))
    split_y = max(top + 1, int(cutoff_y + face_h * 0.34))
    if top < split_y:
        gate_u8[top:split_y, max(0, cx - upper_half_w):min(W, cx + upper_half_w)] = 255
    if split_y < bottom:
        gate_u8[split_y:bottom, max(0, cx - lower_half_w):min(W, cx + lower_half_w)] = 255
    anchor_u8 = cv2.bitwise_and(anchor_u8, gate_u8)
    if int((anchor_u8 > 0).sum()) < 24:
        return current_rgb

    anchor_mask = anchor_u8.astype(np.float32) / 255.0
    reference_restore_u8 = cv2.dilate(
        anchor_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 25)),
        iterations=1,
    )
    reference_restore_u8 = cv2.bitwise_and(
        reference_restore_u8,
        cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        ),
    )
    reference_fill_rgb = self._restore_cloth_overlap_from_source(
        source_rgb=source_rgb,
        current_rgb=source_rgb,
        restore_mask=reference_restore_u8.astype(np.float32) / 255.0,
        final_hair_mask=None,
        tone_reference_rgb=source_rgb,
        tone_reference_mask=cloth_mask,
    )

    yy, xx = np.indices((H, W), dtype=np.float32)
    top_f = float(max(top, 0))
    bottom_f = float(max(bottom, top + 1))
    y_norm = np.clip((yy - top_f) / max(bottom_f - top_f, 1.0), 0.0, 1.0)
    lift_px = face_h * (0.30 - 0.18 * y_norm)
    width_gain = 1.18 - 0.20 * y_norm
    map_x = np.clip(cx + (xx - float(cx)) * width_gain, 0.0, float(W - 1))
    map_y = np.clip(yy + lift_px, 0.0, float(H - 1))
    warped_rgb = cv2.remap(
        reference_fill_rgb,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    visible_cloth_u8 = cv2.bitwise_and(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.bitwise_not(
            cv2.dilate(
                anchor_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                iterations=1,
            )
        ),
    )
    if int((visible_cloth_u8 > 0).sum()) >= 80:
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        plain_gray = float(np.median(source_gray[visible_cloth_u8 > 0]))
        plain_sat = float(np.median(source_sat[visible_cloth_u8 > 0]))
        if plain_gray >= 168.0 and plain_sat <= 84.0:
            plain_fill_color = np.median(source_rgb[visible_cloth_u8 > 0], axis=0).astype(np.uint8)
            plain_rgb = warped_rgb.copy()
            plain_rgb[anchor_u8 > 0] = (
                plain_rgb[anchor_u8 > 0].astype(np.float32) * 0.58
                + plain_fill_color.astype(np.float32) * 0.42
            ).astype(np.uint8)
            warped_rgb = plain_rgb

    restored = self._restore_reference_region(
        current_rgb,
        warped_rgb,
        anchor_mask,
        strength=0.996,
    )
    restored = self._overlay_reference_cloth_fill(
        restored,
        warped_rgb,
        anchor_mask,
        cloth_mask=cloth_mask,
    )
    restored = self._blend_neighbor_cloth_tone(
        restored,
        anchor_mask,
        cloth_mask=cloth_mask,
        reference_rgb=source_rgb,
    )
    restored = self._cv2_refine_cloth_region(
        restored,
        anchor_mask,
        reference_rgb=source_rgb,
        reference_mask=cloth_mask,
    )
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        restored = self._restore_reference_region(
            restored,
            source_rgb,
            np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0),
            strength=0.995,
        )
    return restored

def _tighten_short_under_jaw_mask_to_center(
    self,
    *,
    mask_u8: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    neck_preserve_mask: Optional[np.ndarray] = None,
    min_pixels: int = 24,
) -> np.ndarray:
    H, W = mask_u8.shape[:2]
    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return mask_u8

    focus_mask = self._build_short_under_jaw_crop_core_mask(
        fill_mask=mask_u8.astype(np.float32) / 255.0,
        cloth_mask=cloth_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        neck_preserve_mask=neck_preserve_mask,
    )
    focus_u8 = (np.clip(focus_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255
    if int((focus_u8 > 0).sum()) < min_pixels:
        return mask_u8

    focus_u8 = cv2.dilate(
        focus_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    tightened_u8 = cv2.bitwise_and(mask_u8, focus_u8)
    if int((tightened_u8 > 0).sum()) < min_pixels:
        return mask_u8
    tightened_u8 = cv2.morphologyEx(
        tightened_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
    )
    return tightened_u8

def _build_short_under_jaw_crop_core_mask(
    self,
    *,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = fill_mask.shape[:2]
    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    work = np.clip(fill_mask.astype(np.float32), 0.0, 1.0)
    work = np.clip(
        work * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        work = np.clip(
            work - np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) * 0.98,
            0.0,
            1.0,
        )

    work_u8 = (work > 0.08).astype(np.uint8) * 255
    if int((work_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    top = max(0, int(cutoff_y + face_h * 0.02))
    upper_bottom = min(H, int(cutoff_y + face_h * 0.32))
    lower_bottom = min(H, int(cutoff_y + face_h * 0.96))
    gate_u8 = np.zeros((H, W), dtype=np.uint8)

    upper_half_w = max(10, int(face_w * 0.16))
    middle_half_w = max(14, int(face_w * 0.22))
    lower_half_w = max(18, int(face_w * 0.30))
    if top < upper_bottom:
        gate_u8[top:upper_bottom, max(0, cx - upper_half_w):min(W, cx + upper_half_w)] = 255
    if upper_bottom < lower_bottom:
        split_y = max(upper_bottom, int(cutoff_y + face_h * 0.48))
        gate_u8[upper_bottom:split_y, max(0, cx - middle_half_w):min(W, cx + middle_half_w)] = 255
        gate_u8[split_y:lower_bottom, max(0, cx - lower_half_w):min(W, cx + lower_half_w)] = 255

    ellipse_center = (
        cx,
        min(H - 1, max(0, int(cutoff_y + face_h * 0.44))),
    )
    ellipse_axes = (
        max(12, int(face_w * 0.20)),
        max(10, int(face_h * 0.16)),
    )
    cv2.ellipse(gate_u8, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)
    gate_u8 = cv2.morphologyEx(
        gate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
    )

    core_u8 = cv2.bitwise_and(work_u8, gate_u8)
    if int((core_u8 > 0).sum()) < 24:
        eroded_u8 = cv2.erode(
            work_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
            iterations=1,
        )
        if int((eroded_u8 > 0).sum()) >= 24:
            core_u8 = cv2.bitwise_and(eroded_u8, gate_u8)
        if int((core_u8 > 0).sum()) < 24:
            core_u8 = cv2.bitwise_and(
                work_u8,
                cv2.dilate(
                    gate_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
                    iterations=1,
                ),
            )
    if int((core_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(core_u8, connectivity=8)
    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    center_allow = max(12.0, face_w * 0.24)
    max_component_area = max(40, int(face_w * face_h * 0.18))
    min_component_height = max(6, int(face_h * 0.08))
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        if area > max_component_area:
            continue
        comp_cx = float(centroids[idx][0])
        comp_h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if abs(comp_cx - cx) > center_allow:
            continue
        if comp_h < min_component_height:
            continue
        filtered_u8[labels == idx] = 255
    if int((filtered_u8 > 0).sum()) < 24:
        filtered_u8 = core_u8

    filtered_u8 = cv2.morphologyEx(
        filtered_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)),
    )
    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
        iterations=1,
    )
    out = cv2.GaussianBlur(
        filtered_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.8,
        sigmaY=3.4,
    )
    return np.clip(out * 0.995, 0.0, 1.0).astype(np.float32)

def _compose_strict_short_under_jaw_crop_result(
    self,
    *,
    current_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if (
        generated_rgb.shape[:2] != (H, W)
        or reference_rgb.shape[:2] != (H, W)
        or fill_mask.shape != (H, W)
    ):
        return current_rgb

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return current_rgb

    visible_fill_mask = np.clip(
        fill_mask.astype(np.float32)
        * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        visible_fill_mask = np.clip(
            visible_fill_mask - np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) * 0.98,
            0.0,
            1.0,
        )
    if float(visible_fill_mask.sum()) <= 0.0:
        return current_rgb

    core_mask = self._build_short_under_jaw_crop_core_mask(
        fill_mask=visible_fill_mask,
        cloth_mask=cloth_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        neck_preserve_mask=neck_preserve_mask,
    )
    if float(core_mask.sum()) <= 0.0:
        core_mask = cv2.GaussianBlur(
            np.clip(visible_fill_mask.astype(np.float32), 0.0, 1.0),
            (0, 0),
            sigmaX=2.8,
            sigmaY=3.4,
        ).astype(np.float32)

    boundary_mask = np.clip(
        visible_fill_mask - np.clip(core_mask * 1.08, 0.0, 1.0),
        0.0,
        1.0,
    )
    constrained_base = current_rgb.copy()
    if float(boundary_mask.sum()) > 0.0:
        constrained_base = self._restore_reference_region(
            constrained_base,
            reference_rgb,
            boundary_mask,
            strength=0.992,
        )
        constrained_base = self._overlay_reference_cloth_fill(
            constrained_base,
            reference_rgb,
            boundary_mask,
            cloth_mask=cloth_mask,
        )
        constrained_base = self._blend_neighbor_cloth_tone(
            constrained_base,
            boundary_mask,
            cloth_mask=cloth_mask,
            reference_rgb=reference_rgb,
        )
        constrained_base = self._cv2_refine_cloth_region(
            constrained_base,
            boundary_mask,
            reference_rgb=reference_rgb,
            reference_mask=cloth_mask,
        )

    constrained_generated = generated_rgb.copy()
    if float(boundary_mask.sum()) > 0.0:
        constrained_generated = self._restore_reference_region(
            constrained_generated,
            reference_rgb,
            boundary_mask,
            strength=0.996,
        )
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        constrained_generated = self._restore_reference_region(
            constrained_generated,
            reference_rgb,
            np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0),
            strength=0.997,
        )

    paste_mask = cv2.GaussianBlur(
        np.clip(core_mask.astype(np.float32), 0.0, 1.0),
        (0, 0),
        sigmaX=3.6,
        sigmaY=4.1,
    )[..., np.newaxis]
    paste_mask = np.clip(paste_mask * 0.996, 0.0, 1.0)
    out = (
        constrained_generated.astype(np.float32) * paste_mask
        + constrained_base.astype(np.float32) * (1.0 - paste_mask)
    )
    return np.clip(out, 0, 255).astype(np.uint8)

def _refine_short_under_jaw_crop_region(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    face_crop_pil: Image.Image,
    seed: int,
    control_rgb: Optional[np.ndarray] = None,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or fill_mask.shape != (H, W):
        return current_rgb

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if control_rgb is not None:
        if control_rgb.shape[:2] != (H, W):
            control_rgb = cv2.resize(control_rgb.astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
        else:
            control_rgb = control_rgb.astype(np.uint8)
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return current_rgb

    local_mask = np.clip(
        fill_mask.astype(np.float32)
        * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        local_mask = np.clip(
            local_mask - np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) * 0.98,
            0.0,
            1.0,
        )
    mask_u8 = (local_mask > 0.08).astype(np.uint8) * 255
    if int((mask_u8 > 0).sum()) < 36:
        return current_rgb

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return current_rgb

    crop_left = max(0, int(min(xs.min(), x1) - max(18, face_w * 0.18)))
    crop_right = min(W, int(max(xs.max() + 1, x2) + max(18, face_w * 0.18)))
    crop_top = max(0, int(min(ys.min(), cutoff_y) - max(12, face_h * 0.10)))
    crop_bottom = min(H, int(max(ys.max() + 1, y2) + max(24, face_h * 0.42)))
    if crop_right - crop_left < 48 or crop_bottom - crop_top < 48:
        return current_rgb

    crop_current_rgb = current_rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_source_rgb = source_rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_fill_mask = local_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_cloth_mask = cloth_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_protect_mask = (
        protect_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
        if protect_mask is not None and protect_mask.shape == (H, W)
        else None
    )
    crop_control_rgb = (
        control_rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
        if control_rgb is not None and control_rgb.shape[:2] == (H, W)
        else None
    )
    crop_neck_preserve_mask = (
        neck_preserve_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
        if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W)
        else None
    )

    local_face_bbox = (
        max(0, x1 - crop_left),
        max(0, y1 - crop_top),
        min(crop_right - crop_left, x2 - crop_left),
        min(crop_bottom - crop_top, y2 - crop_top),
    )
    local_cutoff_y = max(0, cutoff_y - crop_top)
    crop_reference_base = self._build_source_conditioned_cloth_base(
        current_rgb=crop_current_rgb,
        source_rgb=crop_source_rgb,
        fill_mask=crop_fill_mask,
        cloth_mask=crop_cloth_mask,
        hair_length="short",
        preserve_mask=crop_neck_preserve_mask,
    )

    crop_generated_rgb = self._sd_refine_removed_region(
        base_rgb=crop_reference_base,
        removal_mask=crop_fill_mask,
        face_bbox=local_face_bbox,
        face_crop_pil=face_crop_pil,
        protect_mask=crop_protect_mask,
        cloth_mask=crop_cloth_mask,
        hair_length="short",
        seed=int(seed),
        reference_rgb=crop_source_rgb,
        refine_mode="short_cloth_crop",
        control_rgb=crop_control_rgb,
    )
    crop_generated_rgb = self._blend_neighbor_cloth_tone(
        crop_generated_rgb,
        crop_fill_mask,
        cloth_mask=crop_cloth_mask,
        reference_rgb=crop_reference_base,
    )
    crop_generated_rgb = self._cv2_refine_cloth_region(
        crop_generated_rgb,
        crop_fill_mask,
        reference_rgb=crop_reference_base,
        reference_mask=crop_cloth_mask,
    )
    crop_generated_rgb = self._apply_short_source_cloth_anchor_restore(
        current_rgb=crop_generated_rgb,
        source_rgb=crop_source_rgb,
        fill_mask=crop_fill_mask,
        cloth_mask=crop_cloth_mask,
        face_bbox=local_face_bbox,
        cutoff_y=local_cutoff_y,
        neck_preserve_mask=crop_neck_preserve_mask,
    )
    blended_crop = self._compose_strict_short_under_jaw_crop_result(
        current_rgb=crop_current_rgb,
        generated_rgb=crop_generated_rgb,
        reference_rgb=crop_source_rgb,
        fill_mask=crop_fill_mask,
        cloth_mask=crop_cloth_mask,
        face_bbox=local_face_bbox,
        cutoff_y=local_cutoff_y,
        neck_preserve_mask=crop_neck_preserve_mask,
    )

    out = current_rgb.copy()
    out[crop_top:crop_bottom, crop_left:crop_right] = blended_crop
    return out

def _generate_short_under_jaw_cloth_insert_region(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    face_crop_pil: Image.Image,
    seed: int,
    control_rgb: Optional[np.ndarray] = None,
    neck_preserve_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or fill_mask.shape != (H, W):
        return current_rgb

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    neck_preserve_mask = self._resize_mask_to_shape(neck_preserve_mask, (H, W))
    if control_rgb is not None:
        if control_rgb.shape[:2] != (H, W):
            control_rgb = cv2.resize(control_rgb.astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
        else:
            control_rgb = control_rgb.astype(np.uint8)
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return current_rgb

    insert_mask = np.clip(
        fill_mask.astype(np.float32)
        * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W):
        insert_mask = np.clip(
            insert_mask - np.clip(neck_preserve_mask.astype(np.float32), 0.0, 1.0) * 0.98,
            0.0,
            1.0,
        )
    insert_u8 = (insert_mask > 0.08).astype(np.uint8) * 255
    if int((insert_u8 > 0).sum()) < 36:
        return current_rgb

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    ys, xs = np.where(insert_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return current_rgb

    crop_left = max(0, int(min(xs.min(), x1) - max(18, face_w * 0.18)))
    crop_right = min(W, int(max(xs.max() + 1, x2) + max(18, face_w * 0.18)))
    crop_top = max(0, int(min(ys.min(), cutoff_y) - max(12, face_h * 0.10)))
    crop_bottom = min(H, int(max(ys.max() + 1, y2) + max(28, face_h * 0.46)))
    if crop_right - crop_left < 48 or crop_bottom - crop_top < 48:
        return current_rgb

    crop_current_rgb = current_rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_source_rgb = source_rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_insert_mask = insert_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_cloth_mask = cloth_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_protect_mask = (
        protect_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
        if protect_mask is not None and protect_mask.shape == (H, W)
        else None
    )
    crop_control_rgb = (
        control_rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
        if control_rgb is not None and control_rgb.shape[:2] == (H, W)
        else None
    )
    crop_neck_preserve_mask = (
        neck_preserve_mask[crop_top:crop_bottom, crop_left:crop_right].copy()
        if neck_preserve_mask is not None and neck_preserve_mask.shape == (H, W)
        else None
    )

    local_face_bbox = (
        max(0, x1 - crop_left),
        max(0, y1 - crop_top),
        min(crop_right - crop_left, x2 - crop_left),
        min(crop_bottom - crop_top, y2 - crop_top),
    )
    local_cutoff_y = max(0, cutoff_y - crop_top)

    crop_insert_u8 = (crop_insert_mask > 0.08).astype(np.uint8) * 255
    crop_source_bgr = cv2.cvtColor(crop_source_rgb, cv2.COLOR_RGB2BGR)
    crop_plate_bgr = cv2.inpaint(crop_source_bgr, crop_insert_u8, 5, cv2.INPAINT_TELEA)
    crop_plate_rgb = cv2.cvtColor(crop_plate_bgr, cv2.COLOR_BGR2RGB)

    visible_cloth_u8 = cv2.bitwise_and(
        (np.clip(crop_cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.bitwise_not(
            cv2.dilate(
                crop_insert_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
                iterations=1,
            )
        ),
    )
    if int((visible_cloth_u8 > 0).sum()) >= 48:
        cloth_fill_color = np.median(crop_source_rgb[visible_cloth_u8 > 0], axis=0).astype(np.uint8)
        crop_plate_rgb[crop_insert_u8 > 0] = (
            crop_plate_rgb[crop_insert_u8 > 0].astype(np.float32) * 0.52
            + cloth_fill_color.astype(np.float32) * 0.48
        ).astype(np.uint8)

    crop_insert_base = crop_current_rgb.copy()
    crop_insert_base = self._restore_reference_region(
        crop_insert_base,
        crop_plate_rgb,
        crop_insert_mask,
        strength=0.998,
    )
    crop_insert_base = self._overlay_reference_cloth_fill(
        crop_insert_base,
        crop_plate_rgb,
        crop_insert_mask,
        cloth_mask=crop_cloth_mask,
    )
    crop_insert_base = self._blend_neighbor_cloth_tone(
        crop_insert_base,
        crop_insert_mask,
        cloth_mask=crop_cloth_mask,
        reference_rgb=crop_plate_rgb,
    )
    if crop_neck_preserve_mask is not None and crop_neck_preserve_mask.shape == crop_insert_mask.shape:
        crop_insert_base = self._restore_reference_region(
            crop_insert_base,
            crop_source_rgb,
            np.clip(crop_neck_preserve_mask.astype(np.float32), 0.0, 1.0),
            strength=0.997,
        )

    crop_generated_rgb = self._sd_refine_removed_region(
        base_rgb=crop_insert_base,
        removal_mask=crop_insert_mask,
        face_bbox=local_face_bbox,
        face_crop_pil=face_crop_pil,
        protect_mask=crop_protect_mask,
        cloth_mask=crop_cloth_mask,
        hair_length="short",
        seed=int(seed),
        reference_rgb=crop_source_rgb,
        refine_mode="short_cloth_insert",
        control_rgb=crop_control_rgb,
    )
    crop_generated_rgb = self._blend_neighbor_cloth_tone(
        crop_generated_rgb,
        crop_insert_mask,
        cloth_mask=crop_cloth_mask,
        reference_rgb=crop_insert_base,
    )
    crop_generated_rgb = self._cv2_refine_cloth_region(
        crop_generated_rgb,
        crop_insert_mask,
        reference_rgb=crop_insert_base,
        reference_mask=crop_cloth_mask,
    )
    crop_generated_rgb = self._apply_short_source_cloth_anchor_restore(
        current_rgb=crop_generated_rgb,
        source_rgb=crop_source_rgb,
        fill_mask=crop_insert_mask,
        cloth_mask=crop_cloth_mask,
        face_bbox=local_face_bbox,
        cutoff_y=local_cutoff_y,
        neck_preserve_mask=crop_neck_preserve_mask,
    )
    blended_crop = self._compose_strict_short_under_jaw_crop_result(
        current_rgb=crop_current_rgb,
        generated_rgb=crop_generated_rgb,
        reference_rgb=crop_source_rgb,
        fill_mask=crop_insert_mask,
        cloth_mask=crop_cloth_mask,
        face_bbox=local_face_bbox,
        cutoff_y=local_cutoff_y,
        neck_preserve_mask=crop_neck_preserve_mask,
    )

    out = current_rgb.copy()
    out[crop_top:crop_bottom, crop_left:crop_right] = blended_crop
    return out

def _stabilize_under_jaw_cloth_fill(
    self,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    fill_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    hair_length: str = "",
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or fill_mask.shape != (H, W):
        return current_rgb

    cloth_mask = self._resize_mask_to_shape(cloth_mask, (H, W))
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return current_rgb

    cloth_cleanup_mask = np.clip(
        fill_mask.astype(np.float32)
        * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    cloth_cleanup_u8 = (cloth_cleanup_mask > 0.08).astype(np.uint8) * 255
    if int((cloth_cleanup_u8 > 0).sum()) < 30:
        return current_rgb

    visible_cloth_u8 = cv2.bitwise_and(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.bitwise_not(
            cv2.dilate(
                cloth_cleanup_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
                iterations=1,
            )
        ),
    )
    if int((visible_cloth_u8 > 0).sum()) < 80:
        return current_rgb

    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    plain_gray = float(np.median(source_gray[visible_cloth_u8 > 0]))
    plain_sat = float(np.median(source_sat[visible_cloth_u8 > 0]))
    use_plain_cloth_force = plain_gray >= 168.0 and plain_sat <= 84.0
    if not use_plain_cloth_force:
        return current_rgb

    plain_fill_rgb = source_rgb.copy()
    plain_fill_color = np.median(source_rgb[visible_cloth_u8 > 0], axis=0).astype(np.uint8)
    plain_fill_rgb[cloth_cleanup_u8 > 0] = plain_fill_color
    reference_fill_rgb = self._restore_reference_region(
        current_rgb,
        plain_fill_rgb,
        cloth_cleanup_mask,
        strength=0.94,
    )
    reference_fill_rgb = self._blend_neighbor_cloth_tone(
        reference_fill_rgb,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
        reference_rgb=source_rgb,
    )
    reference_fill_rgb = self._cv2_refine_cloth_region(
        reference_fill_rgb,
        cloth_cleanup_mask,
        reference_rgb=source_rgb,
        reference_mask=cloth_mask,
    )

    prefer_reference_first = str(hair_length or "").strip().lower() == "short"
    if prefer_reference_first:
        cleaned = self._restore_reference_region(
            current_rgb,
            reference_fill_rgb,
            cloth_cleanup_mask,
            strength=0.985,
        )
    else:
        cleaned = self._overlay_reference_cloth_fill(
            current_rgb,
            reference_fill_rgb,
            cloth_cleanup_mask,
            cloth_mask=cloth_mask,
        )
    cleaned = self._overlay_reference_cloth_fill(
        cleaned,
        reference_fill_rgb,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
    )
    cleaned = self._blend_neighbor_cloth_tone(
        cleaned,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
        reference_rgb=reference_fill_rgb,
    )
    cleaned = self._cv2_refine_cloth_region(
        cleaned,
        cloth_cleanup_mask,
        reference_rgb=reference_fill_rgb,
        reference_mask=cloth_mask,
    )

    mask_ys, mask_xs = np.where(cloth_cleanup_u8 > 0)
    if mask_xs.size >= 24 and mask_ys.size >= 24:
        mask_left = int(mask_xs.min())
        mask_right = int(mask_xs.max()) + 1
        mask_top = int(mask_ys.min())
        mask_bottom = int(mask_ys.max()) + 1
        mask_center_x = int(0.5 * (mask_left + mask_right))
        inner_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        inner_half_w = max(10, int((mask_right - mask_left) * 0.24))
        inner_top = mask_top
        inner_bottom = min(mask_bottom, mask_top + max(16, int((mask_bottom - mask_top) * 0.72)))
        inner_left = max(0, mask_center_x - inner_half_w)
        inner_right = min(W, mask_center_x + inner_half_w)
        if inner_top < inner_bottom and inner_left < inner_right:
            inner_gate_u8[inner_top:inner_bottom, inner_left:inner_right] = 255
            inner_gate_u8 = cv2.bitwise_and(inner_gate_u8, cloth_cleanup_u8)
            if int((inner_gate_u8 > 0).sum()) >= 24:
                inner_gate_mask = inner_gate_u8.astype(np.float32) / 255.0
                cleaned = self._restore_reference_region(
                    cleaned,
                    reference_fill_rgb,
                    inner_gate_mask,
                    strength=0.97,
                )
                cleaned = self._blend_neighbor_cloth_tone(
                    cleaned,
                    inner_gate_mask,
                    cloth_mask=cloth_mask,
                    reference_rgb=reference_fill_rgb,
                )

    cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cleaned_sat = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    ref_gray = cv2.cvtColor(reference_fill_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blackhat = cv2.morphologyEx(
        cleaned_gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
    )
    dark_residual_u8 = (
        (
            (
                (cleaned_gray + 10.0 < ref_gray)
                & (cleaned_sat < 132.0)
            )
            | (
                (cleaned_gray < plain_gray - 16.0)
                & (cleaned_sat < 126.0)
            )
            | (
                (cleaned_gray < plain_gray - 10.0)
                & (blackhat > 7)
                & (cleaned_sat < 140.0)
            )
        ).astype(np.uint8)
        * 255
    )
    dark_residual_u8 = cv2.bitwise_and(dark_residual_u8, cloth_cleanup_u8)
    dark_residual_u8 = cv2.morphologyEx(
        dark_residual_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
    )
    dark_residual_u8 = cv2.dilate(
        dark_residual_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
        iterations=1,
    )
    if int((dark_residual_u8 > 0).sum()) < 24:
        return cleaned

    dark_residual_mask = dark_residual_u8.astype(np.float32) / 255.0
    cleaned = self._restore_reference_region(
        cleaned,
        reference_fill_rgb,
        dark_residual_mask,
        strength=0.98,
    )
    cleaned = self._overlay_reference_cloth_fill(
        cleaned,
        reference_fill_rgb,
        dark_residual_mask,
        cloth_mask=cloth_mask,
    )
    cleaned = self._blend_neighbor_cloth_tone(
        cleaned,
        dark_residual_mask,
        cloth_mask=cloth_mask,
        reference_rgb=reference_fill_rgb,
    )
    return cleaned

def _build_generation_protect_mask(
    self,
    protect_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    hair_length: str,
) -> np.ndarray:
    """
    SD 생성/합성에 사용할 얼굴 보호 마스크.
    short/medium 헤어에서는 목까지 네모나게 막히면 bob 라인이 끊겨 보여서,
    턱 아래는 빠르게 감쇠시키고 중앙 목 부분만 좁게 남긴다.
    """
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
            neck_guard * (0.58 if hair_length == "short" else 0.50),
        )

    if hair_length == "short":
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, erode_k, iterations=1)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)

def _build_removal_protect_mask(
    self,
    protect_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    hair_length: str,
) -> np.ndarray:
    """
    긴 머리 제거(pre-clean) 단계에서 사용할 얼굴 보호 마스크.
    생성 단계보다 목 중앙 보호를 훨씬 약하게 두어, 목 앞쪽으로 내려온 머리 가닥은
    제거 대상으로 남기고 얼굴/턱 주변만 보수적으로 보호한다.
    """
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
    mask = np.minimum(mask, np.clip(face_core * 1.15, 0.0, 1.0))

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
    mask = np.clip(mask - neck_cut, 0.0, 1.0)

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
    y_mid = min(H, int(y2 + face_h * (0.11 if hair_length == "short" else 0.16)))
    y_bottom = min(H, int(y2 + face_h * (0.23 if hair_length == "short" else 0.32)))

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
            int(y2 + face_h * (0.07 if hair_length == "short" else 0.10)),
        )
        ellipse_axes = (
            max(10, int(neck_half * 0.85)),
            max(8, int(face_h * (0.10 if hair_length == "short" else 0.14))),
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

def _filter_short_torso_box_mask(
    self,
    img_rgb: Optional[np.ndarray],
    removal_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    support_mask: Optional[np.ndarray],
    center_support_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
) -> np.ndarray:
    H, W = removal_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    original_px = int((removal_u8 > 0).sum())
    if original_px < 40:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8)
    else:
        cloth_u8 = np.zeros((H, W), dtype=np.uint8)

    support_hint_u8 = np.zeros((H, W), dtype=np.uint8)
    if support_mask is not None and support_mask.shape == (H, W):
        support_hint_u8 = cv2.bitwise_or(
            support_hint_u8,
            cv2.dilate(
                (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)),
                iterations=1,
            ),
        )
    if center_support_mask is not None and center_support_mask.shape == (H, W):
        support_hint_u8 = cv2.bitwise_or(
            support_hint_u8,
            cv2.dilate(
                (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 19)),
                iterations=1,
            ),
        )

    dark_evidence_u8 = np.zeros((H, W), dtype=np.uint8)
    bright_cloth_evidence_u8 = np.zeros((H, W), dtype=np.uint8)
    if img_rgb is not None and img_rgb.shape[:2] == (H, W):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        sat = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        blackhat = cv2.morphologyEx(
            gray.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )
        dark_evidence_u8 = (
            (
                ((gray < 168.0) & ((blur - gray) > 2.4))
                | (blackhat > 8)
            ).astype(np.uint8)
            * 255
        )
        dark_zone_u8 = np.zeros((H, W), dtype=np.uint8)
        dark_top = max(0, int(cutoff_y - face_h * 0.04))
        dark_bottom = min(H, int(cutoff_y + face_h * 0.96))
        dark_left = max(0, int(x1 - face_w * 1.12))
        dark_right = min(W, int(x2 + face_w * 1.12))
        if dark_top < dark_bottom and dark_left < dark_right:
            dark_zone_u8[dark_top:dark_bottom, dark_left:dark_right] = 255
            dark_evidence_u8 = cv2.bitwise_and(dark_evidence_u8, dark_zone_u8)
        if int((cloth_u8 > 0).sum()) > 0:
            dark_evidence_u8 = cv2.bitwise_and(dark_evidence_u8, cloth_u8.astype(np.uint8) * 255)
        bright_cloth_evidence_u8 = (
            (
                (gray > 178.0)
                & (blur > 182.0)
                & (sat < 60.0)
            ).astype(np.uint8)
            * 255
        )
        bright_zone_u8 = np.zeros((H, W), dtype=np.uint8)
        bright_top = max(0, int(cutoff_y + face_h * 0.02))
        bright_bottom = min(H, int(cutoff_y + face_h * 1.04))
        bright_left = max(0, int(x1 - face_w * 1.16))
        bright_right = min(W, int(x2 + face_w * 1.16))
        if bright_top < bright_bottom and bright_left < bright_right:
            bright_zone_u8[bright_top:bright_bottom, bright_left:bright_right] = 255
            bright_cloth_evidence_u8 = cv2.bitwise_and(bright_cloth_evidence_u8, bright_zone_u8)
        if int((cloth_u8 > 0).sum()) > 0:
            bright_cloth_evidence_u8 = cv2.bitwise_and(
                bright_cloth_evidence_u8,
                cloth_u8.astype(np.uint8) * 255,
            )

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(removal_u8, 8)
    boxy_width = max(26, int(face_w * 0.30))
    boxy_height = max(24, int(face_h * 0.30))
    boxy_area = max(120, int(face_w * face_h * 0.075))
    center_half = max(16, int(face_w * 0.24))
    low_top = int(cutoff_y + face_h * 0.16)
    low_bottom = int(cutoff_y + face_h * 0.82)
    center_strand_width = max(16, int(face_w * 0.18))
    center_strand_height = max(30, int(face_h * 0.30))

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 16:
            continue

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        bottom = y + h
        comp_mask = labels == idx
        cloth_overlap = int(cloth_u8[comp_mask].sum())
        overlap_ratio = float(cloth_overlap) / float(area) if area > 0 else 0.0
        comp_cx = float(centroids[idx][0])
        comp_u8 = comp_mask.astype(np.uint8) * 255
        support_overlap = int((cv2.bitwise_and(comp_u8, support_hint_u8) > 0).sum())
        dark_overlap = int((cv2.bitwise_and(comp_u8, dark_evidence_u8) > 0).sum())
        dark_ratio = float(dark_overlap) / float(area) if area > 0 else 0.0
        bright_overlap = int((cv2.bitwise_and(comp_u8, bright_cloth_evidence_u8) > 0).sum())
        bright_ratio = float(bright_overlap) / float(area) if area > 0 else 0.0

        is_boxy = w >= boxy_width and h >= boxy_height and area >= boxy_area
        is_center_box = abs(comp_cx - cx) <= center_half and w >= max(22, int(face_w * 0.28))
        is_center_strand = (
            abs(comp_cx - cx) <= center_half
            and w <= center_strand_width
            and h >= center_strand_height
            and bottom >= int(cutoff_y + face_h * 0.30)
        )
        is_low = y >= low_top or bottom >= low_bottom
        is_side_component = abs(comp_cx - cx) >= max(22, int(face_w * 0.28))
        is_side_blob = (
            is_side_component
            and overlap_ratio >= 0.30
            and w >= max(20, int(face_w * 0.22))
            and area >= max(72, int(face_w * face_h * 0.022))
        )
        is_bright_side_blob = (
            is_side_component
            and overlap_ratio >= 0.26
            and bright_ratio >= 0.18
            and w >= max(24, int(face_w * 0.28))
            and area >= max(96, int(face_w * face_h * 0.024))
            and bottom >= int(cutoff_y + face_h * 0.14)
        )
        if is_center_strand:
            keep_u8[comp_mask] = 255
            continue
        if is_bright_side_blob:
            if support_overlap < max(16, int(area * 0.08)):
                if dark_ratio < 0.16 or bright_ratio > (dark_ratio * 1.8 + 0.06):
                    continue
        if is_side_blob and (dark_ratio < 0.09 or dark_overlap < max(8, int(area * 0.04))):
            if support_overlap < max(14, int(area * 0.08)):
                continue
            if h < max(28, int(face_h * 0.42)):
                continue
        if overlap_ratio > 0.46 and (is_boxy or is_center_box or is_low):
            if support_overlap >= max(18, int(area * 0.08)):
                supported_u8 = cv2.bitwise_and(comp_u8, support_hint_u8)
                supported_u8 = cv2.dilate(
                    supported_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 19)),
                    iterations=1,
                )
                supported_u8 = cv2.bitwise_and(supported_u8, comp_u8)
                keep_u8 = cv2.bitwise_or(keep_u8, supported_u8)
            continue
        if is_low and is_boxy and support_overlap < max(10, int(area * 0.05)):
            continue
        if is_side_component and overlap_ratio > 0.34 and dark_ratio < 0.07 and area >= max(80, int(face_w * face_h * 0.028)):
            continue
        if (
            is_side_component
            and overlap_ratio > 0.30
            and bright_ratio >= 0.22
            and dark_ratio < 0.11
            and support_overlap < max(12, int(area * 0.06))
            and area >= max(110, int(face_w * face_h * 0.030))
        ):
            continue

        keep_u8[comp_mask] = 255

    if int((support_hint_u8 > 0).sum()) > 0:
        deep_torso_u8 = np.zeros((H, W), dtype=np.uint8)
        deep_x1 = max(0, int(x1 - face_w * 1.05))
        deep_x2 = min(W, int(x2 + face_w * 1.05))
        deep_y1 = max(0, int(cutoff_y + face_h * 0.22))
        if deep_x1 < deep_x2 and deep_y1 < H:
            deep_torso_u8[deep_y1:, deep_x1:deep_x2] = 255
            if int((cloth_u8 > 0).sum()) > 0:
                deep_torso_u8 = cv2.bitwise_and(deep_torso_u8, cloth_u8.astype(np.uint8) * 255)
            precise_keep_u8 = cv2.dilate(
                support_hint_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 25)),
                iterations=1,
            )
            precise_keep_u8 = cv2.bitwise_and(precise_keep_u8, deep_torso_u8)
            supported_keep_u8 = cv2.dilate(
                support_hint_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 33)),
                iterations=1,
            )
            supported_keep_u8 = cv2.bitwise_and(supported_keep_u8, deep_torso_u8)
            keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(deep_torso_u8))
            keep_u8 = cv2.bitwise_or(
                keep_u8,
                cv2.bitwise_and(
                    removal_u8,
                    cv2.bitwise_or(precise_keep_u8, supported_keep_u8),
                ),
            )

    kept_px = int((keep_u8 > 0).sum())
    if kept_px <= 0:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    filtered = removal_mask.astype(np.float32) * (keep_u8.astype(np.float32) / 255.0)
    filtered = cv2.GaussianBlur(filtered, (0, 0), sigmaX=1.1, sigmaY=1.1)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)

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

def _restrict_short_removal_to_tail_lanes(
    self,
    removal_mask: np.ndarray,
    support_mask: Optional[np.ndarray],
    center_support_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length != "short":
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    H, W = removal_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    original_px = int((removal_u8 > 0).sum())
    if original_px < 60:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    side_seed_u8 = np.zeros((H, W), dtype=np.uint8)
    if support_mask is not None and support_mask.shape == (H, W):
        side_seed_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255

    center_seed_u8 = np.zeros((H, W), dtype=np.uint8)
    if center_support_mask is not None and center_support_mask.shape == (H, W):
        center_seed_u8 = (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255

    if int((side_seed_u8 > 0).sum()) < 24:
        fallback_side_u8 = removal_u8.copy()
        fallback_corridor_u8 = np.zeros((H, W), dtype=np.uint8)
        fallback_top = max(0, int(cutoff_y))
        fallback_bottom = min(H, int(cutoff_y + face_h * 1.92))
        fallback_left = max(0, int(x1 - face_w * 1.42))
        fallback_right = min(W, int(x2 + face_w * 1.42))
        if fallback_top < fallback_bottom and fallback_left < fallback_right:
            fallback_corridor_u8[fallback_top:fallback_bottom, fallback_left:fallback_right] = 255
            fallback_side_u8 = cv2.bitwise_and(fallback_side_u8, fallback_corridor_u8)

            center_keepout_u8 = np.zeros((H, W), dtype=np.uint8)
            center_half = max(18, int(face_w * 0.24))
            keepout_bottom = min(H, int(cutoff_y + face_h * 0.84))
            if fallback_top < keepout_bottom:
                center_keepout_u8[
                    fallback_top:keepout_bottom,
                    max(0, int(0.5 * (x1 + x2)) - center_half):min(W, int(0.5 * (x1 + x2)) + center_half),
                ] = 255
                fallback_side_u8 = cv2.bitwise_and(fallback_side_u8, cv2.bitwise_not(center_keepout_u8))

            fallback_side_u8 = cv2.morphologyEx(
                fallback_side_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21)),
            )
            fallback_side_u8 = cv2.morphologyEx(
                fallback_side_u8,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            if int((fallback_side_u8 > 0).sum()) >= 60:
                side_seed_u8 = fallback_side_u8

    if int((side_seed_u8 > 0).sum()) < 24 and int((center_seed_u8 > 0).sum()) < 12:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    side_lane_u8 = cv2.dilate(
        side_seed_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 57)),
        iterations=1,
    )
    center_lane_u8 = cv2.dilate(
        center_seed_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 49)),
        iterations=1,
    )
    lane_u8 = cv2.bitwise_or(side_lane_u8, center_lane_u8)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y))
    bottom = min(H, int(cutoff_y + face_h * 1.62))
    left = max(0, int(x1 - face_w * 1.18))
    right = min(W, int(x2 + face_w * 1.18))
    if top >= bottom or left >= right:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)
    corridor_u8[top:bottom, left:right] = 255
    lane_u8 = cv2.bitwise_and(lane_u8, corridor_u8)
    if int((lane_u8 > 0).sum()) < 120:
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    filtered_u8 = cv2.bitwise_and(removal_u8, lane_u8)
    filtered_u8 = cv2.morphologyEx(
        filtered_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17)),
    )
    filtered_px = int((filtered_u8 > 0).sum())
    if filtered_px < max(140, int(original_px * 0.12)):
        return np.clip(removal_mask, 0.0, 1.0).astype(np.float32)

    filtered = removal_mask.astype(np.float32) * (filtered_u8.astype(np.float32) / 255.0)
    filtered = cv2.GaussianBlur(filtered, (0, 0), sigmaX=1.0, sigmaY=1.2)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)

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

def _restore_cloth_overlap_from_source(
    self,
    source_rgb: np.ndarray,
    current_rgb: np.ndarray,
    restore_mask: np.ndarray,
    final_hair_mask: Optional[np.ndarray] = None,
    tone_reference_rgb: Optional[np.ndarray] = None,
    tone_reference_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or restore_mask.shape != (H, W):
        return current_rgb

    mask_u8 = (np.clip(restore_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((mask_u8 > 0).sum()) < 40:
        return current_rgb
    if final_hair_mask is not None and final_hair_mask.shape == (H, W):
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.24).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15)),
            iterations=1,
        )
        mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(final_hair_u8))
    if int((mask_u8 > 0).sum()) < 40:
        return current_rgb

    mask_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    ns = cv2.inpaint(source_rgb, mask_u8, 5, cv2.INPAINT_NS)
    telea = cv2.inpaint(source_rgb, mask_u8, 4, cv2.INPAINT_TELEA)
    refill = cv2.addWeighted(telea, 0.64, ns, 0.36, 0.0)

    ring_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    )
    ring_u8 = cv2.subtract(
        ring_u8,
        cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ),
    )
    ring_bool = ring_u8 > 0
    mask_bool = mask_u8 > 0
    if int(ring_bool.sum()) >= 80 and int(mask_bool.sum()) >= 40:
        refill_lab = cv2.cvtColor(refill, cv2.COLOR_RGB2LAB).astype(np.float32)
        reference_rgb = source_rgb
        if tone_reference_rgb is not None and tone_reference_rgb.shape[:2] == (H, W):
            reference_rgb = tone_reference_rgb
        reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ring_ref_bool = ring_bool.copy()
        if tone_reference_mask is not None and tone_reference_mask.shape == (H, W):
            ref_mask_u8 = cv2.dilate(
                (np.clip(tone_reference_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            masked_ring = np.logical_and(ring_ref_bool, ref_mask_u8 > 0)
            if int(masked_ring.sum()) >= 40:
                ring_ref_bool = masked_ring
        ring_vals = reference_lab[ring_ref_bool]
        mask_vals = refill_lab[mask_bool]
        ring_mean = ring_vals.mean(axis=0)
        mask_mean = mask_vals.mean(axis=0)
        ring_std = ring_vals.std(axis=0)
        mask_std = np.maximum(mask_vals.std(axis=0), 1.0)
        tone_matched = mask_vals.copy()
        tone_matched[:, 0] = np.clip(
            (tone_matched[:, 0] - mask_mean[0]) * np.clip(ring_std[0] / mask_std[0], 0.84, 1.16)
            + mask_mean[0]
            + np.clip(ring_mean[0] - mask_mean[0], -14.0, 14.0) * 0.72,
            0.0,
            255.0,
        )
        tone_matched[:, 1] = np.clip(
            tone_matched[:, 1] + np.clip(ring_mean[1] - mask_mean[1], -5.0, 5.0) * 0.45,
            0.0,
            255.0,
        )
        tone_matched[:, 2] = np.clip(
            tone_matched[:, 2] + np.clip(ring_mean[2] - mask_mean[2], -5.0, 5.0) * 0.45,
            0.0,
            255.0,
        )
        refill_lab[mask_bool] = tone_matched
        refill = cv2.cvtColor(refill_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    alpha = cv2.GaussianBlur(
        (mask_u8 > 0).astype(np.float32),
        (0, 0),
        sigmaX=2.6,
        sigmaY=2.6,
    )[..., np.newaxis]
    alpha = np.clip(alpha * 0.96, 0.0, 1.0)
    out = refill.astype(np.float32) * alpha + current_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _cleanup_region_with_cloth_restore(
    self,
    *,
    source_rgb: np.ndarray,
    current_rgb: np.ndarray,
    cleanup_mask: np.ndarray,
    cloth_mask: Optional[np.ndarray] = None,
    final_hair_mask: Optional[np.ndarray] = None,
    ignore_final_hair_for_cloth_restore: bool = False,
    cleanup_dark_tail: bool = True,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or cleanup_mask.shape != (H, W):
        return current_rgb

    mask_u8 = (np.clip(cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((mask_u8 > 0).sum()) < 40:
        return current_rgb

    cleaned = self._lama_inpaint(current_rgb, mask_u8)
    if cleanup_dark_tail:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, mask_u8)

    if cloth_mask is None or cloth_mask.shape != (H, W):
        return cleaned

    cloth_cleanup_mask = np.clip(
        cleanup_mask.astype(np.float32)
        * (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.float32),
        0.0,
        1.0,
    )
    cloth_cleanup_u8 = (cloth_cleanup_mask > 0.08).astype(np.uint8) * 255
    if int((cloth_cleanup_u8 > 0).sum()) < 30:
        return cleaned

    restore_final_hair_mask = None if ignore_final_hair_for_cloth_restore else final_hair_mask
    reference_fill_u8 = cv2.dilate(
        cloth_cleanup_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25)),
        iterations=1,
    )
    if cloth_mask is not None and cloth_mask.shape == (H, W):
        wide_cloth_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        )
        reference_fill_u8 = cv2.bitwise_and(reference_fill_u8, wide_cloth_u8)
    reference_fill_mask = reference_fill_u8.astype(np.float32) / 255.0
    reference_fill_rgb = self._restore_cloth_overlap_from_source(
        source_rgb=source_rgb,
        current_rgb=source_rgb,
        restore_mask=reference_fill_mask,
        final_hair_mask=None,
        tone_reference_rgb=source_rgb,
        tone_reference_mask=cloth_mask,
    )
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    visible_cloth_u8 = cv2.bitwise_and(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.bitwise_not(
            cv2.dilate(
                cloth_cleanup_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                iterations=1,
            )
        ),
    )
    plain_gray = 0.0
    plain_sat = 255.0
    use_plain_cloth_force = False
    plain_fill_rgb: Optional[np.ndarray] = None
    if int((visible_cloth_u8 > 0).sum()) >= 80:
        plain_gray = float(np.median(source_gray[visible_cloth_u8 > 0]))
        plain_sat = float(np.median(source_sat[visible_cloth_u8 > 0]))
        use_plain_cloth_force = plain_gray >= 168.0 and plain_sat <= 84.0
    if use_plain_cloth_force:
        plain_fill_rgb = source_rgb.copy()
        plain_fill_color = np.median(source_rgb[visible_cloth_u8 > 0], axis=0).astype(np.uint8)
        plain_fill_rgb[reference_fill_u8 > 0] = plain_fill_color
        plain_fill_rgb = self._restore_reference_region(
            reference_fill_rgb,
            plain_fill_rgb,
            reference_fill_mask,
            strength=0.98,
        )
        reference_fill_rgb = self._blend_neighbor_cloth_tone(
            plain_fill_rgb,
            reference_fill_mask,
            cloth_mask=cloth_mask,
            reference_rgb=source_rgb,
        )
        reference_fill_rgb = self._cv2_refine_cloth_region(
            reference_fill_rgb,
            reference_fill_mask,
            reference_rgb=source_rgb,
            reference_mask=cloth_mask,
        )
    cleaned = self._restore_cloth_overlap_from_source(
        source_rgb=source_rgb,
        current_rgb=cleaned,
        restore_mask=cloth_cleanup_mask,
        final_hair_mask=restore_final_hair_mask,
        tone_reference_rgb=source_rgb,
        tone_reference_mask=cloth_mask,
    )
    if use_plain_cloth_force:
        cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY).astype(np.float32)
        cleaned_sat = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        cleanup_bool = cloth_cleanup_u8 > 0
        cleanup_dark_ratio = 0.0
        if int(cleanup_bool.sum()) >= 40:
            cleanup_dark_ratio = float(
                np.mean(
                    (
                        (cleaned_gray < plain_gray - 14.0)
                        & (cleaned_sat < 138.0)
                    )[cleanup_bool]
                )
            )
        if cleanup_dark_ratio >= 0.14:
            cleaned = self._restore_reference_region(
                cleaned,
                plain_fill_rgb if plain_fill_rgb is not None else reference_fill_rgb,
                cloth_cleanup_mask,
                strength=float(np.clip(0.88 + cleanup_dark_ratio * 0.24, 0.88, 0.98)),
            )
    cleaned = self._overlay_reference_cloth_fill(
        cleaned,
        reference_fill_rgb,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
    )
    cleaned = self._blend_neighbor_cloth_tone(
        cleaned,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
        reference_rgb=reference_fill_rgb,
    )
    cleaned = self._cv2_refine_cloth_region(
        cleaned,
        cloth_cleanup_mask,
        reference_rgb=reference_fill_rgb,
        reference_mask=cloth_mask,
    )
    if use_plain_cloth_force:
        cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY).astype(np.float32)
        cleaned_sat = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
        ref_gray = cv2.cvtColor(reference_fill_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dark_residual_u8 = (
            (
                (
                    (cleaned_gray + 12.0 < ref_gray)
                    & (cleaned_sat < 132.0)
                )
                | (
                    (cleaned_gray < plain_gray - 18.0)
                    & (cleaned_sat < 124.0)
                )
            ).astype(np.uint8)
            * 255
        )
        dark_residual_u8 = cv2.bitwise_and(dark_residual_u8, cloth_cleanup_u8)
        dark_residual_u8 = cv2.morphologyEx(
            dark_residual_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
        )
        dark_residual_u8 = cv2.dilate(
            dark_residual_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)),
            iterations=1,
        )
        if int((dark_residual_u8 > 0).sum()) >= 24:
            cleaned = self._restore_reference_region(
                cleaned,
                plain_fill_rgb if plain_fill_rgb is not None else reference_fill_rgb,
                dark_residual_u8.astype(np.float32) / 255.0,
                strength=1.0,
            )
    cleaned = self._overlay_reference_cloth_fill(
        cleaned,
        reference_fill_rgb,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
    )
    return cleaned

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
    center_support_mask: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
    exclusion_mask: Optional[np.ndarray] = None,
    subject_gender_mode: str = "",
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
    cx = int(0.5 * (x1 + x2))
    gender_mode = str(subject_gender_mode or "").strip().lower()
    lateral_only_short_restore = hair_length == "short" and gender_mode != "male"

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y - face_h * 0.04))
    bottom = min(H, int(cutoff_y + face_h * (1.72 if hair_length == "short" else 1.52)))
    left = max(0, int(x1 - face_w * (1.46 if hair_length == "short" else 1.34)))
    right = min(W, int(x2 + face_w * (1.46 if hair_length == "short" else 1.34)))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    exclusion_u8 = np.zeros((H, W), dtype=np.uint8)
    if exclusion_mask is not None and exclusion_mask.shape == (H, W):
        exclusion_u8 = (
            (np.clip(exclusion_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255
        )
        exclusion_u8 = cv2.dilate(
            exclusion_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (27, 45) if hair_length == "short" else (21, 35),
            ),
            iterations=1,
        )
        exclusion_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        exclusion_half_w = max(28, int(face_w * (0.42 if hair_length == "short" else 0.36)))
        exclusion_left = max(0, cx - exclusion_half_w)
        exclusion_right = min(W, cx + exclusion_half_w)
        exclusion_top = max(top, int(cutoff_y + face_h * 0.04))
        exclusion_bottom = min(bottom, int(cutoff_y + face_h * (1.34 if hair_length == "short" else 1.22)))
        if exclusion_top < exclusion_bottom and exclusion_left < exclusion_right:
            exclusion_gate_u8[exclusion_top:exclusion_bottom, exclusion_left:exclusion_right] = 255
            exclusion_u8 = cv2.bitwise_and(exclusion_u8, exclusion_gate_u8)

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (21, 21) if hair_length == "short" else (13, 13),
        ),
        iterations=1,
    )
    cloth_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    use_anchor_fallback = False
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = cv2.dilate(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (25, 25) if hair_length == "short" else (17, 17),
            ),
            iterations=1,
        )
        anchor_u8 = cv2.bitwise_and(anchor_u8, corridor_u8)
        if int((anchor_u8 > 0).sum()) >= 120:
            use_anchor_fallback = True
            anchor_lane_u8 = cv2.dilate(
                anchor_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        max(41, int(face_w * 1.12)) | 1,
                        max(21, int(face_h * 0.28)) | 1,
                    ),
                ),
                iterations=1,
            )
            lane_gate_u8 = np.zeros((H, W), dtype=np.uint8)
            lane_top = max(top, int(cutoff_y + face_h * 0.04))
            lane_bottom = min(bottom, int(cutoff_y + face_h * 1.50))
            lane_left = max(0, int(x1 - face_w * 1.18))
            lane_right = min(W, int(x2 + face_w * 1.18))
            if lane_top < lane_bottom and lane_left < lane_right:
                lane_gate_u8[lane_top:lane_bottom, lane_left:lane_right] = 255
                anchor_lane_u8 = cv2.bitwise_and(anchor_lane_u8, lane_gate_u8)
            else:
                anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    candidate_cloth_u8 = cv2.bitwise_or(cloth_u8, anchor_u8)
    if int((exclusion_u8 > 0).sum()) > 0:
        candidate_cloth_u8 = cv2.bitwise_and(candidate_cloth_u8, cv2.bitwise_not(exclusion_u8))
        anchor_lane_u8 = cv2.bitwise_and(anchor_lane_u8, cv2.bitwise_not(exclusion_u8))
    if int((candidate_cloth_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (19, 27) if hair_length == "short" else (23, 33),
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
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_blur = cv2.GaussianBlur(current_gray, (0, 0), sigmaX=4.2, sigmaY=4.2)
    current_sat = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    source_sat = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    gray_delta = np.abs(current_gray - source_gray)
    strong_diff_u8 = (
        (
            (diff_rgb > (26.0 if hair_length == "short" else 22.0))
            | (gray_delta > (24.0 if hair_length == "short" else 20.0))
        ).astype(np.uint8)
        * 255
    )
    if hair_length == "short":
        smooth_artifact_u8 = (
            (
                (
                    (
                        (current_gray + 12.0 < source_gray)
                        | (current_gray > source_gray + 18.0)
                        | (diff_rgb > 17.0)
                    )
                    & (current_sat < 142.0)
                    & (np.abs(current_gray - current_blur) < 10.0)
                )
            ).astype(np.uint8)
            * 255
        )
        muddy_cloth_u8 = (
            (
                (
                    (diff_rgb > 13.0)
                    | (gray_delta > 12.0)
                    | (current_sat > source_sat + 18.0)
                )
                & (np.abs(current_gray - current_blur) < 12.0)
                & (current_sat < 188.0)
            ).astype(np.uint8)
            * 255
        )
        deep_torso_u8 = np.zeros((H, W), dtype=np.uint8)
        deep_top = max(0, int(cutoff_y + face_h * 0.06))
        deep_bottom = min(H, int(cutoff_y + face_h * 1.52))
        deep_left = max(0, int(x1 - face_w * 1.34))
        deep_right = min(W, int(x2 + face_w * 1.34))
        if deep_top < deep_bottom and deep_left < deep_right:
            deep_torso_u8[deep_top:deep_bottom, deep_left:deep_right] = 255
        muddy_cloth_u8 = cv2.bitwise_and(muddy_cloth_u8, deep_torso_u8)
        strong_diff_u8 = cv2.bitwise_or(strong_diff_u8, smooth_artifact_u8)
        strong_diff_u8 = cv2.bitwise_or(strong_diff_u8, muddy_cloth_u8)
        if center_support_mask is not None and center_support_mask.shape == (H, W):
            center_support_u8 = cv2.dilate(
                (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 31)),
                iterations=1,
            )
            strong_diff_u8 = cv2.bitwise_or(
                strong_diff_u8,
                cv2.bitwise_and(center_support_u8, cv2.bitwise_and(candidate_cloth_u8, removal_u8)),
            )
    strong_diff_u8 = cv2.bitwise_and(strong_diff_u8, candidate_cloth_u8)
    strong_diff_u8 = cv2.bitwise_and(strong_diff_u8, removal_u8)
    if int((strong_diff_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

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
    keep_u8 = cv2.bitwise_and(keep_u8, candidate_cloth_u8)
    if int((exclusion_u8 > 0).sum()) > 0:
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(exclusion_u8))
    pre_tone_keep_u8 = keep_u8.copy()

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
        toned_keep_u8 = cv2.bitwise_and(keep_u8, tone_match_u8)
        toned_px = int((toned_keep_u8 > 0).sum())
        pre_tone_px = int((pre_tone_keep_u8 > 0).sum())
        if hair_length == "short" and pre_tone_px >= 80 and toned_px < max(80, int(pre_tone_px * 0.30)):
            keep_u8 = pre_tone_keep_u8
        else:
            keep_u8 = toned_keep_u8

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
            final_hair_u8[min(H, int(cutoff_y + face_h * 0.40)):, :] = 0
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(final_hair_u8))

    short_lateral_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    short_restore_gate_u8 = np.zeros((H, W), dtype=np.uint8)
    if lateral_only_short_restore:
        lane_top = max(top, int(cutoff_y + face_h * 0.06))
        lane_bottom = min(bottom, int(cutoff_y + face_h * 1.54))
        left_lane_left = max(0, int(x1 - face_w * 1.18))
        left_lane_right = min(W, int(x1 + face_w * 0.20))
        right_lane_left = max(0, int(x2 - face_w * 0.20))
        right_lane_right = min(W, int(x2 + face_w * 1.18))
        if lane_top < lane_bottom and left_lane_left < left_lane_right:
            short_lateral_lane_u8[lane_top:lane_bottom, left_lane_left:left_lane_right] = 255
        if lane_top < lane_bottom and right_lane_left < right_lane_right:
            short_lateral_lane_u8[lane_top:lane_bottom, right_lane_left:right_lane_right] = 255
        short_restore_gate_u8 = short_lateral_lane_u8.copy()
        if center_support_mask is not None and center_support_mask.shape == (H, W):
            center_support_u8 = cv2.dilate(
                (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 41)),
                iterations=1,
            )
            center_lane_u8 = np.zeros((H, W), dtype=np.uint8)
            center_lane_half_w = max(26, int(face_w * 0.20))
            center_lane_left = max(0, cx - center_lane_half_w)
            center_lane_right = min(W, cx + center_lane_half_w)
            center_lane_top = max(top, int(cutoff_y + face_h * 0.06))
            center_lane_bottom = min(bottom, int(cutoff_y + face_h * 1.46))
            if center_lane_top < center_lane_bottom and center_lane_left < center_lane_right:
                center_lane_u8[center_lane_top:center_lane_bottom, center_lane_left:center_lane_right] = 255
                center_support_u8 = cv2.bitwise_and(center_support_u8, center_lane_u8)
                center_support_u8 = cv2.bitwise_and(center_support_u8, candidate_cloth_u8)
                if int((center_support_u8 > 0).sum()) >= 24:
                    short_restore_gate_u8 = cv2.bitwise_or(short_restore_gate_u8, center_support_u8)
        center_dark_column_gate_u8 = np.zeros((H, W), dtype=np.uint8)
        center_column_half_w = max(24, int(face_w * 0.28))
        center_column_left = max(0, cx - center_column_half_w)
        center_column_right = min(W, cx + center_column_half_w)
        center_column_top = max(top, int(cutoff_y + face_h * 0.02))
        center_column_bottom = min(bottom, int(cutoff_y + face_h * 1.42))
        if (
            center_column_top < center_column_bottom
            and center_column_left < center_column_right
        ):
            center_dark_column_gate_u8[
                center_column_top:center_column_bottom,
                center_column_left:center_column_right,
            ] = 255
            center_dark_column_u8 = (
                (
                    (
                        (current_gray + 14.0 < source_gray)
                        | (gray_delta > 12.0)
                        | (diff_rgb > 11.0)
                        | (
                            ((current_blur - current_gray) > 2.0)
                            & (diff_rgb > 8.5)
                        )
                    )
                    & (current_sat < 152.0)
                ).astype(np.uint8)
                * 255
            )
            center_dark_column_u8 = cv2.bitwise_and(
                center_dark_column_u8,
                center_dark_column_gate_u8,
            )
            center_dark_column_u8 = cv2.bitwise_and(
                center_dark_column_u8,
                candidate_cloth_u8,
            )
            center_dark_column_u8 = cv2.bitwise_and(
                center_dark_column_u8,
                removal_u8,
            )
            center_dark_column_u8 = cv2.morphologyEx(
                center_dark_column_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 27)),
            )
            center_dark_column_u8 = cv2.dilate(
                center_dark_column_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13)),
                iterations=1,
            )
            if int((center_dark_column_u8 > 0).sum()) >= 24:
                filtered_center_dark_u8 = np.zeros((H, W), dtype=np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    center_dark_column_u8,
                    8,
                )
                max_center_width = max(64, int(face_w * 0.34))
                min_center_height = max(26, int(face_h * 0.18))
                min_center_area = max(24, int(face_w * face_h * 0.003))
                max_center_offset = max(18, int(face_w * 0.18))
                for idx in range(1, num_labels):
                    x = int(stats[idx, cv2.CC_STAT_LEFT])
                    y = int(stats[idx, cv2.CC_STAT_TOP])
                    w = int(stats[idx, cv2.CC_STAT_WIDTH])
                    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    bottom_y = y + h
                    comp_cx = float(centroids[idx][0])
                    if area < min_center_area:
                        continue
                    if w > max_center_width or h < min_center_height:
                        continue
                    if bottom_y < int(cutoff_y + face_h * 0.20):
                        continue
                    if abs(comp_cx - cx) > max_center_offset:
                        continue
                    filtered_center_dark_u8[labels == idx] = 255
                if int((filtered_center_dark_u8 > 0).sum()) >= 24:
                    filtered_center_dark_u8 = cv2.dilate(
                        filtered_center_dark_u8,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)),
                        iterations=1,
                    )
                    short_restore_gate_u8 = cv2.bitwise_or(
                        short_restore_gate_u8,
                        filtered_center_dark_u8,
                    )
        if int((short_restore_gate_u8 > 0).sum()) > 0:
            keep_u8 = cv2.bitwise_and(keep_u8, short_restore_gate_u8)
            pre_tone_keep_u8 = cv2.bitwise_and(pre_tone_keep_u8, short_restore_gate_u8)

    if int((keep_u8 > 0).sum()) < 80:
        return np.zeros((H, W), dtype=np.float32)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(keep_u8, 8)
    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    min_area = max(80, int(face_w * face_h * 0.010))
    max_area = (
        max(28000, int(face_w * face_h * 1.60))
        if use_anchor_fallback
        else (
            max(18000, int(face_w * face_h * 0.92))
            if lateral_only_short_restore
            else max(3600, int(face_w * face_h * 0.44))
        )
    )
    max_width = (
        max(320, int(face_w * 2.10))
        if use_anchor_fallback
        else (
            max(260, int(face_w * 1.72))
            if lateral_only_short_restore
            else max(120, int(face_w * 1.26))
        )
    )
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
            (11, 19) if hair_length == "short" else (13, 21),
        ),
        iterations=1,
    )
    filtered_u8 = cv2.bitwise_and(filtered_u8, candidate_cloth_u8)
    if lateral_only_short_restore and int((short_restore_gate_u8 > 0).sum()) > 0:
        filtered_u8 = cv2.bitwise_and(filtered_u8, short_restore_gate_u8)
    if int((exclusion_u8 > 0).sum()) > 0:
        filtered_u8 = cv2.bitwise_and(filtered_u8, cv2.bitwise_not(exclusion_u8))
    if hair_length == "short":
        pre_trim_filtered_u8 = filtered_u8.copy()
        filtered_u8 = self._trim_blocky_short_restore_mask_u8(
            mask_u8=filtered_u8,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            min_keep_px=80,
        )
        if int((filtered_u8 > 0).sum()) < 80 and int((pre_trim_filtered_u8 > 0).sum()) >= 80:
            filtered_u8 = pre_trim_filtered_u8
    if int((filtered_u8 > 0).sum()) < 80:
        if not use_anchor_fallback or int((pre_tone_keep_u8 > 0).sum()) < 120:
            return np.zeros((H, W), dtype=np.float32)
        fallback_u8 = self._trim_blocky_short_restore_mask_u8(
            mask_u8=pre_tone_keep_u8,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            min_keep_px=80,
        )
        fallback_px = int((fallback_u8 > 0).sum())
        if fallback_px < 80 or fallback_px > max(9000, int(face_w * face_h * 0.34)):
            return np.zeros((H, W), dtype=np.float32)
        filtered_u8 = fallback_u8
    return (filtered_u8 > 0).astype(np.float32)

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
    anchor_mask: Optional[np.ndarray] = None,
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

    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    use_anchor_fallback = False
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = cv2.dilate(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 27)),
            iterations=1,
        )
        anchor_u8 = cv2.bitwise_and(anchor_u8, corridor_u8)
        if int((anchor_u8 > 0).sum()) >= 120:
            use_anchor_fallback = True
            anchor_lane_u8 = cv2.dilate(
                anchor_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        max(41, int(face_w * 1.18)) | 1,
                        max(19, int(face_h * 0.24)) | 1,
                    ),
                ),
                iterations=1,
            )
            lane_gate_u8 = np.zeros((H, W), dtype=np.uint8)
            lane_top = max(top, int(cutoff_y + face_h * 0.04))
            lane_bottom = min(bottom, int(cutoff_y + face_h * 1.34))
            lane_left = max(0, int(x1 - face_w * 1.18))
            lane_right = min(W, int(x2 + face_w * 1.18))
            if lane_top < lane_bottom and lane_left < lane_right:
                lane_gate_u8[lane_top:lane_bottom, lane_left:lane_right] = 255
                anchor_lane_u8 = cv2.bitwise_and(anchor_lane_u8, lane_gate_u8)
            else:
                anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
            zone_u8 = cv2.bitwise_or(zone_u8, cv2.bitwise_and(removal_u8, anchor_u8))
            zone_u8 = cv2.bitwise_or(zone_u8, cv2.bitwise_and(removal_u8, anchor_lane_u8))

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
    if use_anchor_fallback:
        deep_zone_u8 = cv2.bitwise_or(
            deep_zone_u8,
            cv2.bitwise_and(zone_u8, anchor_u8),
        )
        deep_zone_u8 = cv2.bitwise_or(
            deep_zone_u8,
            cv2.bitwise_and(zone_u8, anchor_lane_u8),
        )

    dark_tail_u8 = (
        ((gray < 156.0) & ((blur - gray) > 2.2)).astype(np.uint8) * 255
    )
    dark_tail_u8 = cv2.bitwise_and(dark_tail_u8, zone_u8)
    anchor_lane_diff_u8 = np.zeros((H, W), dtype=np.uint8)
    column_rescue_u8 = np.zeros((H, W), dtype=np.uint8)
    if use_anchor_fallback:
        anchor_lane_diff_u8 = (
            (
                ((diff_rgb > 11.0) | (gray_delta > 11.0))
                & (lap < 14.0)
                & (gray < 184.0)
            ).astype(np.uint8)
            * 255
        )
        anchor_lane_diff_u8 = cv2.bitwise_and(anchor_lane_diff_u8, zone_u8)
        anchor_lane_diff_u8 = cv2.bitwise_and(anchor_lane_diff_u8, anchor_lane_u8)
        column_seed_u8 = (
            (
                ((diff_rgb > 9.0) | (gray_delta > 9.0))
                & (lap < 18.0)
                & (gray < 196.0)
            ).astype(np.uint8)
            * 255
        )
        column_seed_u8 = cv2.bitwise_and(column_seed_u8, zone_u8)
        column_seed_u8 = cv2.bitwise_and(column_seed_u8, anchor_lane_u8)
        column_seed_u8 = cv2.morphologyEx(
            column_seed_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)),
        )
        column_seed_u8 = cv2.morphologyEx(
            column_seed_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 31)),
        )
        num_col_labels, col_labels, col_stats, col_centroids = cv2.connectedComponentsWithStats(
            column_seed_u8,
            8,
        )
        min_col_area = max(72, int(face_w * face_h * 0.004))
        min_col_height = max(56, int(face_h * 0.28))
        max_col_width = max(132, int(face_w * 0.92))
        max_col_offset = max(260, int(face_w * 1.34))
        for idx in range(1, num_col_labels):
            x = int(col_stats[idx, cv2.CC_STAT_LEFT])
            y = int(col_stats[idx, cv2.CC_STAT_TOP])
            w = int(col_stats[idx, cv2.CC_STAT_WIDTH])
            h = int(col_stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(col_stats[idx, cv2.CC_STAT_AREA])
            bottom_y = y + h
            comp_cx = float(col_centroids[idx][0])
            if area < min_col_area:
                continue
            if h < min_col_height or w > max_col_width:
                continue
            if bottom_y < int(cutoff_y + face_h * 0.22):
                continue
            if abs(comp_cx - cx) > max_col_offset:
                continue
            column_rescue_u8[col_labels == idx] = 255
        if int((column_rescue_u8 > 0).sum()) >= 60:
            column_rescue_u8 = cv2.dilate(
                column_rescue_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 27)),
                iterations=1,
            )

    keep_u8 = cv2.bitwise_or(low_texture_u8, dark_tail_u8)
    keep_u8 = cv2.bitwise_or(keep_u8, deep_zone_u8)
    keep_u8 = cv2.bitwise_or(keep_u8, anchor_lane_diff_u8)
    keep_u8 = cv2.bitwise_or(keep_u8, column_rescue_u8)
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
    if use_anchor_fallback:
        keep_u8 = cv2.bitwise_or(keep_u8, cv2.bitwise_and(deep_zone_u8, anchor_u8))
        keep_u8 = cv2.bitwise_or(keep_u8, cv2.bitwise_and(deep_zone_u8, anchor_lane_u8))

    upper_guard_u8 = np.zeros((H, W), dtype=np.uint8)
    upper_guard_bottom = min(H, int(cutoff_y + face_h * 0.32))
    upper_guard_left = max(0, int(cx - face_w * 0.74))
    upper_guard_right = min(W, int(cx + face_w * 0.74))
    if top < upper_guard_bottom and upper_guard_left < upper_guard_right:
        upper_guard_u8[top:upper_guard_bottom, upper_guard_left:upper_guard_right] = 255
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(upper_guard_u8))

    anchor_fallback_u8 = np.zeros((H, W), dtype=np.uint8)
    if use_anchor_fallback:
        anchor_fallback_u8 = cv2.bitwise_and(zone_u8, cv2.bitwise_or(anchor_u8, anchor_lane_u8))
        anchor_fallback_u8 = cv2.morphologyEx(
            anchor_fallback_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 35)),
        )
        anchor_fallback_u8 = cv2.dilate(
            anchor_fallback_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 31)),
            iterations=1,
        )
        anchor_fallback_u8 = cv2.bitwise_and(anchor_fallback_u8, cv2.bitwise_not(upper_guard_u8))

    if protect_mask is not None and protect_mask.shape == (H, W):
        protect_u8 = cv2.dilate(
            (np.clip(protect_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )
        keep_u8 = cv2.bitwise_and(keep_u8, cv2.bitwise_not(protect_u8))
        anchor_fallback_u8 = cv2.bitwise_and(anchor_fallback_u8, cv2.bitwise_not(protect_u8))
        column_rescue_u8 = cv2.bitwise_and(column_rescue_u8, cv2.bitwise_not(protect_u8))

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
        anchor_fallback_u8 = cv2.bitwise_and(anchor_fallback_u8, cv2.bitwise_not(hair_protect_u8))
        column_rescue_u8 = cv2.bitwise_and(column_rescue_u8, cv2.bitwise_not(hair_protect_u8))

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
    max_width = max(520, int(face_w * 2.60))
    max_offset = max(340, int(face_w * 1.80))
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
        if int((column_rescue_u8 > 0).sum()) >= 120:
            filtered_u8 = column_rescue_u8.copy()
        elif int((anchor_fallback_u8 > 0).sum()) >= 120:
            filtered_u8 = anchor_fallback_u8.copy()
        else:
            if int((keep_u8 > 0).sum()) < 160:
                return np.zeros((H, W), dtype=np.float32)
            filtered_u8 = keep_u8.copy()

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

def _build_side_column_cloth_restore_mask(
    self,
    img_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    candidate_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
    subject_gender_mode: str = "",
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
    gender_mode = str(subject_gender_mode or "").strip().lower()
    lateral_only_short_restore = hair_length == "short" and gender_mode != "male"
    short_lateral_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    if lateral_only_short_restore:
        smooth_u8 = cv2.bitwise_or(bright_smooth_u8, dark_smooth_u8)
        left_lane_left = max(left, int(x1 - face_w * 0.74))
        left_lane_right = min(right, int(x1 + face_w * 0.18))
        right_lane_left = max(left, int(x2 - face_w * 0.18))
        right_lane_right = min(right, int(x2 + face_w * 0.74))
        if left_lane_left < left_lane_right:
            short_lateral_lane_u8[top:bottom, left_lane_left:left_lane_right] = 255
        if right_lane_left < right_lane_right:
            short_lateral_lane_u8[top:bottom, right_lane_left:right_lane_right] = 255
        zone_u8 = cv2.bitwise_and(zone_u8, short_lateral_lane_u8)
        if int((zone_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    min_area = max(120, int(face_w * face_h * 0.014))
    max_area = max(
        7600 if lateral_only_short_restore and hair_length == "short" else 4200,
        int(
            face_w
            * face_h
            * (
                0.34
                if lateral_only_short_restore and hair_length == "short"
                else (0.22 if hair_length == "short" else 0.34)
            )
        ),
    )
    min_height = max(36, int(face_h * 0.12))
    max_width = max(
        176 if lateral_only_short_restore and hair_length == "short" else 120,
        int(
            face_w
            * (
                1.02
                if lateral_only_short_restore and hair_length == "short"
                else (0.78 if hair_length == "short" else 0.54)
            )
        ),
    )
    max_offset = max(120, int(face_w * (0.98 if hair_length == "short" else 0.60)))
    short_center_offset = max(24, int(face_w * 0.30))
    short_lateral_center_offset = max(34, int(face_w * 0.38))
    short_center_width = max(76, int(face_w * 0.34))
    short_center_area = max(1800, int(face_w * face_h * 0.09))
    short_side_width = max(
        148 if lateral_only_short_restore and hair_length == "short" else 96,
        int(face_w * (0.92 if lateral_only_short_restore and hair_length == "short" else 0.56)),
    )
    short_side_area = max(
        7200 if lateral_only_short_restore and hair_length == "short" else 3600,
        int(face_w * face_h * (0.30 if lateral_only_short_restore and hair_length == "short" else 0.18)),
    )
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
            if lateral_only_short_restore and offset <= short_lateral_center_offset:
                continue
            if offset <= short_center_offset:
                if w > short_center_width or area > short_center_area:
                    continue
            else:
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
    if lateral_only_short_restore:
        keep_u8 = cv2.bitwise_and(keep_u8, short_lateral_lane_u8)
    if hair_length == "short":
        keep_u8 = self._trim_blocky_short_restore_mask_u8(
            mask_u8=keep_u8,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            min_keep_px=60,
        )
    if int((keep_u8 > 0).sum()) < 60:
        return np.zeros((H, W), dtype=np.float32)

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
    subject_gender_mode: str = "",
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
    gender_mode = str(subject_gender_mode or "").strip().lower()

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
    lateral_only_short_restore = gender_mode != "male"
    def _collect_direct_mask(
        *,
        lane_inner_mul: float,
        lane_outer_mul: float,
        min_area_ratio: float,
        min_height_mul: float,
        min_height_floor: int,
        max_width_mul: float,
        max_width_floor: int,
        center_reject_mul: float,
        center_reject_floor: int,
        lateral_center_reject_mul: float,
        lateral_center_reject_floor: int,
        trim_min_keep_px: int,
    ) -> np.ndarray:
        local_zone_u8 = zone_u8.copy()
        short_lateral_lane_u8 = np.zeros((H, W), dtype=np.uint8)
        if lateral_only_short_restore:
            left_lane_left = max(left, int(x1 - face_w * lane_outer_mul))
            left_lane_right = min(right, int(x1 + face_w * lane_inner_mul))
            right_lane_left = max(left, int(x2 - face_w * lane_inner_mul))
            right_lane_right = min(right, int(x2 + face_w * lane_outer_mul))
            if left_lane_left < left_lane_right:
                short_lateral_lane_u8[top:bottom, left_lane_left:left_lane_right] = 255
            if right_lane_left < right_lane_right:
                short_lateral_lane_u8[top:bottom, right_lane_left:right_lane_right] = 255
            local_zone_u8 = cv2.bitwise_and(local_zone_u8, short_lateral_lane_u8)
            if int((local_zone_u8 > 0).sum()) < 40:
                return np.zeros((H, W), dtype=np.uint8)

        local_zone_u8 = cv2.morphologyEx(
            local_zone_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        local_zone_u8 = cv2.morphologyEx(
            local_zone_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 13)),
        )

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(local_zone_u8, 8)
        min_area = max(120, int(face_w * face_h * min_area_ratio))
        max_area = max(9200, int(face_w * face_h * 0.34))
        min_height = max(min_height_floor, int(face_h * min_height_mul))
        max_width = max(max_width_floor, int(face_w * max_width_mul))
        max_offset = max(180, int(face_w * 0.98))
        center_reject_offset = max(center_reject_floor, int(face_w * center_reject_mul))
        lateral_center_reject_offset = max(
            lateral_center_reject_floor,
            int(face_w * lateral_center_reject_mul),
        )
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
            if lateral_only_short_restore and abs(comp_cx - cx) <= lateral_center_reject_offset:
                continue
            if (
                abs(comp_cx - cx) <= center_reject_offset
                and w > max(84, int(face_w * 0.38))
                and area > max(1600, int(face_w * face_h * 0.06))
            ):
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < trim_min_keep_px:
            return np.zeros((H, W), dtype=np.uint8)

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
        if lateral_only_short_restore:
            keep_u8 = cv2.bitwise_and(keep_u8, short_lateral_lane_u8)
        keep_u8 = self._trim_blocky_short_restore_mask_u8(
            mask_u8=keep_u8,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            min_keep_px=trim_min_keep_px,
        )
        if int((keep_u8 > 0).sum()) < trim_min_keep_px:
            return np.zeros((H, W), dtype=np.uint8)
        return keep_u8

    keep_u8 = _collect_direct_mask(
        lane_inner_mul=0.08,
        lane_outer_mul=0.72,
        min_area_ratio=0.010,
        min_height_mul=0.28,
        min_height_floor=72,
        max_width_mul=0.78,
        max_width_floor=128,
        center_reject_mul=0.30,
        center_reject_floor=26,
        lateral_center_reject_mul=0.40,
        lateral_center_reject_floor=38,
        trim_min_keep_px=80,
    )
    if lateral_only_short_restore and int((keep_u8 > 0).sum()) < 80:
        keep_u8 = _collect_direct_mask(
            lane_inner_mul=0.18,
            lane_outer_mul=0.76,
            min_area_ratio=0.008,
            min_height_mul=0.20,
            min_height_floor=48,
            max_width_mul=0.94,
            max_width_floor=148,
            center_reject_mul=0.26,
            center_reject_floor=24,
            lateral_center_reject_mul=0.34,
            lateral_center_reject_floor=34,
            trim_min_keep_px=60,
        )
    if int((keep_u8 > 0).sum()) < 60:
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
    subject_gender_mode: str = "",
    support_mask: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
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
    gender_mode = str(subject_gender_mode or "").strip().lower()
    lateral_only_short_restore = gender_mode != "male"

    removal_u8 = (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = cv2.dilate(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 23)),
            iterations=1,
        )
        cloth_u8 = cv2.bitwise_or(cloth_u8, anchor_u8)
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
    left = max(0, int(x1 - face_w * 1.38))
    right = min(W, int(x2 + face_w * 1.38))
    bottom = min(H, int(cutoff_y + face_h * 1.92))
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
    if not lateral_only_short_restore:
        center_lane_top = min(bottom, int(cutoff_y + face_h * 0.34))
        center_half = max(18, int(face_w * 0.20))
        center_x1 = max(left, int(cx - center_half))
        center_x2 = min(right, int(cx + center_half))
        if center_lane_top < bottom and center_x1 < center_x2:
            lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255

    pre_lane_zone_u8 = zone_u8.copy()
    zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
    if int((zone_u8 > 0).sum()) < 60:
        if lateral_only_short_restore:
            return np.zeros((H, W), dtype=np.float32)
        if int((pre_lane_zone_u8 > 0).sum()) < 120:
            return np.zeros((H, W), dtype=np.float32)
        zone_u8 = pre_lane_zone_u8

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
    max_area = max(22000, int(face_w * face_h * 0.48))
    min_height = max(18, int(face_h * 0.10))
    max_width = max(236, int(face_w * 1.44))
    center_keepout = max(16, int(face_w * 0.16))
    deep_center_bottom = int(y2 + face_h * 0.78)
    max_offset = max(244, int(face_w * 1.30))
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
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)),
        iterations=1,
    )
    pre_trim_keep_u8 = cv2.bitwise_and(keep_u8, lane_u8)
    if int((pre_trim_keep_u8 > 0).sum()) < 40 and int((keep_u8 > 0).sum()) >= 40:
        pre_trim_keep_u8 = keep_u8.copy()
    keep_u8 = pre_trim_keep_u8
    keep_u8 = self._trim_blocky_short_restore_mask_u8(
        mask_u8=keep_u8,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        min_keep_px=40,
    )
    if int((keep_u8 > 0).sum()) < 40:
        if int((pre_trim_keep_u8 > 0).sum()) < 40:
            return np.zeros((H, W), dtype=np.float32)
        keep_u8 = pre_trim_keep_u8

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
    anchor_mask: Optional[np.ndarray] = None,
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
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)),
        iterations=1,
    )
    zone_u8 = removal_u8.copy()
    if support_mask is not None and support_mask.shape == (H, W):
        support_u8 = cv2.dilate(
            (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_or(zone_u8, support_u8)
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = cv2.dilate(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 25)),
            iterations=1,
        )
        zone_u8 = cv2.bitwise_or(zone_u8, cv2.bitwise_and(removal_u8, anchor_u8))

    bob_floor = max(
        0,
        min(
            int(y2 + face_h * 0.02),
            int(cutoff_y + face_h * 0.04),
        ),
    )
    bottom = min(H, int(cutoff_y + face_h * 1.72))
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
    side_inner_gap = max(12, int(face_w * 0.06))
    left_lane_right = max(left + 1, int(cx - side_inner_gap))
    right_lane_left = min(right - 1, int(cx + side_inner_gap))
    lane_u8[bob_floor:bottom, left:left_lane_right] = 255
    lane_u8[bob_floor:bottom, right_lane_left:right] = 255
    center_lane_top = min(bottom, int(cutoff_y + face_h * 0.28))
    center_half = max(20, int(face_w * 0.22))
    center_x1 = max(left, int(cx - center_half))
    center_x2 = min(right, int(cx + center_half))
    if center_lane_top < bottom and center_x1 < center_x2:
        lane_u8[center_lane_top:bottom, center_x1:center_x2] = 255
    pre_lane_zone_u8 = zone_u8.copy()
    zone_u8 = cv2.bitwise_and(zone_u8, lane_u8)
    if int((zone_u8 > 0).sum()) < 60:
        if int((pre_lane_zone_u8 > 0).sum()) < 120:
            return np.zeros((H, W), dtype=np.float32)
        zone_u8 = pre_lane_zone_u8

    zone_u8 = cv2.morphologyEx(
        zone_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 27)),
    )
    zone_u8 = cv2.dilate(
        zone_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
        iterations=1,
    )
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    use_lane_mask = True
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(zone_u8, 8)
    min_area = max(36, int(face_w * face_h * 0.0018))
    max_area = max(32000, int(face_w * face_h * 0.56))
    min_height = max(20, int(face_h * 0.10))
    max_width = max(360, int(face_w * 2.04))
    center_keepout = max(20, int(face_w * 0.18))
    deep_center_bottom = int(y2 + face_h * 0.64)

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

    if int((keep_u8 > 0).sum()) < 40:
        if int((zone_u8 > 0).sum()) < 120:
            return np.zeros((H, W), dtype=np.float32)
        keep_u8 = zone_u8.copy()
        use_lane_mask = False

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 33)),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, lane_u8 if use_lane_mask else corridor_u8)
    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.4,
        sigmaY=4.2,
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
    top = max(0, int(cutoff_y + face_h * (0.10 if hair_length == "short" else 0.08)))
    bottom = min(H, int(cutoff_y + face_h * (1.38 if hair_length == "short" else 1.20)))
    left = max(0, int(x1 - face_w * 1.28))
    right = min(W, int(x2 + face_w * 1.28))
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
    max_area = max(1400, int(face_w * face_h * 0.090))
    max_width = max(32, int(face_w * 0.24))
    max_height = max(220, int(face_h * 0.96))
    max_offset = max(220, int(face_w * 1.02))
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
        if fill_ratio > 0.82 and area > 44:
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 8:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)),
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
    anchor_mask: Optional[np.ndarray] = None,
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
    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    use_anchor_fallback = False
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = cv2.dilate(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 25)),
            iterations=1,
        )
        if int((anchor_u8 > 0).sum()) >= 80:
            use_anchor_fallback = True
            anchor_lane_u8 = cv2.dilate(
                anchor_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        max(41, int(face_w * 1.18)) | 1,
                        max(17, int(face_h * 0.24)) | 1,
                    ),
                ),
                iterations=1,
            )
    cloth_u8 = cv2.bitwise_or(cloth_u8, anchor_u8)
    cloth_u8 = cv2.bitwise_or(cloth_u8, anchor_lane_u8)
    zone_u8 = cv2.bitwise_and(hair_u8, cloth_u8)
    zone_u8 = cv2.bitwise_and(zone_u8, removal_u8)
    if int((zone_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.14))
    bottom = min(H, int(cutoff_y + face_h * 1.22))
    left = max(0, int(x1 - face_w * 1.04))
    right = min(W, int(x2 + face_w * 1.04))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255
    zone_u8 = cv2.bitwise_and(zone_u8, corridor_u8)
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
    max_area = (
        max(12000, int(face_w * face_h * 0.56))
        if use_anchor_fallback
        else max(2200, int(face_w * face_h * 0.10))
    )
    max_width = (
        max(168, int(face_w * 0.96))
        if use_anchor_fallback
        else max(58, int(face_w * 0.30))
    )
    min_height = max(24, int(face_h * 0.12))
    max_offset = max(240, int(face_w * 1.00))
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
        if bottom_y < int(cutoff_y + face_h * 0.26):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        keep_u8[labels == idx] = 255

    if int((keep_u8 > 0).sum()) < 24:
        if use_anchor_fallback and int((zone_u8 > 0).sum()) >= 80:
            fallback_u8 = cv2.bitwise_and(zone_u8, anchor_lane_u8)
            fallback_u8 = self._trim_blocky_short_restore_mask_u8(
                mask_u8=fallback_u8,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                min_keep_px=24,
            )
            fallback_px = int((fallback_u8 > 0).sum())
            if 24 <= fallback_px <= max(2800, int(face_w * face_h * 0.14)):
                keep_u8 = fallback_u8
            else:
                return np.zeros((H, W), dtype=np.float32)
        else:
            return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 19) if use_anchor_fallback else (5, 13),
        ),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, corridor_u8)
    if int((keep_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)
    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=1.6,
        sigmaY=2.8,
    ).astype(np.float32)

def _build_center_residual_detector_mask(
    self,
    *,
    current_rgb: np.ndarray,
    source_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    removal_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
    final_hair_mask: Optional[np.ndarray] = None,
    center_support_mask: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(current_rgb.shape[:2], dtype=np.float32)

    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask is None or removal_mask is None:
        return np.zeros((H, W), dtype=np.float32)
    if cloth_mask.shape != (H, W) or removal_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)
    if final_hair_mask is not None and final_hair_mask.shape != (H, W):
        final_hair_mask = None
    if center_support_mask is not None and center_support_mask.shape != (H, W):
        center_support_mask = None
    if anchor_mask is not None and anchor_mask.shape != (H, W):
        anchor_mask = None

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    lane_half = max(20, int(face_w * (0.34 if hair_length == "short" else 0.28)))
    lane_top = max(0, int(cutoff_y + face_h * 0.02))
    lane_bottom = min(H, int(cutoff_y + face_h * (1.34 if hair_length == "short" else 1.12)))
    lane_left = max(0, cx - lane_half)
    lane_right = min(W, cx + lane_half)
    if lane_top >= lane_bottom or lane_left >= lane_right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[lane_top:lane_bottom, lane_left:lane_right] = 255

    deep_top = max(lane_top, int(cutoff_y + face_h * 0.24))
    deep_bottom = min(H, int(cutoff_y + face_h * (1.52 if hair_length == "short" else 1.24)))
    deep_half = max(26, int(face_w * (0.54 if hair_length == "short" else 0.42)))
    deep_left = max(0, cx - deep_half)
    deep_right = min(W, cx + deep_half)
    if deep_top < deep_bottom and deep_left < deep_right:
        corridor_u8[deep_top:deep_bottom, deep_left:deep_right] = 255

    removal_u8 = cv2.dilate(
        (np.clip(removal_mask.astype(np.float32), 0.0, 1.0) > 0.05).astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (19, 27) if hair_length == "short" else (17, 23),
        ),
        iterations=1,
    )
    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (19, 19) if hair_length == "short" else (15, 15),
        ),
        iterations=1,
    )
    support_u8 = np.zeros((H, W), dtype=np.uint8)
    if center_support_mask is not None:
        support_u8 = cv2.dilate(
            (np.clip(center_support_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (17, 35) if hair_length == "short" else (13, 27),
            ),
            iterations=1,
        )
    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    if anchor_mask is not None:
        anchor_u8 = cv2.dilate(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (23, 29) if hair_length == "short" else (17, 23),
            ),
            iterations=1,
        )
        anchor_u8 = cv2.bitwise_and(anchor_u8, corridor_u8)
        if int((anchor_u8 > 0).sum()) >= 80:
            anchor_lane_u8 = cv2.dilate(
                anchor_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        max(41, int(face_w * (1.18 if hair_length == "short" else 0.92))) | 1,
                        max(19, int(face_h * (0.28 if hair_length == "short" else 0.22))) | 1,
                    ),
                ),
                iterations=1,
            )
            anchor_lane_u8 = cv2.bitwise_and(anchor_lane_u8, corridor_u8)

    guide_lane_u8 = cv2.bitwise_or(support_u8, cv2.bitwise_or(anchor_u8, anchor_lane_u8))
    zone_u8 = cv2.bitwise_and(
        corridor_u8,
        cv2.bitwise_and(
            removal_u8,
            cv2.bitwise_or(cloth_u8, guide_lane_u8),
        ),
    )
    if int((zone_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    current_blur = cv2.GaussianBlur(current_gray, (0, 0), sigmaX=5.2, sigmaY=5.2)
    current_hsv = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV)
    source_hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
    current_sat = current_hsv[:, :, 1].astype(np.float32)
    source_sat = source_hsv[:, :, 1].astype(np.float32)
    diff_rgb = np.abs(current_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=2)
    gray_delta = np.abs(current_gray - source_gray)
    lap = np.abs(cv2.Laplacian(current_gray, cv2.CV_32F, ksize=3))
    blackhat = cv2.morphologyEx(
        current_gray.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 19) if hair_length == "short" else (5, 15),
        ),
    ).astype(np.float32)

    diff_candidate_u8 = (
        (
            (diff_rgb > (11.0 if hair_length == "short" else 10.0))
            | (gray_delta > (10.0 if hair_length == "short" else 9.0))
            | (np.abs(current_sat - source_sat) > (16.0 if hair_length == "short" else 14.0))
        ).astype(np.uint8)
        * 255
    )
    dark_candidate_u8 = (
        (
            (
                (current_gray < np.minimum(source_gray + (18.0 if hair_length == "short" else 16.0), 198.0))
                & ((current_blur - current_gray) > (1.4 if hair_length == "short" else 1.6))
            )
            | (blackhat > (6.0 if hair_length == "short" else 7.0))
            | (
                (lap < (18.0 if hair_length == "short" else 16.0))
                & (current_gray < (186.0 if hair_length == "short" else 178.0))
                & (diff_rgb > (9.0 if hair_length == "short" else 8.0))
            )
        ).astype(np.uint8)
        * 255
    )
    candidate_u8 = cv2.bitwise_and(diff_candidate_u8, dark_candidate_u8)
    candidate_u8 = cv2.bitwise_and(candidate_u8, zone_u8)

    if final_hair_mask is not None:
        final_hair_u8 = cv2.dilate(
            (np.clip(final_hair_mask.astype(np.float32), 0.0, 1.0) > 0.16).astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 21) if hair_length == "short" else (11, 17),
            ),
            iterations=1,
        )
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(final_hair_u8, zone_u8))

    if int((guide_lane_u8 > 0).sum()) > 0:
        guided_candidate_u8 = (
            (
                (
                    (diff_rgb > (8.0 if hair_length == "short" else 7.0))
                    | (gray_delta > (8.0 if hair_length == "short" else 7.0))
                    | (np.abs(current_sat - source_sat) > 12.0)
                )
                & (lap < (24.0 if hair_length == "short" else 22.0))
                & (current_gray < (202.0 if hair_length == "short" else 194.0))
            ).astype(np.uint8)
            * 255
        )
        guided_candidate_u8 = cv2.bitwise_and(guided_candidate_u8, zone_u8)
        guided_candidate_u8 = cv2.bitwise_and(guided_candidate_u8, guide_lane_u8)
        candidate_u8 = cv2.bitwise_or(candidate_u8, guided_candidate_u8)

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
            (11, 27) if int((guide_lane_u8 > 0).sum()) > 0 else (7, 17),
        ),
    )
    if int((candidate_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    guided_mode = int((guide_lane_u8 > 0).sum()) > 0
    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    max_area = (
        max(12000, int(face_w * face_h * 0.56))
        if guided_mode
        else max(3200, int(face_w * face_h * 0.16))
    )
    max_width = (
        max(176, int(face_w * 0.92))
        if guided_mode
        else max(86, int(face_w * 0.42))
    )
    min_height = max(28, int(face_h * 0.16))
    max_offset = max(34, int(face_w * (0.24 if hair_length == "short" else 0.20)))
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
        if y < int(cutoff_y + face_h * 0.02):
            continue
        if bottom_y < int(cutoff_y + face_h * 0.18):
            continue
        if abs(comp_cx - cx) > max_offset:
            continue
        fill_ratio = float(area) / float(max(w * h, 1))
        if not guided_mode and fill_ratio > 0.92 and area > max(160, int(face_w * face_h * 0.010)):
            continue
        comp_u8 = (labels == idx).astype(np.uint8) * 255
        if guided_mode:
            guide_overlap = int((cv2.bitwise_and(comp_u8, guide_lane_u8) > 0).sum())
            if guide_overlap < max(12, int(area * 0.04)):
                continue
        keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

    if int((keep_u8 > 0).sum()) < 24:
        fallback_u8 = cv2.bitwise_and(candidate_u8, guide_lane_u8)
        if not guided_mode or int((fallback_u8 > 0).sum()) < 24:
            return np.zeros((H, W), dtype=np.float32)
        if hair_length == "short":
            fallback_u8 = self._trim_blocky_short_restore_mask_u8(
                mask_u8=fallback_u8,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                min_keep_px=24,
            )
        if int((fallback_u8 > 0).sum()) < 24:
            return np.zeros((H, W), dtype=np.float32)
        keep_u8 = fallback_u8

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 23) if hair_length == "short" else (7, 17),
        ),
        iterations=1,
    )
    keep_u8 = cv2.bitwise_and(keep_u8, zone_u8)
    if hair_length == "short":
        trimmed_u8 = self._trim_blocky_short_restore_mask_u8(
            mask_u8=keep_u8,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            min_keep_px=24,
        )
        if int((trimmed_u8 > 0).sum()) >= 24:
            keep_u8 = trimmed_u8
    if int((keep_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    return cv2.GaussianBlur(
        keep_u8.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.8 if hair_length == "short" else 2.1,
        sigmaY=5.2 if hair_length == "short" else 4.0,
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
    top = max(0, int(max(y2 + face_h * 0.12, cutoff_y + face_h * 0.08)))
    bottom = min(H, int(cutoff_y + face_h * 1.56))
    left = max(0, int(x1 - face_w * 1.38))
    right = min(W, int(x2 + face_w * 1.38))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    lane_u8 = np.zeros((H, W), dtype=np.uint8)
    left_lane_left = max(left, int(x1 - face_w * 0.72))
    left_lane_right = min(right, int(x1 + face_w * 0.14))
    right_lane_left = max(left, int(x2 - face_w * 0.14))
    right_lane_right = min(right, int(x2 + face_w * 0.72))
    if left_lane_left < left_lane_right:
        lane_u8[top:bottom, left_lane_left:left_lane_right] = 255
    if right_lane_left < right_lane_right:
        lane_u8[top:bottom, right_lane_left:right_lane_right] = 255
    center_lane_top = min(bottom, int(y2 + face_h * 0.34))
    center_half = max(16, int(face_w * 0.12))
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
            ((gray < 168.0) & (blur < 176.0) & (sat < 124.0))
            | (blackhat > 9)
        ).astype(np.uint8)
        * 255
    )

    candidate_u8 = cv2.bitwise_and(hair_u8, dark_tail_u8)
    if removal_u8 is not None:
        candidate_u8 = cv2.bitwise_or(candidate_u8, cv2.bitwise_and(dark_tail_u8, removal_u8))
    else:
        candidate_u8 = cv2.bitwise_or(candidate_u8, dark_tail_u8)
    if cloth_u8 is not None:
        cloth_support_u8 = cv2.dilate(
            cloth_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)),
            iterations=1,
        )
        deep_lane_u8 = lane_u8.copy()
        deep_lane_u8[:max(0, int(y2 + face_h * 0.12)), :] = 0
        candidate_u8 = cv2.bitwise_and(
            candidate_u8,
            cv2.bitwise_or(cloth_support_u8, deep_lane_u8),
        )
    candidate_u8 = cv2.bitwise_and(candidate_u8, corridor_u8)
    candidate_u8 = cv2.bitwise_and(candidate_u8, lane_u8)
    if int((candidate_u8 > 0).sum()) < 40:
        return np.zeros((H, W), dtype=np.float32)

    lower_start = min(H, int(y2 + face_h * 0.18))
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
    max_width = max(146, int(face_w * 0.94))
    deep_bottom = int(y2 + face_h * 0.24)
    deepest_bottom = int(y2 + face_h * 0.38)
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
            if cloth_overlap < 10 and lane_overlap < max(18, int(area * 0.10)) and bottom_y < int(cutoff_y + face_h * 0.72):
                continue
            if offset < center_keepout and bottom_y < deepest_bottom and cloth_overlap < 12:
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
    top = max(0, int(max(float(y2) - face_h * 0.08, cutoff_y + face_h * 0.04)))
    bottom = min(H, int(y2 + face_h * 1.16))
    left = max(0, int(x1 - face_w * 1.26))
    right = min(W, int(x2 + face_w * 1.26))
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
    max_area = max(24000, int(face_w * face_h * 0.44))
    min_height = max(18, int(face_h * 0.08))
    max_width = max(196, int(face_w * 1.18))
    center_keepout = max(14, int(face_w * 0.12))
    deep_start = int(y2 + face_h * 0.10)
    deep_center_start = int(y2 + face_h * 0.32)

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
        if offset < center_keepout and bottom_y < deep_center_start and cloth_overlap < 10:
            continue
        if offset < center_keepout and w > max(92, int(face_w * 0.56)) and cloth_overlap < 10:
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
    anchor_mask: Optional[np.ndarray] = None,
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
    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    anchor_lane_u8 = np.zeros((H, W), dtype=np.uint8)
    use_anchor_fallback = False

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(cutoff_y + face_h * 0.18))
    bottom = min(H, int(cutoff_y + face_h * 1.24))
    left = max(0, int(x1 - face_w * 1.04))
    right = min(W, int(x2 + face_w * 1.04))
    if top >= bottom or left >= right:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[top:bottom, left:right] = 255

    cloth_u8 = cv2.bitwise_and(cloth_u8, corridor_u8)
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = cv2.dilate(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 25)),
            iterations=1,
        )
        anchor_u8 = cv2.bitwise_and(anchor_u8, corridor_u8)
        if int((anchor_u8 > 0).sum()) >= 80:
            use_anchor_fallback = True
            anchor_lane_u8 = cv2.dilate(
                anchor_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        max(33, int(face_w * 0.92)) | 1,
                        max(15, int(face_h * 0.18)) | 1,
                    ),
                ),
                iterations=1,
            )
            anchor_lane_u8 = cv2.bitwise_and(anchor_lane_u8, corridor_u8)
    zone_u8 = cv2.bitwise_and(cv2.bitwise_or(cloth_u8, anchor_u8), removal_u8)
    zone_u8 = cv2.bitwise_or(zone_u8, cv2.bitwise_and(removal_u8, anchor_lane_u8))
    if int((zone_u8 > 0).sum()) < 30:
        return np.zeros((H, W), dtype=np.float32)
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
    if use_anchor_fallback:
        anchor_candidate_u8 = (
            (
                (gray < 188.0)
                & (sat < 132.0)
                & (lap < 26.0)
            ).astype(np.uint8)
            * 255
        )
        anchor_candidate_u8 = cv2.bitwise_and(anchor_candidate_u8, zone_u8)
        anchor_candidate_u8 = cv2.bitwise_and(anchor_candidate_u8, anchor_lane_u8)
        candidate_u8 = cv2.bitwise_or(candidate_u8, anchor_candidate_u8)
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
            (11, 25) if use_anchor_fallback else (5, 11),
        ),
    )
    if int((candidate_u8 > 0).sum()) < 24:
        return np.zeros((H, W), dtype=np.float32)

    keep_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    max_area = (
        max(24000, int(face_w * face_h * 1.20))
        if use_anchor_fallback
        else max(3000, int(face_w * face_h * 0.12))
    )
    max_width = (
        max(220, int(face_w * 1.26))
        if use_anchor_fallback
        else max(82, int(face_w * 0.42))
    )
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
        if use_anchor_fallback and int((candidate_u8 > 0).sum()) >= 80:
            fallback_u8 = cv2.bitwise_and(candidate_u8, anchor_lane_u8)
            fallback_u8 = self._trim_blocky_short_restore_mask_u8(
                mask_u8=fallback_u8,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                min_keep_px=24,
            )
            fallback_px = int((fallback_u8 > 0).sum())
            if 24 <= fallback_px <= max(4200, int(face_w * face_h * 0.20)):
                keep_u8 = fallback_u8
            else:
                return np.zeros((H, W), dtype=np.float32)
        else:
            return np.zeros((H, W), dtype=np.float32)

    keep_u8 = cv2.dilate(
        keep_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 23) if use_anchor_fallback else (7, 15),
        ),
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
    cx = float(0.5 * (x1 + x2))

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    x_min = max(0, int(x1 - face_w * (1.28 if hair_length == "short" else 1.10)))
    x_max = min(W, int(x2 + face_w * (1.28 if hair_length == "short" else 1.10)))
    y_min = max(0, int(y2 - face_h * 0.03))
    y_max = min(H, int(y2 + face_h * (1.72 if hair_length == "short" else 1.24)))
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[y_min:y_max, x_min:x_max] = 255

    hair_u8 = (np.clip(hair_mask.astype(np.float32), 0.0, 1.0) > 0.35).astype(np.uint8) * 255
    raw_tail_u8 = hair_u8.copy()
    raw_tail_u8[:max(0, int(y2 - face_h * 0.04)), :] = 0
    raw_tail_u8 = cv2.bitwise_and(raw_tail_u8, corridor_u8)
    raw_tail_px = int((raw_tail_u8 > 0).sum())
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

    min_tail_bottom = int(y2 + face_h * (0.10 if hair_length == "short" else 0.08))
    min_height = max(14, int(face_h * (0.10 if hair_length == "short" else 0.08)))

    def _build_raw_tail_fallback_mask() -> np.ndarray:
        if raw_tail_px < max(60, int(face_w * face_h * 0.010)):
            return np.zeros((H, W), dtype=np.uint8)

        fallback_u8 = raw_tail_u8.copy()
        fallback_u8 = cv2.morphologyEx(
            fallback_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 31) if hair_length == "short" else (7, 23),
            ),
        )
        fallback_u8 = cv2.morphologyEx(
            fallback_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        if int((anchor_support_u8 > 0).sum()) >= 40:
            relaxed_anchor_u8 = cv2.dilate(
                anchor_support_u8,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (19, 49) if hair_length == "short" else (15, 37),
                ),
                iterations=1,
            )
            guided_u8 = cv2.bitwise_and(
                fallback_u8,
                cv2.bitwise_or(relaxed_anchor_u8, front_strand_zone_u8),
            )
            if int((guided_u8 > 0).sum()) >= max(48, int(raw_tail_px * 0.10)):
                fallback_u8 = guided_u8

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fallback_u8, 8)
        max_shallow_width = max(88, int(face_w * 1.06))
        center_half = max(18, int(face_w * 0.18))
        max_center_width = max(42, int(face_w * 0.34))
        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom = y + h
            if area < 24:
                continue
            if h < min_height:
                continue
            if bottom < min_tail_bottom:
                continue
            if w >= max_shallow_width and h < max(36, int(face_h * 0.28)):
                continue

            comp_u8 = (labels == idx).astype(np.uint8) * 255
            anchor_overlap = int((cv2.bitwise_and(comp_u8, anchor_support_u8) > 0).sum())
            front_overlap = int((cv2.bitwise_and(comp_u8, front_strand_zone_u8) > 0).sum())
            comp_cx = float(centroids[idx][0])
            is_center_component = abs(comp_cx - cx) <= center_half

            if (
                is_center_component
                and w > max_center_width
                and anchor_overlap < 12
                and front_overlap < 10
                and h < max(46, int(face_h * 0.34))
            ):
                continue
            keep_u8 = cv2.bitwise_or(keep_u8, comp_u8)

        if int((keep_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.uint8)

        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 35) if hair_length == "short" else (7, 27),
            ),
        )
        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 27) if hair_length == "short" else (9, 17),
            ),
            iterations=1,
        )
        return cv2.bitwise_and(keep_u8, corridor_u8)

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
    raw_tail_fallback_u8 = _build_raw_tail_fallback_mask()
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
    if int((dark_u8 > 0).sum()) < 18:
        if int((raw_tail_fallback_u8 > 0).sum()) < 20:
            return np.zeros((H, W), dtype=np.float32)
        keep_u8 = raw_tail_fallback_u8
    else:
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

        keep_px = int((keep_u8 > 0).sum())
        fallback_px = int((raw_tail_fallback_u8 > 0).sum())
        if keep_px < 20:
            if fallback_px < 20:
                return np.zeros((H, W), dtype=np.float32)
            keep_u8 = raw_tail_fallback_u8
        elif (
            fallback_px >= max(48, int(keep_px * 1.10))
            and keep_px < max(160, int(raw_tail_px * 0.22))
        ):
            keep_u8 = cv2.bitwise_or(keep_u8, raw_tail_fallback_u8)

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
    anchor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    H, W = img_rgb.shape[:2]
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return np.zeros((H, W), dtype=np.float32)

    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = int(0.5 * (x1 + x2))

    lane_u8 = np.zeros((H, W), dtype=np.uint8)
    lane_half = max(18, int(face_w * (0.20 if hair_length == "short" else 0.17)))
    lane_x1 = max(0, cx - lane_half)
    lane_x2 = min(W, cx + lane_half)
    lane_y1 = max(0, int(cutoff_y - face_h * 0.04))
    lane_y2 = min(H, int(cutoff_y + face_h * (2.05 if hair_length == "short" else 1.42)))
    if lane_x1 >= lane_x2 or lane_y1 >= lane_y2:
        return np.zeros((H, W), dtype=np.float32)
    lane_u8[lane_y1:lane_y2, lane_x1:lane_x2] = 255

    cloth_u8 = cv2.dilate(
        (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    anchor_u8 = np.zeros((H, W), dtype=np.uint8)
    if anchor_mask is not None and anchor_mask.shape == (H, W):
        anchor_u8 = cv2.erode(
            (np.clip(anchor_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
            iterations=1,
        )
        anchor_u8 = cv2.bitwise_and(anchor_u8, lane_u8)
    zone_u8 = cv2.bitwise_and(lane_u8, cv2.bitwise_or(cloth_u8, anchor_u8))
    if int((zone_u8 > 0).sum()) < 20:
        return np.zeros((H, W), dtype=np.float32)

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
            support_hint_u8 = cv2.bitwise_and(support_hint_u8, zone_u8)
            support_hint_u8 = cv2.morphologyEx(
                support_hint_u8,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (7, 31) if hair_length == "short" else (5, 23),
                ),
            )

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
    dark_u8 = (
        (gray < (162.0 if hair_length == "short" else 154.0))
        & ((blur - gray) > (2.2 if hair_length == "short" else 2.4))
        & (blur > (110.0 if hair_length == "short" else 104.0))
    ).astype(np.uint8) * 255
    dark_u8 = cv2.bitwise_or(dark_u8, blackhat_u8)
    dark_u8 = cv2.bitwise_and(dark_u8, zone_u8)
    if int((support_hint_u8 > 0).sum()) >= 8:
        dark_u8 = cv2.bitwise_and(dark_u8, support_hint_u8)
        if int((dark_u8 > 0).sum()) < 8 and int((support_hint_u8 > 0).sum()) >= 24:
            dark_u8 = support_hint_u8.copy()
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
        if abs(comp_cx - cx) > max(16, int(face_w * 0.18)):
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
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    hair_length: str,
) -> np.ndarray:
    if hair_length not in ("short", "medium"):
        return np.zeros_like(support_mask, dtype=np.float32)

    H, W = support_mask.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    cx = float(0.5 * (x1 + x2))

    support_u8 = (np.clip(support_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((support_u8 > 0).sum()) < 12:
        return np.zeros((H, W), dtype=np.float32)

    raw_support_u8 = support_u8.copy()
    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    x_min = max(0, int(x1 - face_w * (1.34 if hair_length == "short" else 1.14)))
    x_max = min(W, int(x2 + face_w * (1.34 if hair_length == "short" else 1.14)))
    y_min = max(0, int(cutoff_y - face_h * 0.03))
    y_max = min(H, int(cutoff_y + face_h * (1.32 if hair_length == "short" else 1.02)))
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[y_min:y_max, x_min:x_max] = 255
    support_u8 = cv2.bitwise_and(support_u8, corridor_u8)
    raw_support_u8 = cv2.bitwise_and(raw_support_u8, corridor_u8)
    raw_support_px = int((raw_support_u8 > 0).sum())

    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_near_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.06).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        support_on_cloth_u8 = cv2.bitwise_and(support_u8, cloth_near_u8)
        if int((support_on_cloth_u8 > 0).sum()) >= max(12, int(raw_support_px * 0.22)):
            support_u8 = support_on_cloth_u8
        else:
            support_u8 = raw_support_u8.copy()

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
            (7, 13) if hair_length == "short" else (5, 9),
        ),
        iterations=1,
    )

    def _build_relaxed_post_support_mask(source_u8: np.ndarray) -> np.ndarray:
        source_px = int((source_u8 > 0).sum())
        if source_px < 20:
            return np.zeros((H, W), dtype=np.uint8)

        fallback_u8 = source_u8.copy()
        fallback_u8 = cv2.morphologyEx(
            fallback_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 29) if hair_length == "short" else (7, 21),
            ),
        )
        fallback_u8 = cv2.morphologyEx(
            fallback_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        if cloth_mask is not None and cloth_mask.shape == (H, W):
            cloth_hint_u8 = cv2.dilate(
                (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            fallback_on_cloth_u8 = cv2.bitwise_and(fallback_u8, cloth_hint_u8)
            if int((fallback_on_cloth_u8 > 0).sum()) >= max(24, int(source_px * 0.10)):
                fallback_u8 = fallback_on_cloth_u8

        keep_u8 = np.zeros((H, W), dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fallback_u8, 8)
        min_bottom = int(cutoff_y + face_h * 0.02)
        min_height = max(20, int(face_h * 0.16))
        max_shallow_width = max(96, int(face_w * 1.06))
        center_half = max(18, int(face_w * 0.20))
        max_center_width = max(46, int(face_w * 0.36))
        for idx in range(1, num_labels):
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bottom = y + h
            if area < 20:
                continue
            if h < min_height:
                continue
            if bottom < min_bottom:
                continue
            if w >= max_shallow_width and h < max(40, int(face_h * 0.30)):
                continue

            comp_cx = float(centroids[idx][0])
            if (
                abs(comp_cx - cx) <= center_half
                and w > max_center_width
                and h < max(56, int(face_h * 0.40))
            ):
                continue
            keep_u8[labels == idx] = 255

        if int((keep_u8 > 0).sum()) < 12:
            return np.zeros((H, W), dtype=np.uint8)

        keep_u8 = cv2.morphologyEx(
            keep_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 31) if hair_length == "short" else (7, 23),
            ),
        )
        keep_u8 = cv2.dilate(
            keep_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 23) if hair_length == "short" else (11, 17),
            ),
            iterations=1,
        )
        return cv2.bitwise_and(keep_u8, corridor_u8)

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support_u8, 8)
    max_area = max(960, int(face_w * face_h * (0.48 if hair_length == "short" else 0.30)))
    min_bottom = int(cutoff_y + face_h * 0.02)
    for idx in range(1, num_labels):
        y = int(stats[idx, cv2.CC_STAT_TOP])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 8 or area > max_area:
            continue
        if (y + h) < min_bottom:
            continue
        filtered_u8[labels == idx] = 255

    relaxed_filtered_u8 = _build_relaxed_post_support_mask(raw_support_u8)
    filtered_px = int((filtered_u8 > 0).sum())
    relaxed_px = int((relaxed_filtered_u8 > 0).sum())
    if filtered_px < 12:
        if relaxed_px < 12:
            return np.zeros((H, W), dtype=np.float32)
        filtered_u8 = relaxed_filtered_u8
    elif (
        relaxed_px >= max(36, int(filtered_px * 1.35))
        and filtered_px < max(160, int(raw_support_px * 0.18))
    ):
        filtered_u8 = cv2.bitwise_or(filtered_u8, relaxed_filtered_u8)

    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 23) if hair_length == "short" else (11, 17),
        ),
        iterations=1,
    )
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

    raw_support_u8 = support_u8.copy()
    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    x_min = max(0, int(x1 - face_w * (1.08 if hair_length == "short" else 1.02)))
    x_max = min(W, int(x2 + face_w * (1.08 if hair_length == "short" else 1.02)))
    y_min = max(0, int(cutoff_y - face_h * 0.03))
    y_max = min(H, int(cutoff_y + face_h * (1.52 if hair_length == "short" else 1.28)))
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((H, W), dtype=np.float32)
    corridor_u8[y_min:y_max, x_min:x_max] = 255
    support_u8 = cv2.bitwise_and(support_u8, corridor_u8)
    raw_support_u8 = cv2.bitwise_and(raw_support_u8, corridor_u8)
    raw_support_px = int((raw_support_u8 > 0).sum())

    if cloth_mask is not None and cloth_mask.shape == (H, W):
        cloth_hint_u8 = cv2.dilate(
            (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        support_on_cloth_u8 = cv2.bitwise_and(support_u8, cloth_hint_u8)
        if int((support_on_cloth_u8 > 0).sum()) >= max(12, int(raw_support_px * 0.18)):
            support_u8 = support_on_cloth_u8
        else:
            support_u8 = raw_support_u8.copy()

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
    center_half = max(12, int(face_w * (0.24 if hair_length == "short" else 0.20)))
    lane_x1 = max(0, int(0.5 * (x1 + x2)) - center_half)
    lane_x2 = min(W, int(0.5 * (x1 + x2)) + center_half)
    lane_y1 = max(0, int(cutoff_y - face_h * 0.04))
    lane_y2 = min(H, int(cutoff_y + face_h * (1.58 if hair_length == "short" else 0.96)))
    if lane_x1 < lane_x2 and lane_y1 < lane_y2:
        front_lane_u8[lane_y1:lane_y2, lane_x1:lane_x2] = 255

    filtered_u8 = np.zeros((H, W), dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(support_u8, 8)
    max_area = max(520, int(face_w * face_h * (0.30 if hair_length == "short" else 0.20)))
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
        if base_overlap < 8 and not is_front_strand:
            continue
        if hair_length == "short":
            if w > max(34, int(face_w * 0.30)) and not is_front_strand:
                continue
            if is_side_component:
                if base_overlap < max(14, int(area * 0.10)) and front_overlap < 8:
                    continue
                if area > max(156, int(face_w * face_h * 0.050)):
                    continue
        filtered_u8 = cv2.bitwise_or(filtered_u8, comp_u8)

    if int((filtered_u8 > 0).sum()) < 12:
        return np.zeros((H, W), dtype=np.float32)

    filtered_u8 = cv2.dilate(
        filtered_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 21) if hair_length == "short" else (7, 15),
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
        zone = np.maximum(zone, tail_hint * 1.20)
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

    zone_thresh = 0.20 if hair_length == "short" else 0.34
    zone_u8 = (zone > zone_thresh).astype(np.uint8) * 255
    deep_start = min(H, int(cutoff_y + face_h * (0.00 if hair_length == "short" else 0.10)))
    x_min = max(0, int(x1 - face_w * (1.35 if hair_length == "short" else 1.20)))
    x_max = min(W, int(x2 + face_w * (1.35 if hair_length == "short" else 1.20)))
    corridor = np.zeros((H, W), dtype=np.uint8)
    if x_min < x_max and deep_start < H:
        corridor[deep_start:, x_min:x_max] = 255
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
    contrast_thresh = 4.0 if hair_length == "short" else 6.5
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

def _cv2_cleanup_dark_tail_blob(
    img_rgb: np.ndarray,
    dark_tail_u8: np.ndarray,
) -> np.ndarray:
    """Run a small focused cv2 inpaint pass over deep dark residual tail blobs."""
    if dark_tail_u8.shape[:2] != img_rgb.shape[:2]:
        return img_rgb
    if int((dark_tail_u8 > 0).sum()) < 40:
        return img_rgb

    mask_u8 = cv2.dilate(
        dark_tail_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25)),
        iterations=1,
    )
    telea = cv2.inpaint(img_rgb, mask_u8, 7, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(img_rgb, mask_u8, 6, cv2.INPAINT_NS)
    fill = cv2.addWeighted(telea, 0.74, ns, 0.26, 0.0)
    alpha = cv2.GaussianBlur(
        (mask_u8 > 0).astype(np.float32),
        (0, 0),
        sigmaX=3.6,
        sigmaY=3.6,
    )[..., np.newaxis]
    out = fill.astype(np.float32) * alpha + img_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _remove_residual_hair_below_cutoff(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    removal_mask: Optional[np.ndarray] = None,
    shoulder_protect: Optional[np.ndarray] = None,
    neckline_preserve: Optional[np.ndarray] = None,
    lateral_preserve: Optional[np.ndarray] = None,
    hair_length: str = "short",
    center_anchor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    short/medium 변환 후 cutoff 아래에 남은 머리카락을 재검출해 정리.
    """
    H, W = img_rgb.shape[:2]
    cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
    _, y1, _, y2 = face_bbox
    face_h = max(int(y2 - y1), 1)
    soft_zone = max(10, int(face_h * 0.22))
    soft_end = min(H - 1, cutoff_y + soft_zone)

    hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    residual = hair_now.copy()
    residual[:cutoff_y, :] = 0.0

    residual_u8 = (residual > 0.5).astype(np.uint8) * 255
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, open_k)

    if soft_end > cutoff_y:
        ramp = np.ones((H,), dtype=np.float32)
        ramp[:cutoff_y] = 0.0
        ramp[cutoff_y:soft_end + 1] = np.linspace(
            0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
        )
        residual_soft = (residual_u8.astype(np.float32) / 255.0) * ramp[:, np.newaxis]
        residual_u8 = (residual_soft > 0.50).astype(np.uint8) * 255

    dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)
    front_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)
    residual_near_u8 = np.zeros((H, W), dtype=np.uint8)
    if removal_mask is not None and removal_mask.shape == (H, W):
        tail_hint = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        tail_hint_u8 = (tail_hint > 0.0).astype(np.uint8) * 255
        if int((tail_hint_u8 > 0).sum()) > 0:
            deep_start = min(H, int(cutoff_y + face_h * (0.18 if hair_length == "short" else 0.24)))
            deep_zone = np.zeros((H, W), dtype=np.uint8)
            if deep_start < H:
                deep_zone[deep_start:, :] = 255
            if int((residual_u8 > 0).sum()) > 0:
                residual_near_u8 = cv2.dilate(
                    residual_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                    iterations=1,
                )
                tail_hint_u8 = cv2.bitwise_and(
                    tail_hint_u8,
                    cv2.bitwise_or(residual_near_u8, deep_zone),
                )
            else:
                tail_hint_u8 = cv2.bitwise_and(tail_hint_u8, deep_zone)
            residual_u8 = cv2.bitwise_or(residual_u8, tail_hint_u8)

        if hair_length == "short":
            tail_core = self._build_short_tail_core_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
            )
            tail_core_u8 = (tail_core > 0.0).astype(np.uint8) * 255
            if int((tail_core_u8 > 0).sum()) > 0:
                tail_core_gate_u8 = np.zeros((H, W), dtype=np.uint8)
                shallow_limit = min(H, int(cutoff_y + face_h * 0.24))
                if shallow_limit > cutoff_y:
                    center_band_u8 = np.zeros((H, W), dtype=np.uint8)
                    face_w_local = max(int(face_bbox[2] - face_bbox[0]), 1)
                    center_half = max(16, int(face_w_local * 0.22))
                    center_band_u8[
                        cutoff_y:shallow_limit,
                        max(0, int(0.5 * (face_bbox[0] + face_bbox[2])) - center_half):min(W, int(0.5 * (face_bbox[0] + face_bbox[2])) + center_half),
                    ] = 255
                    shallow_gate_u8 = cv2.bitwise_and(residual_near_u8, center_band_u8)
                    tail_core_gate_u8[cutoff_y:shallow_limit, :] = shallow_gate_u8[cutoff_y:shallow_limit, :]
                deep_core_start = min(H, int(cutoff_y + face_h * 0.40))
                if deep_core_start < H:
                    tail_core_gate_u8[deep_core_start:, :] = 255
                tail_core_u8 = cv2.bitwise_and(tail_core_u8, tail_core_gate_u8)
            if int((tail_core_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_or(residual_u8, tail_core_u8)

            front_cleanup = self._build_front_strand_cleanup_mask(
                removal_mask=removal_mask,
                face_bbox=face_bbox,
                cutoff_y=cutoff_y,
                hair_length=hair_length,
                anchor_mask=center_anchor_mask,
            )
            front_cleanup_u8 = (front_cleanup > 0.0).astype(np.uint8) * 255
            if int((front_cleanup_u8 > 0).sum()) > 0:
                residual_u8 = cv2.bitwise_or(residual_u8, front_cleanup_u8)

        dark_tail = self._build_dark_tail_residual_mask(
            img_rgb=img_rgb,
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        dark_tail_u8 = (dark_tail > 0.0).astype(np.uint8) * 255
        if int((dark_tail_u8 > 0).sum()) > 0:
            residual_u8 = cv2.bitwise_or(residual_u8, dark_tail_u8)

    if int((residual_u8 > 0).sum()) < 60:
        return img_rgb

    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 13) if hair_length == "short" else (9, 9),
    )
    residual_u8 = cv2.dilate(residual_u8, dilate_k, iterations=1)

    if shoulder_protect is not None and shoulder_protect.shape == (H, W):
        protect_threshold = 0.62 if hair_length == "short" else 0.34
        protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
        if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
            deep_release_y = min(H, int(cutoff_y + face_h * 0.28))
            if deep_release_y < H:
                protect_u8[deep_release_y:, :] = 0
            protect_u8 = cv2.erode(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
        if int((protect_u8 > 0).sum()) > 0:
            residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(protect_u8))

    if neckline_preserve is not None and neckline_preserve.shape == (H, W):
        preserve_u8 = (neckline_preserve > (0.34 if hair_length == "short" else 0.26)).astype(np.uint8) * 255
        if int((preserve_u8 > 0).sum()) > 0:
            if hair_length == "short":
                deep_release_y = min(H, int(cutoff_y + face_h * 0.22))
                if deep_release_y < H:
                    preserve_u8[deep_release_y:, :] = 0
            preserve_u8 = cv2.dilate(
                preserve_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(preserve_u8))

    if lateral_preserve is not None and lateral_preserve.shape == (H, W):
        lateral_u8 = (lateral_preserve > 0.18).astype(np.uint8) * 255
        if int((lateral_u8 > 0).sum()) > 0:
            lateral_u8 = cv2.dilate(
                lateral_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
                iterations=1,
            )
            residual_u8 = cv2.bitwise_and(residual_u8, cv2.bitwise_not(lateral_u8))

    if int((residual_u8 > 0).sum()) < 60:
        return img_rgb

    cleaned = self._lama_inpaint(img_rgb, residual_u8)
    if int((dark_tail_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, dark_tail_u8)
    if int((front_cleanup_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, front_cleanup_u8)
    return cleaned

def _final_cutoff_cleanup(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    removal_mask: np.ndarray,
    cutoff_y: int,
    shoulder_protect: Optional[np.ndarray] = None,
    neckline_preserve: Optional[np.ndarray] = None,
    lateral_preserve: Optional[np.ndarray] = None,
    hair_length: str = "short",
    center_anchor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    최종 결과에서 cutoff 아래 long-hair 제거 마스크 영역을 한 번 더 정리.
    """
    H, W = img_rgb.shape[:2]
    if removal_mask.shape != (H, W):
        return img_rgb

    x1, y1, x2, y2 = face_bbox
    face_h = max(int(y2 - y1), 1)
    face_w = max(int(x2 - x1), 1)
    cx = int(0.5 * (x1 + x2))
    soft_zone = max(12, int(face_h * 0.25))

    force = removal_mask.copy().astype(np.float32)
    cutoff_y = int(np.clip(cutoff_y, 0, H - 1))
    force[:cutoff_y, :] = 0.0
    soft_end = min(H - 1, cutoff_y + soft_zone)
    if soft_end > cutoff_y:
        ramp = np.ones((H,), dtype=np.float32)
        ramp[:cutoff_y] = 0.0
        ramp[cutoff_y:soft_end + 1] = np.linspace(
            0.0, 1.0, soft_end - cutoff_y + 1, dtype=np.float32
        )
        force = force * ramp[:, np.newaxis]

    # 얼굴 주변 corridor 안에서만 cleanup을 허용해 의상/배경 훼손을 줄인다.
    corridor_ratio = 1.55 if hair_length == "short" else 1.35
    x_min = max(0, int(x1 - face_w * corridor_ratio))
    x_max = min(W, int(x2 + face_w * corridor_ratio))
    corridor = np.zeros((H, W), dtype=np.uint8)
    if x_min < x_max:
        corridor[:, x_min:x_max] = 255

    force_thresh = 0.56 if hair_length == "short" else 0.54
    force_u8 = ((force > force_thresh).astype(np.uint8) * 255)
    force_u8 = cv2.bitwise_and(force_u8, corridor)
    dark_tail_u8 = np.zeros((H, W), dtype=np.uint8)
    front_cleanup_u8 = np.zeros((H, W), dtype=np.uint8)

    # 실제 남아있는 hair 픽셀과 교집합을 우선 적용해 의상/배경 훼손 방지
    hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    hair_now[:cutoff_y, :] = 0.0
    hair_now_u8 = (hair_now > 0.5).astype(np.uint8) * 255
    if int((hair_now_u8 > 0).sum()) > 0:
        hair_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11) if hair_length == "short" else (9, 9),
        )
        hair_now_u8 = cv2.dilate(hair_now_u8, hair_k, iterations=1)

    hair_inter_u8 = cv2.bitwise_and(force_u8, hair_now_u8)
    if hair_length == "short":
        # SegFace miss를 보완하기 위해 side-zone에 한해 high-confidence force를 추가 반영
        center_half = max(18, int(face_w * 0.42))
        side_zone = corridor.copy()
        side_zone[:, max(0, cx - center_half):min(W, cx + center_half)] = 0
        fallback_u8 = ((force > 0.78).astype(np.uint8) * 255)
        fallback_u8 = cv2.bitwise_and(fallback_u8, side_zone)
        force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)
        hair_near_u8 = np.zeros((H, W), dtype=np.uint8)
        if int((hair_now_u8 > 0).sum()) > 0:
            hair_near_u8 = cv2.dilate(
                hair_now_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
                iterations=1,
            )

        tail_hint = self._build_side_tail_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        tail_hint_u8 = (tail_hint > 0.0).astype(np.uint8) * 255
        if int((tail_hint_u8 > 0).sum()) > 0:
            deep_start = min(H, int(cutoff_y + face_h * 0.18))
            deep_zone = np.zeros((H, W), dtype=np.uint8)
            if deep_start < H:
                deep_zone[deep_start:, :] = 255
            forced_tail_u8 = cv2.bitwise_and(
                tail_hint_u8,
                cv2.bitwise_or(hair_near_u8, deep_zone),
            )
            force_u8 = cv2.bitwise_or(force_u8, forced_tail_u8)

        tail_core = self._build_short_tail_core_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
        )
        tail_core_u8 = (tail_core > 0.0).astype(np.uint8) * 255
        if int((tail_core_u8 > 0).sum()) > 0:
            tail_core_gate_u8 = np.zeros((H, W), dtype=np.uint8)
            shallow_limit = min(H, int(cutoff_y + face_h * 0.24))
            if shallow_limit > cutoff_y:
                center_band_u8 = np.zeros((H, W), dtype=np.uint8)
                center_half = max(16, int(face_w * 0.22))
                center_band_u8[
                    cutoff_y:shallow_limit,
                    max(0, cx - center_half):min(W, cx + center_half),
                ] = 255
                shallow_gate_u8 = cv2.bitwise_and(hair_near_u8, center_band_u8)
                tail_core_gate_u8[cutoff_y:shallow_limit, :] = shallow_gate_u8[cutoff_y:shallow_limit, :]
            deep_core_start = min(H, int(cutoff_y + face_h * 0.40))
            if deep_core_start < H:
                tail_core_gate_u8[deep_core_start:, :] = 255
            tail_core_u8 = cv2.bitwise_and(tail_core_u8, tail_core_gate_u8)
        if int((tail_core_u8 > 0).sum()) > 0:
            force_u8 = cv2.bitwise_or(force_u8, tail_core_u8)

        front_cleanup = self._build_front_strand_cleanup_mask(
            removal_mask=removal_mask,
            face_bbox=face_bbox,
            cutoff_y=cutoff_y,
            hair_length=hair_length,
            anchor_mask=center_anchor_mask,
        )
        front_cleanup_u8 = (front_cleanup > 0.0).astype(np.uint8) * 255
        if int((front_cleanup_u8 > 0).sum()) > 0:
            force_u8 = cv2.bitwise_or(force_u8, front_cleanup_u8)
    else:
        # medium도 SegFace miss 보완용 fallback force 일부 허용
        fallback_u8 = ((force > 0.74).astype(np.uint8) * 255)
        fallback_u8 = cv2.bitwise_and(fallback_u8, corridor)
        force_u8 = cv2.bitwise_or(hair_inter_u8, fallback_u8)

    dark_tail = self._build_dark_tail_residual_mask(
        img_rgb=img_rgb,
        removal_mask=removal_mask,
        face_bbox=face_bbox,
        cutoff_y=cutoff_y,
        hair_length=hair_length,
    )
    dark_tail_u8 = (dark_tail > 0.0).astype(np.uint8) * 255
    if int((dark_tail_u8 > 0).sum()) > 0:
        force_u8 = cv2.bitwise_or(force_u8, cv2.bitwise_and(dark_tail_u8, corridor))

    if shoulder_protect is not None and shoulder_protect.shape == (H, W):
        protect_threshold = 0.62 if hair_length == "short" else 0.34
        protect_u8 = (shoulder_protect > protect_threshold).astype(np.uint8) * 255
        if hair_length == "short" and int((protect_u8 > 0).sum()) > 0:
            deep_release_y = min(H, int(cutoff_y + face_h * 0.28))
            if deep_release_y < H:
                protect_u8[deep_release_y:, :] = 0
            protect_u8 = cv2.erode(
                protect_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
        if int((protect_u8 > 0).sum()) > 0:
            force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(protect_u8))

    if neckline_preserve is not None and neckline_preserve.shape == (H, W):
        preserve_u8 = (neckline_preserve > (0.34 if hair_length == "short" else 0.26)).astype(np.uint8) * 255
        if int((preserve_u8 > 0).sum()) > 0:
            if hair_length == "short":
                deep_release_y = min(H, int(cutoff_y + face_h * 0.22))
                if deep_release_y < H:
                    preserve_u8[deep_release_y:, :] = 0
            preserve_u8 = cv2.dilate(
                preserve_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(preserve_u8))

    if lateral_preserve is not None and lateral_preserve.shape == (H, W):
        lateral_u8 = (lateral_preserve > 0.18).astype(np.uint8) * 255
        if int((lateral_u8 > 0).sum()) > 0:
            lateral_u8 = cv2.dilate(
                lateral_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9)),
                iterations=1,
            )
            force_u8 = cv2.bitwise_and(force_u8, cv2.bitwise_not(lateral_u8))

    min_cleanup_px = 28 if hair_length == "short" else 40
    if int((force_u8 > 0).sum()) < min_cleanup_px:
        return img_rgb

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 13) if hair_length == "short" else (7, 7),
    )
    force_u8 = cv2.dilate(force_u8, k, iterations=1)

    # LaMa로 잔존 hair 제거
    cleaned = self._lama_inpaint(img_rgb, force_u8)
    if int((dark_tail_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, dark_tail_u8)
    if int((front_cleanup_u8 > 0).sum()) > 0:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, front_cleanup_u8)
    return cleaned

def _composite(
    self,
    orig_bgr: np.ndarray,
    orig_rgb: np.ndarray,
    gen_pil: Image.Image,            # 512×512 RGB
    hair_mask: np.ndarray,           # H×W float32 (original resolution)
    scale: float,
    pad: Tuple[int, int],            # (pad_left, pad_top)
    original_size: Tuple[int, int],  # (W, H)
    protect_mask: Optional[np.ndarray] = None,  # H×W float32: 이 영역은 alpha=0 강제 (얼굴 보호)
    protect_release_mask: Optional[np.ndarray] = None,
    hair_length: str = "long",
) -> np.ndarray:
    """
    SD 생성 이미지를 원본에 합성.
    - hair mask 영역: SD 생성 결과
    - 그 외 (+ protect_mask): 원본 (얼굴/배경 유지)
    """
    W, H = original_size
    hair_mask = self._resize_mask_to_shape(hair_mask, (H, W))
    protect_mask = self._resize_mask_to_shape(protect_mask, (H, W))
    protect_release_mask = self._resize_mask_to_shape(protect_release_mask, (H, W))
    pad_l, pad_t = pad
    new_w = int(W * scale)
    new_h = int(H * scale)

    # letterbox 제거 → 원본 비율로 crop
    gen_np = np.array(gen_pil)   # 512×512×3 RGB
    gen_cropped = gen_np[pad_t:pad_t + new_h, pad_l:pad_l + new_w]

    # 원본 해상도로 upscale
    gen_orig = cv2.resize(gen_cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    # alpha 블렌딩: short/medium는 경계를 더 또렷하게 유지
    sigma = 6.0
    if hair_length == "short":
        sigma = 4.2
    elif hair_length == "medium":
        sigma = 4.8
    alpha = cv2.GaussianBlur(hair_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if hair_length == "short":
        alpha = np.clip((alpha - 0.10) / 0.90, 0.0, 1.0)
    elif hair_length == "medium":
        alpha = np.clip((alpha - 0.07) / 0.93, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 얼굴/귀/눈 등 보호 영역: alpha를 0으로 강제
    # → Gaussian blur가 얼굴 경계로 번지더라도 원본 픽셀 100% 유지
    if protect_mask is not None:
        # protect_mask도 살짝 dilate해서 경계까지 확실히 보호
        protect_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        protect_dilated = cv2.dilate(protect_mask.astype(np.float32), protect_k)
        if protect_release_mask is not None:
            protect_dilated = np.clip(
                protect_dilated - np.clip(protect_release_mask.astype(np.float32) * 1.35, 0.0, 1.0),
                0.0,
                1.0,
            )
        alpha = alpha * (1.0 - np.clip(protect_dilated, 0.0, 1.0))
    if protect_release_mask is not None:
        release_u8 = (np.clip(protect_release_mask.astype(np.float32), 0.0, 1.0) > 0.04).astype(np.uint8) * 255
        if int((release_u8 > 0).sum()) >= 20:
            release_u8 = cv2.dilate(
                release_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)),
                iterations=1,
            )
            release_alpha = cv2.GaussianBlur(
                release_u8.astype(np.float32) / 255.0,
                (0, 0),
                sigmaX=2.2,
                sigmaY=2.6,
            )
            # Force generated bangs to cover the released fringe band.
            alpha = np.maximum(alpha, np.clip(release_alpha * 0.82, 0.0, 0.92))

    alpha = alpha[..., np.newaxis]   # H×W×1

    orig_f = orig_rgb.astype(np.float32)
    gen_f  = gen_orig.astype(np.float32)
    blend  = gen_f * alpha + orig_f * (1.0 - alpha)
    blend  = np.clip(blend, 0, 255).astype(np.uint8)

    return cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)

def unload(self) -> None:
    """VRAM 해제"""
    import gc
    self._sd_pipe = None
    self._sam2_factory = None
    if self._mp_face:
        self._mp_face.close()
    if self._mp_face_mesh:
        self._mp_face_mesh.close()
    self._mp_face = None
    self._mp_face_mesh = None
    self._loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[SDPipeline] 모델 언로드 완료")

def bind_postprocess_methods_to_pipeline(cls) -> None:
    cls._generate = _generate
    cls._cv2_refine_cloth_region = staticmethod(_cv2_refine_cloth_region)
    cls._overlay_reference_cloth_fill = staticmethod(_overlay_reference_cloth_fill)
    cls._blend_neighbor_cloth_tone = staticmethod(_blend_neighbor_cloth_tone)
    cls._resolve_background_fill_mode = _resolve_background_fill_mode
    cls._sd_refine_removed_region = _sd_refine_removed_region
    cls._filter_short_center_cleanup_mask = _filter_short_center_cleanup_mask
    cls._build_post_cloth_refine_mask = _build_post_cloth_refine_mask
    cls._build_under_jaw_cloth_refine_mask = _build_under_jaw_cloth_refine_mask
    cls._build_short_cloth_only_second_pass_mask = _build_short_cloth_only_second_pass_mask
    cls._build_short_cloth_generation_silhouette_mask = _build_short_cloth_generation_silhouette_mask
    cls._build_short_cloth_control_map = _build_short_cloth_control_map
    cls._build_source_conditioned_cloth_base = _build_source_conditioned_cloth_base
    cls._apply_short_source_cloth_anchor_restore = _apply_short_source_cloth_anchor_restore
    cls._tighten_short_under_jaw_mask_to_center = _tighten_short_under_jaw_mask_to_center
    cls._build_short_under_jaw_crop_core_mask = _build_short_under_jaw_crop_core_mask
    cls._compose_strict_short_under_jaw_crop_result = _compose_strict_short_under_jaw_crop_result
    cls._refine_short_under_jaw_crop_region = _refine_short_under_jaw_crop_region
    cls._generate_short_under_jaw_cloth_insert_region = _generate_short_under_jaw_cloth_insert_region
    cls._stabilize_under_jaw_cloth_fill = _stabilize_under_jaw_cloth_fill
    cls._build_generation_protect_mask = _build_generation_protect_mask
    cls._build_removal_protect_mask = _build_removal_protect_mask
    cls._build_neckline_preserve_mask = _build_neckline_preserve_mask
    cls._build_short_lateral_neck_preserve_mask = _build_short_lateral_neck_preserve_mask
    cls._build_shoulder_protect_mask = _build_shoulder_protect_mask
    cls._build_torso_cloth_preserve_mask = _build_torso_cloth_preserve_mask
    cls._build_bright_cloth_preserve_mask = _build_bright_cloth_preserve_mask
    cls._filter_short_torso_box_mask = _filter_short_torso_box_mask
    cls._build_micro_cloth_artifact_mask = _build_micro_cloth_artifact_mask
    cls._restrict_short_removal_to_tail_lanes = _restrict_short_removal_to_tail_lanes
    cls._build_shoulder_cloth_restore_mask = _build_shoulder_cloth_restore_mask
    cls._restore_cloth_overlap_from_source = _restore_cloth_overlap_from_source
    cls._cleanup_region_with_cloth_restore = _cleanup_region_with_cloth_restore
    cls._build_final_source_cloth_rescue_mask = _build_final_source_cloth_rescue_mask
    cls._build_short_lower_garment_cleanup_mask = _build_short_lower_garment_cleanup_mask
    cls._build_side_column_cloth_restore_mask = _build_side_column_cloth_restore_mask
    cls._build_direct_short_column_restore_mask = _build_direct_short_column_restore_mask
    cls._build_short_below_bob_cloth_restore_mask = _build_short_below_bob_cloth_restore_mask
    cls._build_short_below_bob_generation_block_mask = _build_short_below_bob_generation_block_mask
    cls._build_preclean_side_column_cleanup_mask = _build_preclean_side_column_cleanup_mask
    cls._build_preclean_cloth_hair_cleanup_mask = _build_preclean_cloth_hair_cleanup_mask
    cls._build_residual_strand_cleanup_mask = _build_residual_strand_cleanup_mask
    cls._build_final_hair_lane_cleanup_mask = _build_final_hair_lane_cleanup_mask
    cls._build_center_residual_detector_mask = _build_center_residual_detector_mask
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
    cls._cv2_cleanup_dark_tail_blob = staticmethod(_cv2_cleanup_dark_tail_blob)
    cls._remove_residual_hair_below_cutoff = _remove_residual_hair_below_cutoff
    cls._final_cutoff_cleanup = _final_cutoff_cleanup
    cls._composite = _composite
    cls.unload = unload
