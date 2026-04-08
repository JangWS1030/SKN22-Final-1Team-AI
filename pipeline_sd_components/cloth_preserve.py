"""
MirrAI SD Inpainting — 옷 보존 & 복원 로직
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
postprocess.py에서 분리된 옷 보존/복원 함수 모음.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
    prefer_plain_cloth_fill: bool = False,
    allow_plain_cloth_force: bool = True,
    debug_trace: Optional[List[Tuple[str, np.ndarray]]] = None,
    debug_info: Optional[Dict[str, Any]] = None,
    debug_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    H, W = current_rgb.shape[:2]
    if source_rgb.shape[:2] != (H, W) or cleanup_mask.shape != (H, W):
        return current_rgb

    mask_u8 = (np.clip(cleanup_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((mask_u8 > 0).sum()) < 40:
        return current_rgb

    cleaned = self._lama_inpaint(current_rgb, mask_u8)
    if debug_trace is not None:
        debug_trace.append(("lama_inpaint", cleaned.copy()))
    if cleanup_dark_tail:
        cleaned = self._cv2_cleanup_dark_tail_blob(cleaned, mask_u8)
        if debug_trace is not None:
            debug_trace.append(("dark_tail_cleanup", cleaned.copy()))

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
    if debug_info is not None:
        debug_info.update({
            "cleanup_mask_px": int((mask_u8 > 0).sum()),
            "cloth_cleanup_px": int((cloth_cleanup_u8 > 0).sum()),
            "reference_fill_px": int((reference_fill_u8 > 0).sum()),
            "cleanup_dark_tail": bool(cleanup_dark_tail),
            "prefer_plain_cloth_fill": bool(prefer_plain_cloth_fill),
            "allow_plain_cloth_force": bool(allow_plain_cloth_force),
        })
    if debug_masks is not None:
        debug_masks["cleanup_mask"] = mask_u8.astype(np.float32) / 255.0
        debug_masks["cloth_cleanup"] = cloth_cleanup_u8.astype(np.float32) / 255.0
        debug_masks["reference_fill"] = reference_fill_u8.astype(np.float32) / 255.0
        debug_masks["visible_cloth"] = visible_cloth_u8.astype(np.float32) / 255.0
    if int((visible_cloth_u8 > 0).sum()) >= 80:
        plain_gray = float(np.median(source_gray[visible_cloth_u8 > 0]))
        plain_sat = float(np.median(source_sat[visible_cloth_u8 > 0]))
        use_plain_cloth_force = bool(allow_plain_cloth_force and plain_gray >= 168.0 and plain_sat <= 84.0)
        if prefer_plain_cloth_fill:
            use_plain_cloth_force = True
    if debug_info is not None:
        debug_info.update({
            "plain_gray": float(plain_gray),
            "plain_sat": float(plain_sat),
            "use_plain_cloth_force": bool(use_plain_cloth_force),
        })
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
        if debug_trace is not None:
            debug_trace.append(("plain_fill_reference", reference_fill_rgb.copy()))
        if debug_masks is not None:
            debug_masks["plain_fill_reference"] = reference_fill_mask.astype(np.float32)
    if prefer_plain_cloth_fill and (plain_fill_rgb is not None or reference_fill_rgb is not None):
        cleaned = self._restore_reference_region(
            cleaned,
            plain_fill_rgb if plain_fill_rgb is not None else reference_fill_rgb,
            cloth_cleanup_mask,
            strength=0.985,
        )
        if debug_trace is not None:
            debug_trace.append(("plain_fill_restore", cleaned.copy()))
    else:
        cleaned = self._restore_cloth_overlap_from_source(
            source_rgb=source_rgb,
            current_rgb=cleaned,
            restore_mask=cloth_cleanup_mask,
            final_hair_mask=restore_final_hair_mask,
            tone_reference_rgb=source_rgb,
            tone_reference_mask=cloth_mask,
        )
        if debug_trace is not None:
            debug_trace.append(("source_overlap_restore", cleaned.copy()))
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
            if debug_trace is not None:
                debug_trace.append(("dark_residual_restore", cleaned.copy()))
    cleaned = self._overlay_reference_cloth_fill(
        cleaned,
        reference_fill_rgb,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
    )
    if debug_trace is not None:
        debug_trace.append(("overlay_reference_fill", cleaned.copy()))
    cleaned = self._blend_neighbor_cloth_tone(
        cleaned,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
        reference_rgb=reference_fill_rgb,
    )
    if debug_trace is not None:
        debug_trace.append(("blend_neighbor_tone", cleaned.copy()))
    cleaned = self._cv2_refine_cloth_region(
        cleaned,
        cloth_cleanup_mask,
        reference_rgb=reference_fill_rgb,
        reference_mask=cloth_mask,
    )
    if debug_trace is not None:
        debug_trace.append(("cv2_refine_cloth_region", cleaned.copy()))
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
                strength=0.98,
            )
    cleaned = self._overlay_reference_cloth_fill(
        cleaned,
        reference_fill_rgb,
        cloth_cleanup_mask,
        cloth_mask=cloth_mask,
    )
    if debug_trace is not None:
        debug_trace.append(("overlay_reference_fill_final", cleaned.copy()))
    return cleaned


def bind_cloth_preserve_methods_to_pipeline(cls) -> None:
    """옷 보존/복원 메서드를 MirrAISDPipeline에 바인딩."""
    cls._overlay_reference_cloth_fill = staticmethod(_overlay_reference_cloth_fill)
    cls._blend_neighbor_cloth_tone = staticmethod(_blend_neighbor_cloth_tone)
    cls._restore_cloth_overlap_from_source = _restore_cloth_overlap_from_source
    cls._cleanup_region_with_cloth_restore = _cleanup_region_with_cloth_restore
