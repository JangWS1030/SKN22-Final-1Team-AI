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

from .config import (
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
    FEMALE_SUBJECT_PIPELINE_PROFILE,
    MALE_SUBJECT_PIPELINE_PROFILE,
    NEUTRAL_SUBJECT_PIPELINE_PROFILE,
    PROJECT_ROOT,
    SD_INPAINT_MODEL_ID,
    SD_SIZE,
    SubjectPipelineProfile,
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
    _WHITE_TSHIRT_NEGATIVE_HINTS,
    _WHITE_TSHIRT_POSITIVE_HINTS,
)

logger = logging.getLogger(__name__)

# Extracted from pipeline_sd_inpainting.py to keep MirrAISDPipeline smaller.

_CANONICAL_TARGET_LENGTHS = frozenset({"short", "medium", "long", "bob"})
_CANONICAL_TARGET_VIBES = frozenset({"natural", "chic", "cute", "elegant"})
_CANONICAL_SCALP_TYPES = frozenset({"straight", "waved", "curly", "damaged"})
_CANONICAL_HAIR_COLOURS = frozenset({"black", "brown", "ash", "bleach"})
_CANONICAL_BUDGET_RANGES = frozenset({"low", "mid", "high"})

_MALE_BRANCH_BLOCKED_VOCAB = (
    "bob",
    "lob",
    "mini bob",
    "c-curl bob",
    "feminine bob silhouette",
    "female-coded framing terms",
)
_MALE_BRANCH_NEGATIVE_TERMS = (
    "bob",
    "lob",
    "mini bob",
    "c-curl bob",
    "rounded bob",
    "rounded lob",
    "feminine bob silhouette",
    "feminine short bob",
    "female-coded framing terms",
    "feminine face-framing layers",
    "feminine face-framing panels",
    "curved inward bob ends",
)

_DOWN_FRONT_TEXT_TOKENS = (
    "bang",
    "bangs",
    "fringe",
    "앞머리",
    "시스루",
    "내리는 스타일",
    "다운펌",
    "down style",
    "down perm",
    "front=down",
    "front_styling=down",
    "front_style=down",
)
_LIFTED_FRONT_TEXT_TOKENS = (
    "front=flexible",
    "front_styling=flexible",
    "front_style=flexible",
    "front=up",
    "front_styling=up",
    "front_style=up",
    "front=lifted",
    "front_styling=lifted",
    "front_style=lifted",
    "앞머리 올림",
    "앞머리 올려",
    "올린 앞머리",
    "앞머리 업",
    "이마 보이게",
    "이마를 보이게",
    "이마를 드러내",
    "이마를 드러낸",
    "open forehead",
    "exposed forehead",
    "forehead exposed",
    "lifted front",
    "up style",
    "front up",
    "slicked back",
    "slick back",
    "swept back",
    "swept-back",
    "regent",
)
_NON_PARTED_TEXT_TOKENS = (
    "비가르마",
    "non_parted",
    "non-parted",
    "no_part",
)
_SIDE_PART_TEXT_TOKENS = (
    "parting=either",
    "part=either",
    "parting=flexible",
    "part=flexible",
    "either part",
    "either-part",
    "가르마 상관없",
    "가르마 자유",
    "side part",
    "side-part",
)
_CENTER_PART_TEXT_TOKENS = (
    "middle part",
    "middle-part",
    "center part",
    "center-part",
)


def _normalize_choice(value: Any, allowed: frozenset[str]) -> str:
    lowered = str(value or "").strip().lower()
    return lowered if lowered in allowed else ""


def _normalize_axis_key(key: Any) -> str:
    return str(key or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _infer_front_styling_from_text(text: Any) -> str:
    lowered = _normalize_text(text)
    if not lowered:
        return ""
    if any(token in lowered for token in _LIFTED_FRONT_TEXT_TOKENS):
        return "lifted"
    if any(token in lowered for token in _DOWN_FRONT_TEXT_TOKENS):
        return "down"
    return ""


def _infer_parting_from_text(text: Any) -> str:
    lowered = _normalize_text(text)
    if not lowered:
        return ""
    if any(token in lowered for token in _NON_PARTED_TEXT_TOKENS):
        return "non_parted"
    if any(token in lowered for token in _CENTER_PART_TEXT_TOKENS):
        return "center_part"
    if any(token in lowered for token in _SIDE_PART_TEXT_TOKENS):
        return "side_part"
    return ""


def _canonicalize_axis_value(axis_key: str, value: Any) -> str:
    norm_key = _normalize_axis_key(axis_key)
    normalized = _normalize_axis_key(value)
    lowered = _normalize_text(value)

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
        inferred = _infer_front_styling_from_text(lowered)
        if inferred:
            return inferred

    if norm_key in {"parting", "part"}:
        if normalized in {"non_parted", "nonparted", "no_part"}:
            return "non_parted"
        if normalized in {"side_part", "sidepart", "parted"}:
            return "side_part"
        if normalized in {"middle_part", "middlepart", "center_part", "centerpart"}:
            return "center_part"
        if normalized in {"either", "flexible", "any"}:
            return "side_part"
        inferred = _infer_parting_from_text(lowered)
        if inferred:
            return inferred

    return normalized


def _normalize_style_axes(style_axes: Any) -> Dict[str, Any]:
    if not isinstance(style_axes, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in style_axes.items():
        norm_key = _normalize_axis_key(key)
        if norm_key:
            normalized[norm_key] = value
    return normalized


def _iter_text_fragments(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = " ".join(value.strip().split())
        return [text] if text else []
    if isinstance(value, dict):
        fragments: List[str] = []
        for item in value.values():
            fragments.extend(_iter_text_fragments(item))
        return fragments
    if isinstance(value, (list, tuple, set)):
        fragments: List[str] = []
        for item in value:
            fragments.extend(_iter_text_fragments(item))
        return fragments
    return _iter_text_fragments(str(value))


def _flatten_text(value: Any) -> str:
    fragments = _iter_text_fragments(value)
    return " ".join(fragment for fragment in fragments if fragment).strip()


def _dedupe_prompt_parts(parts: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for part in parts:
        text = " ".join(str(part or "").strip().split()).strip(", ")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _normalize_prompt_context(prompt_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = prompt_context if isinstance(prompt_context, dict) else {}
    canonical = raw.get("canonical_preferences")
    if not isinstance(canonical, dict):
        canonical = {}

    normalized = {
        "structured_payload_present": bool(raw.get("structured_payload_present")),
        "fallback_mode": bool(raw.get("fallback_mode")),
        "gender_branch": _normalize_subject_gender(raw.get("gender_branch")),
        "canonical_preferences": {
            "target_length": _normalize_choice(canonical.get("target_length"), _CANONICAL_TARGET_LENGTHS),
            "target_vibe": _normalize_choice(canonical.get("target_vibe"), _CANONICAL_TARGET_VIBES),
            "scalp_type": _normalize_choice(canonical.get("scalp_type"), _CANONICAL_SCALP_TYPES),
            "hair_colour": _normalize_choice(canonical.get("hair_colour"), _CANONICAL_HAIR_COLOURS),
            "budget_range": _normalize_choice(canonical.get("budget_range"), _CANONICAL_BUDGET_RANGES),
        },
        "style_axes": _normalize_style_axes(raw.get("style_axes")),
        "derived_preferences": raw.get("derived_preferences"),
        "question_answers": raw.get("question_answers"),
        "legacy_fields": raw.get("legacy_fields") if isinstance(raw.get("legacy_fields"), dict) else {},
    }
    normalized["structured_payload_present"] = bool(
        normalized["structured_payload_present"]
        or normalized["gender_branch"]
        or any(normalized["canonical_preferences"].values())
        or normalized["style_axes"]
        or _flatten_text(normalized["derived_preferences"])
        or _flatten_text(normalized["question_answers"])
    )
    return normalized


def _resolve_requested_hair_length(
    hairstyle_text: str,
    prompt_context: Optional[Dict[str, Any]] = None,
) -> str:
    context = _normalize_prompt_context(prompt_context)
    target_length = context["canonical_preferences"].get("target_length", "")
    if target_length == "medium":
        return "medium"
    if target_length == "long":
        return "long"
    if target_length in {"short", "bob"}:
        return "short"
    return _classify_hair_length(hairstyle_text)


def _stringify_structured_request(
    prompt_context: Dict[str, Any],
    *,
    include_legacy_fields: bool = True,
) -> str:
    text_sources: List[str] = []
    if include_legacy_fields:
        text_sources.append(_flatten_text(prompt_context.get("legacy_fields")))
    text_sources.extend([
        _flatten_text(prompt_context.get("question_answers")),
        _flatten_text(prompt_context.get("derived_preferences")),
        _flatten_text(prompt_context.get("style_axes")),
    ])
    return " ".join(
        text
        for text in text_sources
        if text
    ).strip()


def _resolve_axis_value(style_axes: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        norm_key = _normalize_axis_key(key)
        if norm_key not in style_axes:
            continue
        value = style_axes.get(norm_key)
        if isinstance(value, dict):
            for candidate in ("value", "label", "name", "slug", "id"):
                if candidate in value:
                    normalized = _canonicalize_axis_value(norm_key, value.get(candidate))
                    if normalized:
                        return normalized
            flattened = _canonicalize_axis_value(norm_key, _flatten_text(value))
            if flattened:
                return flattened
            continue
        normalized = _canonicalize_axis_value(norm_key, _flatten_text(value))
        if normalized:
            return normalized
    return ""


def _resolve_requested_bangs_state(
    hairstyle_text: str,
    prompt_context: Optional[Dict[str, Any]] = None,
    subject_gender: Optional[str] = None,
) -> bool:
    if _resolve_requested_no_bangs_state(
        hairstyle_text,
        prompt_context,
        subject_gender=subject_gender,
    ):
        return False

    lowered = _normalize_text(hairstyle_text)
    if any(token in lowered for token in ("bang", "bangs", "fringe", "앞머리", "시스루")):
        return True

    context = _normalize_prompt_context(prompt_context)
    style_axes = context["style_axes"]
    structured_text = _stringify_structured_request(
        context,
        include_legacy_fields=True,
    ).lower()
    normalized_gender = _normalize_subject_gender(subject_gender) or context.get("gender_branch", "")

    front_styling = _resolve_axis_value(style_axes, "front_styling", "front_style", "front")
    parting = _resolve_axis_value(style_axes, "parting", "part")

    if not front_styling:
        front_styling = _infer_front_styling_from_text(f"{lowered} {structured_text}")
    if not parting:
        parting = _infer_parting_from_text(structured_text)

    if front_styling in {"down", "down_style", "down_perm", "fringe", "bang", "bangs"}:
        return True

    if _infer_front_styling_from_text(structured_text) == "down":
        return True

    if (
        normalized_gender == "male"
        and parting in {"non_parted", "nonparted", "no_part"}
        and any(
            token in structured_text
            for token in ("비가르마", "soft", "부드", "컬", "curly", "wavy", "wave")
        )
    ):
        return True

    return False


def _resolve_requested_no_bangs_state(
    hairstyle_text: str,
    prompt_context: Optional[Dict[str, Any]] = None,
    subject_gender: Optional[str] = None,
) -> bool:
    no_bangs_tokens = (
        "no bangs",
        "without bangs",
        "no fringe",
        "without fringe",
        "open forehead",
        "exposed forehead",
        "forehead exposed",
        "앞머리 없이",
        "앞머리 없음",
        "앞머리 없는",
        "앞머리 x",
        "앞머리 없는 스타일",
        "이마 보이",
        "이마가 보이",
        "이마를 드러",
    )
    lowered = _normalize_text(hairstyle_text)
    if any(token in lowered for token in no_bangs_tokens):
        return True
    if _infer_front_styling_from_text(lowered) == "lifted":
        return True

    context = _normalize_prompt_context(prompt_context)
    style_axes = context["style_axes"]
    structured_text = _stringify_structured_request(
        context,
        include_legacy_fields=True,
    ).lower()
    normalized_gender = _normalize_subject_gender(subject_gender) or context.get("gender_branch", "")

    front_styling = _resolve_axis_value(style_axes, "front_styling", "front_style", "front")
    parting = _resolve_axis_value(style_axes, "parting", "part")
    if not front_styling:
        front_styling = _infer_front_styling_from_text(f"{lowered} {structured_text}")
    if not parting:
        parting = _infer_parting_from_text(structured_text)

    if front_styling in {
        "lifted",
        "up",
        "up_style",
        "updo",
        "slick_back",
        "back",
        "open_forehead",
        "forehead_open",
    }:
        return True

    if any(token in structured_text for token in no_bangs_tokens):
        return True

    if (
        normalized_gender != "male"
        and front_styling in {"parted", "side_part", "middle_part", "center_part"}
        and parting in {"parted", "side_part", "middle_part", "center_part"}
        and any(
            token in structured_text
            for token in ("앞머리", "fringe", "bang")
        )
    ):
        return False

    return False


# ── 사용자 부정 태그 처리 ─────────────────────────────────────────────────────

# 태그 → SD negative prompt 확장 매핑
_USER_NEGATIVE_TAG_EXPANSIONS: Dict[str, str] = {
    "perm":   "perm, permed hair, tight perm waves, spiral perm, chemical perm, wave perm",
    "curl":   "curly hair, tight curls, spiral curls, ringlet curls, coiled hair",
    "wave":   "wavy hair, beach waves, loose waves, natural waves, wavy texture",
    "bangs":  "full bangs, blunt bangs, heavy fringe, thick curtain bangs, forehead-covering front hair",
    "bang":   "full bangs, blunt bangs, heavy fringe, thick curtain bangs",
    "fringe": "full bangs, heavy fringe, thick fringe, forehead-covering front hair",
    "color":  "hair color change, hair dye, unnatural hair color",
}

# "no X" 패턴 → 정규 태그 추출
_NO_TAG_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("no perm",    "perm"),
    ("no curl",    "curl"),
    ("no curls",   "curl"),
    ("no wave",    "wave"),
    ("no waves",   "wave"),
    ("no bang",    "bangs"),
    ("no bangs",   "bangs"),
    ("no fringe",  "fringe"),
    ("without perm",   "perm"),
    ("without curl",   "curl"),
    ("without wave",   "wave"),
    ("without bang",   "bangs"),
    ("without fringe", "fringe"),
    ("펌 없이",    "perm"),
    ("펌 없는",    "perm"),
    ("컬 없이",    "curl"),
    ("웨이브 없이", "wave"),
    ("앞머리 없이", "bangs"),
    ("앞머리 없는", "bangs"),
)


def _extract_no_tags_from_text(text: str) -> List[str]:
    """텍스트에서 'no X' 패턴을 감지해 정규 태그 목록을 반환 (중복 제거)."""
    lowered = _normalize_text(text)
    found: List[str] = []
    seen: set = set()
    for pattern, tag in _NO_TAG_PATTERNS:
        if pattern in lowered and tag not in seen:
            seen.add(tag)
            found.append(tag)
    return found


def _resolve_requested_no_perm_curl_state(
    hairstyle_text: str,
    prompt_context: Optional[Dict[str, Any]] = None,
) -> bool:
    """'펌 없이' / 'no perm' / 'no curl' / 'no wave' 요청 감지."""
    no_perm_curl_tokens = (
        "no perm", "no curl", "no curls", "no wave", "no waves",
        "without perm", "without curl", "without wave",
        "straight only", "펌 없이", "펌 없는", "컬 없이", "웨이브 없이",
    )
    lowered = _normalize_text(hairstyle_text)
    if any(token in lowered for token in no_perm_curl_tokens):
        return True

    context = _normalize_prompt_context(prompt_context)
    structured_text = _stringify_structured_request(context, include_legacy_fields=True).lower()
    if any(token in structured_text for token in no_perm_curl_tokens):
        return True

    user_negative_tags = context.get("user_negative_tags") or []
    if isinstance(user_negative_tags, (list, tuple)):
        return any(t in ("perm", "curl", "wave") for t in user_negative_tags)
    return False


def _expand_user_negative_tags(tags: List[str]) -> str:
    """정규 태그 목록을 SD negative prompt 문자열로 확장."""
    parts: List[str] = []
    seen: set = set()
    for tag in tags:
        key = str(tag).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        expanded = _USER_NEGATIVE_TAG_EXPANSIONS.get(key)
        if expanded:
            parts.append(expanded)
        else:
            parts.append(key)
    return ", ".join(parts)


def _male_explicit_female_coded_request(
    legacy_text: str,
    prompt_context: Dict[str, Any],
    *,
    include_legacy_text: bool = True,
) -> bool:
    structured_text = _stringify_structured_request(
        prompt_context,
        include_legacy_fields=include_legacy_text,
    )
    text_sources = [structured_text]
    if include_legacy_text:
        text_sources.insert(0, legacy_text)
    combined = " ".join(text for text in text_sources if text).lower()
    if not combined:
        return False
    explicit_terms = (
        "bob",
        "lob",
        "mini bob",
        "c-curl bob",
        "단발",
        "보브",
    )
    return any(term in combined for term in explicit_terms)


def _build_male_structured_style_text(
    prompt_context: Dict[str, Any],
    hair_length: str,
    legacy_text: str,
) -> Tuple[str, List[str]]:
    canonical = prompt_context["canonical_preferences"]
    style_axes = prompt_context["style_axes"]
    structured_text = _stringify_structured_request(
        prompt_context,
        include_legacy_fields=False,
    )
    combined_text = " ".join(
        text for text in [structured_text or legacy_text] if text
    ).lower()
    target_length = canonical.get("target_length") or ("short" if hair_length == "short" else hair_length)
    target_vibe = canonical.get("target_vibe")
    scalp_type = canonical.get("scalp_type")
    two_block = _resolve_axis_value(style_axes, "two_block", "two block")
    front_styling = _resolve_axis_value(style_axes, "front_styling", "front_style", "front")
    parting = _resolve_axis_value(style_axes, "parting", "part")

    if not two_block and "투블럭" in combined_text:
        two_block = "soft" if "soft" in combined_text or "부드" in combined_text else "natural"
    if not front_styling:
        front_styling = _infer_front_styling_from_text(combined_text)
    if not parting:
        parting = _infer_parting_from_text(combined_text)

    explicit_bob_request = _male_explicit_female_coded_request(
        legacy_text,
        prompt_context,
        include_legacy_text=False,
    )
    blocked_vocabulary = [] if explicit_bob_request else list(_MALE_BRANCH_BLOCKED_VOCAB)

    parts: List[str] = []
    if target_length in {"short", "bob"}:
        if two_block == "soft":
            parts.append("male short soft two-block haircut")
        elif two_block:
            parts.append("male short natural two-block haircut")
        else:
            parts.append("male clean short crop haircut")
        parts.append("clean side line")
    elif target_length == "medium":
        if two_block == "soft":
            parts.append("male medium soft two-block haircut")
        elif two_block:
            parts.append("male medium natural two-block haircut")
        else:
            parts.append("male medium layered haircut")
        parts.append("controlled side silhouette")
    else:
        parts.append("male layered hairstyle")

    if front_styling == "down":
        parts.append("down style")
    elif front_styling in {"lifted", "up", "up_style"}:
        parts.append("soft lifted front")
        parts.append("open forehead")
        parts.append("no bangs")
    elif front_styling:
        parts.append("natural front styling")

    if parting in {"non_parted", "nonparted", "no_part"}:
        parts.append("non-parted front")
    elif parting in {"parted", "side_part", "middle_part", "center_part"}:
        parts.append("parted front")

    if scalp_type == "curly":
        if front_styling == "down":
            parts.append("curly texture with a soft down perm finish")
        else:
            parts.append("curly texture")
    elif scalp_type == "waved":
        parts.append("soft volume")
    elif scalp_type == "straight":
        parts.append("clean straight texture")
    elif scalp_type == "damaged":
        parts.append("soft controlled texture")

    if target_vibe == "natural":
        parts.append("natural mood")
    elif target_vibe == "chic":
        parts.append("chic mood")
    elif target_vibe == "cute":
        parts.append("soft clean mood")
    elif target_vibe == "elegant":
        parts.append("refined clean mood")

    parts.extend([
        "clean ear contour",
        "balanced masculine framing" if target_length not in {"short", "bob"} else "",
    ])

    return ", ".join(_dedupe_prompt_parts(parts)), blocked_vocabulary


def _build_female_structured_style_text(
    prompt_context: Dict[str, Any],
    hair_length: str,
    legacy_text: str,
) -> str:
    canonical = prompt_context["canonical_preferences"]
    style_axes = prompt_context["style_axes"]
    structured_text = _stringify_structured_request(
        prompt_context,
        include_legacy_fields=False,
    )
    combined_text = " ".join(
        text for text in [structured_text, legacy_text] if text
    ).lower()
    target_length = canonical.get("target_length")
    target_vibe = canonical.get("target_vibe")
    scalp_type = canonical.get("scalp_type")
    front_styling = _resolve_axis_value(style_axes, "front_styling", "front_style", "front")
    parting = _resolve_axis_value(style_axes, "parting", "part")
    if not front_styling:
        front_styling = _infer_front_styling_from_text(combined_text)
    if not parting:
        parting = _infer_parting_from_text(combined_text)
    no_bangs_requested = _resolve_requested_no_bangs_state(
        legacy_text,
        prompt_context,
        subject_gender="female",
    )
    bangs_requested = _resolve_requested_bangs_state(
        legacy_text,
        prompt_context,
        subject_gender="female",
    )

    seed_parts: List[str] = []
    if target_length == "bob":
        seed_parts.append("short bob")
    elif target_length == "short":
        seed_parts.append("short layered cut")
    elif target_length == "medium":
        seed_parts.append("medium layered hair")
    elif target_length == "long":
        seed_parts.append("long layered hair")

    if target_vibe == "natural":
        seed_parts.append("natural")
    elif target_vibe == "chic":
        seed_parts.append("sleek chic")
    elif target_vibe == "cute":
        seed_parts.append("soft cute")
    elif target_vibe == "elegant":
        seed_parts.append("elegant")

    if scalp_type == "curly":
        seed_parts.append("curly")
    elif scalp_type == "waved":
        seed_parts.append("wavy")
    elif scalp_type == "straight":
        seed_parts.append("straight")

    if no_bangs_requested or front_styling in {"lifted", "up", "up_style"}:
        seed_parts.append("open forehead")
        seed_parts.append("no bangs")
    elif (
        front_styling in {"down", "down_style", "down_perm", "fringe", "bang", "bangs"}
        or bangs_requested
        or any(token in combined_text for token in ("bang", "bangs", "fringe", "앞머리", "시스루"))
    ):
        seed_parts.append("soft see-through bangs")

    if parting in {"side_part", "parted"}:
        seed_parts.append("soft side part")
    elif parting in {"middle_part", "center_part"}:
        seed_parts.append("center part")

    synthetic_text = ", ".join(seed_parts)
    return _normalize_hairstyle_prompt_text(
        synthetic_text or legacy_text,
        hair_length,
        subject_gender="female",
    ) or synthetic_text


def _build_neutral_structured_style_text(
    prompt_context: Dict[str, Any],
    hair_length: str,
    legacy_text: str,
) -> str:
    canonical = prompt_context["canonical_preferences"]
    style_axes = prompt_context["style_axes"]
    structured_text = _stringify_structured_request(
        prompt_context,
        include_legacy_fields=False,
    )
    combined_text = " ".join(
        text for text in [structured_text, legacy_text] if text
    ).lower()
    target_length = canonical.get("target_length") or hair_length
    target_vibe = canonical.get("target_vibe")
    scalp_type = canonical.get("scalp_type")
    front_styling = _resolve_axis_value(style_axes, "front_styling", "front_style", "front")
    parting = _resolve_axis_value(style_axes, "parting", "part")
    if not front_styling:
        front_styling = _infer_front_styling_from_text(combined_text)
    if not parting:
        parting = _infer_parting_from_text(combined_text)

    parts: List[str] = []
    if target_length in {"short", "bob"} or hair_length == "short":
        parts.extend([
            "clean short haircut",
            "controlled short silhouette",
            "hair ending above the neckline",
        ])
    elif target_length == "medium" or hair_length == "medium":
        parts.extend([
            "clean medium haircut",
            "controlled medium silhouette",
        ])
    else:
        parts.append("clean layered hairstyle")

    if front_styling in {"down", "down_style", "down_perm", "fringe", "bang", "bangs"}:
        parts.append("soft front coverage")
    elif front_styling in {"lifted", "up", "up_style"}:
        parts.append("lifted front")
    elif "bang" in combined_text or "fringe" in combined_text or "앞머리" in combined_text:
        parts.append("soft front fringe")

    if parting in {"non_parted", "nonparted", "no_part"}:
        parts.append("non-parted front")
    elif parting in {"parted", "side_part", "middle_part", "center_part"}:
        parts.append("parted front")

    if scalp_type == "straight":
        parts.append("clean straight finish")
    elif scalp_type == "waved":
        parts.append("soft natural wave")
    elif scalp_type == "curly":
        parts.append("soft curly texture")
    elif scalp_type == "damaged":
        parts.append("soft controlled texture")

    if target_vibe == "natural":
        parts.append("natural clean mood")
    elif target_vibe == "chic":
        parts.append("refined chic mood")
    elif target_vibe == "cute":
        parts.append("soft tidy mood")
    elif target_vibe == "elegant":
        parts.append("refined polished mood")

    if hair_length == "short":
        parts.append("balanced side framing")
    return ", ".join(_dedupe_prompt_parts(parts))

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
    explicit = _normalize_subject_gender(subject_gender)
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

def _resolve_subject_pipeline_profile(
    subject_gender: Optional[str],
) -> SubjectPipelineProfile:
    normalized = _normalize_subject_gender(subject_gender)
    if normalized == "male":
        return MALE_SUBJECT_PIPELINE_PROFILE
    if normalized == "female":
        return FEMALE_SUBJECT_PIPELINE_PROFILE
    return NEUTRAL_SUBJECT_PIPELINE_PROFILE

def _normalize_male_short_hairstyle_prompt_text(hairstyle_text: str) -> str:
    raw = " ".join(str(hairstyle_text or "").strip().split())
    lowered = raw.lower()
    hints: List[str] = []

    _lifted_front = any(
        token in lowered
        for token in ("lifted front", "open forehead", "lifted", "swept back", "slick back", "slicked back", "앞머리 올", "올리는")
    )

    if any(token in lowered for token in ("mullet", "wolf cut", "soft mullet")):
        base_style = "modern masculine layered wolf cut with controlled soft mullet balance"
        hints.extend([
            "textured crown and top layers",
            "controlled nape length",
            "soft temple coverage",
        ])
    elif any(
        token in lowered
        for token in ("swept-back", "swept back", "side part", "side-part", "dandy", "two block", "two-block", "comma", "regent")
    ):
        base_style = "clean masculine layered haircut with shorter back and sides"
        if _lifted_front:
            hints.extend([
                "controlled top volume swept back",
                "forehead fully exposed",
                "balanced side silhouette",
            ])
        else:
            hints.extend([
                "controlled top volume",
                "soft front movement",
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
            "controlled top texture",
        ])

    _has_bang_token = "bang" in lowered or "fringe" in lowered
    _bang_negated = "no bang" in lowered or "no fringe" in lowered or "without bang" in lowered or "without fringe" in lowered
    if _has_bang_token and not _bang_negated:
        hints.append("soft masculine fringe with natural forehead coverage")
    elif _lifted_front:
        hints.append("forehead fully exposed, top hair swept upward and back, no hair touching forehead")
    else:
        hints.append("natural masculine hairline with balanced forehead coverage")

    _has_texture_token = any(token in lowered for token in ("wave", "wavy", "curl", "curly", "perm"))
    _texture_negated = (
        "no perm" in lowered or "no curl" in lowered or "no wave" in lowered
        or "without perm" in lowered or "without curl" in lowered or "without wave" in lowered
    )
    if _has_texture_token and not _texture_negated:
        hints.append("light natural texture")
    elif any(token in lowered for token in ("straight", "sleek")):
        hints.append("soft natural finish")

    hints.append("clean ear contour")
    hints.append("no feminine bob silhouette")
    hints.append("no dangling side locks")

    parts = [base_style]
    for hint in hints:
        if hint not in parts:
            parts.append(hint)
    return ", ".join(parts)

def _normalize_male_medium_hairstyle_prompt_text(hairstyle_text: str) -> str:
    raw = " ".join(str(hairstyle_text or "").strip().split())
    lowered = raw.lower()
    hints: List[str] = []

    if any(token in lowered for token in ("mullet", "wolf cut", "soft mullet", "baby mullet", "mini mullet")):
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

    _has_texture_token_med = any(token in lowered for token in ("wave", "wavy", "curl", "curly", "perm"))
    _texture_negated_med = (
        "no perm" in lowered or "no curl" in lowered or "no wave" in lowered
        or "without perm" in lowered or "without curl" in lowered or "without wave" in lowered
    )
    if _has_texture_token_med and not _texture_negated_med:
        hints.append("light natural texture")
    elif any(token in lowered for token in ("straight", "sleek")):
        hints.append("soft natural finish")

    _has_bang_token_med = "bang" in lowered or "fringe" in lowered
    _bang_negated_med = "no bang" in lowered or "no fringe" in lowered or "without bang" in lowered or "without fringe" in lowered
    if _has_bang_token_med and not _bang_negated_med:
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

def _resolve_male_fringe_prompt_hints(hairstyle_text: str) -> Tuple[str, str]:
    lowered = " ".join(str(hairstyle_text or "").strip().lower().split())
    if not any(token in lowered for token in ("bang", "bangs", "fringe", "앞머리")):
        return "", ""

    # "no bangs" / "no fringe" 등 부정 표현이면 fringe 힌트 일체 적용 안 함
    _bang_negated = (
        "no bang" in lowered or "no fringe" in lowered
        or "without bang" in lowered or "without fringe" in lowered
        or "open forehead" in lowered or "lifted front" in lowered
        or "앞머리 없" in lowered
    )
    if _bang_negated:
        return "", ""

    full_fringe_requested = any(
        token in lowered
        for token in (
            "full fringe",
            "heavy fringe",
            "dense fringe",
            "covering the forehead",
            "covering forehead",
            "forehead covering",
            "down fringe",
            "full bangs",
            "heavy bangs",
            "앞머리 덮",
            "이마 덮",
        )
    )
    if full_fringe_requested:
        positive = "full masculine fringe covering most of the forehead"
        negative = (
            "exposed forehead, lifted quiff, pushed-up front hair, slicked-back front, parted curtain fringe, "
        )
    else:
        positive = "soft masculine fringe with visible forehead coverage"
        negative = "exposed forehead, lifted quiff, pushed-up front hair, slicked-back front, "
    return positive, negative

def _normalize_hairstyle_prompt_text(
    hairstyle_text: str,
    hair_length: str,
    subject_gender: Optional[str] = None,
) -> str:
    raw = " ".join(str(hairstyle_text or "").strip().split())
    if not raw:
        return ""
    gender_mode = _infer_subject_gender(raw, subject_gender)
    if gender_mode == "male":
        if hair_length == "short":
            return _normalize_male_short_hairstyle_prompt_text(raw)
        if hair_length == "medium":
            return _normalize_male_medium_hairstyle_prompt_text(raw)
    if hair_length != "short":
        return raw

    lowered = raw.lower()
    hints: List[str] = []
    if "hush" in lowered or "layer" in lowered:
        hints.append("soft internal bob layers above the jawline")
        hints.append("rounded jaw-length bob silhouette")
    if "blunt" in lowered:
        hints.append("clean blunt bob outline")
    _has_bang = "bang" in lowered or "fringe" in lowered
    _bang_neg = "no bang" in lowered or "no fringe" in lowered or "without bang" in lowered or "without fringe" in lowered
    if _has_bang and not _bang_neg:
        hints.append("soft see-through bangs")
    _has_texture = any(token in lowered for token in ("wave", "wavy", "curl", "curly"))
    _texture_neg = (
        "no curl" in lowered or "no wave" in lowered
        or "without curl" in lowered or "without wave" in lowered
    )
    if _has_texture and not _texture_neg:
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

def _estimate_short_silhouette_penalty(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    cutoff_y: int,
    removal_mask: np.ndarray,
) -> Optional[float]:
    H, W = img_rgb.shape[:2]
    if removal_mask.shape != (H, W):
        return None

    hair_now, _, _ = self._segface_hair_mask(img_rgb, face_bbox)
    hair_now = np.clip(hair_now.astype(np.float32), 0.0, 1.0)
    if int((hair_now > 0.18).sum()) < 80:
        return None

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    corridor = np.zeros((H, W), dtype=np.float32)
    corridor_top = max(0, int(y1 - face_h * 0.22))
    corridor_bottom = min(H, int(cutoff_y + face_h * 1.14))
    corridor_left = max(0, int(x1 - face_w * 0.96))
    corridor_right = min(W, int(x2 + face_w * 0.96))
    if corridor_top >= corridor_bottom or corridor_left >= corridor_right:
        return None
    corridor[corridor_top:corridor_bottom, corridor_left:corridor_right] = 1.0

    side_zone = np.zeros((H, W), dtype=np.float32)
    side_top = max(0, int(y1 + face_h * 0.02))
    side_bottom = min(H, int(cutoff_y + face_h * 0.84))
    left_outer = max(0, int(x1 - face_w * 0.88))
    left_inner = min(W, int(x1 + face_w * 0.18))
    right_inner = max(0, int(x2 - face_w * 0.18))
    right_outer = min(W, int(x2 + face_w * 0.88))
    if side_top < side_bottom:
        if left_outer < left_inner:
            side_zone[side_top:side_bottom, left_outer:left_inner] = 1.0
        if right_inner < right_outer:
            side_zone[side_top:side_bottom, right_inner:right_outer] = 1.0

    lower_zone = np.zeros((H, W), dtype=np.float32)
    lower_top = max(0, int(max(cutoff_y, y2 + face_h * 0.08)))
    lower_bottom = min(H, int(cutoff_y + face_h * 1.12))
    lower_left = max(0, int(x1 - face_w * 0.92))
    lower_right = min(W, int(x2 + face_w * 0.92))
    if lower_top < lower_bottom and lower_left < lower_right:
        lower_zone[lower_top:lower_bottom, lower_left:lower_right] = 1.0

    chest_zone = np.zeros((H, W), dtype=np.float32)
    chest_top = max(0, int(cutoff_y + face_h * 0.05))
    chest_bottom = min(H, int(cutoff_y + face_h * 1.02))
    chest_left = max(0, int(x1 + face_w * 0.02))
    chest_right = min(W, int(x2 - face_w * 0.02))
    if chest_top < chest_bottom and chest_left < chest_right:
        chest_zone[chest_top:chest_bottom, chest_left:chest_right] = 1.0

    removal_supported = np.clip(removal_mask.astype(np.float32), 0.0, 1.0).copy()
    removal_supported[:max(0, int(y2 + face_h * 0.04)), :] = 0.0
    removal_supported *= corridor

    corridor_hair = hair_now * corridor
    if float(corridor_hair.sum()) < 20.0:
        return None

    def _zone_penalty(zone_mask: np.ndarray, threshold: float = 0.08) -> float:
        zone_bool = zone_mask > threshold
        if int(zone_bool.sum()) < 20:
            return 0.0
        return float(np.mean(hair_now[zone_bool]))

    side_penalty = _zone_penalty(side_zone)
    lower_penalty = _zone_penalty(lower_zone)
    chest_penalty = _zone_penalty(chest_zone)
    removal_penalty = _zone_penalty(removal_supported)

    hair_bbox = self._mask_bbox(corridor_hair, threshold=0.20)
    bottom_extension_penalty = 0.0
    width_penalty = 0.0
    if hair_bbox is not None:
        hx1, _, hx2, hy2 = [int(v) for v in hair_bbox]
        bottom_extension_penalty = float(
            np.clip((hy2 - int(y2 + face_h * 0.24)) / max(face_h * 0.90, 1.0), 0.0, 1.0)
        )
        width_ratio = float(max(hx2 - hx1, 1)) / float(face_w)
        width_penalty = float(np.clip((width_ratio - 1.46) / 0.72, 0.0, 1.0))

    return float(
        0.28 * side_penalty
        + 0.24 * lower_penalty
        + 0.18 * removal_penalty
        + 0.16 * bottom_extension_penalty
        + 0.08 * chest_penalty
        + 0.06 * width_penalty
    )

def _estimate_accessory_penalty(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
) -> Optional[float]:
    details = _estimate_accessory_penalty_details(
        self,
        img_rgb,
        face_bbox,
        source_accessory_profile=None,
    )
    if not isinstance(details, dict):
        return None
    return float(details.get("total_penalty", 0.0))


def _build_accessory_region_masks(
    face_bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    H, W = [int(v) for v in image_shape]
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    def _rect_mask(top: float, bottom: float, left: float, right: float) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)
        top_i = max(0, int(round(top)))
        bottom_i = min(H, int(round(bottom)))
        left_i = max(0, int(round(left)))
        right_i = min(W, int(round(right)))
        if top_i < bottom_i and left_i < right_i:
            mask[top_i:bottom_i, left_i:right_i] = 255
        return mask

    return {
        "glasses": _rect_mask(
            y1 - face_h * 0.10,
            y1 + face_h * 0.66,
            x1 - face_w * 0.18,
            x2 + face_w * 0.18,
        ),
        "earring": _rect_mask(
            y1 - face_h * 0.16,
            y2 + face_h * 0.54,
            x1 - face_w * 0.92,
            x2 + face_w * 0.92,
        ),
        "necklace": _rect_mask(
            y2 + face_h * 0.04,
            y2 + face_h * 0.92,
            x1 - face_w * 0.72,
            x2 + face_w * 0.72,
        ),
        "headwear": _rect_mask(
            y1 - face_h * 0.92,
            y1 + face_h * 0.18,
            x1 - face_w * 0.74,
            x2 + face_w * 0.74,
        ),
    }


def _estimate_mask_presence_ratio(
    mask: Optional[np.ndarray],
    region_mask: Optional[np.ndarray],
    *,
    face_area: float,
    threshold: float = 0.08,
    dilate_ksize: int = 7,
) -> float:
    if (
        mask is None
        or region_mask is None
        or mask.ndim != 2
        or region_mask.ndim != 2
        or mask.shape != region_mask.shape
    ):
        return 0.0

    work_u8 = (np.clip(mask.astype(np.float32), 0.0, 1.0) > threshold).astype(np.uint8) * 255
    if dilate_ksize >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        work_u8 = cv2.dilate(work_u8, kernel, iterations=1)
    work_u8 = cv2.bitwise_and(work_u8, region_mask.astype(np.uint8))
    return float((work_u8 > 0).sum()) / max(float(face_area), 1.0)


def _estimate_relative_accessory_penalty(
    candidate_value: float,
    source_value: float,
    *,
    tolerance_abs: float,
    tolerance_scale: float,
    ramp: float,
) -> float:
    allowed_value = max(
        float(tolerance_abs),
        float(source_value) + float(tolerance_abs),
        float(source_value) * float(tolerance_scale),
    )
    excess_value = max(float(candidate_value) - allowed_value, 0.0)
    return float(np.clip(excess_value / max(float(ramp), 1e-6), 0.0, 1.0))


def _estimate_headwear_penalty(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    *,
    hair_mask: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
) -> float:
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return 0.0

    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    face_area = float(max(face_w * face_h, 1))
    regions = _build_accessory_region_masks(face_bbox, (H, W))
    headwear_region = regions.get("headwear")
    if headwear_region is None or int((headwear_region > 0).sum()) == 0:
        return 0.0

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    face_crop = gray[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    face_gray_median = float(np.median(face_crop)) if face_crop.size > 0 else 150.0
    dark_threshold = int(np.clip(face_gray_median - 26.0, 42.0, 148.0))

    protect_mask = np.zeros((H, W), dtype=np.uint8)
    if isinstance(hair_mask, np.ndarray) and hair_mask.shape == (H, W):
        protect_mask = np.maximum(
            protect_mask,
            (np.clip(hair_mask.astype(np.float32), 0.0, 1.0) > 0.12).astype(np.uint8) * 255,
        )
    if isinstance(face_mask, np.ndarray) and face_mask.shape == (H, W):
        protect_mask = np.maximum(
            protect_mask,
            (np.clip(face_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
        )
    protect_mask = cv2.dilate(
        protect_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )

    dark_mask = (gray < dark_threshold).astype(np.uint8) * 255
    dark_mask = cv2.bitwise_and(dark_mask, headwear_region)
    dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(protect_mask))
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    if int((dark_mask > 0).sum()) < max(18, int(face_area * 0.008)):
        return 0.0

    comp_mask = (dark_mask > 0).astype(np.uint8)
    comp_count, labels, stats, _ = cv2.connectedComponentsWithStats(comp_mask, connectivity=8)
    if comp_count <= 1:
        return 0.0

    best_idx = 0
    best_area = 0
    for comp_idx in range(1, comp_count):
        comp_area = int(stats[comp_idx, cv2.CC_STAT_AREA])
        if comp_area > best_area:
            best_area = comp_area
            best_idx = comp_idx
    if best_idx <= 0 or best_area < max(18, int(face_area * 0.008)):
        return 0.0

    largest_mask = (labels == best_idx).astype(np.uint8)
    comp_x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    comp_y = int(stats[best_idx, cv2.CC_STAT_TOP])
    comp_w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    comp_h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    comp_y2 = comp_y + comp_h

    forehead_band = np.zeros((H, W), dtype=np.uint8)
    band_top = max(0, int(round(y1 - face_h * 0.02)))
    band_bottom = min(H, int(round(y1 + face_h * 0.18)))
    band_left = max(0, int(round(x1 - face_w * 0.34)))
    band_right = min(W, int(round(x2 + face_w * 0.34)))
    if band_top < band_bottom and band_left < band_right:
        forehead_band[band_top:band_bottom, band_left:band_right] = 255
    forehead_band_area = max(int((forehead_band > 0).sum()), 1)
    forehead_coverage = float(
        (cv2.bitwise_and(largest_mask * 255, forehead_band) > 0).sum()
    ) / float(forehead_band_area)

    edges = cv2.Canny(gray, 32, 96)
    edge_density = float((edges[largest_mask > 0] > 0).mean()) if best_area > 0 else 1.0

    area_score = float(np.clip((float(best_area) / face_area - 0.02) / 0.16, 0.0, 1.0))
    width_score = float(np.clip((float(comp_w) / float(face_w) - 0.52) / 0.90, 0.0, 1.0))
    height_score = float(np.clip((float(comp_h) / float(face_h) - 0.10) / 0.44, 0.0, 1.0))
    bottom_alignment = float(
        np.clip((float(comp_y2) - float(y1 - face_h * 0.24)) / max(float(face_h) * 0.42, 1.0), 0.0, 1.0)
    )
    smoothness_score = float(np.clip((0.18 - edge_density) / 0.18, 0.0, 1.0))

    return float(
        0.24 * area_score
        + 0.22 * width_score
        + 0.16 * height_score
        + 0.18 * bottom_alignment
        + 0.20 * max(forehead_coverage, smoothness_score)
    )


def _estimate_headwear_brim_score(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    *,
    glasses_mask: Optional[np.ndarray] = None,
) -> float:
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return 0.0

    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    face_area = float(max(face_w * face_h, 1))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    face_crop = gray[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    face_gray_median = float(np.median(face_crop)) if face_crop.size > 0 else 150.0
    dark_threshold = int(np.clip(face_gray_median - 24.0, 40.0, 146.0))

    brim_band = np.zeros((H, W), dtype=np.uint8)
    brim_top = max(0, int(round(y1 - face_h * 0.10)))
    brim_bottom = min(H, int(round(y1 + face_h * 0.24)))
    brim_left = max(0, int(round(x1 - face_w * 0.48)))
    brim_right = min(W, int(round(x2 + face_w * 0.48)))
    if brim_top >= brim_bottom or brim_left >= brim_right:
        return 0.0
    brim_band[brim_top:brim_bottom, brim_left:brim_right] = 255

    center_band = np.zeros((H, W), dtype=np.uint8)
    center_top = max(0, int(round(y1 - face_h * 0.04)))
    center_bottom = min(H, int(round(y1 + face_h * 0.18)))
    center_left = max(0, int(round(x1 - face_w * 0.34)))
    center_right = min(W, int(round(x2 + face_w * 0.34)))
    if center_top < center_bottom and center_left < center_right:
        center_band[center_top:center_bottom, center_left:center_right] = 255

    dark_mask = (gray < dark_threshold).astype(np.uint8) * 255
    dark_mask = cv2.bitwise_and(dark_mask, brim_band)
    if isinstance(glasses_mask, np.ndarray) and glasses_mask.shape == (H, W):
        glasses_u8 = cv2.dilate(
            (np.clip(glasses_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(glasses_u8))

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        iterations=1,
    )
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        iterations=1,
    )
    if int((dark_mask > 0).sum()) < max(16, int(face_area * 0.006)):
        return 0.0

    comp_mask = (dark_mask > 0).astype(np.uint8)
    comp_count, labels, stats, _ = cv2.connectedComponentsWithStats(comp_mask, connectivity=8)
    if comp_count <= 1:
        return 0.0

    best_idx = 0
    best_score = -1.0
    center_band_area = max(int((center_band > 0).sum()), 1)
    for comp_idx in range(1, comp_count):
        comp_area = int(stats[comp_idx, cv2.CC_STAT_AREA])
        if comp_area < max(16, int(face_area * 0.006)):
            continue
        comp_w = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
        component_mask = (labels == comp_idx).astype(np.uint8) * 255
        center_overlap = float((cv2.bitwise_and(component_mask, center_band) > 0).sum()) / float(center_band_area)
        width_ratio = float(comp_w) / float(face_w)
        height_ratio = float(comp_h) / float(face_h)
        score = center_overlap * 1.8 + width_ratio - max(height_ratio - 0.28, 0.0) * 2.0
        if score > best_score:
            best_score = score
            best_idx = comp_idx

    if best_idx <= 0:
        return 0.0

    largest_mask = (labels == best_idx).astype(np.uint8)
    best_area = int(stats[best_idx, cv2.CC_STAT_AREA])
    comp_x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    comp_y = int(stats[best_idx, cv2.CC_STAT_TOP])
    comp_w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    comp_h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    center_overlap = float((cv2.bitwise_and(largest_mask * 255, center_band) > 0).sum()) / float(center_band_area)

    edges = cv2.Canny(gray, 32, 96)
    edge_density = float((edges[largest_mask > 0] > 0).mean()) if best_area > 0 else 1.0
    comp_gray = gray[largest_mask > 0]
    gray_std = float(np.std(comp_gray)) if comp_gray.size > 0 else 255.0

    width_score = float(np.clip((float(comp_w) / float(face_w) - 0.62) / 0.42, 0.0, 1.0))
    area_score = float(np.clip((float(best_area) / face_area - 0.018) / 0.09, 0.0, 1.0))
    thinness_score = float(np.clip((0.30 - float(comp_h) / float(face_h)) / 0.16, 0.0, 1.0))
    position_center = float(comp_y) + float(comp_h) * 0.5
    target_center = float(y1) + float(face_h) * 0.08
    position_score = float(
        np.clip(1.0 - abs(position_center - target_center) / max(float(face_h) * 0.18, 1.0), 0.0, 1.0)
    )
    smoothness_score = float(np.clip((0.20 - edge_density) / 0.20, 0.0, 1.0))
    uniformity_score = float(np.clip((24.0 - gray_std) / 24.0, 0.0, 1.0))

    return float(
        0.24 * width_score
        + 0.20 * center_overlap
        + 0.16 * area_score
        + 0.14 * thinness_score
        + 0.14 * position_score
        + 0.06 * smoothness_score
        + 0.06 * uniformity_score
    )


def _build_accessory_profile(
    self,
    img_rgb: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
    *,
    hair_mask: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
    glasses_mask: Optional[np.ndarray] = None,
    earring_mask: Optional[np.ndarray] = None,
    necklace_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return {
            "glasses_ratio": 0.0,
            "earring_ratio": 0.0,
            "necklace_ratio": 0.0,
            "headwear_score": 0.0,
            "headwear_brim_score": 0.0,
        }

    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)
    face_area = float(max(face_w * face_h, 1))
    regions = _build_accessory_region_masks(face_bbox, (H, W))

    glasses_ratio = _estimate_mask_presence_ratio(
        glasses_mask,
        regions.get("glasses"),
        face_area=face_area,
        threshold=0.08,
        dilate_ksize=5,
    )
    earring_ratio = _estimate_mask_presence_ratio(
        earring_mask,
        regions.get("earring"),
        face_area=face_area,
        threshold=0.08,
        dilate_ksize=7,
    )
    necklace_ratio = _estimate_mask_presence_ratio(
        necklace_mask,
        regions.get("necklace"),
        face_area=face_area,
        threshold=0.08,
        dilate_ksize=7,
    )
    headwear_score = _estimate_headwear_penalty(
        self,
        img_rgb,
        face_bbox,
        hair_mask=hair_mask,
        face_mask=face_mask,
    )
    headwear_brim_score = _estimate_headwear_brim_score(
        self,
        img_rgb,
        face_bbox,
        glasses_mask=glasses_mask,
    )
    return {
        "glasses_ratio": float(glasses_ratio),
        "earring_ratio": float(earring_ratio),
        "necklace_ratio": float(necklace_ratio),
        "headwear_score": float(headwear_score),
        "headwear_brim_score": float(headwear_brim_score),
    }


def _estimate_accessory_penalty_details(
    self,
    img_rgb: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    *,
    source_accessory_profile: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return None

    H, W = img_rgb.shape[:2]
    hair_mask, face_mask, _ = self._segface_hair_mask(img_rgb, face_bbox)
    segface_debug = self._last_segface_mask_debug or {}
    glasses_mask = segface_debug.get("glasses_mask")
    earring_mask = segface_debug.get("earring_mask")
    necklace_mask = segface_debug.get("necklace_mask")
    for accessory_mask in (glasses_mask, earring_mask, necklace_mask):
        if accessory_mask is not None and (
            not isinstance(accessory_mask, np.ndarray) or accessory_mask.shape != (H, W)
        ):
            return None

    candidate_profile = _build_accessory_profile(
        self,
        img_rgb,
        face_bbox,
        hair_mask=hair_mask,
        face_mask=face_mask,
        glasses_mask=glasses_mask,
        earring_mask=earring_mask,
        necklace_mask=necklace_mask,
    )
    source_profile = source_accessory_profile if isinstance(source_accessory_profile, dict) else {}
    source_glasses_ratio = float(source_profile.get("glasses_ratio", 0.0) or 0.0)
    source_earring_ratio = float(source_profile.get("earring_ratio", 0.0) or 0.0)
    source_necklace_ratio = float(source_profile.get("necklace_ratio", 0.0) or 0.0)
    source_headwear_score = float(source_profile.get("headwear_score", 0.0) or 0.0)
    source_headwear_brim_score = float(source_profile.get("headwear_brim_score", 0.0) or 0.0)

    headwear_surface_penalty = _estimate_relative_accessory_penalty(
        candidate_profile["headwear_score"],
        source_headwear_score,
        tolerance_abs=0.08,
        tolerance_scale=1.30,
        ramp=0.45,
    )
    headwear_brim_penalty = _estimate_relative_accessory_penalty(
        candidate_profile["headwear_brim_score"],
        source_headwear_brim_score,
        tolerance_abs=0.10,
        tolerance_scale=1.35,
        ramp=0.40,
    )
    headwear_brim_penalty = float(
        max(
            headwear_brim_penalty,
            np.clip(
                (
                    float(candidate_profile["headwear_brim_score"])
                    - max(
                        0.58,
                        source_headwear_brim_score + 0.08,
                        source_headwear_brim_score * 1.18,
                    )
                ) / 0.20,
                0.0,
                1.0,
            ),
        )
    )
    headwear_penalty = float(max(headwear_surface_penalty, headwear_brim_penalty))
    glasses_penalty = _estimate_relative_accessory_penalty(
        candidate_profile["glasses_ratio"],
        source_glasses_ratio,
        tolerance_abs=0.018,
        tolerance_scale=1.55,
        ramp=0.10,
    )
    earring_penalty = _estimate_relative_accessory_penalty(
        candidate_profile["earring_ratio"],
        source_earring_ratio,
        tolerance_abs=0.008,
        tolerance_scale=1.40,
        ramp=0.06,
    )
    necklace_penalty = _estimate_relative_accessory_penalty(
        candidate_profile["necklace_ratio"],
        source_necklace_ratio,
        tolerance_abs=0.012,
        tolerance_scale=1.45,
        ramp=0.09,
    )
    jewelry_penalty = float(np.clip(0.72 * earring_penalty + 0.28 * necklace_penalty, 0.0, 1.0))
    total_penalty = float(
        np.clip(
            0.52 * headwear_penalty
            + 0.30 * glasses_penalty
            + 0.18 * jewelry_penalty,
            0.0,
            1.0,
        )
    )

    exclude = bool(
        headwear_penalty >= float(getattr(self.config, "accessory_exclusion_headwear_threshold", 0.34))
        or glasses_penalty >= float(getattr(self.config, "accessory_exclusion_glasses_threshold", 0.26))
        or jewelry_penalty >= float(getattr(self.config, "accessory_exclusion_jewelry_threshold", 0.28))
        or total_penalty >= float(getattr(self.config, "accessory_exclusion_penalty_threshold", 0.58))
    )
    return {
        "exclude": exclude,
        "total_penalty": total_penalty,
        "headwear_penalty": float(headwear_penalty),
        "headwear_surface_penalty": float(headwear_surface_penalty),
        "headwear_brim_penalty": float(headwear_brim_penalty),
        "glasses_penalty": float(glasses_penalty),
        "earring_penalty": float(earring_penalty),
        "necklace_penalty": float(necklace_penalty),
        "jewelry_penalty": float(jewelry_penalty),
        "candidate_profile": candidate_profile,
        "source_profile": {
            "glasses_ratio": source_glasses_ratio,
            "earring_ratio": source_earring_ratio,
            "necklace_ratio": source_necklace_ratio,
            "headwear_score": source_headwear_score,
            "headwear_brim_score": source_headwear_brim_score,
        },
    }

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

def _extract_source_garment_prompt_hints(
    self,
    source_rgb: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    torso_candidate_mask: Optional[np.ndarray],
    source_cloth_overlap_mask: Optional[np.ndarray],
    hair_mask_for_removal: Optional[np.ndarray],
    protect_mask: Optional[np.ndarray],
    face_bbox: Tuple[int, int, int, int],
) -> Tuple[Dict[str, Any], np.ndarray]:
    H, W = source_rgb.shape[:2]
    empty_mask = np.zeros((H, W), dtype=np.float32)
    if cloth_mask is None or cloth_mask.shape != (H, W):
        return {}, empty_mask

    cloth_u8 = (np.clip(cloth_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255
    if int((cloth_u8 > 0).sum()) < 120:
        return {}, empty_mask

    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    face_w = max(int(x2 - x1), 1)
    face_h = max(int(y2 - y1), 1)

    support_u8 = cloth_u8.copy()
    if torso_candidate_mask is not None and torso_candidate_mask.shape == (H, W):
        torso_u8 = cv2.dilate(
            (np.clip(torso_candidate_mask.astype(np.float32), 0.0, 1.0) > 0.08).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        torso_supported_u8 = cv2.bitwise_and(support_u8, torso_u8)
        if int((torso_supported_u8 > 0).sum()) >= 160:
            support_u8 = torso_supported_u8

    corridor_u8 = np.zeros((H, W), dtype=np.uint8)
    top = max(0, int(y2 - face_h * 0.04))
    bottom = min(H, int(y2 + face_h * 1.95))
    left = max(0, int(x1 - face_w * 1.10))
    right = min(W, int(x2 + face_w * 1.10))
    if top >= bottom or left >= right:
        return {}, empty_mask
    corridor_u8[top:bottom, left:right] = 255
    support_u8 = cv2.bitwise_and(support_u8, corridor_u8)

    exclude_u8 = np.zeros((H, W), dtype=np.uint8)
    exclusion_specs = (
        (source_cloth_overlap_mask, 0.04, (13, 13)),
        (hair_mask_for_removal, 0.08, (11, 11)),
        (protect_mask, 0.08, (9, 9)),
    )
    for mask, threshold, kernel_size in exclusion_specs:
        if mask is None or mask.shape != (H, W):
            continue
        mask_u8 = (np.clip(mask.astype(np.float32), 0.0, 1.0) > threshold).astype(np.uint8) * 255
        if int((mask_u8 > 0).sum()) <= 0:
            continue
        mask_u8 = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size),
            iterations=1,
        )
        exclude_u8 = cv2.bitwise_or(exclude_u8, mask_u8)
    support_u8 = cv2.bitwise_and(support_u8, cv2.bitwise_not(exclude_u8))

    neckline_cut_y = max(0, int(y2 - face_h * 0.03))
    if neckline_cut_y > 0:
        support_u8[:neckline_cut_y, :] = 0

    support_u8 = cv2.erode(
        support_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    support_u8 = cv2.morphologyEx(
        support_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    if int((support_u8 > 0).sum()) < max(180, int(face_w * face_h * 0.010)):
        return {}, empty_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((support_u8 > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return {}, empty_mask
    best_label = 0
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    if best_label <= 0 or best_area < max(180, int(face_w * face_h * 0.010)):
        return {}, empty_mask

    support_u8 = np.zeros((H, W), dtype=np.uint8)
    support_u8[labels == best_label] = 255
    support_mask = support_u8 > 0
    if int(support_mask.sum()) < 180:
        return {}, empty_mask

    lab_img = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_pixels = lab_img[support_mask]
    median_lab = np.median(lab_pixels, axis=0)

    palette_rgb = {
        "white": (245, 244, 239),
        "ivory": (236, 226, 204),
        "cream": (229, 214, 182),
        "beige": (200, 178, 147),
        "gray": (145, 145, 145),
        "black": (46, 46, 46),
        "navy": (49, 65, 101),
        "blue": (74, 112, 181),
        "brown": (122, 87, 66),
    }
    color_name = "neutral"
    best_dist = float("inf")
    for name, rgb in palette_rgb.items():
        rgb_arr = np.array([[rgb]], dtype=np.uint8)
        pal_lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        dist = float(np.linalg.norm(median_lab - pal_lab))
        if dist < best_dist:
            best_dist = dist
            color_name = name

    gray_img = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.Canny(source_rgb, 80, 160)
    edge_density = float((edges[support_mask] > 0).mean())
    light_std = float(gray_img[support_mask].std())
    chroma_std = float(np.sqrt(lab_pixels[:, 1].var() + lab_pixels[:, 2].var()))
    gx_mean = float(np.abs(gx[support_mask]).mean())
    gy_mean = float(np.abs(gy[support_mask]).mean())
    vertical_texture_ratio = gx_mean / max(gy_mean, 1e-4)

    pattern_type = "solid"
    pattern_confidence = 0.62
    if edge_density > 0.028 and vertical_texture_ratio > 1.35:
        pattern_type = "ribbed"
        pattern_confidence = 0.78
    elif edge_density > 0.030 or light_std > 18.0 or chroma_std > 18.0:
        pattern_type = "textured"
        pattern_confidence = 0.68
    elif edge_density < 0.016 and light_std < 13.0 and chroma_std < 14.0:
        pattern_type = "solid"
        pattern_confidence = 0.84

    material_hint: Optional[str] = None
    material_confidence = 0.0
    if pattern_type == "ribbed":
        material_hint = "ribbed knit"
        material_confidence = 0.82
    elif pattern_type == "textured":
        material_hint = "knit fabric"
        material_confidence = 0.66
    elif pattern_type == "solid" and edge_density < 0.018 and light_std < 12.5:
        material_hint = "smooth fabric"
        material_confidence = 0.58

    ys, xs = np.where(support_mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_w = max(int(x_max - x_min + 1), 1)
    top_profile = np.full(W, H, dtype=np.int32)
    for x in range(x_min, x_max + 1):
        col_ys = np.where(support_u8[:, x] > 0)[0]
        if col_ys.size > 0:
            top_profile[x] = int(col_ys[0])

    def _median_top(x_start: int, x_end: int) -> Optional[float]:
        x_start = max(x_min, x_start)
        x_end = min(x_max + 1, x_end)
        if x_end <= x_start:
            return None
        vals = top_profile[x_start:x_end]
        vals = vals[vals < H]
        if vals.size < 6:
            return None
        return float(np.median(vals))

    center_x = int(0.5 * (x_min + x_max))
    center_top = _median_top(center_x - int(bbox_w * 0.12), center_x + int(bbox_w * 0.12))
    left_top = _median_top(x_min + int(bbox_w * 0.08), x_min + int(bbox_w * 0.26))
    right_top = _median_top(x_max - int(bbox_w * 0.26), x_max - int(bbox_w * 0.08))

    neckline_hint: Optional[str] = None
    neckline_confidence = 0.0
    if center_top is not None:
        side_candidates = [v for v in (left_top, right_top) if v is not None]
        if side_candidates:
            side_top = float(np.median(side_candidates))
            center_drop = float(center_top - side_top)
            if center_drop > max(10.0, face_h * 0.06):
                neckline_hint = "v-neck"
                neckline_confidence = 0.79
            elif center_drop > max(4.0, face_h * 0.03):
                neckline_hint = "round"
                neckline_confidence = 0.56

    negative_color_map = {
        "white": ["black clothes", "navy clothes", "blue clothes"],
        "ivory": ["black clothes", "navy clothes", "blue clothes"],
        "cream": ["black clothes", "navy clothes", "blue clothes"],
        "beige": ["black clothes", "navy clothes", "blue clothes"],
        "gray": ["bright blue clothes", "cream clothes"],
        "black": ["white clothes", "cream clothes", "bright blue clothes"],
        "navy": ["white clothes", "cream clothes", "beige clothes"],
        "blue": ["white clothes", "cream clothes", "beige clothes"],
        "brown": ["blue clothes", "navy clothes", "bright white clothes"],
        "neutral": [],
    }

    hints: Dict[str, Any] = {
        "color_name": color_name,
        "pattern_type": pattern_type,
        "material_hint": material_hint,
        "neckline_hint": neckline_hint,
        "negative_color_hints": negative_color_map.get(color_name, []),
        "support_pixels": int(support_mask.sum()),
        "confidence": {
            "color": round(max(0.35, 1.0 - min(best_dist, 42.0) / 42.0), 3),
            "pattern": round(pattern_confidence, 3),
            "material": round(material_confidence, 3),
            "neckline": round(neckline_confidence, 3),
        },
        "stats": {
            "edge_density": round(edge_density, 4),
            "light_std": round(light_std, 3),
            "chroma_std": round(chroma_std, 3),
            "vertical_texture_ratio": round(vertical_texture_ratio, 3),
        },
    }
    return hints, support_u8.astype(np.float32) / 255.0

def _build_prompt(
    hairstyle_text: str,
    color_text: str,
    hair_length: str = "long",
    subject_gender: Optional[str] = None,
    sd_prompt_data: Optional[Dict[str, Any]] = None,
    prompt_context: Optional[Dict[str, Any]] = None,
    source_garment_hints: Optional[Dict[str, Any]] = None,
    white_tshirt_experiment: bool = False,
) -> Tuple[str, str, float, Dict[str, Any]]:
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

    def _compact_prompt_parts(parts: List[str], max_words: int = 48) -> str:
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

    normalized_color = _normalize_color_text(color_text)
    normalized_prompt_context = _normalize_prompt_context(prompt_context)
    explicit_branch = normalized_prompt_context.get("gender_branch")
    gender_mode = explicit_branch or _infer_subject_gender(
        hairstyle_text, subject_gender
    )
    explicit_no_bangs_requested = _resolve_requested_no_bangs_state(
        hairstyle_text,
        normalized_prompt_context,
        subject_gender=gender_mode,
    )
    explicit_no_perm_curl_requested = _resolve_requested_no_perm_curl_state(
        hairstyle_text,
        normalized_prompt_context,
    )
    # 페이로드 negative_prompt / preference.exclude 에서 파싱된 태그
    user_negative_tags: List[str] = list(normalized_prompt_context.get("user_negative_tags") or [])
    # haystyle_text 내 "no X" 패턴도 자동 반영 (backward compat)
    for _tag in _extract_no_tags_from_text(hairstyle_text):
        if _tag not in user_negative_tags:
            user_negative_tags.append(_tag)
    subject_profile = _resolve_subject_pipeline_profile(gender_mode)
    male_fringe_positive_hint = ""
    male_fringe_negative_hint = ""
    if gender_mode == "male":
        male_fringe_positive_hint, male_fringe_negative_hint = _resolve_male_fringe_prompt_hints(
            hairstyle_text
        )
    no_bangs_positive_hint = ""
    no_bangs_negative_hint = ""
    if explicit_no_bangs_requested:
        # 남성: 앞머리 없음 = 짧게 치고 올리는 스타일 → upswept/lifted 키워드 강화
        # 여성: 앞머리 없음 = 기르거나 가르마로 넘기는 스타일 → parted/swept 키워드 사용
        if gender_mode == "male":
            no_bangs_positive_hint = (
                "open forehead, no bangs, forehead fully exposed, "
                "hair lifted upward and back, clean exposed hairline"
            )
            no_bangs_negative_hint = (
                "full bangs, blunt bangs, heavy fringe, thick curtain bangs, forehead-covering front hair, "
                "thick face-covering front panels, dense cheek-covering side fringe, "
                "hair falling onto forehead, forward falling top hair, hair sweeping over forehead, "
                "hair touching forehead, front hair down, hair over eyes, "
            )
        else:
            no_bangs_positive_hint = (
                "open forehead, no bangs, forehead fully exposed, "
                "hair parted away from forehead, side-swept style"
            )
            no_bangs_negative_hint = (
                "full bangs, blunt bangs, heavy fringe, thick curtain bangs, forehead-covering front hair, "
                "thick face-covering front panels, dense cheek-covering side fringe, "
                "hair falling onto forehead, forward falling top hair, hair sweeping over forehead, "
                "hair touching forehead, front hair down, hair over eyes, "
            )
    no_perm_curl_negative_hint = ""
    if explicit_no_perm_curl_requested:
        no_perm_curl_negative_hint = (
            "perm, permed hair, tight perm waves, spiral perm, chemical perm, "
            "curly hair, tight curls, spiral curls, coiled hair, "
            "wavy hair, beach waves, loose waves, wavy texture, "
        )
    # 사용자 전용 negative 태그 확장 (bangs/perm/curl/wave 중복은 위 hint에 흡수)
    _filtered_user_tags = [
        t for t in user_negative_tags
        if t not in ("bangs", "bang", "fringe") or not explicit_no_bangs_requested
        if t not in ("perm", "curl", "wave") or not explicit_no_perm_curl_requested
    ]
    user_negative_hint = _expand_user_negative_tags(_filtered_user_tags)
    if user_negative_hint:
        user_negative_hint += ", "
    structured_payload_used = bool(normalized_prompt_context.get("structured_payload_present"))
    neutral_safe_structured_short = (
        structured_payload_used
        and gender_mode == "neutral"
        and hair_length == "short"
        and not _male_explicit_female_coded_request(
            hairstyle_text,
            normalized_prompt_context,
            include_legacy_text=True,
        )
    )
    male_bob_blocking_enabled = (
        gender_mode == "male"
        and not _male_explicit_female_coded_request(
            hairstyle_text,
            normalized_prompt_context,
            include_legacy_text=not structured_payload_used,
        )
    )
    blocked_vocabulary: List[str] = []
    style_source = "legacy_text"
    normalized_style = ""
    if structured_payload_used and gender_mode == "male":
        normalized_style, blocked_vocabulary = _build_male_structured_style_text(
            normalized_prompt_context,
            hair_length,
            legacy_text=hairstyle_text,
        )
        style_source = "structured_male"
    elif structured_payload_used and gender_mode == "female":
        normalized_style = _build_female_structured_style_text(
            normalized_prompt_context,
            hair_length,
            legacy_text=hairstyle_text,
        )
        style_source = "structured_female"
    elif neutral_safe_structured_short:
        normalized_style = _build_neutral_structured_style_text(
            normalized_prompt_context,
            hair_length,
            legacy_text=hairstyle_text,
        )
        style_source = "structured_neutral"
    if not normalized_style:
        style_source = "legacy_text"
        normalized_style = _normalize_hairstyle_prompt_text(
            hairstyle_text,
            hair_length,
            subject_gender=gender_mode,
        )
    normalized_style_lower = normalized_style.lower()
    if (
        no_bangs_positive_hint
        and "open forehead" in normalized_style_lower
        and "no bangs" in normalized_style_lower
    ):
        no_bangs_positive_hint = ""
    if male_bob_blocking_enabled and not blocked_vocabulary:
        blocked_vocabulary = list(_MALE_BRANCH_BLOCKED_VOCAB)
    preserve_source_garment = True
    garment_priority_parts: List[str] = []
    garment_positive_parts: List[str] = []
    garment_negative_parts: List[str] = []
    if white_tshirt_experiment:
        garment_positive_parts.extend(list(_WHITE_TSHIRT_POSITIVE_HINTS))
        garment_negative_parts.extend(list(_WHITE_TSHIRT_NEGATIVE_HINTS))
    elif preserve_source_garment:
        garment_positive_parts.extend([
            "same original upper garment",
            "natural shoulder garment continuity",
            "consistent upper garment shape",
            "preserved neckline coverage",
        ])
        garment_negative_parts.extend([
            "unrelated new outfit",
            "dramatically changed clothing style",
            "dramatically changed clothing color",
            "salon cape",
            "salon gown",
            "open neckline",
            "deep v-neck",
            "plunging neckline",
            "exposed chest",
            "armor-like chest panel",
            "bib-like front panel",
            "structured breastplate top",
        ])
        if gender_mode == "male":
            garment_negative_parts.extend([
                "dress",
                "blouse",
                "camisole",
                "off-shoulder top",
            ])

    garment_hints = source_garment_hints if isinstance(source_garment_hints, dict) else {}
    if not white_tshirt_experiment and subject_profile.use_source_garment_prompt_hints:
        garment_conf = garment_hints.get("confidence") if isinstance(garment_hints.get("confidence"), dict) else {}
        color_conf = float(garment_conf.get("color", 0.0) or 0.0)
        pattern_conf = float(garment_conf.get("pattern", 0.0) or 0.0)
        material_conf = float(garment_conf.get("material", 0.0) or 0.0)
        neckline_conf = float(garment_conf.get("neckline", 0.0) or 0.0)
        color_name = str(garment_hints.get("color_name") or "").strip().lower()
        pattern_type = str(garment_hints.get("pattern_type") or "").strip().lower()
        material_hint = str(garment_hints.get("material_hint") or "").strip().lower()
        neckline_hint = str(garment_hints.get("neckline_hint") or "").strip().lower()

        garment_negative_parts.extend([
            "armor-like chest panel",
            "bib-like front panel",
            "structured breastplate top",
            "warped clothing",
        ])
        if color_conf >= 0.45:
            if color_name == "white":
                garment_priority_parts.append("clean white upper garment")
            elif color_name in {"ivory", "cream", "beige"}:
                garment_priority_parts.append("soft light upper garment tone")
        if pattern_conf >= 0.68:
            if pattern_type == "solid":
                garment_priority_parts.append("plain unpatterned upper garment")
            elif pattern_type == "ribbed":
                garment_priority_parts.append("subtle ribbed fabric texture")
            elif pattern_type == "textured":
                garment_priority_parts.append("light fabric texture")
        if material_conf >= 0.72:
            if material_hint == "smooth fabric":
                garment_priority_parts.append("soft smooth fabric")
            elif material_hint == "ribbed knit":
                garment_priority_parts.append("fine rib texture")
        if neckline_conf >= 0.56 and neckline_hint == "round":
            garment_priority_parts.append("round crew neckline")
        elif neckline_conf >= 0.72 and neckline_hint == "v-neck":
            garment_negative_parts.extend([
                "deep v-neck",
                "plunging neckline",
                "wide v-neck blouse",
            ])

        negative_color_hints = garment_hints.get("negative_color_hints")
        if isinstance(negative_color_hints, list) and color_name in {"white", "ivory", "cream", "beige"}:
            for item in negative_color_hints:
                text = str(item).strip()
                if text:
                    garment_negative_parts.append(text)
    else:
        garment_negative_parts.extend([
            "deep v-neck",
            "plunging neckline",
            "wide neckline",
            "warped clothing",
        ])

    garment_positive_parts = garment_priority_parts + garment_positive_parts

    deduped_positive_parts: List[str] = []
    seen_positive: set[str] = set()
    for part in garment_positive_parts:
        key = str(part).strip().lower()
        if not key or key in seen_positive:
            continue
        seen_positive.add(key)
        deduped_positive_parts.append(str(part).strip())
    deduped_negative_parts: List[str] = []
    seen_negative: set[str] = set()
    for part in garment_negative_parts:
        key = str(part).strip().lower()
        if not key or key in seen_negative:
            continue
        seen_negative.add(key)
        deduped_negative_parts.append(str(part).strip())

    garment_positive_hint = ", ".join(deduped_positive_parts[:3])
    garment_positive_hint = ", ".join(
        deduped_positive_parts[: subject_profile.garment_positive_hint_limit]
    )
    if hair_length == "short" and not white_tshirt_experiment:
        short_garment_parts = [
            part for part in deduped_positive_parts
            if part in {"same original upper garment"}
        ]
        if short_garment_parts:
            garment_positive_hint = ", ".join(short_garment_parts)
    garment_negative_hint = ", ".join(deduped_negative_parts)
    if garment_negative_hint:
        garment_negative_hint += ", "

    subject_noun = "person"
    if gender_mode == "male":
        subject_noun = "man"
    elif gender_mode == "female":
        subject_noun = "woman"

    # ── DB 프롬프트 데이터가 있으면 우선 사용 ─────────────────────────────
    if sd_prompt_data and sd_prompt_data.get("sd_positive") and not structured_payload_used:
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

        primary_positive = f"professional portrait photo of a {subject_noun} with {style_part}"
        if garment_positive_hint:
            primary_positive = f"{primary_positive}, {garment_positive_hint}"
        positive_parts = [
            primary_positive,
        ]
        if gender_mode == "male":
            positive_parts.append("natural masculine portrait framing")
        if color_pos_hint:
            positive_parts.append(color_pos_hint)
        if male_fringe_positive_hint:
            positive_parts.append(male_fringe_positive_hint)
        if no_bangs_positive_hint:
            positive_parts.append(no_bangs_positive_hint)
        positive_parts.extend([
            "clean neckline",
            "photorealistic, natural lighting, sharp focus",
        ])
        positive = _compact_prompt_parts(positive_parts)
        negative_base = _NEGATIVE_BASE + ", " + _COMMON_STYLE_BLOCK_NEGATIVE
        negative = (
            sd_neg
            + (", " if sd_neg else "")
            + user_negative_hint
            + male_fringe_negative_hint
            + color_neg_hint
            + no_bangs_negative_hint
            + no_perm_curl_negative_hint
            + garment_negative_hint
            + negative_base
        )

        prompt_meta = {
            "resolved_gender_branch": gender_mode,
            "canonical_preferences": normalized_prompt_context.get("canonical_preferences", {}),
            "blocked_vocabulary": blocked_vocabulary,
            "fallback_mode": bool(normalized_prompt_context.get("fallback_mode")),
            "structured_payload_used": structured_payload_used,
            "style_source": "sd_prompt_data",
            "normalized_style": style_part,
        }
        return positive, negative, guidance, prompt_meta

    parts = []
    if normalized_style:
        parts.append(normalized_style)
    if normalized_color:
        parts.append(f"{normalized_color.strip()} hair color")
    style_word_limit = 26 if structured_payload_used else 18
    style = _truncate_words(", ".join(parts) if parts else "natural hairstyle", style_word_limit)

    # 길이별 기본 보강 (직접 입력/DB 프롬프트 폴백 시 사용)
    if hair_length == "short" and gender_mode == "male":
        pos_suffix = (
            ", masculine short cut, balanced forehead, clean temple line, defined sideburn connection, tidy temple transition, no side tails, no jewelry"
        )
        if male_fringe_positive_hint:
            pos_suffix += ", masculine fringe covering the forehead"
        neg_prefix = (
            "bixie, pixie bob, "
            "oversized exposed forehead, exaggerated high hairline, receding hairline, severe slicked-back hair, "
            "earring, earrings, hoop earrings, stud earrings, ear cuff, jewelry, necklace, makeup, "
        )
        guidance = 10.9
    elif hair_length == "short" and neutral_safe_structured_short:
        pos_suffix = (
            ", clean short silhouette, clear neckline, above the neckline, no long side panels, no hair on chest"
        )
        neg_prefix = (
            "very long hair, medium hair, medium length hair, medium-length hair, shoulder-length hair, "
            "shoulder grazing hair, shoulder-grazing hair, collarbone-length hair, lob, rounded bob, oversized bob, "
            "flowing long hair, hair below shoulders, long face-framing panels, curtain side panels, long side tails, "
            "dangling lower tails, hair touching clothes, strands on chest, "
        )
        guidance = 10.4
    elif hair_length == "short":
        pos_suffix = (
            ", precise cropped chin-length bob, hair ending above the jawline, tucked inward ends at the jawline, compact cheek-hugging side silhouette, clear jaw contour, exposed lower neck, above shoulders, no strands below chin, no long tails, no hair on chest, no curtain side panels"
        )
        neg_prefix = (
            "very long hair, medium hair, medium length hair, medium-length hair, shoulder-length hair, "
            "shoulder grazing hair, shoulder-grazing hair, collarbone-length hair, lob, "
            "flowing long hair, hair below shoulders, waist-length hair, side long locks over chest, "
            "center-part curtain hair, center-part long hair, curtain bangs with long side panels, "
            "long straight front panels, long face-framing panels, chest-covering curtain hair, "
            "long vertical side panels, elongated front curtains, dangling front sheets, long front curtains over cheeks, "
            "long hush cut, long wolf cut, mullet tails, long layers below jawline, "
            "hair touching shoulders, hair covering collar, chest-length strands, neckline covered by hair, "
            "hair below jawline, hair below neckline, dangling lower tails, long side tails, nape tails, side columns below chin, "
            "strands touching clothes, side locks on shoulders, hair covering blouse, "
            "overly voluminous hair, puffy hair, oversized bob, wide helmet shape, bulky side volume, "
            "blunt horizontal cut line, helmet hair, bowl-shaped edge, "
        )
        guidance = 12.2
    elif hair_length == "medium" and gender_mode == "male":
        pos_suffix = (
            ", masculine medium cut, balanced forehead, centered volume, natural sideburn connection, tidy temple transition, no side sweep, no jewelry"
        )
        if male_fringe_positive_hint:
            pos_suffix += ", masculine fringe covering the forehead"
        neg_prefix = (
            "dangling earrings, hoop earrings, necklace, jewelry, "
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

    primary_positive = f"professional portrait photo of a {subject_noun} with {style}{pos_suffix}"
    if garment_positive_hint:
        primary_positive = f"{primary_positive}, {garment_positive_hint}"
    positive_parts = [
        primary_positive,
    ]
    if hair_length == "short":
        if gender_mode == "male":
            positive_parts.append("clean short masculine silhouette, hair mass ending above the neckline")
        elif neutral_safe_structured_short:
            positive_parts.append("clean short silhouette, hair mass ending above the neckline")
        else:
            positive_parts.append("strict short bob silhouette, hair mass ending above the neckline")
    if male_fringe_positive_hint:
        positive_parts.append(male_fringe_positive_hint)
    if no_bangs_positive_hint:
        positive_parts.append(no_bangs_positive_hint)
    if color_pos_hint:
        positive_parts.append(color_pos_hint)
    positive_parts.extend([
        "clean neckline",
        "balanced framing",
        "photorealistic portrait",
    ])
    positive = _compact_prompt_parts(positive_parts)
    negative_base = (
        _NEGATIVE_BASE
        + ", cropped head, cropped hair, cut off hair, top of head out of frame, tight close-up portrait, clipped hairstyle"
        + ", "
        + _COMMON_STYLE_BLOCK_NEGATIVE
    )
    male_block_negative = ""
    if gender_mode == "male" and blocked_vocabulary:
        male_block_negative = ", ".join(_MALE_BRANCH_NEGATIVE_TERMS) + ", "
    negative = (
        neg_prefix
        + user_negative_hint
        + male_block_negative
        + male_fringe_negative_hint
        + color_neg_hint
        + no_bangs_negative_hint
        + no_perm_curl_negative_hint
        + garment_negative_hint
        + negative_base
    )

    prompt_meta = {
        "resolved_gender_branch": gender_mode,
        "canonical_preferences": normalized_prompt_context.get("canonical_preferences", {}),
        "blocked_vocabulary": blocked_vocabulary,
        "fallback_mode": bool(normalized_prompt_context.get("fallback_mode")),
        "structured_payload_used": structured_payload_used,
        "style_source": style_source,
        "normalized_style": normalized_style,
    }
    return positive, negative, guidance, prompt_meta

def bind_prompt_methods_to_pipeline(cls) -> None:
    cls._classify_hair_length = staticmethod(_classify_hair_length)
    cls._normalize_color_text = staticmethod(_normalize_color_text)
    cls._normalize_subject_gender = staticmethod(_normalize_subject_gender)
    cls._infer_subject_gender = staticmethod(_infer_subject_gender)
    cls._normalize_prompt_context = staticmethod(_normalize_prompt_context)
    cls._resolve_requested_hair_length = staticmethod(_resolve_requested_hair_length)
    cls._resolve_requested_no_bangs_state = staticmethod(_resolve_requested_no_bangs_state)
    cls._resolve_requested_bangs_state = staticmethod(_resolve_requested_bangs_state)
    cls._resolve_subject_pipeline_profile = staticmethod(_resolve_subject_pipeline_profile)
    cls._normalize_male_short_hairstyle_prompt_text = staticmethod(_normalize_male_short_hairstyle_prompt_text)
    cls._normalize_male_medium_hairstyle_prompt_text = staticmethod(_normalize_male_medium_hairstyle_prompt_text)
    cls._normalize_hairstyle_prompt_text = staticmethod(_normalize_hairstyle_prompt_text)
    cls._resolve_target_hair_lab = staticmethod(_resolve_target_hair_lab)
    cls._estimate_hair_color_distance = _estimate_hair_color_distance
    cls._estimate_short_tail_penalty = _estimate_short_tail_penalty
    cls._estimate_short_silhouette_penalty = _estimate_short_silhouette_penalty
    cls._build_accessory_profile = _build_accessory_profile
    cls._estimate_accessory_penalty_details = _estimate_accessory_penalty_details
    cls._estimate_accessory_penalty = _estimate_accessory_penalty
    cls._estimate_hair_shape_profile = _estimate_hair_shape_profile
    cls._estimate_male_medium_fit_penalty = _estimate_male_medium_fit_penalty
    cls._estimate_mask_mass_center_offset = staticmethod(_estimate_mask_mass_center_offset)
    cls._preserve_original_hair_tone = _preserve_original_hair_tone
    cls._extract_source_garment_prompt_hints = _extract_source_garment_prompt_hints
    cls._build_prompt = staticmethod(_build_prompt)
