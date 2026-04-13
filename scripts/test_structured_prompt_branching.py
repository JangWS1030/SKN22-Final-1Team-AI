#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import handler_sd
from pipeline_sd_components import prompt as prompt_module


def _build_from_payload(payload: dict) -> dict:
    request = handler_sd._extract_generation_request_context(payload)
    hair_length = prompt_module._resolve_requested_hair_length(
        request["hairstyle_text"],
        request["prompt_context"],
    )
    positive, negative, guidance, meta = prompt_module._build_prompt(
        request["hairstyle_text"],
        request["color_text"],
        hair_length,
        subject_gender=request["subject_gender"],
        prompt_context=request["prompt_context"],
    )
    return {
        "request": request,
        "hair_length": hair_length,
        "positive": positive,
        "negative": negative,
        "guidance": guidance,
        "meta": meta,
    }


def _assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"Expected `{needle}` in `{text}`")


def _assert_not_contains(text: str, needle: str) -> None:
    if needle in text:
        raise AssertionError(f"Did not expect `{needle}` in `{text}`")


def main() -> int:
    male_payload = {
        "hairstyle_text": "Sleek Mini Bob",
        "preference_text": "lob",
        "survey_data": {
            "target_length": "short",
            "target_vibe": "natural",
            "scalp_type": "curly",
            "hair_colour": "black",
            "budget_range": "mid",
            "question_answers": [
                "아주 짧고 깔끔하게",
                "자연스러운 투블럭",
                "내리는 스타일",
                "비가르마",
                "컬감",
                "부드러운",
            ],
            "survey_profile": {
                "gender_branch": "male",
                "style_axes": {
                    "two_block": "soft",
                    "front_styling": "down",
                    "parting": "non_parted",
                },
                "derived_preferences": {
                    "finish": "soft",
                },
            },
        },
    }
    male_result = _build_from_payload(male_payload)
    assert male_result["hair_length"] == "short", male_result
    assert male_result["meta"]["resolved_gender_branch"] == "male", male_result
    assert male_result["meta"]["style_source"] == "structured_male", male_result
    assert male_result["request"]["structured_payload_used"] is True, male_result
    normalized_male_style = str(male_result["meta"]["normalized_style"]).lower()
    positive_male = male_result["positive"].lower()
    negative_male = male_result["negative"].lower()
    for blocked in ("bob", "lob", "mini bob", "c-curl bob"):
        _assert_not_contains(normalized_male_style, blocked)
        _assert_not_contains(positive_male, blocked)
    _assert_contains(normalized_male_style, "soft two-block")
    _assert_contains(normalized_male_style, "down style")
    _assert_contains(normalized_male_style, "non-parted front")
    _assert_contains(normalized_male_style, "curly texture")
    _assert_contains(negative_male, "mini bob")

    female_payload = {
        "survey_data": {
            "target_length": "bob",
            "target_vibe": "chic",
            "scalp_type": "straight",
            "hair_colour": "brown",
            "budget_range": "high",
            "survey_profile": {
                "gender_branch": "female",
                "style_axes": {
                    "parting": "parted",
                },
            },
        },
    }
    female_result = _build_from_payload(female_payload)
    assert female_result["hair_length"] == "short", female_result
    assert female_result["meta"]["resolved_gender_branch"] == "female", female_result
    assert female_result["meta"]["style_source"] == "structured_female", female_result
    _assert_contains(female_result["positive"].lower(), "bob")

    legacy_payload = {
        "preference_text": "short bob cut, chic",
        "color_text": "ash brown",
    }
    legacy_result = _build_from_payload(legacy_payload)
    assert legacy_result["request"]["structured_payload_used"] is False, legacy_result
    assert legacy_result["meta"]["style_source"] == "legacy_text", legacy_result
    _assert_contains(legacy_result["positive"].lower(), "bob")

    print(
        json.dumps(
            {
                "male_prompt": {
                    "positive": male_result["positive"],
                    "negative_has_block": "mini bob" in negative_male,
                    "normalized_style": male_result["meta"]["normalized_style"],
                },
                "female_prompt": {
                    "positive": female_result["positive"],
                },
                "legacy_prompt": {
                    "positive": legacy_result["positive"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
