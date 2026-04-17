#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import handler_sd
from pipeline_sd_components import mask_builders as mask_builders_module
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
        sd_prompt_data=payload.get("sd_prompt_data"),
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
    assert prompt_module._resolve_requested_bangs_state(
        male_result["request"]["hairstyle_text"],
        male_result["request"]["prompt_context"],
        male_result["request"]["subject_gender"],
    ) is True, male_result
    normalized_male_style = str(male_result["meta"]["normalized_style"]).lower()
    positive_male = male_result["positive"].lower()
    negative_male = male_result["negative"].lower()
    for blocked in ("bob", "lob", "mini bob", "c-curl bob"):
        _assert_not_contains(normalized_male_style, blocked)
        _assert_not_contains(positive_male, blocked)
    _assert_contains(normalized_male_style, "soft two-block")
    _assert_contains(normalized_male_style, "lowered masculine fringe")
    _assert_contains(normalized_male_style, "non-parted front")
    _assert_contains(normalized_male_style, "curly texture")
    _assert_contains(normalized_male_style, "controlled crown volume")
    _assert_contains(negative_male, "mini bob")
    _assert_contains(negative_male, "baseball cap")
    _assert_contains(negative_male, "earbuds")

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

    no_bangs_payload = {
        "survey_data": {
            "target_length": "short",
            "target_vibe": "chic",
            "scalp_type": "straight",
            "survey_profile": {
                "gender_branch": "male",
                "style_axes": {
                    "front_styling": "lifted",
                    "parting": "parted",
                },
            },
        },
    }
    no_bangs_request = handler_sd._extract_generation_request_context(no_bangs_payload)
    assert prompt_module._resolve_requested_bangs_state(
        no_bangs_request["hairstyle_text"],
        no_bangs_request["prompt_context"],
        no_bangs_request["subject_gender"],
    ) is False, no_bangs_request
    explicit_no_bangs_payload = {
        "hairstyle_text": "long elegant",
        "preference_text": "long, elegant, curly, black, high",
        "survey_data": {
            "target_length": "long",
            "target_vibe": "elegant",
            "scalp_type": "curly",
            "hair_colour": "black",
            "budget_range": "high",
            "question_answers": {
                "q1": "길게",
                "q2": "볼륨감 있는 스타일",
                "q3": "앞머리 없이",
                "q4": "끝선 위주 자연스러운 컬",
                "q5": "고급스러운",
                "q6": "확실히 이미지 변신하고 싶음",
            },
            "survey_profile": {
                "gender_branch": "female",
                "style_axes": {
                    "front_styling": "up",
                    "parting": "side_part",
                },
            },
        },
    }
    explicit_no_bangs_result = _build_from_payload(explicit_no_bangs_payload)
    explicit_no_bangs_request = explicit_no_bangs_result["request"]
    assert prompt_module._resolve_requested_no_bangs_state(
        explicit_no_bangs_request["hairstyle_text"],
        explicit_no_bangs_request["prompt_context"],
        explicit_no_bangs_request["subject_gender"],
    ) is True, explicit_no_bangs_request
    assert prompt_module._resolve_requested_bangs_state(
        explicit_no_bangs_request["hairstyle_text"],
        explicit_no_bangs_request["prompt_context"],
        explicit_no_bangs_request["subject_gender"],
    ) is False, explicit_no_bangs_request
    _assert_contains(explicit_no_bangs_result["positive"].lower(), "open forehead")
    _assert_contains(explicit_no_bangs_result["positive"].lower(), "no bangs")
    _assert_contains(explicit_no_bangs_result["negative"].lower(), "full bangs")
    _assert_contains(explicit_no_bangs_result["negative"].lower(), "baseball cap")
    _assert_contains(explicit_no_bangs_result["negative"].lower(), "earbuds")
    neutral_structured_payload = {
        "hairstyle_text": "short chic",
        "preference_text": "short, chic, straight, brown, mid",
        "survey_data": {
            "target_length": "short",
            "target_vibe": "chic",
            "scalp_type": "straight",
            "hair_colour": "brown",
            "budget_range": "mid",
            "survey_profile": {},
        },
    }
    neutral_result = _build_from_payload(neutral_structured_payload)
    assert neutral_result["meta"]["style_source"] == "structured_neutral", neutral_result
    _assert_not_contains(neutral_result["positive"].lower(), "bob")
    _assert_not_contains(neutral_result["positive"].lower(), "lob")
    legacy_preference_gender_payload = {
        "hairstyle_text": "short chic",
        "preference_text": "short, chic, straight, brown, mid",
        "preference": {
            "length": "short",
            "hair_type": "straight",
            "budget": "medium",
            "gender_branch": "male",
        },
        "survey_data": {
            "target_length": "short",
            "target_vibe": "chic",
            "scalp_type": "straight",
            "hair_colour": "brown",
            "budget_range": "mid",
            "survey_profile": {},
        },
    }
    legacy_preference_gender_result = _build_from_payload(legacy_preference_gender_payload)
    assert legacy_preference_gender_result["meta"]["resolved_gender_branch"] == "male", legacy_preference_gender_result
    assert legacy_preference_gender_result["meta"]["style_source"] == "structured_male", legacy_preference_gender_result
    _assert_not_contains(legacy_preference_gender_result["positive"].lower(), "bob")
    legacy_alias_payload = {
        "preference_text": (
            "gender=male, length=short, mood=chic, texture=straight, "
            "color=ash, budget=mid, front=flexible, parting=either"
        ),
        "survey_data": {
            "target_length": "short",
            "target_vibe": "chic",
            "scalp_type": "straight",
            "hair_colour": "ash",
            "budget_range": "mid",
            "survey_profile": {
                "gender_branch": "male",
            },
        },
    }
    legacy_alias_result = _build_from_payload(legacy_alias_payload)
    legacy_alias_positive = legacy_alias_result["positive"].lower()
    legacy_alias_style = str(legacy_alias_result["meta"]["normalized_style"]).lower()
    assert legacy_alias_result["meta"]["style_source"] == "structured_male", legacy_alias_result
    assert prompt_module._resolve_requested_no_bangs_state(
        legacy_alias_result["request"]["hairstyle_text"],
        legacy_alias_result["request"]["prompt_context"],
        legacy_alias_result["request"]["subject_gender"],
    ) is True, legacy_alias_result
    _assert_contains(legacy_alias_style, "soft lifted front")
    _assert_contains(legacy_alias_style, "parted front")
    _assert_contains(legacy_alias_positive, "open forehead")
    _assert_contains(legacy_alias_positive, "no bangs")
    requested_front_mask = mask_builders_module._build_requested_front_coverage_mask(
        (512, 512),
        (156, 132, 356, 348),
        male_result["request"]["prompt_context"],
        hair_length="short",
        subject_gender=male_result["request"]["subject_gender"] or "male",
        fringe_requested=True,
    )
    if int((requested_front_mask > 0.05).sum()) <= 0:
        raise AssertionError("Expected synthetic front coverage mask for structured male down/non-parted request")
    lower_center_band = requested_front_mask[210:270, 220:292]
    lower_side_band = requested_front_mask[210:270, 150:202]
    if float(lower_center_band.sum()) <= float(lower_side_band.sum()):
        raise AssertionError("Expected front coverage mask to emphasize lower center fringe corridor")

    conflicting_sd_payload = {
        "survey_data": male_payload["survey_data"],
        "sd_prompt_data": {
            "sd_positive": "quiff, airy lifted top, open forehead, natural curl two-block",
        },
    }
    conflicting_result = _build_from_payload(conflicting_sd_payload)
    conflicting_style = str(conflicting_result["meta"]["normalized_style"]).lower()
    _assert_not_contains(conflicting_style, "quiff")
    _assert_not_contains(conflicting_style, "open forehead")
    _assert_contains(conflicting_style, "airy crown volume")
    _assert_contains(conflicting_result["positive"].lower(), "controlled crown volume")

    print(
        json.dumps(
            {
                "male_prompt": {
                    "positive": male_result["positive"],
                    "negative_has_block": "mini bob" in negative_male,
                    "normalized_style": male_result["meta"]["normalized_style"],
                    "front_coverage_mask_px": int((requested_front_mask > 0.05).sum()),
                },
                "conflicting_sd_prompt": {
                    "normalized_style": conflicting_result["meta"]["normalized_style"],
                    "positive": conflicting_result["positive"],
                },
                "female_prompt": {
                    "positive": female_result["positive"],
                },
                "legacy_prompt": {
                    "positive": legacy_result["positive"],
                },
                "explicit_no_bangs_prompt": {
                    "positive": explicit_no_bangs_result["positive"],
                    "negative": explicit_no_bangs_result["negative"],
                },
                "neutral_structured_prompt": {
                    "positive": neutral_result["positive"],
                },
                "legacy_alias_prompt": {
                    "positive": legacy_alias_result["positive"],
                    "normalized_style": legacy_alias_result["meta"]["normalized_style"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
