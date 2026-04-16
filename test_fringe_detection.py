#!/usr/bin/env python3
"""
테스트: 앞머리 제거 파라미터 처리 확인 (no fringe)
"""

from pipeline_sd_components import prompt as prompt_module

# 사용자 입력 파라미터
hairstyle_text = "male haircut, masculine salon style, short crop, soft two-block, open forehead, non-parted crop, soft volume, natural mood"
color_text = "brown"
preference_text = "gender=male, length=short, mood=natural, texture=waved, color=brown, budget=low, two_block=soft, front=lifted, parting=non_parted, short crop, non-parted crop, male salon vocabulary only"

# prompt_context 구성 (실제 파이프라인에서 어떻게 구성되는지 확인 필요)
prompt_context = {
    "structured_payload_present": True,
    "gender_branch": "male",
    "canonical_preferences": {
        "target_length": "short",
        "target_vibe": "natural",
        "scalp_type": "waved",
        "hair_colour": "brown",
        "budget_range": "low",
    },
    "style_axes": {
        "front": "down",
        "parting": "non_parted",
        "two_block": "soft",
    },
    "derived_preferences": preference_text,
    "question_answers": {},
    "legacy_fields": {
        "hairstyle_text": hairstyle_text,
        "color_text": color_text,
    },
}

# 테스트 1: bangs_requested 확인
print("[TEST 1] _resolve_requested_bangs_state")
print("=" * 60)
bangs_requested = prompt_module._resolve_requested_bangs_state(
    hairstyle_text, prompt_context, subject_gender="male"
)
print(f"hairstyle_text: {hairstyle_text}")
print(f"preference_text: {preference_text}")
print(f"bangs_requested: {bangs_requested}")
print()

# 테스트 2: no_bangs_requested 확인
print("[TEST 2] _resolve_requested_no_bangs_state")
print("=" * 60)
no_bangs_requested = prompt_module._resolve_requested_no_bangs_state(
    hairstyle_text, prompt_context, subject_gender="male"
)
print(f"no_bangs_requested: {no_bangs_requested}")
print()

# 테스트 3: front_styling 추론
print("[TEST 3] Front styling inference")
print("=" * 60)
front_styling_from_hairstyle = prompt_module._infer_front_styling_from_text(
    hairstyle_text
)
front_styling_from_pref = prompt_module._infer_front_styling_from_text(preference_text)
print(f"front_styling from hairstyle_text: {front_styling_from_hairstyle}")
print(f"front_styling from preference_text: {front_styling_from_pref}")
print()

# 테스트 4: parting 추론
print("[TEST 4] Parting inference")
print("=" * 60)
parting_from_hairstyle = prompt_module._infer_parting_from_text(hairstyle_text)
parting_from_pref = prompt_module._infer_parting_from_text(preference_text)
print(f"parting from hairstyle_text: {parting_from_hairstyle}")
print(f"parting from preference_text: {parting_from_pref}")
print()

# 테스트 5: 정규화된 프롬프트 컨텍스트
print("[TEST 5] Normalized prompt context")
print("=" * 60)
normalized = prompt_module._normalize_prompt_context(prompt_context)
print(f"gender_branch: {normalized.get('gender_branch')}")
print(f"canonical_preferences: {normalized.get('canonical_preferences')}")
print(f"style_axes: {normalized.get('style_axes')}")
print()
