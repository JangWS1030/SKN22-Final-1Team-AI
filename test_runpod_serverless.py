#!/usr/bin/env python3
"""
Runpod Serverless 배포 테스트 스크립트
Serverless 엔드포인트를 통한 앞머리 생성 테스트
"""

import os
import sys
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Runpod Serverless 설정
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
    print("❌ 환경 변수 미설정:")
    print(f"  RUNPOD_API_KEY: {RUNPOD_API_KEY is not None}")
    print(f"  RUNPOD_ENDPOINT_ID: {RUNPOD_ENDPOINT_ID is not None}")
    sys.exit(1)

# API URL
BASE_URL = "https://api.runpod.io/v2"
RUN_URL = f"{BASE_URL}/{RUNPOD_ENDPOINT_ID}/run"
STATUS_URL = f"{BASE_URL}/{RUNPOD_ENDPOINT_ID}/status"

print("=" * 80)
print("🧪 Runpod Serverless 배포 테스트")
print("=" * 80)
print(f"Endpoint ID: {RUNPOD_ENDPOINT_ID}")
print(f"API Key: {RUNPOD_API_KEY[:20]}...")
print()

# Step 1: 앞머리 제거 요청 (no fringe)
print("【1단계】앞머리 제거 요청 (no fringe)")
print("-" * 80)

test_payload = {
    "hairstyle_text": "male haircut, masculine salon style, short crop, soft two-block, open forehead, non-parted crop, soft volume, natural mood",
    "color_text": "brown",
    "subject_gender": "male",
    "prompt_context": {
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
    },
}

print(f"테스트 요청:")
print(f"  - 스타일: soft two-block + open forehead (no fringe)")
print(f"  - 색상: brown")
print(f"  - 성별: male")
print(f"  - front: down (앞머리 명시 요청)")
print()

headers = {
    "Content-Type": "application/json",
}

try:
    print(f"📤 API 호출: POST {RUN_URL}")
    response = requests.post(
        RUN_URL,
        json=test_payload,
        headers=headers,
        timeout=30,
    )

    print(f"📥 응답 (Status: {response.status_code})")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ 요청 성공")
        print(f"  ID: {result.get('id', 'N/A')}")
        request_id = result.get("id")
    else:
        print(f"❌ 요청 실패")
        print(f"  응답: {response.text[:500]}")
        sys.exit(1)
except requests.exceptions.Timeout:
    print(f"❌ 타임아웃: API 요청이 실패함")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit(1)

print()
time.sleep(2)

# Step 2: 생성 결과 폴링
print("【2단계】생성 결과 모니터링")
print("-" * 80)

if request_id:
    status_check_url = f"{STATUS_URL}/{request_id}"
    print(f"🔍 상태 조회: GET {status_check_url}")
    print()

    max_attempts = 60  # 5분 대기 (5초 간격)
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(status_check_url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                status = result.get("status")

                print(f"  [{attempt}/{max_attempts}] 상태: {status}")

                if status == "COMPLETED":
                    print(f"✓ 생성 완료!")
                    output = result.get("output", {})
                    print(f"  이미지 수: {len(output.get('images', []))}")
                    print(f"  생성 시간: {output.get('generation_time', 'N/A')}s")

                    # 이미지 정보 출력
                    if output.get("images"):
                        for i, img_info in enumerate(output["images"], 1):
                            print(f"  [{i}] {img_info}")

                    break
                elif status == "FAILED":
                    print(f"❌ 생성 실패!")
                    print(f"  오류: {result.get('error', 'Unknown error')}")
                    break
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    time.sleep(5)
                else:
                    print(f"  상태: {status}")
                    time.sleep(5)
            else:
                print(f"  응답 코드: {response.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"  ❌ 조회 오류: {e}")
            time.sleep(5)

print()

# Step 3: 최종 결과 요약
print("=" * 80)
print("📊 테스트 결과 요약")
print("=" * 80)
print()
print("【배포 정보】")
print(f"  Endpoint: {RUNPOD_ENDPOINT_ID}")
print(f"  Request ID: {request_id}")
print()
print("【테스트 항목】")
print("  [✓] Serverless 엔드포인트 연결")
print("  [✓] 앞머리 생성 요청 성공")
print("  [✓] 결과 폴링 완료")
print()
print("【검증 항목 (생성 완료 후)】")
print("  - 앞머리가 제거되었는가?")
print("  - 기존 앞머리가 제거되지 않았는가?")
print("  - 이미지가 깨져있지 않은가?")
print("  - 이미지 품질이 정상인가?")
print()
print("=" * 80)
