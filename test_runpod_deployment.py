#!/usr/bin/env python3
"""
Runpod 배포 테스트 스크립트
새 Pod에 대한 헬스 체크 및 앞머리 생성 테스트
"""

import os
import sys
import requests
import json
import time
from pathlib import Path
from urllib.parse import urljoin

# 테스트 설정
POD_IP = (
    os.getenv("POD_IP")
    or input("Pod IP 주소를 입력하세요 (예: xxx.xxx.xxx.xxx:5000): ").strip()
)
if not POD_IP:
    print("❌ Pod IP가 필요합니다.")
    sys.exit(1)

# URL 정규화
if not POD_IP.startswith("http"):
    POD_IP = f"http://{POD_IP}"
if not POD_IP.endswith(":5000"):
    if ":" not in POD_IP.split("://")[-1]:  # 포트가 없으면 추가
        POD_IP = f"{POD_IP}:5000"

print("=" * 80)
print("🧪 Runpod 배포 테스트")
print("=" * 80)
print(f"Pod URL: {POD_IP}")
print()

# Step 1: 헬스 체크
print("【1단계】헬스 체크")
print("-" * 80)
try:
    response = requests.get(f"{POD_IP}/health", timeout=10)
    if response.status_code == 200:
        health_data = response.json()
        print(f"✓ API 서버 정상 작동")
        print(f"  상태: {health_data.get('status', 'N/A')}")
        print(f"  모델: {health_data.get('model', 'N/A')}")
    else:
        print(f"❌ 헬스 체크 실패 (Status: {response.status_code})")
        sys.exit(1)
except requests.exceptions.Timeout:
    print(f"❌ 타임아웃: Pod 서버가 응답하지 않습니다.")
    print(f"   URL 확인: {POD_IP}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit(1)

print()
time.sleep(1)

# Step 2: 앞머리 제거 테스트 (no fringe)
print("【2단계】앞머리 제거 테스트 (no fringe)")
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

try:
    response = requests.post(
        f"{POD_IP}/api/generate",
        json=test_payload,
        timeout=120,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ 생성 요청 성공 (Status: {response.status_code})")
        print(f"  Request ID: {result.get('request_id', 'N/A')}")
        print(f"  Status: {result.get('status', 'N/A')}")
        request_id = result.get("request_id")
    else:
        print(f"❌ 생성 실패 (Status: {response.status_code})")
        print(f"  응답: {response.text[:500]}")
        sys.exit(1)
except requests.exceptions.Timeout:
    print(f"❌ 타임아웃: 생성 요청이 시간 초과됨 (>120초)")
    print(f"   Pod에서 생성 중일 수 있으니 잠시 후 다시 확인하세요.")
    sys.exit(1)
except Exception as e:
    print(f"❌ 오류: {e}")
    sys.exit(1)

print()
time.sleep(2)

# Step 3: 생성 결과 확인
print("【3단계】생성 결과 확인")
print("-" * 80)

if request_id:
    try:
        for attempt in range(1, 11):
            print(f"  [{attempt}/10] 결과 조회 중...", end=" ")
            response = requests.get(f"{POD_IP}/api/result/{request_id}", timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "completed":
                    print("✓ 완료!")
                    print(f"    이미지 경로: {result.get('image_path', 'N/A')}")
                    print(f"    생성 시간: {result.get('generation_time', 'N/A')}s")
                    break
                elif result.get("status") == "processing":
                    print("처리 중...")
                    time.sleep(5)
                else:
                    print(f"상태: {result.get('status', 'N/A')}")
                    time.sleep(3)
            elif response.status_code == 202:
                print("처리 중...")
                time.sleep(5)
            else:
                print(f"상태 코드: {response.status_code}")
                time.sleep(3)
    except Exception as e:
        print(f"❌ 오류: {e}")

print()

# Step 4: 최종 결과 요약
print("=" * 80)
print("📊 테스트 결과 요약")
print("=" * 80)
print()
print("✓ 배포 테스트 완료!")
print()
print("【검증 항목】")
print("  [✓] API 서버 정상 작동")
print("  [✓] 앞머리 생성 요청 성공")
print("  [✓] 요청 처리 중...")
print()
print("【다음 단계】")
print("1. 생성된 이미지 다운로드")
print("2. 앞머리(fringe) 확인:")
print("   - 앞머리가 제거되었는가?")
print("   - 기존 앞머리가 제거되지 않았는가?")
print("   - 이미지가 깨져있지 않은가?")
print("3. 결과 검증 완료")
print()
print("=" * 80)
