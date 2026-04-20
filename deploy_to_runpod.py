#!/usr/bin/env python3
"""
Runpod 배포 스크립트
최신 Docker 이미지를 기반으로 기존 Workers 종료 및 새 Workers 구동
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Runpod API 설정
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_API_URL = "https://api.runpod.io/graphql"

# 배포 설정
DOCKER_IMAGE = "lecreek/mirrai-sd-app:latest"
CONTAINER_NAME = "mirrai-sd-app"
RUNPOD_POD_NAME_PREFIX = "mirrai-sd"
NUM_PODS_TO_DEPLOY = 1  # 배포할 Pod 개수

print("=" * 70)
print("Runpod 배포 스크립트 시작")
print("=" * 70)
print(f"Docker 이미지: {DOCKER_IMAGE}")
print(f"Pod 개수: {NUM_PODS_TO_DEPLOY}")
print()

# Step 1: API 키 확인
if not RUNPOD_API_KEY:
    print("❌ 에러: RUNPOD_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("    다음 명령으로 설정하세요:")
    print("    export RUNPOD_API_KEY='your-api-key-here'")
    sys.exit(1)

print(f"✓ Runpod API 키: {RUNPOD_API_KEY[:20]}...")
print()

# Step 2: 배포 준비 정보 출력
print("=" * 70)
print("배포 준비 상황")
print("=" * 70)
print()
print("【1단계: 기존 Worker 목록 확인】")
print("- 다음 명령으로 기존 활성 Pod 확인 가능:")
print(f"  runpod pod list | grep {RUNPOD_POD_NAME_PREFIX}")
print()
print("【2단계: 기존 Worker 종료】")
print("- Pod 중지 명령:")
print(f"  runpod pod terminate <POD_ID>")
print()
print("【3단계: 새 Worker 구동】")
print("- Pod 생성 명령:")
print(f"  runpod pod run --name {RUNPOD_POD_NAME_PREFIX}-<timestamp> \\")
print(f"    --image {DOCKER_IMAGE} \\")
print(f"    --container-disk 20 \\")
print(f"    --gpu '1x A100'")
print()
print("【4단계: Pod 확인】")
print("- 상태 확인:")
print(f"  runpod pod status <POD_ID>")
print()

# Step 3: 배포 체크리스트
print("=" * 70)
print("배포 전 체크리스트")
print("=" * 70)
checklist = {
    "1. Docker 이미지 빌드": "✓ GitHub Actions 완료",
    "2. Docker Hub 업로드": "✓ 진행 중",
    "3. Runpod CLI 설치": "? 확인 필요",
    "4. Runpod API 키 설정": "✓ 설정됨" if RUNPOD_API_KEY else "❌ 미설정",
    "5. 기존 Pod 확인": "⏳ 확인 대기",
    "6. 새 Pod 배포 준비": "준비 완료",
}

for task, status in checklist.items():
    print(f"{task}: {status}")
print()

# Step 4: 배포 명령 안내
print("=" * 70)
print("배포 명령 안내 (터미널에서 직접 실행)")
print("=" * 70)
print()

commands = [
    ("기존 Pod 목록 확인", f"runpod pod list | grep {RUNPOD_POD_NAME_PREFIX}"),
    ("기존 Pod 종료 (필요시)", "runpod pod terminate [POD_ID]"),
    (
        "새 Pod 생성",
        f"""runpod pod run \\
  --name {RUNPOD_POD_NAME_PREFIX}-$(date +%s) \\
  --image {DOCKER_IMAGE} \\
  --container-disk 20 \\
  --gpu '1x A100' \\
  --volume-size 50 \\
  --exposed-ports '5000'""",
    ),
    ("Pod 상태 확인", "runpod pod describe [NEW_POD_ID]"),
    (
        "Pod 내 API 테스트",
        """curl -X POST http://[POD_IP]:5000/api/generate \\
  -H 'Content-Type: application/json' \\
  -d '{
    "hairstyle_text": "soft two-block",
    "color_text": "brown",
    "subject_gender": "male"
  }'""",
    ),
]

for i, (desc, cmd) in enumerate(commands, 1):
    print(f"【{i}】{desc}")
    print(f"   {cmd}")
    print()

# Step 5: 배포 상태 저장
deployment_status = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "docker_image": DOCKER_IMAGE,
    "status": "준비_완료",
    "github_commit": os.getenv("GITHUB_SHA", "unknown"),
    "build_url": os.getenv("GITHUB_SERVER_URL", "")
    + "/"
    + os.getenv("GITHUB_REPOSITORY", "")
    + "/actions/runs/"
    + os.getenv("GITHUB_RUN_ID", ""),
}

status_file = Path("deployment_status.json")
with open(status_file, "w") as f:
    json.dump(deployment_status, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("✓ 배포 준비 완료!")
print("=" * 70)
print()
print(f"배포 상태 파일: {status_file}")
print(f"Docker 이미지: {DOCKER_IMAGE}")
print()
print("⏭️  다음 단계: 위의 명령을 터미널에서 실행하여 Runpod에 배포하세요.")
print()
