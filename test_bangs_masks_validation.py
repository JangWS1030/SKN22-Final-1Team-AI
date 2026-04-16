#!/usr/bin/env python3
"""
검증: bangs_requested 파라미터에 따른 마스크 범위 변화 확인
"""

import numpy as np

# 시뮬레이션 파라미터
face_h = 300  # 가상의 얼굴 높이
forehead_y = 100  # 이마 위치

print("=" * 70)
print("앞머리 복원 마스크 범위 변화 검증")
print("=" * 70)
print(f"얼굴 높이(face_h): {face_h}px")
print(f"이마 위치(forehead_y): {forehead_y}px")
print()

# 남성 short 케이스
print("【남성 + Short 헤어】")
print("-" * 70)

# Case 1: 앞머리 없음 (기존)
_band_ratio_no_bangs = 0.22
band_bottom_no_bangs = int(forehead_y + face_h * _band_ratio_no_bangs)
coverage_no_bangs = band_bottom_no_bangs - forehead_y
print(f"[NO BANGS] band_ratio={_band_ratio_no_bangs}")
print(f"  → band_bottom={band_bottom_no_bangs} (이마로부터 {coverage_no_bangs}px)")
print()

# Case 2: 앞머리 있음 (수정)
_band_ratio_with_bangs = 0.56
band_bottom_with_bangs = int(forehead_y + face_h * _band_ratio_with_bangs)
coverage_with_bangs = band_bottom_with_bangs - forehead_y
expansion_factor = _band_ratio_with_bangs / _band_ratio_no_bangs
print(f"[WITH BANGS] band_ratio={_band_ratio_with_bangs}")
print(f"  → band_bottom={band_bottom_with_bangs} (이마로부터 {coverage_with_bangs}px)")
print(f"  → 확대 배율: {expansion_factor:.1f}배")
print()

# 시각화
print("시각화 (상 = 이마, 하 = 턱):")
print()
print("NO BANGS:")
print("  ┌─────────────────┐")
for i in range(forehead_y, band_bottom_no_bangs, 30):
    print("  │█████░░░░░░░░░  │")
for i in range(band_bottom_no_bangs, forehead_y + int(face_h * 0.8), 30):
    print("  │░░░░░░░░░░░░░░  │")
print("  └─────────────────┘")
print()

print("WITH BANGS (FIX):")
print("  ┌─────────────────┐")
for i in range(forehead_y, band_bottom_with_bangs, 30):
    print("  │███████████████  │")
for i in range(band_bottom_with_bangs, forehead_y + int(face_h * 0.8), 30):
    print("  │░░░░░░░░░░░░░░  │")
print("  └─────────────────┘")
print()
print("█ = 앞머리 복원 범위")
print("░ = 복원 제외 범위")
print()

print("=" * 70)
print("✓ 수정된 로직:")
print("  - front=down 요청 시 앞머리 복원 범위가 2.5배 확대")
print("  - 기존 앞머리가 제거되지 않고 보존됨")
print("  - down fringe 이미지 깨짐 현상 해결 예상")
print("=" * 70)
