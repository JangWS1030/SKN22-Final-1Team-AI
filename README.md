# MirrAI SD Inpainting RunPod Repo

현재 저장소는 실제 운영 중인 SD 인페인팅 서비스 경로만 남긴 정리본입니다.  
레거시 생성 파이프라인, 미사용 랜드마크 백엔드, 구형 RunPod 보조 경로는 제거했고, 현재 기준 실행/배포 흐름은 `SD runtime -> Docker -> GitHub Actions -> RunPod`입니다.

## 저장소 범위

- SD 인페인팅 추론 경로 유지
- RunPod 배포/스모크 테스트 경로 유지
- generation 학습/평가 스크립트 유지
- 로컬 테스트, 학습 산출물, 대용량 모델 파일은 업로드 제외

## 핵심 파일

- `handler_sd.py`: RunPod serverless 엔트리포인트
- `pipeline_sd_inpainting.py`: 실제 SD 추론 파이프라인
- `runtime_download.py`: 런타임 모델 캐시 준비
- `download_weights_sd.py`: SD 관련 가중치 다운로드 보조 스크립트
- `entrypoint_sd.sh`: 컨테이너 시작 스크립트
- `Dockerfile.sd.base`: 공통 의존성 베이스 이미지
- `Dockerfile.sd.app`: 현재 CI/CD에서 사용하는 앱 이미지
- `Dockerfile.sd`: 수동 단일 이미지 빌드용

## 현재 파이프라인 구성

- SegFace 기반 hair mask
- SAM2 기반 refinement
- MediaPipe FaceMesh 기반 face protect mask
- Stable Diffusion inpainting
- ControlNet canny
- IP-Adapter face

## 의존성

- `requirements.txt`: 현재 서비스 런타임 의존성
- `requirements-train.txt`: 학습/평가용 추가 의존성
- `requirements-dev.txt`: 로컬 개발용 추가 의존성

런타임 설치:

```bash
python -m pip install -r requirements.txt
```

학습/평가 환경 설치:

```bash
python -m pip install -r requirements-train.txt
```

## 로컬/업로드 제외 범위

아래 항목은 로컬 전용으로 유지하고 업로드 대상에서 제외합니다.

- `tests/`
- `images/`
- `output/`, `cmd/`
- `dataset_build/`
- `pretrained_models/`
- 생성된 `.docx`

## 남겨둔 문서

- `README_release_runbook.md`: 현재 GitHub Actions + RunPod 릴리스 절차
- `README_runpod_volume.md`: RunPod cold start 완화용 volume 설정 메모
- `docs/pipeline_runtime_config.md`: 현재 파이프라인이 실제로 읽는 runtime config 기준 문서

## 실행 예시

RunPod handler 로컬 실행:

```bash
python handler_sd.py
```

헬스체크:

```bash
python tests/test_runpod.py --health-check
```

샘플 요청:

```bash
python tests/test_runpod.py \
  --image images/1234.jpeg \
  --hairstyle "wolf cut, layered bangs" \
  --color "ash brown" \
  --top-k 1
```

단발/중단발 마스크 비교:

```bash
python tests/test_runpod.py \
  --image images/1234.jpeg \
  --hairstyle "short chin-length bob cut, hush cut" \
  --top-k 1 \
  --bg-fill sd \
  --mask-refine-mode segface_priority
```

마스크 비교 모드:

- `sam2`: 기본 경로
- `segface_priority`: SegFace 코어 유지 + SAM2 경계 보정만 약하게 반영
- `segface_only`: SegFace 마스크만 사용

## CI/CD

push 시 현재 기준으로 아래 워크플로가 동작합니다.

- `.github/workflows/build-sd-base.yml`
- `.github/workflows/build-sd-app.yml`
- `.github/workflows/release-runpod.yml`

배포 세부 절차는 `README_release_runbook.md`를 기준으로 봅니다.

## 스크립트 범위

현재 `scripts/`에는 아래 범주의 스크립트가 남아 있습니다.

- RunPod 릴리스/스모크 테스트
- generation 학습 데이터 전처리
- generation 학습 실행
- generation 벤치마크/평가
- 문서 자동 생성 및 결과 정리

## 로컬 전용 / 업로드 제외 범위

아래 항목은 로컬 유지 대상이고 git/docker 업로드 대상에서 제외합니다.

- `tests/`
- `images/`
- `output/`
- `cmd/`
- `dataset_build/`
- `pretrained_models/`
- 생성된 `.docx`

## 참고 문서

- `README_release_runbook.md`: GitHub Actions + RunPod 릴리스 절차
- `README_runpod_volume.md`: RunPod cold start 완화용 volume 설정 메모
- `docs/`: generation 학습/평가 관련 문서

## 검증 기준

현재 정리 이후 기본 확인 명령은 아래 두 개입니다.

```bash
python -m py_compile handler_sd.py pipeline_sd_inpainting.py runtime_download.py scripts/runpod_release.py
python tests/test_runpod.py --health-check
```
