# 수동 릴리스 런북

현재 저장소 기준 릴리스 절차는 `GitHub Actions -> Docker Hub -> RunPod release -> smoke test` 순서입니다.

## 준비

- `gh`, `docker`, `python`
- `.env`에 `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- GitHub secrets에 `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- RunPod cache 설정 유지:
  - endpoint `networkVolumeId`를 지우지 말 것
  - template env의 `HF_HOME`, `TORCH_HOME`, `MIRRAI_PRELOAD_ON_STARTUP`를 지우지 말 것

로컬 확인:

```bash
gh auth status
docker version
python -c "import requests,dotenv; print('deps_ok')"
```

## 배포 전 확인

```bash
python test_runpod.py --health-check
```

얼굴형 분석 확인:

```bash
python test_runpod.py --analyze-face --image images/1234.jpg --include-visualization
```

샘플 생성 확인:

```bash
python test_runpod.py \
  --image images/1234.jpg \
  --hairstyle "short chin-length bob cut, hush cut" \
  --color "ash beige" \
  --top-k 1 \
  --filename-prefix baseline_check
```

## 빌드

push 시 자동으로 다음 워크플로가 동작합니다.

- `.github/workflows/build-sd-base.yml`
- `.github/workflows/build-sd-app.yml`
- `.github/workflows/release-runpod.yml`

수동 확인:

```bash
gh run list --limit 10
gh run watch <run-id> --exit-status
gh run view <run-id> --log-failed
```

## 수동 RunPod 릴리스

자동 릴리스 대신 직접 실행하려면 아래 둘 중 하나를 사용합니다.

빠른 로컬 빌드:

```bash
python scripts/build_sd_image.py \
  --image-repo byoungj/sd \
  --tag manual-20260326-170000 \
  --push
```

`Dockerfile.sd.app` 기반이라 코드 변경만 있을 때는 전체 의존성을 다시 빌드하지 않아 훨씬 빠릅니다.
현재 앱 이미지에는 `pipeline_sd_inpainting.py`와 함께 `pipeline_sd_components/`도 포함되어야 합니다.
처음 한 번 `base-latest`가 없으면 아래처럼 base를 같이 만들 수 있습니다.

```bash
python scripts/build_sd_image.py \
  --image-repo byoungj/sd \
  --tag manual-20260326-170000 \
  --ensure-base \
  --base-tag v1 \
  --push
```

```bash
python scripts/runpod_release.py --image-tag latest
```

릴리스 직후 아래 항목이 그대로 유지됐는지 확인합니다.

- endpoint `networkVolumeId=h6lcfsxdt0`
- template env에 `HF_HOME=/runpod-volume/huggingface`
- template env에 `TORCH_HOME=/runpod-volume/torch`
- template env에 `MIRRAI_PRELOAD_ON_STARTUP=1`

위 값이 빠졌다면 바로 아래 복구 스크립트를 실행합니다.

```bash
python scripts/runpod_restore_cache_config.py \
  --network-volume-id h6lcfsxdt0 \
  --preload-on-startup
```

```powershell
.\scripts\runpod_release.ps1 -ImageTag latest
```

dry-run:

```bash
python scripts/runpod_release.py --image-tag latest --dry-run
```

## 최종 검증

```bash
python -m py_compile scripts/runpod_release.py test_runpod.py pipeline_sd_inpainting.py pipeline_sd_components/loading.py pipeline_sd_components/output.py pipeline_sd_components/prompt.py
python test_runpod.py --health-check
```

필요하면 wrapper도 사용합니다.

```powershell
.\scripts\runpod_smoke.ps1 -HealthCheck
.\scripts\runpod_smoke.ps1
```

## 메모

- 현재 릴리스 스크립트는 `scripts/runpod_release.py` 하나로 정리했습니다.
- RunPod cache volume 정보가 빠지면 콜드스타트가 다시 길어지므로, endpoint/template 갱신 시 volume 및 cache env 보존 여부를 항상 확인합니다.
- 예전 MCP 연결 문서, self-hosted runner 문서, 구형 endpoint update 스크립트는 저장소에서 제거했습니다.
