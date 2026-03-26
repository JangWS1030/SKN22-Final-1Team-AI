# 수동 릴리스 런북

현재 저장소 기준 릴리스 절차는 `GitHub Actions -> Docker Hub -> RunPod release -> smoke test` 순서입니다.

## 준비

- `gh`, `docker`, `python`
- `.env`에 `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`
- GitHub secrets에 `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`

로컬 확인:

```bash
gh auth status
docker version
python -c "import requests,dotenv; print('deps_ok')"
```

## 배포 전 확인

```bash
python tests/test_runpod.py --health-check
```

샘플 생성 확인:

```bash
python tests/test_runpod.py \
  --image images/1234.jpeg \
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
  --image-repo sikersiker/sd \
  --tag manual-20260326-170000 \
  --push
```

`Dockerfile.sd.app` 기반이라 코드 변경만 있을 때는 전체 의존성을 다시 빌드하지 않아 훨씬 빠릅니다.
처음 한 번 `base-latest`가 없으면 아래처럼 base를 같이 만들 수 있습니다.

```bash
python scripts/build_sd_image.py \
  --image-repo sikersiker/sd \
  --tag manual-20260326-170000 \
  --ensure-base \
  --base-tag v1 \
  --push
```

```bash
python scripts/runpod_release.py --image-tag latest
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
python -m py_compile scripts/runpod_release.py tests/test_runpod.py
python tests/test_runpod.py --health-check
```

필요하면 wrapper도 사용합니다.

```powershell
.\scripts\runpod_smoke.ps1 -HealthCheck
.\scripts\runpod_smoke.ps1
```

## 메모

- 현재 릴리스 스크립트는 `scripts/runpod_release.py` 하나로 정리했습니다.
- 예전 MCP 연결 문서, self-hosted runner 문서, 구형 endpoint update 스크립트는 저장소에서 제거했습니다.
