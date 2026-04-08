# Pipeline Runtime Config

## 목적

현재 `MirrAISDPipeline` 실행 경로에서 실제로 읽는 runtime 설정만 정리한 문서입니다.

이 문서는 `pipeline_sd_inpainting.py`, `handler_sd.py`, `scripts/run_generation_benchmark.py` 기준으로 유지합니다.

## 현재 사용 중인 `SDInpaintConfig` 필드

- `num_inference_steps`
  - SD inpainting step 수
- `controlnet_conditioning_scale`
  - ControlNet canny conditioning 강도
- `ip_adapter_scale`
  - IP-Adapter face conditioning 강도
- `canny_low`
  - Canny lower threshold
- `canny_high`
  - Canny upper threshold
- `mask_dilate_px`
  - SD 입력용 hair mask dilation 크기
- `seeds`
  - 추론 seed 리스트
- `device`
  - 실행 디바이스
- `dtype`
  - 실행 dtype
- `lora_path`
  - 런타임 LoRA 경로
- `lora_scale`
  - 런타임 LoRA scale
- `use_sam2`
  - SAM2 refinement 사용 여부
- `bg_fill_mode`
  - short/medium 변환 시 제거 영역 복원 전략
- `enable_post_cloth_refine`
  - 후처리 cloth refine 사용 여부

## 내부 고정 동작

아래 값들은 더 이상 `SDInpaintConfig`로 받지 않고 파이프라인 내부 로직으로 고정되어 있습니다.

- `guidance_scale`
  - `sd_prompt_data.sd_guidance`가 있으면 외부 입력을 사용하고, 없으면 `_build_prompt()` 내부 로직으로 결정
- `face_crop_padding`
  - `_crop_face()` 내부의 고정 비율 사용
- `enable_xformers`
  - IP-Adapter attention processor와 충돌 이력이 있어 강제 비활성

## 환경 변수 override

- `ENABLE_SAM2`
  - handler cold start 시 `use_sam2` 결정
- `LORA_PATH`
  - handler 기본 LoRA 경로 override
- `LORA_SCALE`
  - handler 기본 LoRA scale override
- `SEGFACE_HAIR_THRESHOLD`
  - custom SegFace hair threshold override

## Benchmark 스크립트 기준 입력

`scripts/run_generation_benchmark.py`는 현재 아래 설정만 CLI로 받습니다.

- `--num-inference-steps`
- `--controlnet-conditioning-scale`
- `--ip-adapter-scale`
- `--lora-path`
- `--lora-scale`

`guidance_scale`은 benchmark CLI에서 직접 받지 않습니다. 현재 파이프라인은 외부 `sd_prompt_data`가 있으면 그 값을 쓰고, 없으면 prompt 길이 분류 결과에 따라 내부에서 계산합니다.
