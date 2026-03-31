# Pipeline Runtime Config

This document tracks the environment variables that are actually relevant to the
current RunPod serverless SD runtime.

Scope:
- `handler_sd.py`
- `pipeline_sd_inpainting.py`
- `pipeline_sd_components/loading.py`
- `runtime_download.py`
- `utils/sam2_runtime.py`
- local release and smoke-test scripts under `scripts/`

## Endpoint Runtime: Recommended

These are the highest-signal variables for reducing cold-start time or keeping
the serverless runtime pointed at the correct model sources.

- `HF_HOME`
  - Recommended cache root for Hugging Face downloads. Point this at a mounted
    persistent volume such as `/runpod-volume/huggingface`.
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
  - Required when private or gated Hugging Face assets are used.
- `SAM2_CHECKPOINT_PATH`
  - Recommended when SAM2 should load from a persistent file on the mounted
    volume, for example `/runpod-volume/pretrained_models/sam2.pt`.
- `SEGFACE_HF_REPO_ID`
  - Default custom SegFace model repository.
- `SEGFACE_HF_FILENAME`
  - Default custom SegFace checkpoint filename.
- `MIRRAI_LORA_HF_REPO_ID`
  - Default runtime LoRA source.
- `MIRRAI_LORA_HF_FILENAME`
  - Default runtime LoRA filename.
- `ENABLE_SAM2`
  - Enables or disables SAM2 refinement at handler startup.

## Endpoint Runtime: Optional Overrides

These variables are read by the current runtime, but they are not normally
required for a healthy deployment.

- `LORA_PATH`
- `LORA_SCALE`
- `SEGFACE_HF_SUBFOLDER`
- `SEGFACE_HF_TOKEN`
- `SEGFACE_CKPT_PATH`
- `SEGFACE_MODEL_VARIANT`
- `SEGFACE_INPUT_RESOLUTION`
- `SEGFACE_BASE_CKPT_PATH`
- `SEGFACE_BASE_HF_REPO_ID`
- `SEGFACE_BASE_HF_SUBFOLDER`
- `SEGFACE_BASE_HF_FILENAME`
- `SEGFACE_BASE_HF_TOKEN`
- `SEGFACE_BASE_MODEL_VARIANT`
- `SEGFACE_BASE_INPUT_RESOLUTION`
- `SEGFACE_HAIR_THRESHOLD`
- `SEGFACE_CUSTOM_HAIR_THRESHOLD`
- `SEGFACE_CROP_SCALE`
- `SEGFACE_CROP_CENTER_Y_OFFSET`
- `SEGFACE_CUSTOM_HAIR_RATIO_MIN`
- `MASK_REFINE_MODE`
- `MIRRAI_PRELOAD_ON_STARTUP`
- `MIRRAI_BUILD_TAG`
- `SAM2_HF_REPO_ID`
- `SAM2_HF_FILENAME`
- `SAM2_MODEL_CONFIG`
- `LLM_REFINED_TRENDS_PATH`
- `MEDIAPIPE_FACE_LANDMARKER_MODEL`
- `MEDIAPIPE_AUTO_DOWNLOAD_FACE_LANDMARKER`
- `MEDIAPIPE_MODEL_CACHE_DIR`
- `TORCH_CUDA_ARCH_LIST`

Notes:
- `SEGFACE_HAIR_THRESHOLD` is applied by the handler after pipeline load.
- `SEGFACE_CUSTOM_HAIR_THRESHOLD` is also read during SegFace loading, so both
  names can affect the same behavior from slightly different entry points.

## Runtime Metadata From RunPod

These are read by the runtime, but they are typically injected by RunPod rather
than manually managed in endpoint settings.

- `RUNPOD_ENDPOINT_ID`
- `RUNPOD_POD_ID`
- `RUNPOD_GPU_TYPE_ID`
- `RUNPOD_GPU_SIZE`
- `RUNPOD_HANDLER_FILE`
- `RUNPOD_WEBHOOK_GET_JOB`
- `RUNPOD_WEBHOOK_PING`
- `RUNPOD_WEBHOOK_POST_OUTPUT`
- `RUNPOD_WEBHOOK_POST_STREAM`

## Local Tooling Only

These are mainly used by release and test scripts on a developer machine.

- `RUNPOD_API_KEY`
  - Used by local release and smoke-test tooling.
- `RUNPOD_ENDPOINT_ID`
  - Also used by local release and smoke-test tooling when targeting an
    endpoint from outside RunPod.

## Legacy Or Unused By Current Repo Runtime

These names are not consumed by the current repository runtime code path.
Keeping them is usually harmless, but they add noise and can confuse future
operators.

- `SEG_PTH_GITHUB_OWNER`
- `SEG_PTH_GITHUB_REPO`
- `SEG_PTH_GITHUB_PATH`
- `SEG_PTH_GITHUB_REF`
- `SEG_PTH_PATH`
- `GITHUB_TOKEN`
- `ENABLE_STARTUP_GIT_PULL`
- `RUNPOD_DEBUG_LEVEL`

Special case:
- `MODEL_DOWNLOAD_TIMEOUT`
  - Present in `Dockerfile.sd.app`, but not actively consumed by the current
    Python runtime code.
- `TORCH_HOME`
  - Not read by the repository code. It may still be harmless as a library
    cache hint, but there is no observed effect in the current SD serverless
    path.

## Recommended Endpoint Env Set

For the current serverless deployment, a compact endpoint configuration is
usually enough:

```dotenv
HF_HOME=/runpod-volume/huggingface
HF_TOKEN=<secret>
SAM2_CHECKPOINT_PATH=/runpod-volume/pretrained_models/sam2.pt
SEGFACE_HF_REPO_ID=siik/segface_hair_khairstyle
SEGFACE_HF_FILENAME=best.pt
MIRRAI_LORA_HF_REPO_ID=siik/mirrai-hair-swap-stage4-garment-reveal-lora-20260330
MIRRAI_LORA_HF_FILENAME=pytorch_lora_weights.safetensors
ENABLE_SAM2=1
```

Optional but situational:

```dotenv
LORA_SCALE=1.0
SEGFACE_HAIR_THRESHOLD=0.5
MASK_REFINE_MODE=sam2
```

## Audit Script

Use the env audit helper to classify a `.env` file or endpoint env dump:

```bash
python scripts/audit_serverless_env.py --env-file .env
```

It reports:
- runtime keys that matter now
- local-tooling-only keys
- legacy keys that can be removed
- unknown keys that should be reviewed manually

## Secret Hygiene

Do not paste live secrets into Git, issue comments, or chat logs.

If a live `HF_TOKEN`, `RUNPOD_API_KEY`, or GitHub token has been shared in
plain text, rotate it immediately and replace the old secret in RunPod, GitHub,
and local `.env` files.
