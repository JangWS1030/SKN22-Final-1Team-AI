# =============================================================================
# MirrAI SD App Image (Code only)
# byoungj/sd:v*
#
# BASE_IMAGE는 CI에서 주입:
#   --build-arg BASE_IMAGE=<DOCKER_USERNAME>/sd:base-latest
# =============================================================================

ARG BASE_IMAGE=sd:base-latest
FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

# SD 앱 코드만 복사
COPY --chmod=755 entrypoint_sd.sh entrypoint.sh
COPY handler_sd.py             ./
COPY handler_runpod_diag.py    ./
COPY pipeline_sd_inpainting.py ./
COPY runtime_download.py       ./
COPY utils/env_loader.py       utils/env_loader.py
COPY utils/sam2_runtime.py     utils/sam2_runtime.py
COPY utils/trend_prompt.py     utils/trend_prompt.py
COPY models/__init__.py        models/__init__.py
COPY models/segface/           models/segface/
COPY data/                     data/

# Normalize Windows CRLF line endings so the Linux entrypoint can execute.
RUN sed -i 's/\r$//' entrypoint.sh

# `pretrained_models/` contains local-only assets in some environments.
# The runtime creates/downloads the required files on demand.
RUN mkdir -p pretrained_models

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    MODEL_DOWNLOAD_TIMEOUT=600 \
    ENABLE_SAM2=1

ENTRYPOINT ["./entrypoint.sh"]
