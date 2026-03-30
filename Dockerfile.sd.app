# =============================================================================
# MirrAI SD App Image (Code only)
# byoungj/sd:v*
#
# BASE_IMAGE는 CI에서 주입:
#   --build-arg BASE_IMAGE=<DOCKER_USERNAME>/sd:base-latest
# =============================================================================

ARG BASE_IMAGE=sd:base-latest
ARG BUILD_TAG=dev
FROM ${BASE_IMAGE} AS runtime
ARG BUILD_TAG

WORKDIR /app

# SD 앱 코드만 복사
COPY --chmod=755 entrypoint_sd.sh entrypoint.sh
COPY handler_sd.py             ./
COPY style_recommender.py      ./
COPY handler_runpod_diag.py    ./
COPY pipeline_sd_inpainting.py ./
COPY pipeline_sd_components/   pipeline_sd_components/
COPY runtime_download.py       ./
COPY utils/                    utils/
COPY rag_pipeline/             rag_pipeline/
COPY models/__init__.py        models/__init__.py
COPY models/segface/           models/segface/
COPY data/                     data/

# Normalize Windows CRLF line endings so the Linux entrypoint can execute.
RUN sed -i 's/\r$//' entrypoint.sh

# `pretrained_models/` contains local-only assets in some environments.
# Runtime LoRA is loaded from Hugging Face by default, so the app image keeps
# only code and lightweight config.
RUN mkdir -p pretrained_models

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    MODEL_DOWNLOAD_TIMEOUT=600 \
    ENABLE_SAM2=1 \
    MIRRAI_BUILD_TAG=${BUILD_TAG}

ENTRYPOINT ["./entrypoint.sh"]
