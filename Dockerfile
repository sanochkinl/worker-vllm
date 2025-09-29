# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 0) Базовые пакеты (TLS, компиляторы)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential python3 python3-pip python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# 1) Инструменты pip
RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

RUN python3 -m pip install --no-cache-dir uv

SHELL ["/bin/bash","-lc"]
RUN set -euxo pipefail

# torch first (still recommended)
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.3.1 torchvision==0.18.1

# now uv + vLLM nightly with the flag
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install -U vllm \
      --extra-index-url https://wheels.vllm.ai/nightly \
      --torch-backend=auto && \
    uv pip install qwen-vl-utils==0.0.14

# 4) Твои зависимости
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -r /requirements.txt && \
    python3 -m pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git" && \
    python3 -m pip install --no-cache-dir accelerate qwen-omni-utils -U

# 5) FlashInfer — ОТДЕЛЬНЫЙ RUN (и индекс должен совпадать: cu121 + torch2.3)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir \
      -i https://flashinfer.ai/whl/cu121/torch2.3 \
      flashinfer

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0

ENV PYTHONPATH="/:/vllm-workspace"


COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
