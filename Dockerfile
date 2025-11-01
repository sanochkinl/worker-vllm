FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential python3 python3-pip python3-dev git curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python3 -m pip install --upgrade pip && \
#     python3 -m pip install --upgrade -r /requirements.txt

# vLLM nightly
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -U vllm \
      --extra-index-url https://wheels.vllm.ai/nightly

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -r /requirements.txt && \
    python3 -m pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git"
    # python3 -m pip install --no-cache-dir qwen-vl-utils==0.0.14 && \
    # python3 -m pip install --no-cache-dir accelerate qwen-omni-utils -U

# Install vLLM (switching back to pip installs since issues that required building fork are fixed and space optimization is not as important since caching) and FlashInfer
# RUN python3 -m pip install --no-cache-dir flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3



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
