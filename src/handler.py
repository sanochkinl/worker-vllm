import os
import asyncio

# ---- ENV & PATHS (set BEFORE importing engine/vllm) -------------------------
# Multiprocessing: avoid "spawn" guard crashes on serverless
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

# Optional but recommended cache locations (mount a volume at /models on RunPod)
os.environ.setdefault("HF_HOME", "/models/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/models/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/models/hf/datasets")
os.environ.setdefault("TMPDIR", "/models/tmp")

# Create disk cache dirs if you're using: --mm-processor-cache-type disk --mm-processor-cache-path /models/mmcache
for _p in ("/models/mmcache", "/models/hf", "/models/hf/datasets", "/models/tmp"):
    try:
        os.makedirs(_p, exist_ok=True)
    except Exception:
        pass

# Optional: make logs chattier during bring-up
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import runpod  # imported after envs are set

# Import engine types but do NOT instantiate yet (avoid spawn at import time)
from engine import vLLMEngine as _vLLMEngineClass
from engine import OpenAIvLLMEngine as _OpenAIvLLMEngineClass

# ---- LAZY SINGLETON INIT ----------------------------------------------------
_vllm_engine = None
_openai_engine = None
_init_lock = asyncio.Lock()

async def get_engines():
    """Create engines lazily on first request, once."""
    global _vllm_engine, _openai_engine
    if _vllm_engine is not None and _openai_engine is not None:
        return _vllm_engine, _openai_engine

    async with _init_lock:
        if _vllm_engine is None or _openai_engine is None:
            # If your vLLMEngine reads settings from env, ensure they are set here.
            # Example (uncomment if your engine supports these env overrides):
            # os.environ.setdefault("MODEL_NAME", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
            # os.environ.setdefault("EXTRA_VLLM_ARGS",
            #     "--trust-remote-code --quantization awq --dtype float16 "
            #     "--kv-cache-dtype fp8 --gpu-memory-utilization 0.92 "
            #     "--max-model-len 8192 --max-num-seqs 8 "
            #     "--mm-processor-cache-type disk --mm-processor-cache-path /models/mmcache "
            #     "--async-scheduling"
            # )

            # Instantiate the core engine now (safe point)
            _vllm_engine = _vLLMEngineClass()
            _openai_engine = _OpenAIvLLMEngineClass(_vllm_engine)
    return _vllm_engine, _openai_engine

# ---- HANDLER ---------------------------------------------------------------
from utils import JobInput  # ok to import now

async def handler(job):
    vllm_engine, openai_engine = await get_engines()

    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine

    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

# ---- SERVERLESS START -------------------------------------------------------
# Use a defensive concurrency modifier that works before engine init.
# You can override with env MAX_CONCURRENCY if you want.
_default_conc = int(os.environ.get("MAX_CONCURRENCY", "8"))

def _concurrency_modifier(_):
    # If engine is already built and exposes a cap, use it; else fall back.
    try:
        if _vllm_engine is not None and hasattr(_vllm_engine, "max_concurrency"):
            return _vllm_engine.max_concurrency
    except Exception:
        pass
    return _default_conc

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": _concurrency_modifier,
        "return_aggregate_stream": True,
    }
)
