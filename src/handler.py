import os

# ✅ Critical: avoid multiprocessing "spawn" crash
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# ✅ Do not shadow the class name
vllm_engine = vLLMEngine()
openai_engine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda _: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
