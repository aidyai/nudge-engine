"""PhoneLLM Mini on vLLM, served by Modal as an OpenAI-compatible API.

One-time setup (replace the values):
    uv run modal secret create huggingface-secret HF_TOKEN=hf_xxx
    uv run modal secret create nudge-models MODELS_API_KEY=<a long random string>

Deploy:
    uv run modal deploy deploy/llm.py

Logs (request lines and throughput stats, never prompts or replies):
    uv run modal app logs nudge-llm        # last 100 lines
    uv run modal app logs nudge-llm -f     # follow live

Cost rules built in: nothing runs while idle (min_containers=0), a container stops 2 minutes after
its last request, and at most one container can ever run (max_containers=1).

Modal prints a URL like https://<workspace>--nudge-llm-server.us-east.modal.direct
Put that URL plus /v1 in .env as LLM_BASE_URL, and the same random string as MODELS_API_KEY.

The vLLM flags below are the recipe from the model card of pipecat-ai/phonellm-mini-alpha-2.
Two things the card says not to do, and we do not do: --enforce-eager (CUDA graphs make decoding
5 to 7 times faster) and --reasoning-parser (the model runs with thinking off).
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

MINUTES = 60
PORT = 8000
MODEL = "pipecat-ai/phonellm-mini-alpha-2"
SERVED_NAME = "phonellm-mini-alpha-2"
MODEL_REVISION = os.environ.get("MODEL_REVISION")  # pin to a commit hash after the first good deploy
VLLM_VERSION = "0.30.0"  # the version the model card's recipe was measured on
# BF16 weights are about 64 GB. The card lists the RTX PRO 6000 (96 GB) for production serving.
GPU = os.environ.get("LLM_GPU", "RTX-PRO-6000")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env(
        {
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",  # throughput lines in the logs every 10 seconds
        }
    )
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("nudge-llm")


def build_vllm_command(api_key: str, revision: str | None = None) -> list[str]:
    """The `vllm serve` command. A plain function so a test can check the flags."""
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--api-key",
        api_key,
        "--uvicorn-log-level=info",
        # the model card's recipe:
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "256",
        "--enable-prefix-caching",  # on, as the card says. The hit rate is measured in evals/bench.py
        "--async-scheduling",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]
    if revision:
        cmd += ["--revision", revision]
    return cmd


def wait_until_healthy(process: subprocess.Popen, timeout: float = 20 * MINUTES) -> None:
    """Block until vLLM answers /health. Fail loudly if it crashes instead of hanging."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (code := process.poll()) is not None:
            raise RuntimeError(f"vLLM exited with code {code} while starting")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM was not healthy after {timeout:.0f} seconds")


@app.server(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("nudge-models")],
    port=PORT,
    startup_timeout=20 * MINUTES,  # time to download weights and start vLLM
    scaledown_window=2 * MINUTES,  # stay up this long after the last request, then stop paying
    min_containers=0,  # nothing runs (or costs) while idle. Warm it before a calling window.
    max_containers=1,  # a runaway client can never start a second GPU
    target_concurrency=32,
    exit_grace_period=15,
    unauthenticated=True,  # the URL is public; vLLM itself checks the API key
)
class Server:
    @modal.enter()
    def start(self):
        self.process = subprocess.Popen(build_vllm_command(os.environ["MODELS_API_KEY"], MODEL_REVISION))
        wait_until_healthy(self.process)

    @modal.exit()
    def stop(self):
        self.process.terminate()
