"""PhoneLLM Mini on vLLM, served by Modal as an OpenAI-compatible API.

One-time setup (replace the values):
    uv run modal secret create huggingface-secret HF_TOKEN=hf_xxx
    uv run modal secret create nudge-models MODELS_API_KEY=<a long random string>

Deploy:
    uv run modal deploy deploy/llm.py

Logs (request lines and throughput stats, never prompts or replies):
    uv run modal app logs nudge-llm        # last 100 lines
    uv run modal app logs nudge-llm -f     # follow live

Cold start options (all measured by evals/bench.py, see docs/performance.md):
  - caches always on: weights, vLLM's torch.compile cache and FlashInfer's autotune cache live in Volumes.
    Triton kernels are NOT moved there: we tried (TRITON_CACHE_DIR in the volume) and the next cold start went
    from about 9 minutes to over 18, with "Cubin file saved by TritonBundler not found" errors. See docs/deploy.md.
  - LLM_SNAPSHOT=1 at deploy time: Modal's GPU memory snapshot (alpha). vLLM starts once, warms up, goes to
    sleep, and Modal saves the process. Later cold starts restore it instead of redoing the warm-up.
        LLM_SNAPSHOT=1 uv run modal deploy deploy/llm.py
  - LLM_MIN_CONTAINERS=1: keep one replica warm for a calling window (it bills while up).

Cost rules built in: nothing runs while idle (min_containers=0), a container stops 2 minutes after
its last request, and at most one container can ever run (max_containers=1).

Modal prints a URL like https://<workspace>--nudge-llm-server.us-east.modal.direct
Put that URL plus /v1 in .env as LLM_BASE_URL, and the same random string as MODELS_API_KEY.

The vLLM flags below are the recipe from the model card of pipecat-ai/phonellm-mini-alpha-2.
Two things the card says not to do, and we do not do: --enforce-eager (CUDA graphs make decoding
5 to 7 times faster) and --reasoning-parser (the model runs with thinking off).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MINUTES = 60
PORT = 8000
MODEL = "pipecat-ai/phonellm-mini-alpha-2"
SERVED_NAME = "phonellm-mini-alpha-2"
MODEL_REVISION = os.environ.get("MODEL_REVISION")  # pin to a commit hash after the first good deploy
VLLM_VERSION = "0.30.0"  # the version the model card's recipe was measured on
SNAPSHOT = os.environ.get("LLM_SNAPSHOT") == "1"  # opt in: it is an alpha feature, so we measure it before trusting it
MIN_CONTAINERS = int(os.environ.get("LLM_MIN_CONTAINERS", "0"))
# BF16 weights are about 64 GB. The card lists the RTX PRO 6000 (96 GB) for production serving.
GPU = os.environ.get("LLM_GPU", "RTX-PRO-6000")

IMAGE_ENV = {
    "HF_HUB_CACHE": "/root/.cache/huggingface",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_LOG_STATS_INTERVAL": "10",  # throughput lines in the logs every 10 seconds
}
if SNAPSHOT:
    IMAGE_ENV |= {
        "VLLM_SERVER_DEV_MODE": "1",  # turns on the /sleep and /wake_up endpoints that snapshotting needs
        "TORCHINDUCTOR_COMPILE_THREADS": "1",  # Modal: makes torch.compile compatible with snapshots
    }

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env(IMAGE_ENV)
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("nudge-llm")


HF_ROOT = Path(os.environ.get("HF_HUB_CACHE", "/root/.cache/huggingface"))
LOCAL_MODEL = Path("/tmp/phonellm")  # the container's own disk, fast, thrown away when it stops


def find_snapshot(root: Path, model: str = MODEL, revision: str | None = None) -> Path | None:
    """Where Hugging Face's cache keeps the files of one model version, or None if it has not been downloaded yet."""
    repo = root / ("models--" + model.replace("/", "--"))
    for name in [revision] if revision else []:
        if name and (repo / "snapshots" / name).is_dir():
            return repo / "snapshots" / name
    ref = repo / "refs" / (revision or "main")
    if ref.is_file() and (repo / "snapshots" / ref.read_text().strip()).is_dir():
        return repo / "snapshots" / ref.read_text().strip()
    found = sorted((repo / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime) if (repo / "snapshots").is_dir() else []
    return found[-1] if found else None


def copy_weights(root: Path = HF_ROOT, dest: Path = LOCAL_MODEL, revision: str | None = None, workers: int = 16, say=print) -> Path | None:
    """Copy the model files from the shared volume to the container's own disk, many files at once.

    A cold start read the 59 GB straight from the volume: 23 seconds on one machine and 17 minutes on another
    (about 57 MB/s). Reading whole files in parallel is the cheap way to find out if the slow reads were the
    volume's random access. Returns the folder to serve, or None to serve from the volume as before.
    """
    source = find_snapshot(root, revision=revision)
    if source is None:
        say("[nudge] no downloaded weights yet: vLLM will download them (the next start will copy them)")
        return None
    files = [p for p in source.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dest).free < size * 1.2:
        say("[nudge] not enough local disk to copy the weights: serving from the volume")
        return None

    def copy(path: Path) -> None:
        target = dest / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)  # follows the cache's links to the real file

    started = time.time()
    say(f"[nudge] copying {size / 1e9:.1f} GB of weights to local disk, {workers} files at a time")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(copy, files))
    seconds = time.time() - started
    say(f"[nudge] copied in {seconds:.0f} s ({size / 1e6 / max(seconds, 0.001):.0f} MB/s)")
    return dest


def build_vllm_command(api_key: str, revision: str | None = None, sleep_mode: bool = False, model: str = MODEL) -> list[str]:
    """The `vllm serve` command. A plain function so a test can check the flags."""
    cmd = [
        "vllm",
        "serve",
        model,
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
    if revision and model == MODEL:  # a local folder has no revisions
        cmd += ["--revision", revision]
    if sleep_mode:
        cmd += ["--enable-sleep-mode"]  # lets us park the engine before Modal takes the snapshot
    return cmd


def wait_until_healthy(process: subprocess.Popen, timeout: float = 45 * MINUTES) -> None:
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


def call_vllm(path: str, api_key: str, body: dict | None = None, timeout: float = 600) -> None:
    """POST to the vLLM server running in this container. Used for warm-up, sleep and wake_up."""
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=data,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    urllib.request.urlopen(request, timeout=timeout).read()


def warm_up(api_key: str, rounds: int = 3) -> None:
    """A few real requests, so the kernels they need are compiled before the snapshot is taken."""
    body = {"model": SERVED_NAME, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16}
    for _ in range(rounds):
        call_vllm("/v1/chat/completions", api_key, body)


SERVER = dict(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("nudge-models")],
    port=PORT,
    startup_timeout=45 * MINUTES,  # cold starts took 9, 18 and 29 minutes in testing; Modal kills the container at this limit
    scaledown_window=2 * MINUTES,  # stay up this long after the last request, then stop paying
    min_containers=MIN_CONTAINERS,  # 0: nothing runs (or costs) while idle. Set LLM_MIN_CONTAINERS=1 for a calling window.
    max_containers=1,  # a runaway client can never start a second GPU
    target_concurrency=32,
    exit_grace_period=15,
    # Normally the URL is public and vLLM checks the API key. Snapshot mode needs vLLM's dev endpoints, which have NO key
    # check and include one that runs arbitrary methods on the engine, so there Modal's proxy must stop strangers first.
    unauthenticated=not SNAPSHOT,
)


def save_caches() -> None:
    try:
        vllm_cache.commit()  # keep the compiled kernels for the next cold start
    except Exception as e:  # never let a cache problem break shutdown
        print(f"could not save the vLLM cache: {e}")


if SNAPSHOT:

    @app.server(**SERVER, enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True})
    class Server:
        @modal.enter(snap=True)
        def start(self):
            key = os.environ["MODELS_API_KEY"]
            model = str(copy_weights(revision=MODEL_REVISION) or MODEL)
            self.process = subprocess.Popen(build_vllm_command(key, MODEL_REVISION, sleep_mode=True, model=model))
            wait_until_healthy(self.process)
            print("[nudge] snapshot step 1/3: vLLM is healthy", flush=True)
            warm_up(key)
            print("[nudge] snapshot step 2/3: warm-up requests done", flush=True)
            call_vllm("/sleep?level=1", key)  # weights move to CPU memory, the GPU is emptied, then Modal snapshots
            print("[nudge] snapshot step 3/3: vLLM is asleep, Modal takes the snapshot now", flush=True)

        @modal.enter(snap=False)
        def wake(self):
            print("[nudge] restored from the snapshot, waking vLLM", flush=True)
            call_vllm("/wake_up", os.environ["MODELS_API_KEY"])
            wait_until_healthy(self.process)
            print("[nudge] awake and healthy", flush=True)

        @modal.exit()
        def stop(self):
            self.process.terminate()
            save_caches()

else:

    @app.server(**SERVER)
    class Server:
        @modal.enter()
        def start(self):
            model = str(copy_weights(revision=MODEL_REVISION) or MODEL)
            self.process = subprocess.Popen(build_vllm_command(os.environ["MODELS_API_KEY"], MODEL_REVISION, model=model))
            wait_until_healthy(self.process)
            print("[nudge] vLLM is healthy", flush=True)

        @modal.exit()
        def stop(self):
            self.process.terminate()
            save_caches()
