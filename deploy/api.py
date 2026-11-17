"""The engine API on Modal. Needs the secret `nudge-env` (from your .env).

    uv run modal deploy deploy/api.py

One container, because call state lives in memory. Scales to zero.
"""

from importlib.metadata import version

import modal

PINS = [f"{n}=={version(n)}" for n in ("fastapi", "httpx", "sqlalchemy", "pydantic", "pydantic-settings", "pyyaml", "tzdata")]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"pipecat-ai[deepgram,silero,websocket]=={version('pipecat-ai')}", *PINS, f"psycopg[binary]=={version('psycopg')}")
    .add_local_dir("engine", "/root/engine")
)
app = modal.App("nudge-api")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("nudge-env")],
    min_containers=0,
    max_containers=1,
    scaledown_window=10 * 60,
    timeout=15 * 60,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from engine.main import build_app

    return build_app()
