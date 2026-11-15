"""The engine API on Modal: the website's calls, speech, and phone calls, in one app.

Needs the Modal secret `nudge-env`, made from your .env:
    uv run modal secret create nudge-env --from-dotenv .env --force
Deploy:   uv run modal deploy deploy/api.py
Logs:     uv run modal app logs nudge-api -f
URL:      https://<your-workspace>--nudge-api-web.modal.run   (open /health to check)

One container only, because call state lives in its memory. It scales to zero, so nothing costs money
while nobody is using it, and a call left idle for ten minutes is forgotten. A call can last 15 minutes.
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
