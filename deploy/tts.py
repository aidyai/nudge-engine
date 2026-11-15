"""Chatterbox Turbo text-to-speech on Modal.

POST {"text": "..."} with the header  Authorization: Bearer <MODELS_API_KEY>
and you get back raw 16-bit mono PCM audio at 24 kHz (also stated in the X-Sample-Rate header).

One-time setup (the two secrets are the same ones deploy/llm.py uses):
    uv run modal secret create huggingface-secret HF_TOKEN=hf_xxx
    uv run modal secret create nudge-models MODELS_API_KEY=<a long random string>

Optional: a voice to clone. About 10 seconds of clear speech from a person who gave consent:
    uv run modal volume create nudge-voices
    uv run modal volume put nudge-voices ./voice.wav /voice.wav

Deploy:
    uv run modal deploy deploy/tts.py
Modal prints the endpoint URL. Put it in .env as TTS_URL.

Logs (one JSON line per request: sizes and timings, never the text):
    uv run modal app logs nudge-tts
    uv run modal app logs nudge-tts -f

Cost rules built in: nothing runs while idle, a container stops 2 minutes after its last request,
at most one container can run, and the default GPU is the cheap L4 (set TTS_GPU=A10 if it struggles).

Based on Modal's own Chatterbox Turbo example, with a bearer-key check added.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path

import modal
import numpy as np
from fastapi import Header, HTTPException, Response
from pydantic import BaseModel, Field

MINUTES = 60
SAMPLE_RATE = 24_000
MAX_CHARS = 600  # protects the GPU from one huge request
VOICE_FILE = Path("/voices/voice.wav")
GPU = os.environ.get("TTS_GPU", "L4")  # about $0.80 an hour; Modal's example uses the A10 at about $1.10

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "chatterbox-tts==0.1.7",
    "fastapi[standard]==0.124.4",
    "peft==0.18.0",
    "setuptools<82",
)

voices = modal.Volume.from_name("nudge-voices", create_if_missing=True)
app = modal.App("nudge-tts", image=image)

logging.basicConfig(level=logging.INFO, format="%(message)s")  # Modal shows stdout and stderr as logs
logger = logging.getLogger("nudge.tts")
NEVER_LOG = {"text", "transcript", "audio", "prompt", "api_key", "authorization"}

with image.imports():  # only imported inside the container
    import torchaudio.functional as audio_functional
    from chatterbox.tts_turbo import ChatterboxTurboTTS


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


def is_authorized(header: str | None, expected_key: str) -> bool:
    """True only for 'Bearer <the shared key>'. An empty key on the server side never matches."""
    if not expected_key or not header or not header.startswith("Bearer "):
        return False
    return secrets.compare_digest(header.removeprefix("Bearer ").encode(), expected_key.encode())


def log_event(event: str, **fields: object) -> None:
    """One JSON line in the Modal logs. Refuses fields that could carry personal data or secrets."""
    banned = NEVER_LOG & set(fields)
    if banned:
        raise ValueError(f"refusing to log {sorted(banned)}")
    logger.info(json.dumps({"event": event, **fields}, separators=(",", ":"), sort_keys=True))


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Float audio in [-1, 1] to little-endian 16-bit PCM bytes."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


@app.cls(
    gpu=GPU,
    scaledown_window=2 * MINUTES,  # stays warm 2 minutes after the last request
    max_containers=1,  # a runaway client can never start a second GPU
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("nudge-models")],
    volumes={"/voices": voices},
)
@modal.concurrent(max_inputs=8)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    @modal.fastapi_endpoint(method="POST", docs=False)
    def speak(self, body: SpeakRequest, authorization: str | None = Header(default=None)) -> Response:
        if not is_authorized(authorization, os.environ.get("MODELS_API_KEY", "")):
            log_event("tts_rejected", reason="bad_or_missing_key")
            raise HTTPException(status_code=401, detail="bad or missing API key")
        started = time.monotonic()
        audio = self.synthesize(body.text)
        log_event(
            "tts_request",
            chars=len(body.text),
            audio_seconds=round(len(audio) / (2 * SAMPLE_RATE), 2),
            latency_ms=round((time.monotonic() - started) * 1000),
        )
        return Response(
            content=audio,
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(SAMPLE_RATE)},
        )

    def synthesize(self, text: str) -> bytes:
        options = {"audio_prompt_path": str(VOICE_FILE)} if VOICE_FILE.exists() else {}
        wav = self.model.generate(text, **options)  # a torch tensor, shape [1, samples]
        if self.model.sr != SAMPLE_RATE:
            wav = audio_functional.resample(wav, self.model.sr, SAMPLE_RATE)
        return float_to_pcm16(wav.squeeze().cpu().numpy())
