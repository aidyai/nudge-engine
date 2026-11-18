"""Chatterbox Turbo text-to-speech on Modal: POST {"text": ...}, get 16-bit mono PCM at 24 kHz back.

uv run modal deploy deploy/tts.py
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
GPU = os.environ.get("TTS_GPU", "A10")  # about $1.10 an hour. The L4 is cheaper but its capacity ran out, so the voice never started

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "chatterbox-tts==0.1.7",
        "fastapi[standard]==0.124.4",
        "peft==0.18.0",
        "setuptools<82",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)  # weights are downloaded once into the volume below, not on every cold start

voices = modal.Volume.from_name("nudge-voices", create_if_missing=True)
weights = modal.Volume.from_name("chatterbox-cache", create_if_missing=True)
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
    volumes={"/voices": voices, "/cache": weights},
)
@modal.concurrent(max_inputs=8)
class Chatterbox:
    @modal.enter()
    def load(self):
        voices.reload()  # sees a voice.wav uploaded after the volume was last mounted
        logger.info("voice: %s", "cloned from /voices/voice.wav" if VOICE_FILE.exists() else "the model's default")
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        weights.commit()  # keep the downloaded weights for the next cold start
        self.synthesize("Warming up.")  # the first real request is then not the slow one

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
