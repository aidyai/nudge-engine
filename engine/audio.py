"""Small audio helpers."""

from __future__ import annotations

import io
import wave

TTS_SAMPLE_RATE = 24_000  # deploy/tts.py always returns 16-bit mono PCM at this rate


def pcm_to_wav(pcm: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV header so a browser (or Deepgram) can play it."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
