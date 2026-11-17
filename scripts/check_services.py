"""Check every outside service answers to your keys. Prints OK, SKIP or FAIL, never a secret.

uv run --env-file .env python -m scripts.check_services
"""

from __future__ import annotations

import difflib
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from engine.audio import TTS_SAMPLE_RATE, pcm_to_wav
from engine.voices import KEYTERMS

PATIENCE_S = 45 * 60  # a cold brain took 9, 18 and 29 minutes to start (Modal's limit for it is 45)
SECRET_NAMES = ("DEEPGRAM_API_KEY", "MODELS_API_KEY", "HF_TOKEN", "DATABASE_URL", "API_KEY", "LLM_PROXY_KEY", "LLM_PROXY_SECRET")
TEST_SENTENCE = "Hello, you owe twenty-five thousand naira, due on Tuesday."


class Skip(Exception):
    """Not set up yet. Not a failure."""


def _need(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"{missing[0]} is empty in .env")


def similarity(a: str, b: str) -> float:
    """0 to 1: how alike two sentences are, ignoring case and punctuation."""

    def words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    return difflib.SequenceMatcher(None, words(a), words(b)).ratio()


# the checks
def check_deepgram() -> str:
    _need("DEEPGRAM_API_KEY")
    r = httpx.get(
        "https://api.deepgram.com/v1/projects",
        headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
        timeout=20,
    )
    r.raise_for_status()
    return "key accepted"


def check_database() -> str:
    _need("DATABASE_URL")
    url = os.environ["DATABASE_URL"]
    if not url.startswith("postgresql+psycopg://"):
        raise RuntimeError("DATABASE_URL must start with postgresql+psycopg://")
    from sqlalchemy import create_engine, text

    with create_engine(url, connect_args={"connect_timeout": 15}).connect() as conn:
        conn.execute(text("SELECT 1"))
    return "connected"


def check_modal() -> str:
    if os.environ.get("MODAL_TOKEN_ID") or (Path.home() / ".modal.toml").exists():
        return "login found"
    raise RuntimeError("not logged in: run `uv run modal setup`")


def check_huggingface() -> str:
    _need("HF_TOKEN")
    r = httpx.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
        timeout=20,
    )
    r.raise_for_status()
    return f"token belongs to {r.json().get('name', 'someone')}"


def check_llm() -> str:
    if not os.environ.get("LLM_BASE_URL"):
        raise Skip("not deployed yet: run `uv run modal deploy deploy/llm.py`, then set LLM_BASE_URL")
    _need("MODELS_API_KEY")
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    body = {
        "model": os.environ.get("LLM_MODEL", "phonellm-mini-alpha-2"),
        "messages": [{"role": "user", "content": "Say hello in five words."}],
        "temperature": 0,
        "max_tokens": 32,
    }
    headers = {"Authorization": f"Bearer {os.environ['MODELS_API_KEY']}"}
    if os.environ.get("LLM_PROXY_KEY") and os.environ.get("LLM_PROXY_SECRET"):
        headers |= {"Modal-Key": os.environ["LLM_PROXY_KEY"], "Modal-Secret": os.environ["LLM_PROXY_SECRET"]}
    print("       waiting for the LLM server; a cold start can take several minutes...", flush=True)
    deadline = time.time() + PATIENCE_S
    while True:
        response = None
        try:
            response = httpx.post(f"{base}/chat/completions", headers=headers, json=body, timeout=60)
        except (httpx.TimeoutException, httpx.ConnectError):
            pass  # the server is still starting
        if response is not None and response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            return f"model replied: {str(reply).strip()[:50]!r}"
        if response is not None and response.status_code in (401, 403) and "Modal-Key" not in headers:
            raise RuntimeError("the brain wants a Modal proxy token: set LLM_PROXY_KEY and LLM_PROXY_SECRET (see docs/deploy.md)")
        if response is not None and response.status_code not in (502, 503, 504):
            response.raise_for_status()
        if time.time() > deadline:
            raise RuntimeError(f"the server was still not ready after {PATIENCE_S // 60} minutes")
        time.sleep(10)


def _speak(text: str) -> bytes:
    _need("MODELS_API_KEY")
    r = httpx.post(
        os.environ["TTS_URL"],
        json={"text": text},
        headers={"Authorization": f"Bearer {os.environ['MODELS_API_KEY']}"},
        timeout=300,  # the first request starts a GPU container
        follow_redirects=True,  # Modal answers 303 when a cold start takes over 150 s, and the result comes from the new address
    )
    r.raise_for_status()
    if r.headers.get("x-sample-rate") != str(TTS_SAMPLE_RATE) or not r.content or len(r.content) % 2:
        raise RuntimeError("the speech server returned audio in an unexpected format")
    return r.content


def check_tts() -> str:
    if not os.environ.get("TTS_URL"):
        raise Skip("not deployed yet: run `uv run modal deploy deploy/tts.py`, then set TTS_URL")
    pcm = _speak(TEST_SENTENCE)
    Path("tts_check.wav").write_bytes(pcm_to_wav(pcm))
    return f"{len(pcm) / (2 * TTS_SAMPLE_RATE):.1f} seconds of audio saved to tts_check.wav (play it!)"


def check_round_trip() -> str:
    if not os.environ.get("TTS_URL"):
        raise Skip("needs the TTS server first")
    _need("DEEPGRAM_API_KEY")
    wav = pcm_to_wav(_speak(TEST_SENTENCE))
    r = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-3", "keyterm": KEYTERMS},
        headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": "audio/wav"},
        content=wav,
        timeout=60,
    )
    r.raise_for_status()
    heard = r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    score = similarity(TEST_SENTENCE, heard)
    if "naira" not in heard.lower():  # the word every call depends on
        raise RuntimeError(f"Deepgram heard {heard!r}: it did not catch the word naira")
    if score < 0.8:
        raise RuntimeError(f"Deepgram heard {heard!r}, which is not close to what was said")
    return f"said and heard back: {heard!r}"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("Deepgram (speech to text)", check_deepgram),
    ("Neon (database)", check_database),
    ("Modal (your GPUs)", check_modal),
    ("Hugging Face (model files)", check_huggingface),
    ("Your LLM on Modal (vLLM)", check_llm),
    ("Your TTS on Modal (Chatterbox)", check_tts),
    ("Speech round trip", check_round_trip),
]


def _scrub(message: str) -> str:
    for name in SECRET_NAMES:
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "***")
    return message


def main() -> int:
    failures = 0
    for label, check in CHECKS:
        try:
            print(f"[ OK ] {label}: {check()}")
        except Skip as e:
            print(f"[SKIP] {label}: {e}")
        except httpx.HTTPStatusError as e:
            failures += 1
            print(f"[FAIL] {label}: the service said HTTP {e.response.status_code} (check the key or URL)")
        except Exception as e:  # noqa: BLE001 - a checker must report, never crash
            failures += 1
            print(f"[FAIL] {label}: {_scrub(str(e))[:160]}")
    print("\nAll good." if not failures else f"\n{failures} to fix before we continue.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
