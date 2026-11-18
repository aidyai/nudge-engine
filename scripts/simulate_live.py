"""A live call without a person: a pretend borrower streams its voice over the websocket, like the browser does.

    uv run --env-file .env python -m scripts.simulate_live --label live

Silence = from the borrower's last sound to the agent's first sound. Saved to evals/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
import wave
from collections.abc import Callable
from urllib.parse import quote

import numpy as np
import soxr

from evals.stats import summarize
from scripts.simulate_call import LINES, RESULTS, Engine, borrower_birthday

MIC_RATE = 16_000
CHUNK_S = 0.04  # the page sends 40 ms of microphone at a time
VOICE_DONE_S = 1.0  # no agent audio for this long means the agent has finished speaking
PATIENCE_S = 60  # the longest we wait for the agent to start answering


def mic_audio(wav: bytes) -> bytes:
    """A wav from the voice model, as the raw 16 kHz audio a microphone would send."""
    with wave.open(io.BytesIO(wav)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return soxr.resample(samples, w.getframerate(), MIC_RATE).astype("<i2").tobytes()


class Listener:
    """Collects what the agent sends: audio with the time it arrived, and the JSON lines about the call."""

    def __init__(self, socket, clock: Callable[[], float] = time.monotonic):
        self.socket, self.clock = socket, clock
        self.audio_at: list[float] = []
        self.said: list[dict] = []
        self.heard: list[str] = []

    async def run(self) -> None:
        try:
            while True:
                message = await self.socket.recv()
                if isinstance(message, bytes):
                    self.audio_at.append(self.clock())
                    continue
                body = json.loads(message)
                if body["type"] == "agent":
                    self.said.append(body)
                elif body["type"] == "user":
                    self.heard.append(body["text"])
        except Exception:  # the socket closed: the call is over
            return

    async def wait_for(self, ready: Callable[[], bool], seconds: float, tick: float = 0.02) -> bool:
        deadline = self.clock() + seconds
        while not ready():
            if self.clock() > deadline:
                return False
            await asyncio.sleep(tick)
        return True

    async def agent_finished(self, since: int, pace: float = 1.0) -> None:
        """Wait until the agent has started speaking and then gone quiet."""
        if await self.wait_for(lambda: len(self.audio_at) > since, PATIENCE_S * pace):
            await self.wait_for(lambda: self.clock() - self.audio_at[-1] > VOICE_DONE_S * pace, PATIENCE_S * pace)


async def speak_into(socket, listener: Listener, pcm: bytes, pace: float = 1.0) -> tuple[float, float | None]:
    """Stream the borrower's voice, then silence, in real time. Returns when the last sound ended and when the agent first answered."""
    step = int(MIC_RATE * CHUNK_S) * 2
    before = len(listener.audio_at)
    for start in range(0, len(pcm), step):
        await socket.send(pcm[start : start + step])
        await asyncio.sleep(CHUNK_S * pace)
    ended = listener.clock()
    quiet = b"\x00\x00" * int(MIC_RATE * CHUNK_S)
    while len(listener.audio_at) <= before and listener.clock() - ended < PATIENCE_S * pace:  # the agent hears silence until it answers
        await socket.send(quiet)
        await asyncio.sleep(CHUNK_S * pace)
    answered = listener.audio_at[before] if len(listener.audio_at) > before else None
    return ended, answered


async def live_call(socket, lines: list[str], speak: Callable[[str], bytes], say=print, pace: float = 1.0) -> dict:
    """One whole call. `socket` has send(bytes) and recv(); `speak(text)` gives the borrower's voice as a wav."""
    listener = Listener(socket)
    reader = asyncio.create_task(listener.run())
    try:
        await listener.agent_finished(0, pace)
        say(f"AGENT: {listener.said[0]['text'] if listener.said else '(nothing)'}")
        turns = []
        for line in lines:
            if listener.said and listener.said[-1]["done"]:
                break
            said_before, heard_before, audio_before = len(listener.said), len(listener.heard), len(listener.audio_at)
            ended, answered = await speak_into(socket, listener, mic_audio(speak(line)), pace)
            await listener.agent_finished(audio_before, pace)
            await listener.wait_for(lambda n=said_before: len(listener.said) > n, 2 * pace)
            if answered is None:
                say(f"YOU said: {line!r}\n  the agent did not answer in time")
                break
            reply = listener.said[said_before] if len(listener.said) > said_before else {"text": "", "state": "?", "outcome": None}
            heard = listener.heard[heard_before] if len(listener.heard) > heard_before else ""
            silence_ms = (answered - ended) * 1000
            say(
                f"YOU said: {line!r}\n  heard as: {heard!r}\nAGENT: {reply['text']}\n  step: {reply['state']} | silence {silence_ms:.0f} ms"
            )
            turns.append({"said": line, "heard": heard, "reply": reply["text"], "state": reply["state"], "silence_ms": silence_ms})
        last = listener.said[-1] if listener.said else {}
        return {
            "outcome": last.get("outcome"),
            "verified": last.get("verified"),
            "done": last.get("done"),
            "turns": turns,
            "silence_ms": summarize([t["silence_ms"] for t in turns]),
        }
    finally:
        reader.cancel()


async def run(engine: Engine, lines: list[str], say=print) -> dict:
    import websockets

    ticket, _ = engine._timed("POST", "/realtime/tickets")
    body = ticket.json()
    say(f"You are {body['borrower']['name']}.")
    async with websockets.connect(f"{body['ws_url']}?ticket={quote(body['ticket'])}", max_size=None) as socket:
        return await live_call(socket, lines, lambda text: engine.speak(text)[0], say)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="live", help="name for the saved result")
    args = parser.parse_args(argv)
    missing = [n for n in ("API_URL", "API_KEY", "DATABASE_URL") if not os.environ.get(n)]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}. Run the setup cell first (it saves API_URL).")
        return 1
    dob = borrower_birthday(os.environ["DATABASE_URL"])
    result = asyncio.run(run(Engine(os.environ["API_URL"], os.environ["API_KEY"]), [line.format(dob=dob) for line in LINES]))
    RESULTS.mkdir(exist_ok=True)
    meta = {"kind": "live", "label": args.label, "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    path = RESULTS / f"{args.label}-live.json"
    path.write_text(json.dumps({"meta": meta, **result}, indent=2))
    s = result["silence_ms"]
    if s["n"]:
        spread = f"{s['p50']:.0f} / {s['p95']:.0f} / {s['p99']:.0f} ms"
        print(f"\noutcome: {result['outcome']}  |  silence after the borrower stops (p50 / p95 / p99): {spread}")
    print(f"saved {path}")
    return 0 if result["outcome"] == "promise_captured" else 2


if __name__ == "__main__":
    sys.exit(main())
