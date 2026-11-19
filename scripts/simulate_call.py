"""A phone call without the phone: a pretend borrower speaks, the engine hears it and answers out loud.

    uv run --env-file .env python -m scripts.simulate_call --label call

borrower voice -> ears (Deepgram) -> brain -> voice. Silence = ears + brain + voice, saved to evals/results/.
The agent's side is written to simulated_call.wav.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import wave
from pathlib import Path

import httpx

from evals.stats import summarize

LINES = [
    "Yes, speaking.",
    "{dob}",
    "I can pay on Friday.",
    "Yes, that is right.",
    "Thank you.",
]  # the happy path, as a borrower would say it
MONTHS = "January February March April May June July August September October November December".split()
RESULTS = Path(__file__).resolve().parent.parent / "evals" / "results"
STILL_STARTING = (408, 502, 503, 504)
DEMO_ARCHETYPE = "cooperative_due_3d"  # the borrower the API's demo call rings


ORDINALS = (
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth "
    "sixteenth seventeenth eighteenth nineteenth twentieth twenty-first twenty-second twenty-third twenty-fourth "
    "twenty-fifth twenty-sixth twenty-seventh twenty-eighth twenty-ninth thirtieth thirty-first"
).split()


def spoken_birthday(day_month: str) -> str:
    """'03-11' (day-month, as stored) becomes 'the third of November'. Words, because a cloned voice can read '21' as '2000'."""
    day, month = day_month.split("-")
    return f"the {ORDINALS[int(day) - 1]} of {MONTHS[int(month) - 1]}"


class Engine:
    """The deployed API, as a caller sees it. Each method returns what it got and how long it took in milliseconds."""

    def __init__(self, url: str, key: str, timeout: float = 600, patience_s: float = 600, pause_s: float = 10):
        self.http = httpx.Client(base_url=url.rstrip("/"), headers={"X-API-Key": key}, timeout=timeout, follow_redirects=True)
        self.patience_s, self.pause_s = patience_s, pause_s

    def _timed(self, method: str, path: str, **kwargs) -> tuple[httpx.Response, float]:
        started = time.perf_counter()
        while True:  # a sleeping model answers 408, 502, 503 or 504 while it wakes: wait and ask again
            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.TimeoutException:
                response = None
            if response is not None and response.status_code not in STILL_STARTING:
                break
            if time.perf_counter() - started > self.patience_s:
                raise SystemExit(f"{path} did not answer in {self.patience_s:.0f} s. Is the model awake? Check Modal.")
            time.sleep(self.pause_s)
        response.raise_for_status()
        return response, (time.perf_counter() - started) * 1000

    def start(self) -> tuple[dict, float]:
        r, ms = self._timed("POST", "/calls")
        return r.json(), ms

    def speak(self, text: str) -> tuple[bytes, float]:
        r, ms = self._timed("POST", "/speech/say", json={"text": text})
        return r.content, ms

    def hear(self, wav: bytes) -> tuple[str, float]:
        r, ms = self._timed("POST", "/speech/transcribe", content=wav, headers={"Content-Type": "audio/wav"})
        return r.json()["transcript"], ms

    def turn(self, call_id: str, text: str) -> tuple[dict, float]:
        r, ms = self._timed("POST", f"/calls/{call_id}/turns", json={"text": text})
        return r.json(), ms


def simulate(engine, lines: list[str], say=print) -> dict:
    """Play one call. `engine` is an Engine (or a fake with the same four methods)."""
    call, start_ms = engine.start()
    say(f"AGENT: {call['reply']}")
    greeting_audio, greeting_tts_ms = engine.speak(call["reply"])
    agent_audio = [greeting_audio]
    turns = []
    for line in lines:
        if call.get("done"):
            break
        borrower_audio, _ = engine.speak(line)  # the borrower's voice: not part of what the caller waits for
        heard, stt_ms = engine.hear(borrower_audio)
        reply, brain_ms = engine.turn(call["call_id"], heard)
        audio, tts_ms = engine.speak(reply["reply"])
        agent_audio.append(audio)
        say(f"YOU said: {line!r}\n  heard as: {heard!r}\nAGENT: {reply['reply']}")
        say(
            f"  step: {reply['state']} | verified: {reply['verified']} | outcome: {reply['outcome']} "
            f"| ears {stt_ms:.0f} ms, brain {brain_ms:.0f} ms, voice {tts_ms:.0f} ms"
        )
        turns.append(
            {
                "said": line,
                "heard": heard,
                "reply": reply["reply"],
                "state": reply["state"],
                "stt_ms": stt_ms,
                "brain_ms": brain_ms,
                "tts_ms": tts_ms,
                "silence_ms": stt_ms + brain_ms + tts_ms,
            }
        )
        call = {**call, **reply}
    return {
        "outcome": call.get("outcome"),
        "verified": call.get("verified"),
        "done": call.get("done"),
        "start_ms": start_ms,
        "greeting_tts_ms": greeting_tts_ms,
        "turns": turns,
        "silence_ms": summarize([t["silence_ms"] for t in turns]),
        "audio": agent_audio,
    }


def join_wavs(parts: list[bytes], pause_s: float = 0.6) -> bytes:
    """One wav out of many (same sample rate), with a short silence between them."""
    out, params = [], None
    for part in parts:
        with wave.open(io.BytesIO(part)) as w:
            params = params or w.getparams()
            out.append(w.readframes(w.getnframes()))
    if params is None:
        return b""
    gap = b"\x00\x00" * int(params.framerate * pause_s)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setparams(params)
        w.writeframes(gap.join(out))
    return buffer.getvalue()


def borrower_birthday(database_url: str) -> str:
    from sqlalchemy import create_engine, text

    with create_engine(database_url).connect() as conn:
        row = conn.execute(
            text("select birth_day_month from lender.borrowers where archetype = :a and not opted_out order by id limit 1"),
            {"a": DEMO_ARCHETYPE},
        ).first()
    if row is None:
        raise SystemExit("No demo borrower in the database: run the seed step (the setup cell does it).")
    return spoken_birthday(row[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="call", help="name for the saved result")
    args = parser.parse_args(argv)
    missing = [n for n in ("API_URL", "API_KEY", "DATABASE_URL") if not os.environ.get(n)]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}. Run the setup cell first (it saves API_URL).")
        return 1
    dob = borrower_birthday(os.environ["DATABASE_URL"])
    result = simulate(Engine(os.environ["API_URL"], os.environ["API_KEY"]), [line.format(dob=dob) for line in LINES])
    audio = result.pop("audio")
    Path("simulated_call.wav").write_bytes(join_wavs(audio))
    RESULTS.mkdir(exist_ok=True)
    meta = {"kind": "call", "label": args.label, "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    path = RESULTS / f"{args.label}-call.json"
    path.write_text(json.dumps({"meta": meta, **result}, indent=2))
    s = result["silence_ms"]
    print(f"\noutcome: {result['outcome']}  |  silence after the borrower stops (p50 / p95 / p99): "
          f"{s['p50']:.0f} / {s['p95']:.0f} / {s['p99']:.0f} ms")  # fmt: skip
    print(f"saved {path} and simulated_call.wav")
    return 0 if result["outcome"] == "promise_captured" else 2


if __name__ == "__main__":
    sys.exit(main())
