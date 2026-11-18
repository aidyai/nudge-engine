import asyncio
import io
import json
import wave

import numpy as np

from scripts import simulate_live as live


def wav(seconds: float = 0.3, rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.random.default_rng(1).integers(-9000, 9000, int(rate * seconds))).astype("<i2").tobytes())
    return buffer.getvalue()


class FakeSocket:
    """An engine that greets, then answers each borrower line once it hears them stop (silence after sound)."""

    def __init__(self, replies):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.replies = list(replies)
        self.heard_sound = False
        self.queue_agent(self.replies.pop(0))

    def queue_agent(self, reply, user=None):
        if user:
            self.queue.put_nowait(json.dumps({"type": "user", "text": user}))
        self.queue.put_nowait(json.dumps({"type": "agent", "verified": False, **reply}))
        for _ in range(3):
            self.queue.put_nowait(b"\x01\x00" * 100)

    async def send(self, data: bytes):
        if any(data):
            self.heard_sound = True
        elif self.heard_sound and self.replies:
            self.heard_sound = False
            self.queue_agent(self.replies.pop(0), user="what you said")

    async def recv(self):
        return await self.queue.get()


def test_the_borrower_voice_becomes_16k_microphone_audio():
    pcm = live.mic_audio(wav(0.3, 24_000))
    assert len(pcm) == 2 * 4800 and len(pcm) % 2 == 0  # 0.3 s at 16 kHz, 16-bit


async def test_a_live_call_measures_the_silence_before_each_answer():
    replies = [
        {"text": "Hello, this is Idara.", "state": "verify", "outcome": None, "done": False},
        {"text": "Thank you. Your birthday?", "state": "verify", "outcome": None, "done": False},
        {"text": "Done. Goodbye.", "state": "close", "outcome": "promise_captured", "done": True},
    ]
    out = []
    result = await live.live_call(
        FakeSocket(replies), ["Yes.", "21 January", "never reached"], lambda text: wav(), say=out.append, pace=0.02
    )
    assert result["outcome"] == "promise_captured" and result["done"] is True
    assert [t["reply"] for t in result["turns"]] == ["Thank you. Your birthday?", "Done. Goodbye."]  # it stopped when the agent hung up
    assert all(t["heard"] == "what you said" and t["silence_ms"] >= 0 for t in result["turns"])
    assert result["silence_ms"]["n"] == 2


async def test_an_agent_that_never_answers_is_reported_not_waited_for_forever(monkeypatch):
    monkeypatch.setattr(live, "PATIENCE_S", 0.3)
    replies = [{"text": "Hello.", "state": "verify", "outcome": None, "done": False}]  # no second reply
    result = await live.live_call(FakeSocket(replies), ["Yes."], lambda text: wav(), say=lambda *_: None, pace=1.0)
    assert result["turns"] == [] and result["outcome"] is None
