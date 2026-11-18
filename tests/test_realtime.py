import json

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    OutputTransportMessageFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import run_test
from starlette.websockets import WebSocketDisconnect

from engine.agent import start_call
from engine.api import create_app
from engine.config import Settings
from engine.phone import AgentVoice
from engine.realtime import BrowserSerializer, make_ticket, read_ticket, save_latency
from tests.conftest import TODAY
from tests.test_agent import Script, say

KEY = {"X-API-Key": "secret"}


def settings(**values):
    return Settings(_env_file=None, **{"api_key": "secret", **values})  # pyright: ignore[reportCallIssue]


def client(pg_engine, runner=None, **cfg):
    kwargs = {"realtime_runner": runner} if runner else {}
    return TestClient(create_app(settings(**cfg), pg_engine, Script(), **kwargs))


# ---------------------------------------------------------------- the ticket


def test_a_ticket_gives_back_the_borrower_it_was_made_for():
    assert read_ticket("secret", make_ticket("secret", "B001", now=1000), now=1010) == "B001"


@pytest.mark.parametrize("ticket", ["", "B001", "B001.2000.deadbeef", "B002.1060.x"])
def test_a_forged_or_broken_ticket_gives_nothing(ticket):
    assert read_ticket("secret", ticket, now=1000) is None


def test_a_ticket_runs_out_and_needs_the_same_secret():
    ticket = make_ticket("secret", "B001", now=1000)
    assert read_ticket("secret", ticket, now=1061) is None
    assert read_ticket("another", ticket, now=1010) is None
    assert read_ticket("", make_ticket("", "B001", now=1000), now=1010) is None


# ---------------------------------------------------------------- audio and messages to and from the page


async def test_the_page_gets_audio_as_bytes_and_everything_else_as_json():
    s = BrowserSerializer()
    assert await s.serialize(TTSAudioRawFrame(b"\x01\x02", 24000, 1)) == b"\x01\x02"
    assert json.loads(await s.serialize(InterruptionFrame()) or "") == {"type": "interrupt"}  # pyright: ignore[reportArgumentType]
    assert json.loads(await s.serialize(OutputTransportMessageFrame(message={"type": "user", "text": "hi"})) or "") == {  # pyright: ignore[reportArgumentType]
        "type": "user",
        "text": "hi",
    }
    assert await s.serialize(TextFrame("ignored")) is None
    assert await s.serialize(OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "metrics"})) is None


async def test_microphone_bytes_become_16k_audio_and_text_is_ignored():
    s = BrowserSerializer()
    frame = await s.deserialize(b"\x00\x01" * 10)
    assert isinstance(frame, InputAudioRawFrame) and frame.sample_rate == 16000 and frame.num_channels == 1
    assert await s.deserialize("hello") is None
    assert isinstance(await s.deserialize('{"type": "interrupt"}'), InterruptionFrame)  # a tap on the screen cuts the agent off


# ---------------------------------------------------------------- routes


def test_a_ticket_needs_the_key(pg_engine, db):
    c = client(pg_engine)
    assert c.post("/realtime/tickets").status_code == 401
    r = c.post("/realtime/tickets", headers=KEY)
    body = r.json()
    assert r.status_code == 200 and body["ws_url"].endswith("/realtime/stream")
    assert read_ticket("secret", body["ticket"]) is not None and body["borrower"]["birthday"]


def test_a_good_ticket_starts_the_call_for_its_borrower(pg_engine, db):
    ran = []

    async def runner(websocket, borrower_id, cfg, engine, chat):
        ran.append(borrower_id)

    c = client(pg_engine, runner)
    ticket = c.post("/realtime/tickets", headers=KEY).json()["ticket"]
    with c.websocket_connect(f"/realtime/stream?ticket={ticket}"):
        pass
    assert len(ran) == 1 and ran[0].startswith("B")


@pytest.mark.parametrize("ticket", ["", "nonsense", "B001.9999999999.abc"])
def test_a_bad_ticket_is_turned_away(pg_engine, db, ticket):
    ran = []

    async def runner(*args):
        ran.append(args)

    c = client(pg_engine, runner)
    with pytest.raises(WebSocketDisconnect), c.websocket_connect(f"/realtime/stream?ticket={ticket}") as ws:
        ws.receive_text()
    assert ran == []


def test_the_placeholder_key_never_opens_a_call(pg_engine, db):
    c = client(pg_engine, api_key="change-me")
    ticket = make_ticket("change-me", "B001")
    with pytest.raises(WebSocketDisconnect), c.websocket_connect(f"/realtime/stream?ticket={ticket}") as ws:
        ws.receive_text()


# ---------------------------------------------------------------- what the page is told, and the timing


async def test_the_page_is_told_what_was_heard_and_what_the_agent_says(db):
    call = start_call(db, "B001", today=TODAY, check_dialing=False)
    ctx = LLMContext(messages=[{"role": "user", "content": "Yes, speaking."}])
    voice = AgentVoice(call, db, Script(say("Thank you. What is your day and month of birth?")), announce=True)
    down, _ = await run_test(
        voice,
        frames_to_send=[LLMContextFrame(ctx)],
        expected_down_frames=[OutputTransportMessageFrame, OutputTransportMessageFrame, TTSSpeakFrame],
    )
    heard, said = down[0].message, down[1].message  # pyright: ignore[reportAttributeAccessIssue]
    assert heard == {"type": "user", "text": "Yes, speaking."}
    assert said["type"] == "agent" and said["text"].startswith("Thank you") and said["done"] is False


async def test_the_silence_after_the_caller_stops_is_saved_with_its_parts(db):
    call = start_call(db, "B001", today=TODAY, check_dialing=False)
    await save_latency(db, call, LatencyBreakdown(total_secs=1.234))
    row = db.exec_driver_sql("select latency_ms, payload from nudge.events where type = 'silence'").one()
    assert row[0] == 1234 and row[1]["parts"] == []


# ---------------------------------------------------------------- the whole pipeline, with a fake mouth and fake ears


class FakeEars(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class FakeMouth(FrameProcessor):
    """Turns each sentence into 100 ms of silence, like a voice would turn it into speech."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            await self.push_frame(TTSAudioRawFrame(b"\x00\x00" * 2400, 24000, 1))
        else:
            await self.push_frame(frame, direction)


def test_a_live_call_greets_the_caller_with_text_and_audio(pg_engine, db, monkeypatch):
    from engine import realtime

    monkeypatch.setattr(realtime, "make_stt", lambda cfg: FakeEars())
    monkeypatch.setattr(realtime, "make_tts", lambda cfg: FakeMouth())
    greeting = "Hello, this is Idara, an automated assistant from Nudge Finance. Am I speaking with Musa Bello?"
    c = TestClient(create_app(settings(), pg_engine, Script(say(greeting))))
    ticket = c.post("/realtime/tickets", headers=KEY).json()["ticket"]
    with c.websocket_connect(f"/realtime/stream?ticket={ticket}") as ws:
        first = json.loads(ws.receive_text())
        audio = ws.receive_bytes()
    assert first["type"] == "agent" and first["text"] == greeting and first["done"] is False
    assert len(audio) > 0
