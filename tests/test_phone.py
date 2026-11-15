from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import EndWorkerFrame, LLMContextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test
from starlette.websockets import WebSocketDisconnect

from engine.agent import start_call
from engine.api import create_app
from engine.config import Settings
from engine.phone import AgentVoice, last_user_text, twiml
from tests.conftest import TODAY
from tests.test_agent import Script, say

KEY = {"X-API-Key": "secret"}
TO = "+2348012345678"


def settings(**values):
    base = dict(
        api_key="secret",
        twilio_account_sid="ACtest",
        twilio_auth_token="tok",
        twilio_from_number="+15005550006",
        allowed_numbers=f"{TO}, +2348098765432",
    )
    return Settings(_env_file=None, **{**base, **values})  # pyright: ignore[reportCallIssue]


class Twilio:
    """A fake Twilio. Records the request it was sent."""

    def __init__(self, status=201, body=None):
        self.requests, self.status, self.body = [], status, body or {"sid": "CA123"}

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    def form(self):
        return {k: v[0] for k, v in parse_qs(self.requests[-1].content.decode()).items()}


def client(pg_engine, db, twilio, runner=None, **cfg):
    kwargs = {"phone_runner": runner} if runner else {}
    app = create_app(settings(**cfg), pg_engine, Script(), http=httpx.Client(transport=httpx.MockTransport(twilio)), **kwargs)
    return TestClient(app)


# ---------------------------------------------------------------- small pure pieces


def test_twiml_connects_the_call_audio_to_our_websocket_with_a_passcode():
    xml = twiml("wss://host.example/ws", "abc123")
    assert '<Stream url="wss://host.example/ws">' in xml and '<Parameter name="nonce" value="abc123"/>' in xml


def test_the_last_thing_the_caller_said():
    ctx = LLMContext(
        messages=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]
    )
    assert last_user_text(ctx) == "second"
    assert last_user_text(LLMContext(messages=[{"role": "user", "content": [{"type": "text", "text": "hello there"}]}])) == "hello there"
    assert last_user_text(LLMContext(messages=[])) == ""


# ---------------------------------------------------------------- ringing a number


def test_a_call_needs_the_key(pg_engine, db):
    twilio = Twilio()
    c = client(pg_engine, db, twilio)
    assert c.post("/phone/calls", json={"to": TO}).status_code == 401
    assert c.post("/phone/calls", json={"to": TO}, headers={"X-API-Key": "nope"}).status_code == 401
    assert (
        client(pg_engine, db, twilio, api_key="change-me")
        .post("/phone/calls", json={"to": TO}, headers={"X-API-Key": "change-me"})
        .status_code
        == 503
    )
    assert twilio.requests == []


def test_only_numbers_on_the_list_can_be_called(pg_engine, db):
    twilio = Twilio()
    c = client(pg_engine, db, twilio)
    assert c.post("/phone/calls", json={"to": "+2347000000000"}, headers=KEY).status_code == 403
    assert twilio.requests == []  # a stranger's number never reaches Twilio


@pytest.mark.parametrize("bad", ["08012345678", "+0123", "hello", "+234801234567890123"])
def test_numbers_must_be_in_international_format(pg_engine, db, bad):
    assert client(pg_engine, db, Twilio()).post("/phone/calls", json={"to": bad}, headers=KEY).status_code == 422


def test_a_good_request_makes_twilio_call_with_the_right_instructions(pg_engine, db):
    twilio = Twilio()
    r = client(pg_engine, db, twilio).post("/phone/calls", json={"to": TO}, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert body["call_sid"] == "CA123" and body["answer_as"] and body["birthday_to_say"].split()[0].isdigit()
    sent = twilio.requests[-1]
    assert str(sent.url) == "https://api.twilio.com/2010-04-01/Accounts/ACtest/Calls.json"
    assert sent.headers["authorization"].startswith("Basic ")
    form = twilio.form()
    assert form["To"] == TO and form["From"] == "+15005550006"
    assert "wss://testserver/phone/stream" in form["Twiml"] and 'name="nonce"' in form["Twiml"]


def test_twilio_problems_are_explained(pg_engine, db):
    twilio = Twilio(status=400, body={"code": 21219, "message": "The number is unverified."})
    r = client(pg_engine, db, twilio).post("/phone/calls", json={"to": TO}, headers=KEY)
    assert r.status_code == 502 and "unverified" in r.json()["detail"]


def test_missing_twilio_settings_say_so(pg_engine, db):
    assert client(pg_engine, db, Twilio(), twilio_auth_token="").post("/phone/calls", json={"to": TO}, headers=KEY).status_code == 503


# ---------------------------------------------------------------- Twilio connecting back


def start_messages(nonce):
    return [
        {"event": "connected", "protocol": "Call", "version": "1.0.0"},
        {"event": "start", "sequenceNumber": "1", "streamSid": "MZ123",
         "start": {"accountSid": "ACtest", "streamSid": "MZ123", "callSid": "CA123", "tracks": ["inbound"],
                   "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}, "customParameters": {"nonce": nonce}}},
    ]  # fmt: skip


def connect(c, nonce):
    with c.websocket_connect("/phone/stream") as ws:
        for message in start_messages(nonce):
            ws.send_json(message)
        ws.receive_text()  # blocks until the server closes, which raises WebSocketDisconnect


def test_a_connection_with_an_unknown_passcode_is_turned_away(pg_engine, db):
    ran = []

    async def runner(*args):
        ran.append(args)

    c = client(pg_engine, db, Twilio(), runner=runner)
    with pytest.raises(WebSocketDisconnect):
        connect(c, "guessed-passcode")
    assert ran == []


def test_a_real_call_runs_once_and_its_passcode_cannot_be_reused(pg_engine, db):
    ran = []

    async def runner(websocket, call_data, borrower_id, cfg, engine, chat):
        ran.append((call_data["stream_id"], call_data["call_id"], borrower_id))

    twilio = Twilio()
    c = client(pg_engine, db, twilio, runner=runner)
    c.post("/phone/calls", json={"to": TO}, headers=KEY)
    nonce_value = twilio.form()["Twiml"].split('name="nonce" value="')[1].split('"')[0]
    with c.websocket_connect("/phone/stream") as ws:
        for message in start_messages(nonce_value):
            ws.send_json(message)
    assert len(ran) == 1 and ran[0][:2] == ("MZ123", "CA123") and ran[0][2].startswith("B")
    with pytest.raises(WebSocketDisconnect):
        connect(c, nonce_value)  # the same passcode a second time
    assert len(ran) == 1


# ---------------------------------------------------------------- the agent inside the pipeline


async def test_the_agent_answers_when_the_caller_finishes_a_turn(db):
    call = start_call(db, "B001", today=TODAY, check_dialing=False)
    ctx = LLMContext(messages=[{"role": "user", "content": "Yes, speaking."}])
    voice = AgentVoice(call, db, Script(say("Thank you. What is your day and month of birth?")))
    down, up = await run_test(voice, frames_to_send=[LLMContextFrame(ctx)], expected_down_frames=[TTSSpeakFrame])
    assert down[0].text == "Thank you. What is your day and month of birth?" and up == []  # pyright: ignore[reportAttributeAccessIssue]


async def test_the_call_hangs_up_after_the_agent_is_done(db):
    call = start_call(db, "B001", today=TODAY, check_dialing=False)
    ctx = LLMContext(messages=[{"role": "user", "content": "Please stop calling me."}])
    voice = AgentVoice(call, db, Script())  # the opt-out is handled by code, the model is never asked
    down, up = await run_test(
        voice, frames_to_send=[LLMContextFrame(ctx)], expected_down_frames=[TTSSpeakFrame], expected_up_frames=[EndWorkerFrame]
    )
    assert "will not call you again" in down[0].text  # pyright: ignore[reportAttributeAccessIssue]
    assert call.done and call.outcome == "opted_out"
