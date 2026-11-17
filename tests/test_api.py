import httpx
import pytest
from fastapi.testclient import TestClient

from engine.api import create_app
from engine.config import Settings
from tests.test_agent import Script, say

KEY = {"X-API-Key": "secret"}
GREETING = "Hello, this is an automated assistant calling for Sample Lender. Am I speaking with the borrower?"
sim_months = "January February March April May June July August September October November December".split()
PCM = b"\x01\x00" * 2400
TRANSCRIPT = {"results": {"channels": [{"alternatives": [{"transcript": "I can pay on Friday", "confidence": 0.97}]}]}}


class Upstream:
    """Fake voice server and fake Deepgram. Each test can replace either one."""

    def __init__(self):
        self.calls = []
        self.tts = lambda r: httpx.Response(200, content=PCM, headers={"x-sample-rate": "24000"})
        self.deepgram = lambda r: httpx.Response(200, json=TRANSCRIPT)

    def __call__(self, request):
        self.calls.append(request)
        return {"tts.test": self.tts, "api.deepgram.com": self.deepgram}[request.url.host](request)


def make_client(pg_engine, db, *messages, upstream=None, **cfg):
    values = dict(api_key="secret", models_api_key="m", deepgram_api_key="dg", tts_url="https://tts.test/speak") | cfg
    script, upstream = Script(*messages), upstream or Upstream()
    http = httpx.Client(transport=httpx.MockTransport(upstream))
    return TestClient(create_app(Settings(_env_file=None, **values), pg_engine, script, http=http)), script, upstream  # pyright: ignore[reportCallIssue]


# ---------------------------------------------------------------- basics


def test_health_needs_no_key_and_docs_are_hidden(pg_engine, db):
    client, *_ = make_client(pg_engine, db)
    assert client.get("/health").json() == {"status": "ok"}
    assert all(client.get(p).status_code == 404 for p in ("/docs", "/redoc", "/openapi.json"))


def test_the_key_is_required_and_the_placeholder_never_works(pg_engine, db):
    client, script, upstream = make_client(pg_engine, db)
    assert client.post("/calls").status_code == 401
    assert client.post("/calls", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/speech/say", json={"text": "hi"}).status_code == 401
    unsafe, *_ = make_client(pg_engine, db, api_key="change-me")
    assert unsafe.post("/calls", headers={"X-API-Key": "change-me"}).status_code == 503
    assert script.offered == [] and upstream.calls == []  # nothing reached a paid service


# ---------------------------------------------------------------- calls


def test_starting_a_call_returns_the_greeting_and_the_state(pg_engine, db):
    client, *_ = make_client(pg_engine, db, say(GREETING))
    r = client.post("/calls", headers=KEY).json()
    assert r["reply"] == GREETING and r["state"] == "verify" and r["verified"] is False and r["done"] is False
    assert len(r["call_id"]) == 36
    assert r["borrower"]["name"] and r["borrower"]["birthday"].split()[1] in sim_months  # who the demo borrower is, so the page can say it


def test_a_turn_goes_to_the_agent(pg_engine, db):
    client, *_ = make_client(pg_engine, db, say(GREETING), say("Thank you. What is your day and month of birth?"))
    call_id = client.post("/calls", headers=KEY).json()["call_id"]
    r = client.post(f"/calls/{call_id}/turns", json={"text": "Yes, speaking."}, headers=KEY).json()
    assert r["reply"].startswith("Thank you") and r["call_id"] == call_id


def test_unknown_calls_and_bad_text_are_refused(pg_engine, db):
    client, *_ = make_client(pg_engine, db, say(GREETING))
    call_id = client.post("/calls", headers=KEY).json()["call_id"]
    assert client.post("/calls/nope/turns", json={"text": "hi"}, headers=KEY).status_code == 404
    assert client.post(f"/calls/{call_id}/turns", json={"text": ""}, headers=KEY).status_code == 422
    assert client.post(f"/calls/{call_id}/turns", json={"text": "x" * 501}, headers=KEY).status_code == 422
    assert client.post(f"/calls/{call_id}/turns", json={"text": "hi"}).status_code == 401


def test_a_finished_call_does_not_ask_the_model_again(pg_engine, db):
    client, script, _ = make_client(pg_engine, db, say(GREETING))
    call_id = client.post("/calls", headers=KEY).json()["call_id"]
    r = client.post(f"/calls/{call_id}/turns", json={"text": "Please stop calling me"}, headers=KEY).json()
    assert r["done"] is True and r["outcome"] == "opted_out"
    again = client.post(f"/calls/{call_id}/turns", json={"text": "hello?"}, headers=KEY).json()
    assert again["done"] is True and again["reply"] == "" and len(script.offered) == 1


def test_a_sleeping_model_gives_a_clear_503(pg_engine, db):
    def broken(messages, tools):
        raise RuntimeError("model server answered 503")

    client = TestClient(create_app(Settings(_env_file=None, api_key="secret"), pg_engine, broken))  # pyright: ignore[reportCallIssue]
    r = client.post("/calls", headers=KEY)
    assert r.status_code == 503 and "waking up" in r.json()["detail"]


def test_a_missing_demo_borrower_says_so(pg_engine, db):
    client = TestClient(create_app(Settings(_env_file=None, api_key="secret"), pg_engine, Script(), demo_archetype="nobody"))  # pyright: ignore[reportCallIssue]
    assert client.post("/calls", headers=KEY).status_code == 503


# ---------------------------------------------------------------- speech


def test_a_recording_becomes_text(pg_engine, db):
    client, _, upstream = make_client(pg_engine, db)
    r = client.post("/speech/transcribe", content=b"fake-webm", headers={**KEY, "Content-Type": "audio/webm;codecs=opus"})
    assert r.json() == {"transcript": "I can pay on Friday", "confidence": 0.97}
    sent = upstream.calls[0]
    assert sent.headers["authorization"] == "Token dg" and sent.headers["content-type"] == "audio/webm;codecs=opus"
    assert sent.url.params["model"] == "nova-3"
    assert "naira" in sent.url.params.get_list("keyterm")  # Deepgram heard "naira" as "twenty dra" without it


@pytest.mark.parametrize(("headers", "body", "status"), [
    ({"Content-Type": "text/plain"}, b"hello", 415),
    ({"Content-Type": "audio/webm"}, b"", 400),
    ({"Content-Type": "audio/webm"}, b"x" * 4_000_001, 413),
])  # fmt: skip
def test_bad_recordings_are_refused_before_anything_is_paid_for(pg_engine, db, headers, body, status):
    client, _, upstream = make_client(pg_engine, db)
    assert client.post("/speech/transcribe", content=body, headers={**KEY, **headers}).status_code == status
    assert upstream.calls == []


def test_deepgram_rejecting_the_key_is_explained(pg_engine, db):
    upstream = Upstream()
    upstream.deepgram = lambda r: httpx.Response(401)
    client, *_ = make_client(pg_engine, db, upstream=upstream)
    r = client.post("/speech/transcribe", content=b"abc", headers={**KEY, "Content-Type": "audio/webm"})
    assert r.status_code == 502 and "DEEPGRAM_API_KEY" in r.json()["detail"]


def test_text_becomes_a_playable_wav(pg_engine, db):
    client, *_ = make_client(pg_engine, db)
    r = client.post("/speech/say", json={"text": "Hello."}, headers=KEY)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"


@pytest.mark.parametrize(
    "reply", [httpx.Response(401), httpx.Response(500), httpx.Response(200, content=PCM, headers={"x-sample-rate": "16000"})]
)
def test_a_broken_voice_server_is_explained(pg_engine, db, reply):
    upstream = Upstream()
    upstream.tts = lambda r: reply
    client, *_ = make_client(pg_engine, db, upstream=upstream)
    assert client.post("/speech/say", json={"text": "Hello."}, headers=KEY).status_code == 502


def test_the_voice_refuses_empty_and_oversized_text(pg_engine, db):
    client, *_ = make_client(pg_engine, db)
    assert client.post("/speech/say", json={"text": ""}, headers=KEY).status_code == 422
    assert client.post("/speech/say", json={"text": "x" * 601}, headers=KEY).status_code == 422
