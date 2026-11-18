"""The web API: the website's calls, speech, and phone routes. Everything but /health needs X-API-Key.

Call state lives in memory, so this runs as one container.
"""

import secrets
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from engine import agent, records
from engine.audio import pcm_to_wav
from engine.intake import NewBorrower, WhoToCall, create_borrower, describe, pick_borrower
from engine.phone import add_phone_routes, run_phone_call
from engine.realtime import add_realtime_routes, run_browser_call
from engine.voices import KEYTERMS, SynthesisError, pcm_from_reply

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
MAX_CALLS_KEPT = 50
MAX_AUDIO_BYTES = 4_000_000


class Turn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class Say(BaseModel):
    text: str = Field(min_length=1, max_length=600)


def create_app(
    cfg, engine, chat, http=None, phone_runner=run_phone_call, realtime_runner=run_browser_call, demo_archetype="cooperative_due_3d"
):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)  # the URL is public: no docs
    http = http or httpx.Client()
    calls = {}  # call id -> agent.Call

    def check(key):
        if not cfg.api_key or cfg.api_key == "change-me":
            raise HTTPException(503, "the server has no API_KEY configured")
        if not key or not secrets.compare_digest(key.encode(), cfg.api_key.encode()):
            raise HTTPException(401, "bad or missing API key")

    def view(call, reply):
        return {
            "call_id": call.id,
            "reply": reply,
            "state": call.state,
            "verified": call.verified,
            "outcome": call.outcome,
            "done": call.done,
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/calls")
    def list_calls(limit: int = 50, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        with engine.connect() as db:
            return records.list_calls(db, min(max(limit, 1), 200))

    @app.get("/calls/{call_id}")
    def one_call(call_id: str, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        try:
            uuid.UUID(call_id)
        except ValueError:
            raise HTTPException(404, "no such call") from None
        with engine.connect() as db:
            detail = records.call_detail(db, call_id)
        if detail is None:
            raise HTTPException(404, "no such call")
        return detail

    @app.post("/borrowers")
    def new_borrower(form: NewBorrower, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        try:
            form.day_month()
        except ValueError as e:
            raise HTTPException(422, "that day and month do not exist together, like 31 February") from e
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as db:
            borrower_id = create_borrower(db, form)
            return {"borrower_id": borrower_id, "borrower": describe(pick_borrower(db, borrower_id, demo_archetype))}

    @app.post("/calls")
    def start_call(body: WhoToCall | None = None, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as db:
            row = pick_borrower(db, body.borrower_id if body else None, demo_archetype)
            if row is None:
                raise HTTPException(503, "no such borrower in the database: run the seed command, or fill in the form again")
            call = agent.start_call(db, row.id, check_dialing=False)  # a demo: the calling-hours rule is for the real scheduler
            try:
                reply = agent.greet(call, db, chat)
            except RuntimeError as e:
                raise HTTPException(503, "the model is waking up or unreachable, try again shortly") from e
        calls[call.id] = call
        if len(calls) > MAX_CALLS_KEPT:
            calls.pop(next(iter(calls)))
        return {**view(call, reply), "borrower": describe(row)}

    @app.post("/calls/{call_id}/turns")
    def take_turn(call_id: str, body: Turn, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        call = calls.get(call_id)
        if call is None:
            raise HTTPException(404, "no such call")
        if call.done:
            return view(call, "")
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as db:
            try:
                reply = agent.turn(call, db, body.text, chat)
            except RuntimeError as e:
                raise HTTPException(503, "the model is waking up or unreachable, try again shortly") from e
        return view(call, reply)

    @app.post("/speech/transcribe")
    async def transcribe(request: Request, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        if not cfg.deepgram_api_key:
            raise HTTPException(503, "DEEPGRAM_API_KEY is not set")
        kind = request.headers.get("content-type", "")
        if not kind.startswith("audio/"):
            raise HTTPException(415, "send the recording with an audio content type")
        audio = await request.body()
        if not audio:
            raise HTTPException(400, "the recording was empty")
        if len(audio) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "the recording is too large")
        headers = {"Authorization": f"Token {cfg.deepgram_api_key}", "Content-Type": kind}
        try:
            r = await run_in_threadpool(
                lambda: http.post(
                    DEEPGRAM_URL,
                    params={"model": "nova-3", "smart_format": "true", "punctuate": "true", "keyterm": KEYTERMS},
                    headers=headers,
                    content=audio,
                    timeout=30,
                )
            )
        except httpx.HTTPError as e:
            raise HTTPException(502, "could not reach Deepgram") from e
        if r.status_code in (401, 403):
            raise HTTPException(502, "Deepgram rejected the key (check DEEPGRAM_API_KEY)")
        if r.status_code != 200:
            raise HTTPException(502, f"Deepgram answered HTTP {r.status_code}")
        try:
            best = r.json()["results"]["channels"][0]["alternatives"][0]
        except (ValueError, KeyError, IndexError):
            raise HTTPException(502, "Deepgram sent a reply we could not read") from None
        return {"transcript": best.get("transcript", ""), "confidence": best.get("confidence")}

    @app.post("/speech/say")
    def say(body: Say, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        if not cfg.tts_url or not cfg.models_api_key:
            raise HTTPException(503, "TTS_URL and MODELS_API_KEY are not set")
        try:
            r = http.post(
                cfg.tts_url,
                json={"text": body.text},
                headers={"Authorization": f"Bearer {cfg.models_api_key}"},
                timeout=120,
                follow_redirects=True,  # a cold start makes Modal answer 303
            )  # the first request starts a GPU
            pcm = pcm_from_reply(r)
        except httpx.HTTPError as e:
            raise HTTPException(502, "could not reach the voice server") from e
        except SynthesisError as e:
            raise HTTPException(502, str(e)) from e
        return Response(pcm_to_wav(pcm), media_type="audio/wav")

    add_phone_routes(app, cfg, engine, chat, http, check, runner=phone_runner, demo_archetype=demo_archetype)
    add_realtime_routes(app, cfg, engine, chat, check, runner=realtime_runner, demo_archetype=demo_archetype)
    return app
