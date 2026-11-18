"""Phone calls: Twilio rings a number and streams the audio here, and the same agent does the talking.

Only numbers in ALLOWED_NUMBERS can be called.
"""

import asyncio
import secrets

from fastapi import Header, HTTPException, Request, WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndWorkerFrame, Frame, LLMContextFrame, OutputTransportMessageFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner
from pydantic import BaseModel, Field
from sqlalchemy import text

from engine import agent, lender
from engine.voices import make_stt, make_tts

MONTHS = "January February March April May June July August September October November December".split()


def twiml(ws_url, nonce):
    """What Twilio does when the call is answered: connect the audio to our websocket, with a one-time passcode."""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{ws_url}">'
        f'<Parameter name="nonce" value="{nonce}"/></Stream></Connect></Response>'
    )


def last_user_text(context):
    for m in reversed(context.get_messages()):
        if m.get("role") == "user":
            c = m["content"]
            return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


class AgentVoice(FrameProcessor):
    """Sits where the LLM normally sits. When the caller finishes a turn, ask the agent and speak its reply."""

    def __init__(self, call, db, chat, announce=False):
        super().__init__()
        self.call, self.db, self.chat = call, db, chat
        self.announce = announce  # tell the web page what was said (the phone has no page)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):  # the caller finished talking
            if heard := last_user_text(frame.context).strip():
                await self.tell({"type": "user", "text": heard})
                await self.say(await asyncio.to_thread(agent.turn, self.call, self.db, heard, self.chat))
        else:
            await self.push_frame(frame, direction)

    async def tell(self, message):
        if self.announce:
            await self.push_frame(OutputTransportMessageFrame(message=message))

    async def say(self, reply):
        if reply:
            await self.tell(
                {
                    "type": "agent",
                    "text": reply,
                    "state": self.call.state,
                    "verified": self.call.verified,
                    "outcome": self.call.outcome,
                    "done": self.call.done,
                }
            )
            await self.push_frame(TTSSpeakFrame(reply))
        if self.call.done:  # hang up after the last words are spoken
            await self.push_frame(EndWorkerFrame(), FrameDirection.UPSTREAM)


async def run_phone_call(websocket, call_data, borrower_id, cfg, engine, chat):
    db = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    call = agent.start_call(db, borrower_id, check_dialing=False)
    logger.info(f"phone call {call.id} started")
    voice = AgentVoice(call, db, chat)
    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=cfg.twilio_account_sid,
        auth_token=cfg.twilio_auth_token,
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer),
    )
    user_turns, _ = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.7)))
    )  # 0.7 s of silence ends the caller's turn
    worker = PipelineWorker(
        Pipeline([transport.input(), make_stt(cfg), user_turns, voice, make_tts(cfg), transport.output()]),
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000, enable_metrics=True),
    )  # phone audio is 8 kHz

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        await voice.say(await asyncio.to_thread(agent.greet, call, db, chat))  # the agent speaks first

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await worker.cancel()

    try:
        await WorkerRunner(handle_sigint=False).run(worker)
    finally:
        if not call.done:
            lender.end_call(db, call.id, call.outcome or "hung_up")
        db.close()
        logger.info(f"phone call {call.id} ended, outcome {call.outcome}")


class CallRequest(BaseModel):
    to: str = Field(pattern=r"^\+[1-9]\d{7,14}$")  # international format, like +2348012345678


def add_phone_routes(app, cfg, engine, chat, http, check, runner=run_phone_call, demo_archetype="cooperative_due_3d"):
    pending = {}  # one-time passcode -> borrower id, for calls we placed and Twilio has not connected yet

    @app.post("/phone/calls")
    def ring(body: CallRequest, request: Request, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        if body.to not in cfg.allowed_list:
            raise HTTPException(403, "that number is not on the allowed list (ALLOWED_NUMBERS)")
        if not (cfg.twilio_account_sid and cfg.twilio_auth_token and cfg.twilio_from_number):
            raise HTTPException(503, "Twilio is not configured")
        with engine.connect() as db:
            who = db.execute(
                text(
                    "select id, full_name, birth_day_month from lender.borrowers where archetype = :a and not opted_out order by id limit 1"
                ),
                {"a": demo_archetype},
            ).first()
        if who is None:
            raise HTTPException(503, "no demo borrower in the database: run the seed command")
        nonce = secrets.token_urlsafe(16)
        pending[nonce] = who.id
        if len(pending) > 20:
            pending.pop(next(iter(pending)))
        ws_url = f"wss://{request.headers.get('host') or request.url.netloc}/phone/stream"
        r = http.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{cfg.twilio_account_sid}/Calls.json",
            auth=(cfg.twilio_account_sid, cfg.twilio_auth_token),
            data={"To": body.to, "From": cfg.twilio_from_number, "Twiml": twiml(ws_url, nonce)},
            timeout=20,
        )
        if r.status_code >= 300:
            pending.pop(nonce, None)
            try:
                why = r.json().get("message", "")
            except ValueError:
                why = ""
            raise HTTPException(502, f"Twilio refused the call: {why or r.status_code}")
        day, month = who.birth_day_month.split("-")
        return {"call_sid": r.json().get("sid"), "answer_as": who.full_name, "birthday_to_say": f"{int(day)} {MONTHS[int(month) - 1]}"}

    @app.websocket("/phone/stream")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        try:
            kind, data = await parse_telephony_websocket(websocket)
        except Exception:
            await websocket.close(code=1008)
            return
        borrower_id = pending.pop((data.get("body") or {}).get("nonce", ""), None)  # one use only
        if kind != "twilio" or borrower_id is None:
            await websocket.close(code=1008)  # not a call we placed
            return
        await runner(websocket, data, borrower_id, cfg, engine, chat)
