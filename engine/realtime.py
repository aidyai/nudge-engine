"""Live calls from the browser: audio streams both ways over one websocket, and the same agent does the talking.

The browser cannot send the API key on a websocket, so the website asks for a ticket first (signed, one minute) and connects with it.
"""

import asyncio
import hashlib
import hmac
import json
import time

from fastapi import Header, HTTPException, Request, WebSocket
from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from engine import agent, lender
from engine.audio import TTS_SAMPLE_RATE
from engine.intake import WhoToCall, describe, pick_borrower
from engine.phone import AgentVoice
from engine.voices import make_stt, make_tts

MIC_SAMPLE_RATE = 16_000  # what the browser sends: raw 16-bit mono audio
TICKET_SECONDS = 60


def sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_ticket(secret: str, borrower_id: str, now: float | None = None) -> str:
    expires = int((now if now is not None else time.time()) + TICKET_SECONDS)
    payload = f"{borrower_id}.{expires}"
    return f"{payload}.{sign(secret, payload)}"


def read_ticket(secret: str, ticket: str, now: float | None = None) -> str | None:
    """The borrower id inside a good ticket, or None if it is forged, broken or too old."""
    payload, _, signature = ticket.rpartition(".")
    borrower_id, _, expires = payload.rpartition(".")
    if not secret or not hmac.compare_digest(signature, sign(secret, payload)) or not expires.isdigit():
        return None
    return borrower_id if int(expires) >= (now if now is not None else time.time()) else None


class BrowserSerializer(FrameSerializer):
    """Audio goes as raw bytes. Everything the page needs to show goes as one line of JSON."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if self.should_ignore_frame(frame):  # Pipecat's own protocol messages are not for this page
            return None
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "interrupt"})
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=MIC_SAMPLE_RATE, num_channels=1)
        try:
            message = json.loads(data)
        except ValueError:
            return None
        if isinstance(message, dict) and message.get("type") == "interrupt":  # the person tapped the screen to cut the agent off
            return InterruptionFrame()
        return None


async def save_latency(db, call, breakdown):
    """The wait after the caller stopped, split into its parts (Pipecat measures them). Saved on the call."""
    parts = {"from": str(breakdown.measured_from), "parts": breakdown.turn_contribution_lines()}
    lender.log_event(db, call.id, "silence", parts, round(breakdown.total_secs * 1000))


async def run_browser_call(websocket, borrower_id, cfg, engine, chat):
    db = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    call = agent.start_call(db, borrower_id, check_dialing=False)
    logger.info(f"live call {call.id} started")
    voice = AgentVoice(call, db, chat, announce=True)
    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_breakdown")
    async def on_breakdown(observer, breakdown):
        await save_latency(db, call, breakdown)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=BrowserSerializer()),
    )
    user_turns, _ = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),  # the voice detector says "sound stopped"...
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]  # ...smart-turn says "they are finished"
            ),
        ),
    )
    worker = PipelineWorker(
        Pipeline([transport.input(), make_stt(cfg), user_turns, voice, make_tts(cfg), transport.output()]),
        params=PipelineParams(audio_in_sample_rate=MIC_SAMPLE_RATE, audio_out_sample_rate=TTS_SAMPLE_RATE, enable_metrics=True),
        observers=[latency],
    )

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
        logger.info(f"live call {call.id} ended, outcome {call.outcome}")


def add_realtime_routes(app, cfg, engine, chat, check, runner=run_browser_call, demo_archetype="cooperative_due_3d"):
    @app.post("/realtime/tickets")
    def ticket(request: Request, body: WhoToCall | None = None, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        with engine.connect() as db:
            who = pick_borrower(db, body.borrower_id if body else None, demo_archetype)
        if who is None:
            raise HTTPException(503, "no such borrower in the database: run the seed command, or fill in the form again")
        host = request.headers.get("host") or request.url.netloc
        scheme = "ws" if host.startswith(("localhost", "127.0.0.1")) else "wss"  # plain ws only when testing on your own machine
        return {
            "ticket": make_ticket(cfg.api_key, who.id),
            "ws_url": f"{scheme}://{host}/realtime/stream",
            "borrower": describe(who),
        }

    @app.websocket("/realtime/stream")
    async def stream(websocket: WebSocket, ticket: str = ""):
        borrower_id = read_ticket(cfg.api_key, ticket) if cfg.api_key != "change-me" else None
        await websocket.accept()
        if borrower_id is None:
            await websocket.close(code=1008)  # not a ticket we gave out, or it ran out
            return
        await runner(websocket, borrower_id, cfg, engine, chat)
