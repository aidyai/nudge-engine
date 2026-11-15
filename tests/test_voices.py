import json

import httpx
import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame

from engine.audio import TTS_SAMPLE_RATE
from engine.config import Settings
from engine.voices import CHUNK_BYTES, ChatterboxTTSService, SynthesisError, make_stt, make_tts

AUDIO = b"\x01\x00" * 4800  # 0.2 seconds of fake 16-bit audio at 24 kHz


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


def tts_with(handler) -> ChatterboxTTSService:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatterboxTTSService(url="https://tts.test/speak", api_key="secret", http_client=client)


def good_reply(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=AUDIO, headers={"x-sample-rate": str(TTS_SAMPLE_RATE)})


# ------------------------------------------------------------ the speech endpoint client


async def test_synthesize_sends_the_text_and_the_key():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return good_reply(request)

    audio = await tts_with(handler).synthesize("Hello there.")
    assert audio == AUDIO
    assert json.loads(seen[0].content) == {"text": "Hello there."}
    assert seen[0].headers["authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(401), "rejected the API key"),
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, content=AUDIO, headers={"x-sample-rate": "16000"}), "unexpected sample rate"),
        (httpx.Response(200, content=b"", headers={"x-sample-rate": "24000"}), "empty or broken"),
        (httpx.Response(200, content=b"\x01\x00\x02", headers={"x-sample-rate": "24000"}), "empty or broken"),
    ],
)
async def test_bad_replies_become_plain_errors(response, reason):
    with pytest.raises(SynthesisError, match=reason):
        await tts_with(lambda request: response).synthesize("Hello.")


async def test_a_network_failure_becomes_a_plain_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(SynthesisError, match="could not reach"):
        await tts_with(handler).synthesize("Hello.")


async def test_run_tts_streams_whole_samples_as_audio_frames():
    frames = [f async for f in tts_with(good_reply).run_tts("Hello.", "ctx-1")]
    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames and b"".join(f.audio for f in audio_frames) == AUDIO
    assert all(len(f.audio) % 2 == 0 and f.sample_rate == TTS_SAMPLE_RATE for f in audio_frames)
    assert all(len(f.audio) <= CHUNK_BYTES for f in audio_frames)


async def test_run_tts_reports_a_failure_as_an_error_frame_not_a_crash():
    frames = [f async for f in tts_with(lambda request: httpx.Response(500)).run_tts("Hello.", "ctx-1")]
    assert len(frames) == 1 and isinstance(frames[0], ErrorFrame)


# ------------------------------------------------------------ building the services from settings


def test_make_stt_needs_a_key_and_uses_nova_3():
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        make_stt(settings())
    stt = make_stt(settings(deepgram_api_key="dg-key"))
    assert stt._settings.model == "nova-3-general"  # pyright: ignore[reportPrivateUsage]


def test_make_tts_needs_its_url_and_key():
    with pytest.raises(ValueError, match="TTS_URL"):
        make_tts(settings())
    assert make_tts(settings(tts_url="https://tts.test/speak", models_api_key="k"))
