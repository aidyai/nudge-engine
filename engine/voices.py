"""The ears (Deepgram) and the voice (Chatterbox on Modal), as Pipecat services for phone calls."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from engine.audio import TTS_SAMPLE_RATE
from engine.config import Settings

CHUNK_BYTES = TTS_SAMPLE_RATE * 2 // 10  # 100 ms of audio per frame, always a whole number of samples


# Words Deepgram should favour. Without this it heard "naira" as "twenty dra" (found with scripts/check_services.py).
KEYTERMS = ["naira", "kobo"]


def make_stt(settings: Settings) -> DeepgramSTTService:
    """Deepgram Nova-3, streaming, English."""
    if not settings.deepgram_api_key:
        raise ValueError("DEEPGRAM_API_KEY is empty in .env")
    return DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            language=Language.EN,
            interim_results=True,
            punctuate=True,
            keyterm=KEYTERMS,
        ),
    )


class SynthesisError(Exception):
    """The text-to-speech endpoint did not give us usable audio."""


def pcm_from_reply(response: httpx.Response) -> bytes:
    """Check what the voice server sent back and return the raw audio. Raises SynthesisError with a plain reason."""
    if response.status_code == 401:
        raise SynthesisError("the speech server rejected the API key (check MODELS_API_KEY)")
    if response.status_code != 200:
        raise SynthesisError(f"the speech server answered HTTP {response.status_code}")
    if response.headers.get("x-sample-rate") != str(TTS_SAMPLE_RATE):
        raise SynthesisError("the speech server returned an unexpected sample rate")
    if not response.content or len(response.content) % 2:
        raise SynthesisError("the speech server returned empty or broken audio")
    return response.content


class ChatterboxTTSService(TTSService):
    """Pipecat text-to-speech that calls the Chatterbox endpoint from deploy/tts.py."""

    Settings = TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        timeout_s: float = 20.0,
        **kwargs,
    ):
        default_settings = self.Settings(model="chatterbox-turbo", voice=None, language=None)
        super().__init__(
            sample_rate=TTS_SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )
        self._url, self._api_key, self._timeout_s = url, api_key, timeout_s
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient()

    def can_generate_metrics(self) -> bool:
        return True

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._close()

    async def _close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def synthesize(self, text: str) -> bytes:
        """One sentence in, raw 16-bit PCM out. Raises SynthesisError with a plain reason."""
        try:
            response = await self._http.post(
                self._url,
                json={"text": text},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout_s,
                follow_redirects=True,  # a cold start makes Modal answer 303: the audio is at the address it gives
            )
        except httpx.HTTPError as e:
            raise SynthesisError(f"could not reach the speech server ({type(e).__name__})") from e
        return pcm_from_reply(response)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            audio = await self.synthesize(text)
        except SynthesisError as e:
            yield ErrorFrame(error=str(e))
            return
        await self.start_tts_usage_metrics(text)
        await self.stop_ttfb_metrics()
        for start in range(0, len(audio), CHUNK_BYTES):
            chunk = audio[start : start + CHUNK_BYTES]
            yield TTSAudioRawFrame(chunk, TTS_SAMPLE_RATE, 1, context_id=context_id)


def make_tts(settings: Settings) -> ChatterboxTTSService:
    if not settings.tts_url or not settings.models_api_key:
        raise ValueError("TTS_URL and MODELS_API_KEY must be set in .env (deploy deploy/tts.py first)")
    return ChatterboxTTSService(url=settings.tts_url, api_key=settings.models_api_key)
