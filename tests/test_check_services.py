import io
import wave

import pytest

from engine.audio import pcm_to_wav
from scripts import check_services as cs


def test_pcm_to_wav_makes_a_readable_wav():
    pcm = b"\x01\x00" * 2400
    with wave.open(io.BytesIO(pcm_to_wav(pcm))) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24_000)
        assert wav.readframes(wav.getnframes()) == pcm


@pytest.mark.parametrize(
    ("said", "heard", "close"),
    [
        (
            "Hello, you owe twenty-five thousand naira.",
            "Hello. You owe 25,000 naira.",
            False,
        ),  # digits differ
        ("Hello, you owe twenty-five thousand naira.", "hello you owe twenty five thousand naira", True),
        ("Hello, you owe twenty-five thousand naira.", "Goodbye my friend", False),
    ],
)
def test_similarity_ignores_case_and_punctuation(said, heard, close):
    assert (cs.similarity(said, heard) >= 0.8) is close


def test_with_nothing_set_up_the_checker_fails_cleanly_and_skips_the_models(monkeypatch, tmp_path, capsys):
    for name in (*cs.SECRET_NAMES, "LLM_BASE_URL", "TTS_URL", "MODAL_TOKEN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.modal.toml in there
    assert cs.main() == 1
    out = capsys.readouterr().out
    assert "[FAIL] Deepgram" in out and "is empty in .env" in out
    assert "[SKIP] Your LLM on Modal" in out
    assert "[SKIP] Your TTS on Modal" in out
    assert "[SKIP] Speech round trip" in out


def test_secrets_never_reach_the_output(monkeypatch):
    monkeypatch.setenv("MODELS_API_KEY", "super-secret-value")
    assert "super-secret-value" not in cs._scrub("the key super-secret-value was rejected")
