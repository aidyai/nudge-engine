import io
import wave

from scripts import simulate_call as sim


def wav(frames: int = 100, rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


class FakeEngine:
    """Stands in for the API: ears that mishear one word, a brain that finishes after the third turn."""

    def __init__(self):
        self.log = []
        self.turns = 0

    def start(self):
        self.log.append("start")
        return {"call_id": "c1", "reply": "Hello, this is Sample Lender.", "done": False}, 50.0

    def speak(self, text):
        self.log.append(f"speak:{text}")
        return wav(), 200.0

    def hear(self, audio):
        self.log.append("hear")
        return "heard text", 100.0

    def turn(self, call_id, text):
        self.log.append(f"turn:{text}")
        self.turns += 1
        done = self.turns == 3
        return {"reply": f"reply {self.turns}", "state": "close" if done else "verify", "verified": True,
                "outcome": "promise_captured" if done else None, "done": done}, 300.0  # fmt: skip


def test_the_borrowers_words_go_through_the_ears_before_the_brain():
    engine = FakeEngine()
    sim.simulate(engine, ["Yes.", "3 November"], say=lambda _: None)
    assert "turn:heard text" in engine.log  # the brain gets what the ears heard, not the script
    assert "turn:Yes." not in engine.log
    assert engine.log.index("hear") < engine.log.index("turn:heard text")


def test_the_silence_is_ears_plus_brain_plus_voice():
    result = sim.simulate(FakeEngine(), ["Yes."], say=lambda _: None)
    turn = result["turns"][0]
    assert turn["silence_ms"] == 100.0 + 300.0 + 200.0
    assert result["silence_ms"]["p50"] == 600.0


def test_the_call_stops_when_the_agent_ends_it():
    result = sim.simulate(FakeEngine(), ["a", "b", "c", "d", "e"], say=lambda _: None)
    assert len(result["turns"]) == 3 and result["done"] and result["outcome"] == "promise_captured"


def test_the_agents_side_of_the_call_is_kept_for_listening():
    result = sim.simulate(FakeEngine(), ["Yes."], say=lambda _: None)
    assert len(result["audio"]) == 2  # the greeting and one reply


def test_wavs_join_into_one_with_a_pause_between():
    joined = sim.join_wavs([wav(100), wav(100)], pause_s=0.5)
    with wave.open(io.BytesIO(joined)) as w:
        assert w.getframerate() == 24_000 and w.getnframes() == 100 + 100 + 12_000


def test_a_stored_birthday_is_said_like_a_person_says_it():
    assert sim.spoken_birthday("03-11") == "3 November"
    assert sim.spoken_birthday("25-12") == "25 December"


def test_it_says_what_is_missing_instead_of_crashing(monkeypatch, capsys):
    for name in ("API_URL", "API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert sim.main([]) == 1
    assert "API_URL" in capsys.readouterr().out
