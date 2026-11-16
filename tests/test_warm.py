"""The warm-window script: the schedule it builds and the calls it makes, with Modal replaced by fakes."""

import modal
import pytest

from deploy import warm


def test_the_models_start_before_the_window_and_stop_when_it_closes():
    assert warm.window_crons("08:00", "18:00") == ("45 7 * * 1-5", "0 18 * * 1-5")
    assert warm.window_crons("09:30", "17:45", lead_minutes=30) == ("0 9 * * 1-5", "45 17 * * 1-5")


def test_the_start_never_goes_before_midnight():
    assert warm.window_crons("00:05", "12:00", lead_minutes=15)[0] == "0 0 * * 1-5"


def test_the_schedule_comes_from_the_policy_file():
    from engine.policy import load_policy

    window = load_policy().calling_window
    assert warm.window_crons(window.start, window.end) == (warm._on_cron, warm._off_cron)


class Fake:
    def __init__(self):
        self.calls = []

    def update_autoscaler(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.parametrize("count", [0, 1])
def test_turning_on_and_off_reaches_both_the_brain_and_the_voice(monkeypatch, count):
    brain, voice = Fake(), Fake()
    names = {}

    def server_from_name(app, name):
        names["brain"] = (app, name)
        return brain

    class FakeVoice:
        speak = voice

    def cls_from_name(app, name):
        names["voice"] = (app, name)
        return FakeVoice

    monkeypatch.setattr(modal.Server, "from_name", server_from_name)
    monkeypatch.setattr(modal.Cls, "from_name", cls_from_name)
    assert warm.set_warm(count) == [f"brain: min_containers={count}", f"voice: min_containers={count}"]
    assert brain.calls == voice.calls == [{"min_containers": count}]
    assert names == {"brain": ("nudge-llm", "Server"), "voice": ("nudge-tts", "Chatterbox")}


def test_the_app_names_match_what_the_deploy_files_create():
    import inspect

    from deploy import llm, tts

    assert warm.BRAIN[0] == llm.app.name and f"class {warm.BRAIN[1]}:" in inspect.getsource(llm)
    assert warm.VOICE[0] == tts.app.name and f"class {warm.VOICE[1]}:" in inspect.getsource(tts)
    assert "def speak(" in inspect.getsource(tts)  # the method whose autoscaler we change
