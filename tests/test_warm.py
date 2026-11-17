"""The on/off script: the calls it makes, with Modal replaced by fakes."""

import modal
import pytest

from deploy import warm


class Fake:
    def __init__(self):
        self.calls = []

    def update_autoscaler(self, **kwargs):
        self.calls.append(kwargs)

    @property
    def autoscaler(self):
        return self

    def update(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.parametrize("count", [0, 1])
def test_turning_on_and_off_reaches_both_the_brain_and_the_voice(monkeypatch, count):
    brain, voice = Fake(), Fake()
    names = {}

    def server_from_name(app, name):
        names["brain"] = (app, name)
        return brain

    def cls_from_name(app, name):
        names["voice"] = (app, name)
        return lambda: voice  # calling the class gives the instance, which is where Modal wants the update

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
    assert "def speak(" in inspect.getsource(tts)


def test_nothing_in_the_warm_script_runs_by_itself():
    """Billing must only start when a person asks. No schedules, no timers."""
    import inspect

    source = inspect.getsource(warm)
    assert "Cron" not in source and "schedule=" not in source and "Period" not in source
