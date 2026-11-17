import pytest

from scripts import launch

VOICE = "Created https://me--nudge-tts-chatterbox.modal.run\n"
BRAIN = "Created https://me--nudge-llm-server.modal.run.\n"
API = "Created https://me--nudge-api-web.modal.run\n"


class FakeJob:
    def __init__(self, shell, cmd, env):
        self.shell, self.cmd, self.env = shell, cmd, env

    def wait(self):
        return self.shell.answer(self.cmd, self.env)


class FakeShell:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[list[str], dict | None]] = []
        self.fail_on = fail_on

    def answer(self, cmd, env):
        text = " ".join(cmd)
        if self.fail_on and self.fail_on in text:
            return 1, "boom"
        for name, out in (("llm.py", BRAIN), ("tts.py", VOICE), ("api.py", API)):
            if name in text:
                return 0, out
        return 0, ""

    def run(self, cmd, env=None):
        self.calls.append((cmd, env))
        return self.answer(cmd, env)

    def start(self, cmd, env=None):
        self.calls.append((cmd, env))
        return FakeJob(self, cmd, env)

    def text(self):
        return [" ".join(c) for c, _ in self.calls]


@pytest.fixture
def env_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text("API_KEY=a\nHF_TOKEN=h\n")
    return p


def test_it_does_every_step_in_order_and_saves_the_urls(env_file):
    shell = FakeShell()
    env = launch.launch(shell, env_path=env_file)
    steps = shell.text()
    order = [
        "modal secret create huggingface-secret",
        "modal secret create nudge-models",
        "deploy/llm.py",
        "deploy/tts.py",
        "modal secret create nudge-env",
        "deploy/api.py",
        "deploy/warm.py::on",
        "scripts.check_services",
    ]
    positions = [next(i for i, s in enumerate(steps) if o in s) for o in order]
    assert positions == sorted(positions)
    assert env["TTS_URL"] == "https://me--nudge-tts-chatterbox.modal.run"
    assert env["LLM_BASE_URL"] == "https://me--nudge-llm-server.modal.run/v1"
    assert env["API_URL"] == "https://me--nudge-api-web.modal.run"


def test_the_seed_step_only_runs_when_asked(env_file):
    plain = FakeShell()
    launch.launch(plain, env_path=env_file)
    assert not any("engine.seed" in s for s in plain.text())
    seeded = FakeShell()
    launch.launch(seeded, seed=True, env_path=env_file)
    assert "engine.seed --init-schema" in seeded.text()[0]


def test_snapshot_mode_sets_the_flag_for_the_brain_only(env_file):
    env_file.write_text("LLM_PROXY_KEY=wk-1\nLLM_PROXY_SECRET=ws-1\n")
    shell = FakeShell()
    launch.launch(shell, snapshot=True, env_path=env_file)
    brain = next(env for cmd, env in shell.calls if "deploy/llm.py" in " ".join(cmd))
    assert brain == {"LLM_SNAPSHOT": "1"}
    voice = next(env for cmd, env in shell.calls if "deploy/tts.py" in " ".join(cmd))
    assert not voice


def test_a_normal_launch_turns_the_snapshot_off(env_file):
    shell = FakeShell()
    launch.launch(shell, env_path=env_file)
    brain = next(env for cmd, env in shell.calls if "deploy/llm.py" in " ".join(cmd))
    assert brain == {"LLM_SNAPSHOT": "0"}


def test_snapshot_mode_without_the_proxy_token_stops_before_deploying(env_file):
    shell = FakeShell()
    with pytest.raises(launch.Failed, match="proxy token"):
        launch.launch(shell, snapshot=True, env_path=env_file)
    assert shell.calls == []


@pytest.mark.parametrize("bad", ["deploy/tts.py", "deploy/llm.py", "deploy/api.py", "deploy/warm.py::on", "scripts.check_services"])
def test_it_stops_at_the_first_failure(env_file, bad):
    shell = FakeShell(fail_on=bad)
    with pytest.raises(launch.Failed):
        launch.launch(shell, env_path=env_file)
    assert not any("scripts.check_services" in s for s in shell.text()[: shell.text().index(next(s for s in shell.text() if bad in s))])


def test_the_secret_command_does_not_put_the_secret_in_the_command_line(env_file):
    shell = FakeShell()
    launch.launch(shell, env_path=env_file)
    cmd = next(s for s in shell.text() if "create huggingface-secret" in s)
    assert "$HF_TOKEN" in cmd and "h --force" not in cmd


def test_find_url_ignores_trailing_punctuation():
    assert launch.find_url("see https://a--nudge-llm-x.modal.run.", "nudge-llm") == "https://a--nudge-llm-x.modal.run"
    assert launch.find_url("nothing here", "nudge-llm") is None


def test_child_output_is_unbuffered_so_lines_appear_when_printed():
    job = launch.RealJob(["python", "-c", "import os; print(os.environ['PYTHONUNBUFFERED'])"], None, echo=False)
    assert job.wait() == (0, "1\n")
