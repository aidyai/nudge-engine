import pytest

from engine.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LLM_BASE_URL", "ALLOWED_NUMBERS", "API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def fresh() -> Settings:
    return Settings(_env_file=None)  # pyright: ignore[reportCallIssue]  # ignore any real .env


def test_defaults_are_safe():
    s = fresh()
    assert s.api_key == "change-me"  # the API refuses this placeholder
    assert s.llm_model == "phonellm-mini-alpha-2"
    assert s.allowed_list == []  # nobody can be phoned until you list them


def test_the_environment_overrides(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.modal.direct/v1")
    assert fresh().llm_base_url == "https://example.modal.direct/v1"


def test_allowed_numbers_are_split_and_trimmed(monkeypatch):
    monkeypatch.setenv("ALLOWED_NUMBERS", " +2348012345678 ,+2348098765432,, ")
    assert fresh().allowed_list == ["+2348012345678", "+2348098765432"]
