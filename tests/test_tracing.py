"""Tracing is optional: with no keys every step just runs, and with a client every step is recorded."""

import contextlib

import pytest

from engine import tracing


@pytest.fixture(autouse=True)
def fresh():
    tracing.client.cache_clear()
    yield
    tracing.client.cache_clear()


def test_without_keys_everything_is_a_no_op(monkeypatch):
    monkeypatch.setattr(tracing, "get_settings", lambda: type("S", (), {"langfuse_public_key": "", "langfuse_secret_key": ""})())
    assert tracing.client() is None
    with tracing.span("turn", session="c1", input="hello") as step:
        step.update(output="x")
        step.score("silence_ms", 900)
    tracing.flush()


def test_with_a_client_the_steps_and_scores_are_recorded(monkeypatch):
    seen = []

    class Observation:
        def update(self, **fields):
            seen.append(("update", fields))

    class Client:
        @contextlib.contextmanager
        def start_as_current_observation(self, **fields):
            seen.append(("start", fields["name"], fields["as_type"]))
            yield Observation()

        def score_current_trace(self, **fields):
            seen.append(("score", fields["name"], fields["value"]))

    monkeypatch.setattr(tracing, "client", lambda: Client())
    with tracing.span("turn", input="hello") as step:
        step.update(output="hi")
        step.score("task_passed", 1)
    assert seen == [("start", "turn", "span"), ("update", {"output": "hi"}), ("score", "task_passed", 1.0)]
