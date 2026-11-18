"""Optional Langfuse tracing. Without LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY nothing here does anything.

with tracing.span("turn", session=call.id, input=text) as s:
    ...
    s.update(output=reply)
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from typing import Any

from loguru import logger

from engine.config import get_settings


@lru_cache
def client():
    """The Langfuse client, or None when no keys are set (or the library is missing)."""
    cfg = get_settings()
    if not (cfg.langfuse_public_key and cfg.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(public_key=cfg.langfuse_public_key, secret_key=cfg.langfuse_secret_key, base_url=cfg.langfuse_base_url)
    except Exception as e:  # tracing must never stop a call
        logger.warning(f"tracing is off: {type(e).__name__}: {e}")
        return None


class Handle:
    """What a span gives back. Every method is safe to call when tracing is off."""

    def __init__(self, lf=None, observation=None):
        self.lf, self.observation = lf, observation

    def update(self, **fields) -> None:
        if self.observation is not None:
            self.observation.update(**fields)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """A number on the whole trace, like task_passed or silence_ms."""
        if self.lf is not None:
            self.lf.score_current_trace(name=name, value=float(value), data_type="NUMERIC", comment=comment)


@contextlib.contextmanager
def span(name: str, *, session: str | None = None, tags: list[str] | None = None, as_type: Any = "span", **fields: Any):
    """One step of a call. Steps inside it become its children. `session` groups every step of one call."""
    lf = client()
    if lf is None:
        yield Handle()
        return
    from langfuse import propagate_attributes

    with contextlib.ExitStack() as stack:
        if session or tags:
            stack.enter_context(propagate_attributes(session_id=session, tags=tags))
        observation = stack.enter_context(lf.start_as_current_observation(name=name, as_type=as_type, **fields))
        yield Handle(lf, observation)


def flush() -> None:
    """Send what is waiting. Call it before a short script ends."""
    if (lf := client()) is not None:
        lf.flush()
