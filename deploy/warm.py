"""Keep the brain and the voice warm during calling hours, and let them sleep the rest of the day.

Why: the brain takes about 9 minutes to wake from cold (measured, see docs/performance.md). No caller can wait
that long. But Nudge's calls are scheduled, so we know when they start. We wake the models before the calling
window opens and let them go to sleep when it closes. A warm model bills while it is warm, so this is on only
during the window.

By hand (a demo, a test, the notebook):
    uv run modal run deploy/warm.py::on      # starts one brain and one voice now (the brain takes about 9 minutes)
    uv run modal run deploy/warm.py::off     # lets them sleep again: do this when you finish, it stops the billing

Automatic (calling hours come from engine/policy.yaml):
    uv run modal deploy deploy/warm.py
    A small scheduled job turns them on 15 minutes before the window opens and off when it closes, Monday to Friday.
    Change DAYS below to match the lender.

This uses Modal's runtime autoscaler update. It lasts until the next deploy of deploy/llm.py or deploy/tts.py,
which put the settings back to zero. That is fine: the next morning's job turns them on again.
"""

from __future__ import annotations

import modal

from engine.policy import load_policy

LEAD_MINUTES = 15  # how long before the window opens the models start. The brain needs about 9, so this has some slack
DAYS = "1-5"  # cron day-of-week: 1-5 is Monday to Friday
BRAIN = ("nudge-llm", "Server")
VOICE = ("nudge-tts", "Chatterbox")

image = (
    modal.Image.debian_slim(python_version="3.12").uv_pip_install("pydantic", "pyyaml", "tzdata").add_local_dir("engine", "/root/engine")
)
app = modal.App("nudge-warm", image=image)


def _hour_minute(t) -> tuple[int, int]:
    """Hour and minute from a time (what the policy file gives us) or from text like '08:00'."""
    if hasattr(t, "hour"):
        return t.hour, t.minute
    h, m = str(t).split(":")
    return int(h), int(m)


def window_crons(start, end, lead_minutes: int = LEAD_MINUTES, days: str = DAYS) -> tuple[str, str]:
    """('turn on', 'turn off') cron strings from a calling window like start='08:00', end='18:00'."""
    sh, sm = _hour_minute(start)
    eh, em = _hour_minute(end)
    on = max(0, sh * 60 + sm - lead_minutes)  # never before midnight
    return f"{on % 60} {on // 60} * * {days}", f"{em} {eh} * * {days}"


def set_warm(count: int) -> list[str]:
    """Ask Modal to keep `count` copies of the brain and the voice running (0 lets them sleep). Returns what it did."""
    done = []
    modal.Server.from_name(*BRAIN).update_autoscaler(min_containers=count)
    done.append(f"brain: min_containers={count}")
    modal.Cls.from_name(*VOICE)().speak.update_autoscaler(min_containers=count)
    done.append(f"voice: min_containers={count}")
    return done


_window = load_policy().calling_window
_on_cron, _off_cron = window_crons(_window.start, _window.end)


@app.function(schedule=modal.Cron(_on_cron, timezone=_window.timezone))
def warm_up():
    print("; ".join(set_warm(1)))


@app.function(schedule=modal.Cron(_off_cron, timezone=_window.timezone))
def cool_down():
    print("; ".join(set_warm(0)))


@app.local_entrypoint()
def on():
    print("; ".join(set_warm(1)))
    print("Starting. The brain takes about 9 minutes. When you are done, run: modal run deploy/warm.py::off")


@app.local_entrypoint()
def off():
    print("; ".join(set_warm(0)))
    print("They will go to sleep after about 2 minutes. Billing stops then.")
