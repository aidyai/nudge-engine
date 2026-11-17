"""Turn the brain and the voice on and off by hand. Nothing here runs by itself.

    uv run modal run deploy/warm.py::on      # keeps one brain and one voice awake now (the brain takes minutes to start)
    uv run modal run deploy/warm.py::off     # lets them sleep again: do this when you finish, it stops the billing

`off` does not delete anything. The apps stay deployed and the next request wakes them, it just starts from cold.
A model that is awake bills by the hour. This uses Modal's runtime autoscaler update, which lasts until the next
deploy of deploy/llm.py or deploy/tts.py (a deploy puts the settings back to zero, so a redeploy also stops it).
"""

from __future__ import annotations

import modal

BRAIN = ("nudge-llm", "Server")
VOICE = ("nudge-tts", "Chatterbox")

image = modal.Image.debian_slim(python_version="3.12")
app = modal.App("nudge-warm", image=image)


def set_warm(count: int) -> list[str]:
    """Ask Modal to keep `count` copies of the brain and the voice running (0 lets them sleep). Returns what it did."""
    done = []
    modal.Server.from_name(*BRAIN).autoscaler.update(min_containers=count)
    done.append(f"brain: min_containers={count}")
    modal.Cls.from_name(*VOICE)().update_autoscaler(min_containers=count)  # on the class instance: Modal refuses it on a method
    done.append(f"voice: min_containers={count}")
    return done


@app.local_entrypoint()
def on():
    print("; ".join(set_warm(1)))
    print("Starting. The brain takes minutes. They bill while awake. When you are done, run: modal run deploy/warm.py::off")


@app.local_entrypoint()
def off():
    print("; ".join(set_warm(0)))
    print("They will go to sleep after about 2 minutes. Billing stops then.")
