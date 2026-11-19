# How the pieces work

The details behind the [README](../README.md).

## Live calls

The website's demo is a live call, not a recording per sentence. The browser streams the microphone to `/realtime/stream` and gets the agent's voice back as it is made (`engine/realtime.py`). The pipeline is Pipecat:

```
microphone -> Silero (is there sound?) -> Deepgram (words) -> smart-turn-v3 (are they finished?) -> agent -> voice -> speaker
```

- **Silero VAD** and **smart-turn-v3** come with Pipecat and run on the CPU of the API container, so they have no cold start of their own. The brain and the voice still wake from sleep.
- Talking over the agent interrupts it.
- The browser cannot send the API key on a websocket, so the website asks for a signed ticket first (`POST /realtime/tickets`, good for one minute).
- Pipecat measures each turn. The wait after the caller stops is saved on the call, split into its parts, as a `silence` event.
- `uv run python -m scripts.simulate_live` plays a whole call over the websocket and measures the silence, like the browser would.

Not done: the Pipecat **backchannel detector** ("mm-hm" must not interrupt) and the tone models. They need their own wiring and testing and we have not verified either. The voice still speaks a whole sentence at a time, which is the main reason the wait is long.

## Tracing in Langfuse

Put `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` (the notebook asks for them; the engine itself still runs without them, sending nothing) (a free [Langfuse Cloud](https://cloud.langfuse.com) project is enough) and every call is traced: one session per call, a step for each turn, each brain call, each tool and each rule check. Eval runs go there too, tagged with their label, with the plain scores (`rules_held`, `task_passed`, `model_ms_max`) attached to each run. Without the keys nothing is sent and nothing changes.

## Who to call

The website has a form where a company types the company name, the person, their birthday and the loan. It sends `POST /borrowers` (needs the key), which saves a borrower and a loan (migration `0002_company`, run by the SETUP step), and the call routes take an optional `borrower_id`. Seeding for evals wipes these custom borrowers. A spreadsheet upload is shown on the website but not built.

## Backchannel detector

`scripts/probe_backchannel.py` prints the inputs and outputs of `pipecat-ai/backchannel-detector` so it can be wired in without guessing. It is not in the pipeline yet; the model is English only and was trained on human conversation, so it needs measuring first.

## Reading calls back

Every call, sentence, tool call, rule check and promise is saved to the database. `GET /calls` lists them with totals and `GET /calls/<id>` returns one in full (phone numbers are masked). The website's dashboard reads these two routes, so it shows what really happened.

## Evals

```
uv run modal run evals/run.py                              # eight scripted borrowers, once
uv run modal run evals/run.py --runs 5 --label baseline    # five runs each, saved
```

Eight scripted borrowers talk to the real deployed model: a cooperative one, one who fails verification, one who says stop, one who wants a person, one who lost their job, one who says they already paid, one who tries to trick the agent, and one who offers too little.

Two scores, kept apart. **Rules** are what must never break: no digit spoken, nothing about the loan before verification, at most one promise. A broken rule fails the run. **Task** is whether the call reached the right outcome with the right tools; it is reported with a 95% interval, because "5 of 5" is not the same as "100%". The checks are plain code, not an LLM judge, so the same call scores the same way every time.

The method (scripted multi-turn calls, per-turn tool checks, latency percentiles) comes from [aiewf-eval](https://github.com/kwindla/aiewf-eval). We kept only the part that fits a small agent.

## Performance

```
uv run modal run evals/bench.py::llm --label rtx6000-bf16 --gpu RTX-PRO-6000 --price <usd per hour>
uv run modal run evals/bench.py::tts --label a10 --gpu A10 --price <usd per hour>
uv run python -m evals.report --write      # builds docs/performance.md
```

The brain benchmark replays the five requests of a real call (real prompt, real tools) with 1, 4, 8 and 16 calls at once, and reports time to first token (P50/P95/P99, goal under 700 ms), tokens per second, requests per second, the prefix-cache hit rate, the cost per 1,000 requests, and the cold start. `--tools fixed` sends all eight tools on every request, so you can show what the per-state tool lists buy. Chatterbox cannot stream, so the voice benchmark times whole sentences and reports the real-time factor.
