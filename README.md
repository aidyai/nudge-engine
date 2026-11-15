# nudge-engine

The backend for Nudge, a voice agent that phones a borrower before a loan payment is due, reminds them, asks what they plan to do, and records the promise.

It runs against a fake lender and made-up borrowers. Nothing here touches real money or real customers. The website that goes with it is the `nudge` repo.

![Architecture](docs/architecture.svg)

## The idea

The model decides what to say and which tool to call. Plain code decides what is allowed. The prompt asks for good behaviour, but nothing depends on the model obeying it.

- The model only sees the tools that fit the current step of the call.
- Every tool call goes through one function (`run_tool` in `engine/agent.py`), which checks the step and the rules first.
- Every reply is checked before it is spoken: no digits, nothing threatening, no loan details before the caller is verified, and every amount and date must be copied word for word from a tool result.
- Nothing is saved until the caller answers a read-back sentence that the code wrote. A repeated "yes" cannot save twice.
- "Stop calling me" and "let me speak to a person" are handled by code, not by the model.

![Steps of a call](docs/call-states.svg)

## What is in here

```
engine/    the agent, the rules, the API, phone calls
deploy/    what runs on Modal: llm.py (the brain), tts.py (the voice), api.py
evals/     scripted borrowers that talk to the real model and check the rules held
db/        database migrations
scripts/   check_services.py, to see that your keys and URLs work
tests/     unit tests, no keys and no GPU needed
docs/      deploying, and phone calls
```

If you read one file, read `engine/agent.py`. It is one file, top to bottom.

## Run it on your machine

You need Python 3.12, [uv](https://docs.astral.sh/uv/) (the Python equivalent of npm), and a Postgres database. A free Neon database works.

```
uv sync
cp .env.example .env     # fill in DATABASE_URL and API_KEY to start
uv run python -m engine.seed --init-schema
uv run uvicorn engine.main:build_app --factory --reload --port 8000
```

Open http://localhost:8000/health. The seed command creates the tables and 45 fake borrowers.

To have a real conversation you also need the models. The brain and the voice need GPUs, so they run on Modal. The API can run on your machine and point at them. `docs/deploy.md` has the steps.

The tests need none of that:

```
uv run pytest
```

## Deploy

`docs/deploy.md`. In short: the models and the API run on Modal, and a push to `main` redeploys them after the tests pass. Deploying never starts a GPU, only a real request does.

## Evals

```
uv run modal run evals/run.py
```

Six scripted borrowers talk to the real deployed model: a cooperative one, one who fails verification, one who says stop, one who wants a person, one who tries to trick the agent, and one who offers too little. After each conversation the script checks the rules that must always hold: no digit was spoken, no loan detail before verification, at most one promise, and the expected outcome. It prints PASS or FAIL for each.

## Phone calls

Twilio can ring a number and the same agent talks to whoever answers. Only numbers you list in `ALLOWED_NUMBERS` can be called. See `docs/phone.md`.

## Not done yet

- A scheduler that decides who to call and when.
- Reconciling payments against promises.
- A real lender integration. The lender here is a few database tables.
- Languages other than English.
- A proper latency report. The terminal and Modal logs show timings, nothing more.
