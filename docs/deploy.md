# Deploying

Three things run on Modal: the brain (`deploy/llm.py`), the voice (`deploy/tts.py`) and the API (`deploy/api.py`).
The first time, do it by hand. You see errors straight away, and it is faster. After that, a push to `main` redeploys everything.

## First time

1. **Accounts.** Modal (add a payment method, and set a monthly budget under Usage and Billing), Hugging Face, Deepgram, and a Postgres database (Neon is fine).
   Put the keys in `.env`. Make two long random strings for `MODELS_API_KEY` and `API_KEY`:
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`

2. **Two Modal secrets** for the models:
   ```
   uv run modal secret create huggingface-secret HF_TOKEN=hf_xxx
   uv run modal secret create nudge-models MODELS_API_KEY=<your first random string>
   ```

3. **The brain and the voice.** Start these first, the brain takes the longest because it downloads a 30B model.
   ```
   uv run modal deploy deploy/tts.py
   uv run modal deploy deploy/llm.py
   ```
   Each prints a URL. In `.env`, set `TTS_URL` to the voice URL, and `LLM_BASE_URL` to the brain URL with `/v1` on the end.

4. **The database.** `uv run --env-file .env python -m engine.seed --init-schema`

5. **The API.**
   ```
   uv run modal secret create nudge-env --from-dotenv .env --force
   uv run modal deploy deploy/api.py
   ```
   Open `https://<your-workspace>--nudge-api-web.modal.run/health`. It should answer `{"status":"ok"}`.
   Run step 5's first command again whenever you change `.env`.

6. **Check it.** `uv run modal run evals/run.py` runs the eight scripted conversations. The first run waits while the brain wakes up.

7. **The website.** In the `nudge` repo, set `ENGINE_URL` to the API URL and `ENGINE_API_KEY` to your `API_KEY`.

## After that

Add these in GitHub under Settings, Secrets and variables, Actions:

| Secret | Value |
|---|---|
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal, Settings, API Tokens |
| `HF_TOKEN` | your Hugging Face token |
| `MODELS_API_KEY` | the first random string |

Then every push to `main` runs the tests and the secret scan, and if they pass, redeploys the voice, the brain and the API.
Pull requests from forks never see the secrets.

## Seeing what happened

```
uv run modal app logs nudge-api -f
uv run modal app logs nudge-llm
uv run modal app logs nudge-tts
```

The voice logs only sizes and timings, never the text.

## Cost

Nothing runs while idle: each service stops a few minutes after its last request, and each is limited to one container.
While running, the brain costs a few dollars an hour and the voice under one. Modal's free plan includes monthly credit, and the budget you set caps the rest.
The first request after a quiet spell is slow because the brain has to start.
