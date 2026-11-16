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

## Cold starts and cost

A stopped brain costs nothing, and the price is the first request: the first cold start took about 10 minutes (about 2 minutes downloading 59 GB of weights, about 8 minutes compiling and warming up). Everything below is measured by `evals/bench.py`, so the claims in `docs/performance.md` come from runs, not from this page.

- **Caches (always on).** The weights, vLLM's `torch.compile` cache and FlashInfer's autotune cache live in Modal Volumes. Measured on the RTX PRO 6000: the first cold start took 507 s to start the engine (about 380 s of that is warming up Mamba2 Triton kernels), the second 542 s. The caches save the 2 minute download and about 8 s of compile, not the warm-up.
- **What we tried and removed.** Putting Triton's kernel cache (`TRITON_CACHE_DIR`) in the volume. The next cold start printed `Cubin file saved by TritonBundler not found` errors and took 1,097 s instead of about 520 s, and Modal killed it at the old 20 minute startup limit. It stays out. The startup limit is now 30 minutes.
- **GPU snapshot (opt in).** `LLM_SNAPSHOT=1 uv run modal deploy deploy/llm.py`. vLLM starts once, warms up, goes to sleep, and Modal snapshots the process, so later cold starts restore it instead of redoing the roughly 6 minute warm-up. It is an alpha Modal feature, the first starts after a deploy build the snapshot (Modal says it takes a handful, usually under five, before it pays off), and a redeploy throws old snapshots away.
  **Security.** Snapshot mode needs `VLLM_SERVER_DEV_MODE=1`. vLLM's security page says never to use it in production: its dev endpoints have no API-key check, and one of them (`/collective_rpc`) runs arbitrary methods on the engine. So in snapshot mode the brain is deployed with `unauthenticated=False`: Modal's proxy rejects any request without a proxy token (401) before vLLM sees it. Make a token at https://modal.com/settings/proxy-auth-tokens and put its two halves in `.env` as `LLM_PROXY_KEY` (`wk-...`) and `LLM_PROXY_SECRET` (`ws-...`). The engine, the evals and the benchmark send them as `Modal-Key` and `Modal-Secret` headers. A test checks that the default deploy has no dev endpoints and that snapshot mode is never public.
- **A warm replica for a calling window.** `LLM_MIN_CONTAINERS=1 uv run modal deploy deploy/llm.py`. It bills while it is up, so set it back to 0 afterwards.
- **One GPU, always.** `max_containers=1` and the 2 minute scale-down window cap the bill. Raise them only when a benchmark says you need to.

What this does not do: FP8 or other quantisation, a second region, or routing that keeps one caller on one replica. They are the next experiments, and each one would be a new line in the performance table.

## Warm window (what makes cold starts a non-problem for scheduled calls)

Modal's own tools for cold starts, and what we use:

| Modal tool | What it does | Where we use it |
|---|---|---|
| `min_containers` | keeps N copies running, so nothing starts from cold | `deploy/warm.py`, only during calling hours |
| `autoscaler.update` / `update_autoscaler` | changes those settings while running, without a redeploy | `deploy/warm.py` calls it |
| scheduled functions (`modal.Cron`) | runs a function at set times | `deploy/warm.py` turns the models on 15 minutes before the window opens and off when it closes |
| `scaledown_window` | how long an idle copy stays up (2 minutes for us; Modal allows up to 20) | `deploy/llm.py`, `deploy/tts.py` |
| `buffer_containers` | spare idle copies for sudden bursts | not used: we run one GPU on purpose |
| Volumes | weights and compile caches survive between starts | both models |
| GPU memory snapshot (alpha) | saves the warmed-up process | `LLM_SNAPSHOT=1`, opt in, see above |

Nudge's calls are scheduled, so we know when they start. `uv run modal deploy deploy/warm.py` deploys the schedule. It reads the calling window from `engine/policy.yaml`. By hand: `uv run modal run deploy/warm.py::on` and `::off`.

The cost is the GPU price per hour times the hours of the window each day. Look up the price on modal.com/pricing. Redeploying `deploy/llm.py` or `deploy/tts.py` puts the settings back to zero, and the next scheduled start turns them on again.

## One command

`python -m scripts.launch` does the whole setup: secrets, voice, brain, API, wakes the models, checks them. Add `--snapshot` for the GPU snapshot brain (needs `LLM_PROXY_KEY` and `LLM_PROXY_SECRET` in `.env`) and `--seed` for the fake borrowers. The notebook has a cell that runs it.

`modal run evals/bench.py::cold --label snapshot --runs 5 --note snapshot` measures the cold start by itself: wait for sleep, wake, time, repeat. Results go into `docs/performance.md`.
