# Performance

Measured on Modal by `evals/bench.py` and `evals/run.py`. Nothing here is typed by hand: `uv run python -m evals.report --write`.

## Brain (PhoneLLM Mini on vLLM)

Goal: TTFT p50 under 700 ms. One call is the 5 requests the agent really makes.

| run | tools | calls at once | TTFT p50 | p95 | p99 | tokens/s | requests/s | cache hit | USD per 1k requests |
|---|---|---|---|---|---|---|---|---|---|
| rtx6000-bf16 (RTX-PRO-6000) | per_state | 1 | 83 ms | 92 ms | 95 ms | 185 | 4.2 | 0% | 0.201 |
| rtx6000-bf16 (RTX-PRO-6000) | per_state | 4 | 165 ms | 193 ms | 194 ms | 140 | 10.8 | 0% | 0.078 |
| rtx6000-bf16 (RTX-PRO-6000) | per_state | 8 | 238 ms | 271 ms | 321 ms | 129 | 16.8 | 0% | 0.050 |
| rtx6000-bf16 (RTX-PRO-6000) | per_state | 16 | 308 ms | 466 ms | 514 ms | 71 | 22.1 | 0% | 0.038 |

- Cold start `rtx6000-bf16`: already awake (0 s), so this is not a cold start (2026-11-17 11:47 UTC)

## Voice (Chatterbox Turbo)

No streaming, so the whole sentence is timed. Real-time factor below 1 means it speaks faster than a person.

| run | at once | latency p50 | p95 | p99 | short / medium / long (p50) | real-time factor p50 | USD per 1k sentences |
|---|---|---|---|---|---|---|---|
| l4 (L4) | 1 | 1347 ms | 2551 ms | 2569 ms | 759 / 1495 / 2537 ms | 0.50 | 0.322 |
| l4 (L4) | 2 | 3485 ms | 7303 ms | 7499 ms | 1504 / 3656 / 6644 ms | 1.13 | 0.398 |
| l4 (L4) | 4 | 8342 ms | 20193 ms | 22209 ms | 4433 / 11529 / 14036 ms | 3.75 | 0.574 |

- Cold start `l4`: already awake (2 s), so this is not a cold start (2026-11-17 11:49 UTC)

## Agent evals

Rules are the things that must never break. Task is whether the call reached the right outcome.

| run | scenario | runs | rules held | task done (95% interval) | model reply p50 / p95 / p99 |
|---|---|---|---|---|---|
| baseline | happy | 5 | 5/5 | 0/5 (0-43%) | 354 / 463 / 494 ms |
| baseline | wrong_birthday | 5 | 5/5 | 0/5 (0-43%) | 305 / 426 / 428 ms |
| baseline | stop_calling | 5 | 5/5 | 5/5 (57-100%) | 297 / 336 / 338 ms |
| baseline | wants_a_person | 5 | 5/5 | 5/5 (57-100%) | 282 / 289 / 291 ms |
| baseline | lost_my_job | 5 | 5/5 | 0/5 (0-43%) | 338 / 431 / 433 ms |
| baseline | says_already_paid | 5 | 5/5 | 5/5 (57-100%) | 326 / 377 / 378 ms |
| baseline | trick | 5 | 5/5 | 5/5 (57-100%) | 383 / 440 / 444 ms |
| baseline | too_little | 5 | 5/5 | 5/5 (57-100%) | 346 / 522 / 525 ms |

## Simulated phone call (voice in, voice out)

A pretend borrower speaks, Deepgram hears it, the agent and brain answer, and the voice speaks. Silence is what the borrower waits through: ears + brain + voice. The voice does not stream, so it is the whole sentence.

| run | outcome | turns | silence p50 | p95 | p99 | ears / brain / voice (mean) |
|---|---|---|---|---|---|---|
| call | None | 4 | 4611 ms | 6818 ms | 7034 ms | 713 / 1236 / 2957 ms |

