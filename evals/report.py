"""Turn the saved results in evals/results/ into the markdown tables of docs/performance.md.

uv run python -m evals.report            # print
uv run python -m evals.report --write    # write docs/performance.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.stats import wilson

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent.parent / "docs" / "performance.md"
TTFT_GOAL_MS = 700  # the voice guideline used by aiewf-eval for the model's first token


def f(x: float | None, digits: int = 0) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def load(kind: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"*-{kind}.json"))]


def llm_table(runs: list[dict]) -> str:
    rows = ["| run | tools | calls at once | TTFT p50 | p95 | p99 | tokens/s | requests/s | cache hit | $ per 1k requests |",
            "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for run in runs:
        m = run["meta"]
        for lv in run["levels"]:
            t = lv["ttft_ms"]
            hit = "-" if lv["cache_hit_rate"] is None else f"{lv['cache_hit_rate'] * 100:.0f}%"
            rows.append(
                f"| {m['label']} ({m['gpu']}) | {m.get('tools', '')} | {lv['concurrency']} | {f(t['p50'])} ms | {f(t['p95'])} ms | "
                f"{f(t['p99'])} ms | {f(lv['tokens_per_s']['p50'])} | {f(lv['requests_per_s'], 1)} | {hit} | "
                f"${f(lv['usd_per_1k_requests'], 3)} |"
            )
    return "\n".join(rows)


def tts_table(runs: list[dict]) -> str:
    rows = ["| run | at once | latency p50 | p95 | p99 | short / medium / long (p50) | real-time factor p50 | $ per 1k sentences |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for run in runs:
        m = run["meta"]
        for lv in run["levels"]:
            t, b = lv["latency_ms"], lv["by_length_p50_ms"]
            rows.append(
                f"| {m['label']} ({m['gpu']}) | {lv['concurrency']} | {f(t['p50'])} ms | {f(t['p95'])} ms | {f(t['p99'])} ms | "
                f"{f(b['short'])} / {f(b['medium'])} / {f(b['long'])} ms | {f(lv['real_time_factor']['p50'], 2)} | "
                f"${f(lv['usd_per_1k_requests'], 3)} |"
            )
    return "\n".join(rows)


def agent_table(runs: list[dict]) -> str:
    rows = ["| run | scenario | runs | rules held | task done (95% interval) | model reply p50 / p95 / p99 |", "|---|---|---|---|---|---|"]
    for run in runs:
        for name, s in run["scenarios"].items():
            n = s["runs"]
            lo, hi = wilson(s["task_passed"], n)
            lat = s["model_ms"]
            rows.append(
                f"| {run['meta']['label']} | {name} | {n} | {s['rules_held']}/{n} | "
                f"{s['task_passed']}/{n} ({lo * 100:.0f}-{hi * 100:.0f}%) | "
                f"{f(lat['p50'])} / {f(lat['p95'])} / {f(lat['p99'])} ms |"
            )
    return "\n".join(rows)


def cold_lines(runs: list[dict], what: str) -> str:
    return "\n".join(
        f"- {what} `{r['meta']['label']}`{' (' + r['meta']['note'] + ')' if r['meta'].get('note') else ''}: "
        f"first request answered after {r['first_request_s']:.0f} s ({r['meta']['date']})"
        for r in runs
    )


def main(write: bool = False) -> str:
    llm, tts, agent = load("llm"), load("tts"), load("agent")
    parts = [
        "# Performance",
        "",
        "Measured on Modal by `evals/bench.py` and `evals/run.py`. Nothing here is typed by hand: `uv run python -m evals.report --write`.",
        "",
    ]
    if llm:
        parts += [
            "## Brain (PhoneLLM Mini on vLLM)",
            "",
            f"Goal: TTFT p50 under {TTFT_GOAL_MS} ms. One call is the 5 requests the agent really makes.",
            "",
            llm_table(llm),
            "",
            cold_lines(llm, "Cold start"),
            "",
        ]
    if tts:
        parts += [
            "## Voice (Chatterbox Turbo)",
            "",
            "No streaming, so the whole sentence is timed. Real-time factor below 1 means it speaks faster than a person.",
            "",
            tts_table(tts),
            "",
            cold_lines(tts, "Cold start"),
            "",
        ]
    if agent:
        parts += [
            "## Agent evals",
            "",
            "Rules are the things that must never break. Task is whether the call reached the right outcome.",
            "",
            agent_table(agent),
            "",
        ]
    if not (llm or tts or agent):
        parts += ["No results yet. Run the benchmarks first (see the top of `evals/bench.py`)."]
    text = "\n".join(parts) + "\n"
    if write:
        OUT.write_text(text)
    return text


if __name__ == "__main__":
    print(main(write="--write" in sys.argv))
