"""Speed and cost benchmarks for the two models. Runs on Modal, next to the servers, so the numbers
are the servers' own and not the distance from your laptop.

The method is borrowed from kwindla/aiewf-eval: time to first token as percentiles (P50/P95/P99), and
a voice target of about 700 ms for the model's first token. We keep only that part.

Brain (PhoneLLM). One "call" is the 5 requests the agent really makes, with the real prompt and tools:
    uv run --env-file .env modal run evals/bench.py::llm --label rtx6000-bf16 --gpu RTX-PRO-6000 --price 3.03
    --tools fixed          every request carries all 8 tools (the "before" in the tools experiment)
    --levels 1,4,8,16      how many calls run at the same time
    --note snapshot        what was different about this deploy (shown next to the cold start in the report)

Voice (Chatterbox). It has no streaming, so we time the whole sentence and divide by the audio length:
    uv run --env-file .env modal run evals/bench.py::tts --label l4 --gpu L4 --price 0.80

Start from a stopped server to measure a cold start (wait 3 minutes after the last request).
--price is Modal's price for that GPU per hour, from modal.com/pricing. Results go to evals/results/.
Then:  uv run python -m evals.report --write
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("httpx", "pydantic", "pydantic-settings", "pyyaml", "sqlalchemy", "tzdata", "psycopg[binary]")
    .add_local_dir("engine", "/root/engine")
    .add_local_dir("evals", "/root/evals")
)
app = modal.App("nudge-bench")
RESULTS = Path(__file__).parent / "results"
VLLM_VERSION = "0.30.0"
SAMPLE_RATE = 24_000

# ------------------------------------------------------------------ the brain: what one call asks of the model


def build_requests(tools_mode: str = "per_state", model: str = "phonellm-mini-alpha-2") -> list[dict]:
    """The 5 requests of a typical call, using the agent's real prompt, states and tools."""
    from engine.agent import STATES, TOOLS
    from engine.prompts import agent_prompt

    assert tools_mode in ("per_state", "fixed")

    def tools_for(state: str) -> list[dict]:
        return list(TOOLS.values()) if tools_mode == "fixed" else [TOOLS[n] for n in STATES[state]]

    def call(i: str, name: str, args: dict) -> dict:
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}  # fmt: skip

    def result(i: str, body: dict) -> dict:
        return {"role": "tool", "tool_call_id": i, "content": json.dumps(body)}

    system = {"role": "system", "content": agent_prompt("Ngozi Okafor", "Sample Lender", date(2026, 11, 9))}
    hello = {"role": "user", "content": "[The call just connected. Greet the person now.]"}
    greeting = {
        "role": "assistant",
        "content": "Hello, this is an automated assistant calling for Sample Lender. Am I speaking with Ngozi Okafor?",
    }
    yes = {"role": "user", "content": "Yes, speaking. My birthday is fourteen March."}
    loan = {"amount": "twenty-five thousand naira", "due": "Friday, the thirteenth of November"}
    offer = {"status": "ok", "readback": "To confirm: you will pay twenty-five thousand naira on Friday, the thirteenth of November."}
    told = {
        "role": "assistant",
        "content": "Thank you. You owe twenty-five thousand naira, due Friday the thirteenth. What would you like to do?",
    }
    pay = {"role": "user", "content": "I can pay on Friday."}
    readback = {"role": "assistant", "content": offer["readback"] + " Is that right?"}
    ok = {"role": "user", "content": "Yes, that is right."}

    steps = [
        ("verify", [system, hello]),
        ("verify", [system, hello, greeting, yes]),
        ("explain", [system, hello, greeting, yes, call("c1", "verify_identity", {"answer": "14 March"}),
                     result("c1", {"status": "verified"}), call("c2", "get_loan", {}), result("c2", loan)]),  # fmt: skip
        ("arrange", [system, hello, greeting, yes, call("c1", "verify_identity", {"answer": "14 March"}),
                     result("c1", {"status": "verified"}), call("c2", "get_loan", {}), result("c2", loan), told, pay,
                     call("c3", "propose_option", {"option": "promise", "date": "Friday"}), result("c3", offer)]),  # fmt: skip
        ("confirm", [system, hello, greeting, yes, call("c1", "verify_identity", {"answer": "14 March"}),
                     result("c1", {"status": "verified"}), call("c2", "get_loan", {}), result("c2", loan), told, pay,
                     call("c3", "propose_option", {"option": "promise", "date": "Friday"}), result("c3", offer),
                     readback, ok]),  # fmt: skip
    ]  # fmt: skip
    return [
        {"model": model, "messages": m, "tools": tools_for(state), "temperature": 0, "max_tokens": 300,
         "stream": True, "stream_options": {"include_usage": True}}
        for state, m in steps
    ]  # fmt: skip


def parse_sse(lines: list[str]) -> dict:
    """What we need from a streamed reply: the text, the tool names, and the token count if the server sent it."""
    text, tools, tokens = "", [], None
    for line in lines:
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        if chunk.get("usage"):
            tokens = chunk["usage"].get("completion_tokens")
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                if name := (tc.get("function") or {}).get("name"):
                    tools.append(name)
    return {"text": text, "tools": tools, "tokens": tokens}


async def one_request(client, url: str, key: str, body: dict) -> dict:
    """Send one streamed request. ttft = time to the first content or tool-call chunk."""
    started = time.perf_counter()
    first, lines = None, []
    async with client.stream("POST", f"{url}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}) as r:
        if r.status_code != 200:
            await r.aread()
            return {"ok": False, "status": r.status_code}
        async for line in r.aiter_lines():
            lines.append(line)
            if first is None and line.startswith("data:") and re.search(r'"(content|tool_calls)":\s*(\[|")', line):
                first = time.perf_counter()
    total = time.perf_counter() - started
    got = parse_sse(lines)
    ttft = (first or time.perf_counter()) - started
    tokens = got["tokens"] or 0
    decode = tokens / (total - ttft) if tokens and total > ttft else None
    return {"ok": True, "ttft_ms": ttft * 1000, "total_ms": total * 1000, "tokens": tokens, "tps": decode}


async def scrape(client, url: str, key: str) -> dict:
    """vLLM's prefix-cache counters from /metrics. {} if the server does not show them."""
    try:
        r = await client.get(url.removesuffix("/v1") + "/metrics", headers={"Authorization": f"Bearer {key}"})
    except Exception:
        return {}
    out = {}
    for name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
        m = re.search(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)", r.text, re.M)
        if m:
            out[name] = float(m.group(1))
    return out


def hit_rate(before: dict, after: dict) -> float | None:
    hits = after.get("vllm:prefix_cache_hits_total", 0) - before.get("vllm:prefix_cache_hits_total", 0)
    queries = after.get("vllm:prefix_cache_queries_total", 0) - before.get("vllm:prefix_cache_queries_total", 0)
    return hits / queries if queries > 0 else None


def summarize_level(concurrency: int, rows: list[dict], wall_s: float, price_per_hour: float, cache: float | None) -> dict:
    from evals.stats import summarize

    ok = [r for r in rows if r["ok"]]
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "ttft_ms": summarize([r["ttft_ms"] for r in ok]),
        "total_ms": summarize([r["total_ms"] for r in ok]),
        "tokens_per_s": summarize([r["tps"] for r in ok if r["tps"]]),
        "requests_per_s": len(ok) / wall_s if wall_s else None,
        "cache_hit_rate": cache,
        "usd_per_1k_requests": price_per_hour * (wall_s / 3600) / len(ok) * 1000 if ok else None,
    }


async def run_sessions(url: str, key: str, requests: list[dict], concurrency: int, sessions: int) -> list[dict]:
    import httpx

    gate = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=120) as client:

        async def session() -> None:
            async with gate:
                for body in requests:  # one caller's requests go one after the other, like a real call
                    rows.append(await one_request(client, url, key, body))

        await asyncio.gather(*(session() for _ in range(sessions)))
    return rows


async def bench_llm_async(url: str, key: str, model: str, tools_mode: str, levels: list[int], price: float) -> dict:
    import httpx

    requests = build_requests(tools_mode, model)
    started = time.perf_counter()  # the first request also wakes a stopped server: that is the cold start
    async with httpx.AsyncClient(timeout=1200) as client:
        deadline = time.time() + 1200
        while True:
            r = await one_request(client, url, key, requests[0])
            if r["ok"] or r["status"] not in (502, 503, 504) or time.time() > deadline:
                break
            await asyncio.sleep(5)
        first_request_s = time.perf_counter() - started
        if not r["ok"]:
            raise RuntimeError(f"model server answered {r['status']}")
        await asyncio.gather(*(one_request(client, url, key, b) for b in requests))  # warm up the cache
        out = []
        for c in levels:
            before = await scrape(client, url, key)
            t0 = time.perf_counter()
            rows = await run_sessions(url, key, requests, c, sessions=max(6, 2 * c))
            wall = time.perf_counter() - t0
            out.append(summarize_level(c, rows, wall, price, hit_rate(before, await scrape(client, url, key))))
    return {"first_request_s": first_request_s, "levels": out}


@app.function(image=image, secrets=[modal.Secret.from_name("nudge-env")], timeout=60 * 60)
def bench_llm(tools_mode: str, levels: list[int], price: float) -> dict:
    from engine.config import Settings

    cfg = Settings()
    return asyncio.run(bench_llm_async(cfg.llm_base_url, cfg.models_api_key, cfg.llm_model, tools_mode, levels, price))


# ------------------------------------------------------------------ the voice: whole sentences, no streaming

SENTENCES = {
    "short": ["Thank you.", "One moment, please.", "Goodbye."],
    "medium": [
        "Hello, this is an automated assistant calling for Sample Lender.",
        "You owe twenty-five thousand naira, due on Friday.",
        "I understand, thank you for telling me.",
    ],
    "long": [
        "To confirm: you will pay twenty-five thousand naira on Friday, the thirteenth of November. Is that right?",
        "I am sorry to hear that. I will ask a colleague to call you back, and they will be glad to help you.",
    ],
}


async def speak_once(client, url: str, key: str, text: str) -> dict:
    started = time.perf_counter()
    r = await client.post(url, json={"text": text}, headers={"Authorization": f"Bearer {key}"}, follow_redirects=True)
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code}
    ms = (time.perf_counter() - started) * 1000
    audio_s = len(r.content) / (2 * SAMPLE_RATE)
    return {"ok": True, "ms": ms, "audio_s": audio_s, "rtf": ms / 1000 / audio_s if audio_s else None}


async def run_sentences(url: str, key: str, concurrency: int, rounds: int) -> tuple[list[dict], float]:
    import httpx

    gate = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    sentences = [(bucket, s) for bucket, group in SENTENCES.items() for s in group] * rounds
    async with httpx.AsyncClient(timeout=300) as client:

        async def one(bucket: str, s: str) -> None:
            async with gate:
                rows.append({"bucket": bucket, **await speak_once(client, url, key, s)})

        t0 = time.perf_counter()
        await asyncio.gather(*(one(b, s) for b, s in sentences))
    return rows, time.perf_counter() - t0


def summarize_voice_level(concurrency: int, rows: list[dict], wall_s: float, price_per_hour: float) -> dict:
    from evals.stats import summarize

    ok = [r for r in rows if r["ok"]]
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "latency_ms": summarize([r["ms"] for r in ok]),
        "by_length_p50_ms": {b: summarize([r["ms"] for r in ok if r["bucket"] == b])["p50"] for b in SENTENCES},
        "real_time_factor": summarize([r["rtf"] for r in ok if r["rtf"]]),  # below 1 = faster than speech
        "requests_per_s": len(ok) / wall_s if wall_s else None,
        "usd_per_1k_requests": price_per_hour * (wall_s / 3600) / len(ok) * 1000 if ok else None,
    }


async def bench_tts_async(url: str, key: str, levels: list[int], price: float) -> dict:
    import httpx

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=1200) as client:
        deadline = time.time() + 1200
        while True:
            r = await speak_once(client, url, key, "Hello.")
            if r["ok"] or r["status"] not in (502, 503, 504, 429) or time.time() > deadline:
                break
            await asyncio.sleep(5)
    if not r["ok"]:
        raise RuntimeError(f"voice server answered {r['status']}")
    first_request_s = time.perf_counter() - started
    out = []
    for c in levels:
        rows, wall = await run_sentences(url, key, c, rounds=2)
        out.append(summarize_voice_level(c, rows, wall, price))
    return {"first_request_s": first_request_s, "levels": out}


@app.function(image=image, secrets=[modal.Secret.from_name("nudge-env")], timeout=60 * 60)
def bench_tts(levels: list[int], price: float) -> dict:
    from engine.config import Settings

    cfg = Settings()
    return asyncio.run(bench_tts_async(cfg.tts_url, cfg.models_api_key, levels, price))


# ------------------------------------------------------------------ saving


def meta_for(kind: str, label: str, gpu: str, price: float, **extra: object) -> dict:
    return {"kind": kind, "label": label, "gpu": gpu, "usd_per_gpu_hour": price,
            "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), **extra}  # fmt: skip


def save(kind: str, label: str, meta: dict, data: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{label}-{kind}.json"
    path.write_text(json.dumps({"meta": meta, **data}, indent=2))
    return path


def parse_levels(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


@app.local_entrypoint()
def llm(label: str, gpu: str, price: float, levels: str = "1,4,8,16", tools: str = "per_state", note: str = ""):
    data = bench_llm.remote(tools, parse_levels(levels), price)
    meta = meta_for("llm", label, gpu, price, model="phonellm-mini-alpha-2", vllm=VLLM_VERSION, tools=tools, note=note)
    print("saved", save("llm", label, meta, data))


@app.local_entrypoint()
def tts(label: str, gpu: str, price: float, levels: str = "1,2,4"):
    data = bench_tts.remote(parse_levels(levels), price)
    meta = meta_for("tts", label, gpu, price, model="chatterbox-turbo")
    print("saved", save("tts", label, meta, data))
