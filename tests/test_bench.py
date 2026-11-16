"""The benchmark's own logic, checked against fake data so a Modal run is not the first time it executes."""

import json

import pytest

from evals import bench


def test_a_call_is_five_requests_with_only_the_tools_of_each_state():
    reqs = bench.build_requests("per_state")
    assert len(reqs) == 5
    names = [[t["function"]["name"] for t in r["tools"]] for r in reqs]
    assert names[0] == ["verify_identity", "opt_out", "hand_off", "end_call"]
    assert names[-1] == ["confirm_option", "opt_out", "hand_off"]
    assert all(r["stream"] and r["model"] == "phonellm-mini-alpha-2" for r in reqs)


def test_the_fixed_mode_sends_every_tool_every_time():
    sizes = {len(r["tools"]) for r in bench.build_requests("fixed")}
    assert sizes == {8}


def test_each_request_extends_the_one_before_so_the_cache_can_help():
    reqs = bench.build_requests("per_state")
    for a, b in zip(reqs, reqs[1:], strict=False):
        assert b["messages"][: len(a["messages"])] == a["messages"]


def test_streamed_replies_are_read():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[],"usage":{"completion_tokens":2}}',
        "data: [DONE]",
    ]
    assert bench.parse_sse(lines) == {"text": "Hello", "tools": [], "tokens": 2}
    call = ['data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"get_loan"}}]}}]}']
    assert bench.parse_sse(call)["tools"] == ["get_loan"]


def test_cache_hit_rate_comes_from_counter_differences():
    before = {"vllm:prefix_cache_hits_total": 10, "vllm:prefix_cache_queries_total": 100}
    after = {"vllm:prefix_cache_hits_total": 70, "vllm:prefix_cache_queries_total": 200}
    assert bench.hit_rate(before, after) == pytest.approx(0.6)
    assert bench.hit_rate({}, {}) is None


def test_a_level_is_summarised_with_cost():
    rows = [{"ok": True, "ttft_ms": 100 + i, "total_ms": 400, "tokens": 20, "tps": 60.0} for i in range(10)]
    rows.append({"ok": False, "status": 500})
    lv = bench.summarize_level(4, rows, wall_s=36.0, price_per_hour=3.6, cache=0.9)
    assert lv["requests"] == 11 and lv["errors"] == 1
    assert lv["ttft_ms"]["p50"] == pytest.approx(104.5)
    assert lv["requests_per_s"] == pytest.approx(10 / 36)
    assert lv["usd_per_1k_requests"] == pytest.approx(3.6 * (36 / 3600) / 10 * 1000)


def test_a_voice_level_reports_real_time_factor():
    rows = [{"ok": True, "bucket": "short", "ms": 500.0, "audio_s": 1.0, "rtf": 0.5}] * 4
    lv = bench.summarize_voice_level(1, rows, wall_s=2.0, price_per_hour=0.8)
    assert lv["real_time_factor"]["p50"] == 0.5 and lv["by_length_p50_ms"]["short"] == 500.0
    assert lv["by_length_p50_ms"]["long"] is None


def test_results_are_saved_under_the_label(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RESULTS", tmp_path)
    meta = bench.meta_for("llm", "x", "L4", 0.8)
    path = bench.save("llm", "x", meta, {"levels": []})
    assert path.name == "x-llm.json" and json.loads(path.read_text())["meta"]["gpu"] == "L4"


def test_levels_are_parsed():
    assert bench.parse_levels("1, 4,8") == [1, 4, 8]


def test_cold_run_waits_then_wakes_each_time():
    from evals import bench

    log: list[str] = []
    times = iter([500.0, 90.0, 80.0])
    out = bench.cold_run(lambda: (log.append("wake"), next(times))[1], lambda s: log.append(f"sleep {s}"), 3, 200, say=lambda _: None)
    assert out == [500.0, 90.0, 80.0]
    assert log == ["sleep 200", "wake"] * 3
