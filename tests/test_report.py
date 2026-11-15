from evals import report

LEVEL = {
    "concurrency": 4,
    "ttft_ms": {"p50": 120.0, "p95": 240.0, "p99": 300.0},
    "tokens_per_s": {"p50": 55.0},
    "requests_per_s": 12.5,
    "cache_hit_rate": 0.9,
    "usd_per_1k_requests": 0.1234,
}
META = {
    "label": "rtx6000-bf16",
    "gpu": "RTX-PRO-6000",
    "tools": "per_state",
    "date": "2026-11-20 10:00 UTC",
}


def test_the_brain_table_shows_percentiles_and_cost():
    table = report.llm_table([{"meta": META, "levels": [LEVEL]}])
    assert "| rtx6000-bf16 (RTX-PRO-6000) | per_state | 4 | 120 ms | 240 ms | 300 ms | 55 | 12.5 | 90% | $0.123 |" in table


def test_a_missing_number_is_a_dash_not_a_crash():
    assert report.f(None) == "-" and report.f(3.14159, 2) == "3.14"


def test_the_agent_table_shows_an_interval():
    scenario = {"runs": 10, "rules_held": 10, "task_passed": 9, "model_ms": {"p50": 800, "p95": 1500, "p99": 2000}}
    table = report.agent_table([{"meta": {"label": "baseline"}, "scenarios": {"happy": scenario}}])
    assert "10/10" in table and "9/10 (60-98%)" in table


def test_with_no_results_it_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    assert "No results yet" in report.main()
