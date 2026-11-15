import pytest
from hypothesis import given
from hypothesis import strategies as st

from evals.stats import pct, summarize, wilson


def test_percentiles_of_a_known_list():
    xs = list(range(1, 101))
    assert pct(xs, 50) == pytest.approx(50.5)
    assert pct(xs, 95) == pytest.approx(95.05)
    assert pct(xs, 99) == pytest.approx(99.01)
    assert pct(xs, 0) == 1 and pct(xs, 100) == 100


def test_one_value_is_every_percentile():
    assert pct([7.0], 50) == pct([7.0], 99) == 7.0


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        pct([], 50)


@given(st.lists(st.floats(0, 1e6), min_size=1, max_size=50))
def test_percentiles_never_go_backwards(xs):
    s = summarize(xs)
    assert min(xs) <= s["p50"] <= s["p95"] <= s["p99"] <= s["max"]


def test_summary_of_nothing_has_no_numbers():
    assert summarize([])["p50"] is None and summarize([])["n"] == 0


def test_a_pass_rate_interval_is_wide_when_samples_are_few():
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and 0.72 < lo < 0.73


def test_the_interval_narrows_with_more_runs():
    lo5, hi5 = wilson(40, 50)
    lo1, hi1 = wilson(400, 500)
    assert (hi1 - lo1) < (hi5 - lo5)
    assert lo5 < 0.8 < hi5


def test_no_runs_means_we_know_nothing():
    assert wilson(0, 0) == (0.0, 1.0)
