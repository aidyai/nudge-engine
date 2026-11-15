from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from engine.policy import (
    can_dial,
    check_attempt_limits,
    check_call_length,
    check_calling_window,
    check_consent,
    check_tool_precondition,
    compute_offers,
    first_denial,
    load_policy,
    validate_option,
)

LAGOS = ZoneInfo("Africa/Lagos")
CFG = load_policy()
TODAY = date(2026, 11, 9)
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=LAGOS)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=LAGOS)


# ------------------------------------------------------------ the numbers file


def test_policy_yaml_loads_with_real_times():
    # Guards the YAML trap where an unquoted 08:00 becomes the number 480.
    assert CFG.calling_window.start == time(8, 0)
    assert CFG.calling_window.end == time(18, 0)
    assert CFG.attempts.per_day == 2
    assert CFG.offers.part_payment_min_fraction == Decimal("0.5")


# ------------------------------------------------------------ rule 1: calling window


@pytest.mark.parametrize(
    ("hour", "minute", "allowed"),
    [(7, 59, False), (8, 0, True), (12, 0, True), (17, 59, True), (18, 0, False), (23, 30, False)],
)
def test_calling_window_edges(hour, minute, allowed):
    assert check_calling_window(at(9, hour, minute), CFG).allowed is allowed


def test_calling_window_converts_from_utc():
    # 07:00 UTC is 08:00 in Lagos.
    assert check_calling_window(datetime(2026, 11, 9, 7, 0, tzinfo=UTC), CFG).allowed
    assert not check_calling_window(datetime(2026, 11, 9, 6, 59, tzinfo=UTC), CFG).allowed


def test_calling_window_refuses_naive_datetimes():
    with pytest.raises(ValueError):
        check_calling_window(datetime(2026, 11, 9, 12, 0), CFG)


@given(st.integers(min_value=0, max_value=1439))
def test_calling_window_matches_the_clock_for_every_minute(minute_of_day):
    when = at(9, minute_of_day // 60, minute_of_day % 60)
    assert check_calling_window(when, CFG).allowed == (480 <= minute_of_day < 1080)


# ------------------------------------------------------------ rule 2: consent


@pytest.mark.parametrize(
    ("calls_allowed", "opted_out", "allowed", "reason"),
    [
        (True, False, True, "consent_ok"),
        (False, False, False, "no_consent"),
        (True, True, False, "opted_out"),
        (False, True, False, "opted_out"),
    ],
)
def test_consent(calls_allowed, opted_out, allowed, reason):
    decision = check_consent(calls_allowed, opted_out)
    assert (decision.allowed, decision.reason) == (allowed, reason)


# ------------------------------------------------------------ rule 3: attempt limits


def test_no_previous_attempts_is_fine():
    assert check_attempt_limits([], NOW, CFG).allowed


def test_minimum_gap_between_attempts():
    assert check_attempt_limits([at(9, 11, 0)], NOW, CFG).reason == "too_soon"
    assert check_attempt_limits([at(9, 10, 0)], NOW, CFG).allowed  # exactly two hours ago


def test_daily_limit():
    decision = check_attempt_limits([at(9, 8), at(9, 10)], NOW, CFG)
    assert (decision.allowed, decision.reason) == (False, "daily_limit")


def test_weekly_limit():
    week = [at(day, 10) for day in (8, 7, 6, 5, 4)]
    assert check_attempt_limits(week, NOW, CFG).reason == "weekly_limit"


def test_an_attempt_exactly_seven_days_ago_no_longer_counts():
    week = [at(2, 12)] + [at(day, 10) for day in (8, 7, 6, 5)]
    assert check_attempt_limits(week, NOW, CFG).allowed


def test_attempt_limits_refuse_naive_datetimes():
    with pytest.raises(ValueError):
        check_attempt_limits([datetime(2026, 11, 9, 8, 0)], NOW, CFG)


def test_can_dial_returns_every_decision_in_order():
    decisions = can_dial(calls_allowed=True, opted_out=False, previous_attempts=[], now=NOW, cfg=CFG)
    assert [d.rule for d in decisions] == ["P2_CONSENT", "P1_CALLING_WINDOW", "P3_ATTEMPT_LIMITS"]
    assert first_denial(decisions) is None


def test_can_dial_blocks_an_opted_out_borrower_even_inside_the_window():
    decisions = can_dial(calls_allowed=True, opted_out=True, previous_attempts=[], now=NOW, cfg=CFG)
    denial = first_denial(decisions)
    assert denial is not None and denial.rule == "P2_CONSENT"


# ------------------------------------------------------------ rule 4: disclosure


@pytest.mark.parametrize("tool", ["get_loan", "get_offers", "propose_option", "confirm_option"])
def test_loan_tools_need_verification(tool):
    assert check_tool_precondition(tool, verified=False, locked=False).reason == "not_verified"
    assert check_tool_precondition(tool, verified=True, locked=False).allowed


def test_verify_identity_is_open_until_locked():
    assert check_tool_precondition("verify_identity", verified=False, locked=False).allowed
    assert check_tool_precondition("verify_identity", verified=False, locked=True).reason == "identity_locked"


@pytest.mark.parametrize("tool", ["hand_off", "opt_out", "end_call"])
def test_a_locked_call_can_always_exit(tool):
    assert check_tool_precondition(tool, verified=False, locked=True).allowed


# ------------------------------------------------------------ rule 6: offers


def offers_for(amount: str, due: date = TODAY + timedelta(days=1)):
    return compute_offers(Decimal(amount), due, TODAY, CFG)


def test_offer_limits():
    offers = offers_for("25000")
    assert offers.pay_now_ngn == Decimal("25000.00")
    assert offers.part_payment is not None
    assert (offers.part_payment.min_ngn, offers.part_payment.max_ngn) == (
        Decimal("12500.00"),
        Decimal("24999.99"),
    )
    assert offers.extension_max_days == 3
    assert offers.promise_latest == TODAY + timedelta(days=7)


def test_no_part_payment_when_the_amount_is_too_small_to_split():
    assert offers_for("0.01").part_payment is None


@pytest.mark.parametrize(
    ("amount", "allowed", "reason"),
    [
        ("12500.00", True, "part_payment_ok"),
        ("12499.99", False, "below_minimum"),
        ("24999.99", True, "part_payment_ok"),
        ("25000.00", False, "above_maximum_use_pay_now"),
    ],
)
def test_part_payment_edges(amount, allowed, reason):
    decision = validate_option("part_payment", Decimal(amount), None, offers_for("25000"))
    assert (decision.allowed, decision.reason) == (allowed, reason)


def test_part_payment_needs_an_amount():
    assert validate_option("part_payment", None, None, offers_for("25000")).reason == "amount_required"


def test_pay_now_must_match_the_amount_due():
    offers = offers_for("25000")
    assert validate_option("pay_now", None, None, offers).allowed
    assert validate_option("pay_now", Decimal("25000"), None, offers).allowed
    assert validate_option("pay_now", Decimal("20000"), None, offers).reason == "amount_mismatch"


def test_extension_counts_from_the_due_date():
    offers = offers_for("25000", due=TODAY + timedelta(days=1))  # due 10 Nov
    assert validate_option("extension", None, date(2026, 11, 13), offers).allowed  # +3 days
    assert validate_option("extension", None, date(2026, 11, 14), offers).reason == "extension_too_long"
    assert validate_option("extension", None, date(2026, 11, 10), offers).reason == "extension_not_later"


def test_extension_for_an_overdue_loan_counts_from_today():
    offers = offers_for("25000", due=TODAY - timedelta(days=2))
    assert validate_option("extension", None, TODAY + timedelta(days=3), offers).allowed
    assert validate_option("extension", None, TODAY + timedelta(days=4), offers).reason == "extension_too_long"


def test_promise_dates():
    offers = offers_for("25000")
    assert validate_option("promise", None, TODAY, offers).allowed
    assert validate_option("promise", None, TODAY + timedelta(days=7), offers).allowed
    assert validate_option("promise", None, TODAY + timedelta(days=8), offers).reason == "too_far_ahead"
    assert validate_option("promise", None, TODAY - timedelta(days=1), offers).reason == "date_in_past"


@pytest.mark.parametrize("option", ["extension", "promise"])
def test_dated_options_need_a_date(option):
    assert validate_option(option, None, None, offers_for("25000")).reason == "date_required"


def test_callback_is_always_allowed_and_inventions_are_not():
    offers = offers_for("25000")
    assert validate_option("callback", None, None, offers).allowed
    assert validate_option("waiver", Decimal("0"), None, offers).reason == "unknown_option"


@given(
    st.decimals(min_value=Decimal("1"), max_value=Decimal("10000000"), places=2),
    st.data(),
)
def test_part_payment_is_allowed_exactly_inside_the_limits(amount_due, data):
    offers = compute_offers(amount_due, TODAY, TODAY, CFG)
    limits = offers.part_payment
    assume(limits is not None)
    assert limits is not None
    assert limits.min_ngn >= amount_due / 2  # never below the minimum fraction
    assert limits.min_ngn < amount_due / 2 + Decimal("0.01")
    proposed = data.draw(st.decimals(min_value=Decimal("0.01"), max_value=amount_due + 100, places=2))
    decision = validate_option("part_payment", proposed, None, offers)
    assert decision.allowed == (limits.min_ngn <= proposed <= limits.max_ngn)


# ------------------------------------------------------------ rule 9: call length


def test_call_length():
    assert check_call_length(299, CFG).allowed
    assert check_call_length(300, CFG).reason == "time_up"


# ------------------------------------------------------------ decisions are loggable


def test_decisions_become_events():
    event = check_consent(False, False).to_event()
    assert event == {"allowed": False, "rule": "P2_CONSENT", "reason": "no_consent", "detail": {}}
