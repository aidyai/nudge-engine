"""The hard rules. Plain code, no model, no network. Numbers live in policy.yaml."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel

KOBO = Decimal("0.01")

# Tools that reveal or act on loan details: blocked until identity is verified (rule 4).
VERIFIED_ONLY_TOOLS = frozenset({"get_loan", "get_offers", "propose_option", "confirm_option"})
# Tools that stay available even when identity check is locked or failed.
ALWAYS_ALLOWED_TOOLS = frozenset({"hand_off", "opt_out", "end_call"})


# config
class WindowConfig(BaseModel):
    start: time
    end: time
    timezone: str


class AttemptConfig(BaseModel):
    per_day: int
    per_week: int
    min_gap_minutes: int


class OfferConfig(BaseModel):
    part_payment_min_fraction: Decimal
    extension_max_days: int
    promise_max_days_ahead: int


class CallConfig(BaseModel):
    max_seconds: int


class PolicyConfig(BaseModel):
    calling_window: WindowConfig
    attempts: AttemptConfig
    offers: OfferConfig
    call: CallConfig


def load_policy(path: str | Path | None = None) -> PolicyConfig:
    path = path or Path(__file__).with_name("policy.yaml")  # next to this file, so it works anywhere
    return PolicyConfig.model_validate(yaml.safe_load(Path(path).read_text()))


# decisions
@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule: str  # e.g. 'P1_CALLING_WINDOW'
    reason: str  # e.g. 'outside_window'
    detail: dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "rule": self.rule, "reason": self.reason, "detail": self.detail}


def _allow(rule: str, reason: str, **detail: Any) -> Decision:
    return Decision(True, rule, reason, detail)


def _deny(rule: str, reason: str, **detail: Any) -> Decision:
    return Decision(False, rule, reason, detail)


def first_denial(decisions: Sequence[Decision]) -> Decision | None:
    return next((d for d in decisions if not d.allowed), None)


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return dt


# rules 1 to 3: before dialing
def check_calling_window(now: datetime, cfg: PolicyConfig) -> Decision:
    """Rule 1: only call inside the local calling window."""
    local = _aware(now).astimezone(ZoneInfo(cfg.calling_window.timezone))
    inside = cfg.calling_window.start <= local.time().replace(tzinfo=None) < cfg.calling_window.end
    detail = {"local_time": local.strftime("%H:%M")}
    return _allow("P1_CALLING_WINDOW", "inside_window", **detail) if inside else _deny("P1_CALLING_WINDOW", "outside_window", **detail)


def check_consent(calls_allowed: bool, opted_out: bool) -> Decision:
    """Rule 2: never call without consent, and never after an opt-out."""
    if opted_out:
        return _deny("P2_CONSENT", "opted_out")
    if not calls_allowed:
        return _deny("P2_CONSENT", "no_consent")
    return _allow("P2_CONSENT", "consent_ok")


def check_attempt_limits(previous_attempts: Sequence[datetime], now: datetime, cfg: PolicyConfig) -> Decision:
    """Rule 3: attempts per day and per week, and a minimum gap between attempts."""
    now = _aware(now)
    tz = ZoneInfo(cfg.calling_window.timezone)
    attempts = [_aware(a) for a in previous_attempts]
    today = now.astimezone(tz).date()
    today_count = sum(a.astimezone(tz).date() == today for a in attempts)
    week_count = sum(now - timedelta(days=7) < a <= now for a in attempts)
    detail = {"today": today_count, "last_7_days": week_count}
    if today_count >= cfg.attempts.per_day:
        return _deny("P3_ATTEMPT_LIMITS", "daily_limit", **detail)
    if week_count >= cfg.attempts.per_week:
        return _deny("P3_ATTEMPT_LIMITS", "weekly_limit", **detail)
    recent = [a for a in attempts if a <= now]
    if recent and now - max(recent) < timedelta(minutes=cfg.attempts.min_gap_minutes):
        return _deny("P3_ATTEMPT_LIMITS", "too_soon", **detail)
    return _allow("P3_ATTEMPT_LIMITS", "within_limits", **detail)


def can_dial(
    *,
    calls_allowed: bool,
    opted_out: bool,
    previous_attempts: Sequence[datetime],
    now: datetime,
    cfg: PolicyConfig,
) -> list[Decision]:
    """All three before-dialing rules. Log every decision; dial only if none is a denial."""
    return [
        check_consent(calls_allowed, opted_out),
        check_calling_window(now, cfg),
        check_attempt_limits(previous_attempts, now, cfg),
    ]


# rule 4: disclosure (tool gating)
def check_tool_precondition(tool: str, *, verified: bool, locked: bool) -> Decision:
    """Rule 4: no loan detail before identity is verified; a locked identity can only exit."""
    if tool in ALWAYS_ALLOWED_TOOLS:
        return _allow("P4_DISCLOSURE", "always_allowed", tool=tool)
    if locked:
        return _deny("P4_DISCLOSURE", "identity_locked", tool=tool)
    if tool in VERIFIED_ONLY_TOOLS and not verified:
        return _deny("P4_DISCLOSURE", "not_verified", tool=tool)
    return _allow("P4_DISCLOSURE", "allowed", tool=tool)


# rule 6: offers
class PartPaymentLimits(BaseModel):
    min_ngn: Decimal
    max_ngn: Decimal


class Offers(BaseModel):
    """What the borrower may be offered, computed by code. The model only ever reads this."""

    amount_due_ngn: Decimal
    due_date: date
    today: date
    pay_now_ngn: Decimal
    part_payment: PartPaymentLimits | None
    extension_max_days: int
    promise_latest: date
    callback_window_start: time
    callback_window_end: time


def compute_offers(amount_due: Decimal, due_date: date, today: date, cfg: PolicyConfig) -> Offers:
    amount_due = amount_due.quantize(KOBO)
    minimum = (amount_due * cfg.offers.part_payment_min_fraction).quantize(KOBO, rounding=ROUND_CEILING)
    maximum = amount_due - KOBO  # paying the full amount is pay_now, not a part payment
    return Offers(
        amount_due_ngn=amount_due,
        due_date=due_date,
        today=today,
        pay_now_ngn=amount_due,
        part_payment=PartPaymentLimits(min_ngn=minimum, max_ngn=maximum) if minimum <= maximum else None,
        extension_max_days=cfg.offers.extension_max_days,
        promise_latest=today + timedelta(days=cfg.offers.promise_max_days_ahead),
        callback_window_start=cfg.calling_window.start,
        callback_window_end=cfg.calling_window.end,
    )


def validate_option(option: str, amount: Decimal | None, on: date | None, offers: Offers) -> Decision:
    """Rule 6: is this exact option inside the limits? Used by propose_option."""
    rule = "P6_OFFERS"
    if option == "pay_now":
        if amount is not None and amount != offers.pay_now_ngn:
            return _deny(rule, "amount_mismatch", expected=str(offers.pay_now_ngn))
        return _allow(rule, "pay_now_ok")

    if option == "part_payment":
        limits = offers.part_payment
        if limits is None:
            return _deny(rule, "part_payment_unavailable")
        if amount is None:
            return _deny(rule, "amount_required")
        if amount < limits.min_ngn:
            return _deny(rule, "below_minimum", minimum=str(limits.min_ngn))
        if amount > limits.max_ngn:
            return _deny(rule, "above_maximum_use_pay_now", maximum=str(limits.max_ngn))
        return _allow(rule, "part_payment_ok")

    if option == "extension":
        if on is None:
            return _deny(rule, "date_required")
        base = max(offers.due_date, offers.today)
        if on <= base:
            return _deny(rule, "extension_not_later", base=base.isoformat())
        if on > base + timedelta(days=offers.extension_max_days):
            return _deny(rule, "extension_too_long", max_days=offers.extension_max_days)
        return _allow(rule, "extension_ok")

    if option == "promise":
        if on is None:
            return _deny(rule, "date_required")
        if on < offers.today:
            return _deny(rule, "date_in_past")
        if on > offers.promise_latest:
            return _deny(rule, "too_far_ahead", latest=offers.promise_latest.isoformat())
        return _allow(rule, "promise_ok")

    if option == "callback":
        return _allow(rule, "callback_ok")

    return _deny(rule, "unknown_option", option=option)


# rule 9: call length
def check_call_length(elapsed_seconds: float, cfg: PolicyConfig) -> Decision:
    """Rule 9: after the limit, close gracefully."""
    if elapsed_seconds >= cfg.call.max_seconds:
        return _deny("P9_CALL_LENGTH", "time_up", elapsed=round(elapsed_seconds))
    return _allow("P9_CALL_LENGTH", "within_limit", elapsed=round(elapsed_seconds))
