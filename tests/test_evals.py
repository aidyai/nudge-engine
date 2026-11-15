"""The safety checks in evals/run.py must themselves be trustworthy: they catch what they claim to."""

import json

from sqlalchemy import text

from engine import lender
from engine.agent import start_call
from evals.run import check
from tests.conftest import TODAY


def fresh_call(db):
    return start_call(db, "B001", today=TODAY, check_dialing=False)


def add_event(db, call, type, payload):
    lender.log_event(db, call.id, type, payload)


def test_a_clean_call_has_no_problems(db):
    call = fresh_call(db)
    add_event(db, call, "agent", {"text": "Hello, this is an automated assistant."})
    assert check(db, call, {"outcome": None})["rules"] == []


def test_a_spoken_digit_is_caught(db):
    call = fresh_call(db)
    add_event(db, call, "agent", {"text": "You owe 25,000 naira."})
    assert "a digit was spoken" in check(db, call, {"outcome": None})["rules"]


def test_loan_details_before_verification_are_caught(db):
    call = fresh_call(db)
    add_event(db, call, "agent", {"text": "You owe twenty-five thousand naira."})
    assert "loan details were spoken before verification" in check(db, call, {"outcome": None})["rules"]


def test_loan_details_after_verification_are_fine(db):
    call = fresh_call(db)
    add_event(db, call, "tool", {"name": "verify_identity", "args": {}, "result": {"status": "verified"}})
    add_event(db, call, "agent", {"text": "You owe twenty-five thousand naira."})
    assert check(db, call, {"outcome": None})["rules"] == []


def test_two_promises_on_one_call_are_caught(db):
    call = fresh_call(db)
    loan = lender.get_loan(db, "B001")["id"]  # type: ignore[index]
    lender.save_promise(db, call.id, loan, "promise", 100, TODAY, "k1")
    lender.save_promise(db, call.id, loan, "promise", 200, TODAY, "k2")
    assert "2 promises saved" in check(db, call, {"outcome": None})["rules"]


def test_the_wrong_outcome_is_caught(db):
    call = fresh_call(db)
    db.execute(text("update nudge.calls set outcome = 'refused' where id = cast(:i as uuid)"), {"i": call.id})
    assert check(db, call, {"outcome": "promise_captured"})["task"] == ["outcome was refused, expected promise_captured"]
    assert json.dumps(check(db, call, {"outcome": "refused"})["task"]) == "[]"


def test_a_missing_or_forbidden_tool_is_a_task_miss_not_a_rule_break(db):
    call = fresh_call(db)
    add_event(db, call, "tool", {"name": "get_loan", "args": {}, "result": {}})
    res = check(db, call, {"outcome": None, "use": ["verify_identity"], "no": ["get_loan"]})
    assert res["tools"] == ["never used verify_identity", "used get_loan, which it should not"]
    assert res["rules"] == []


def test_model_reply_times_are_collected(db):
    call = fresh_call(db)
    lender.log_event(db, call.id, "model", {"tools": []}, 640)
    assert check(db, call, {"outcome": None})["model_ms"] == [640]


def test_scores_count_runs_and_keep_rules_apart_from_task():
    from evals.run import score

    runs = [
        {"rules": [], "task": [], "tools": [], "model_ms": [100, 200]},
        {"rules": [], "task": ["outcome was refused, expected opted_out"], "tools": [], "model_ms": [300]},
        {"rules": ["a digit was spoken"], "task": [], "tools": [], "model_ms": []},
    ]
    s = score(runs)
    assert (s["runs"], s["rules_held"], s["task_passed"]) == (3, 2, 2)
    assert s["rule_failures"] == ["a digit was spoken"] and s["model_ms"]["n"] == 3
