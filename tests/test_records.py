from datetime import date

from engine import lender, records
from tests.test_api import GREETING, KEY, make_client, say


def a_finished_call(db):
    call_id = lender.new_call(db, "B001", "v2")
    log = lambda kind, payload, ms=None: lender.log_event(db, call_id, kind, payload, ms)  # noqa: E731
    log("model", {"tools": []}, 300)
    log("agent", {"text": "Hello."})
    log("user", {"text": "Yes, speaking."})
    log("model", {"tools": ["get_loan"]}, 200)
    log("model", {"tools": []}, 100)
    log("tool", {"name": "get_loan", "args": {}, "result": {"amount_due": "twenty-five thousand naira"}}, 5)
    log("policy", {"allowed": True, "rule": "P6_OFFERS", "reason": "promise_ok", "detail": {}})
    log("output_check", {"problem": "a digit was spoken"})
    log("payment_link_sent", {"amount": 100})
    log("agent", {"text": "You owe twenty-five thousand naira."})
    loan = lender.get_loan(db, "B001")
    lender.save_promise(db, call_id, loan["id"], "promise", 25000, date(2026, 11, 13), "k1")  # type: ignore[index]
    lender.save_handoff(db, call_id, "hardship", "Lost a job.")
    lender.end_call(db, call_id, "promise_captured")
    return call_id


def test_a_call_is_read_back_the_way_it_was_written(db):
    detail = records.call_detail(db, a_finished_call(db))
    assert detail is not None
    assert detail["name"] and detail["outcome"] == "promise_captured" and detail["prompt_version"] == "v2"
    assert [t["who"] for t in detail["transcript"]] == ["agent", "caller", "agent"]
    assert [t.get("reply_ms") for t in detail["transcript"]] == [300, None, 300]  # model time added up per reply
    assert detail["tools"][0]["name"] == "get_loan"
    assert [c["allowed"] for c in detail["checks"]] == [True, False]  # the rule that passed, the reply that broke one
    assert detail["promise"]["option"] == "promise" and detail["handoff"]["reason"] == "hardship"
    assert detail["payment_link_sent"] is True


def test_the_phone_number_is_never_sent_in_full(db):
    phone = lender.must(db, "select phone from lender.borrowers where id = 'B001'")["phone"]
    shown = records.call_detail(db, a_finished_call(db))["phone"]  # type: ignore[index]
    assert phone not in shown and shown.endswith(phone[-4:])


def test_the_list_shows_newest_first_with_totals(db):
    first, second = a_finished_call(db), a_finished_call(db)
    out = records.list_calls(db)
    assert [str(c["id"]) for c in out["calls"]] == [second, first]
    assert out["stats"]["calls"] == 2 and out["stats"]["promises"] == 1 and out["stats"]["handoffs"] == 2  # same key: one promise


def test_a_callback_that_was_only_scheduled_is_not_a_call(db):
    lender.schedule_callback(db, "B001", __import__("datetime").datetime(2026, 11, 10, 9, 0, tzinfo=__import__("datetime").UTC))
    assert records.list_calls(db)["calls"] == []


def test_the_routes_need_the_key_and_refuse_unknown_calls(pg_engine, db):
    client, *_ = make_client(pg_engine, db, say(GREETING))
    assert client.get("/calls").status_code == 401
    assert client.get("/calls/0b0f9f0e-6d6e-4a3b-9d27-3f5f2b1d9a10").status_code == 401
    assert client.get("/calls/0b0f9f0e-6d6e-4a3b-9d27-3f5f2b1d9a10", headers=KEY).status_code == 404
    assert client.get("/calls/not-a-uuid", headers=KEY).status_code == 404


def test_a_demo_call_shows_up_on_the_dashboard_routes(pg_engine, db):
    client, *_ = make_client(pg_engine, db, say(GREETING))
    call_id = client.post("/calls", headers=KEY).json()["call_id"]
    listed = client.get("/calls", headers=KEY).json()
    assert listed["calls"][0]["id"] == call_id and listed["stats"]["calls"] == 1
    detail = client.get(f"/calls/{call_id}", headers=KEY).json()
    assert detail["transcript"][0]["text"] == GREETING and detail["status"] == "in_progress"
