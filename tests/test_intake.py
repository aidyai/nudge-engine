import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from engine.agent import start_call
from engine.api import create_app
from engine.config import Settings
from engine.intake import NewBorrower
from tests.test_agent import Script, say

KEY = {"X-API-Key": "secret"}
FORM = {
    "company": "Acme  Microfinance",
    "full_name": "Ada   Obi",
    "phone": "+234 803 000 0000",
    "birth_day": 4,
    "birth_month": 9,
    "amount_due_ngn": "45000.50",
    "due_date": "2026-12-01",
}


def client(pg_engine, replies=()):
    return TestClient(create_app(Settings(_env_file=None, api_key="secret"), pg_engine, Script(*replies)))  # pyright: ignore[reportCallIssue]


def test_the_form_tidies_names_and_keeps_the_day_and_month_together():
    form = NewBorrower(**FORM)
    assert (form.company, form.full_name, form.day_month()) == ("Acme Microfinance", "Ada Obi", "04-09")


@pytest.mark.parametrize(
    "change",
    [
        {"phone": "hello"},
        {"amount_due_ngn": "0"},
        {"amount_due_ngn": "-5"},
        {"birth_month": 13},
        {"company": " "},
        {"due_date": "tomorrow"},
    ],
)
def test_a_bad_form_is_refused(change):
    with pytest.raises(ValidationError):
        NewBorrower(**{**FORM, **change})


def test_a_day_that_does_not_exist_is_refused():
    with pytest.raises(ValueError):
        NewBorrower(**{**FORM, "birth_day": 31, "birth_month": 2}).day_month()


def test_adding_a_borrower_needs_the_key(pg_engine, db):
    assert client(pg_engine).post("/borrowers", json=FORM).status_code == 401


def test_a_bad_form_gets_a_422_not_a_crash(pg_engine, db):
    c = client(pg_engine)
    assert c.post("/borrowers", headers=KEY, json={**FORM, "phone": "x"}).status_code == 422
    assert c.post("/borrowers", headers=KEY, json={**FORM, "birth_day": 31, "birth_month": 2}).status_code == 422


def test_a_company_can_add_a_borrower_and_the_agent_calls_for_that_company(pg_engine, db):
    c = client(pg_engine, [say("Hello, this is Idara from Acme Microfinance.")])
    added = c.post("/borrowers", headers=KEY, json=FORM).json()
    assert added["borrower"] == {"name": "Ada Obi", "company": "Acme Microfinance", "birthday": "4 September"}
    started = c.post("/calls", headers=KEY, json={"borrower_id": added["borrower_id"]}).json()
    assert started["borrower"]["name"] == "Ada Obi"
    loan = db.exec_driver_sql(
        "select amount_due_ngn, due_date::text from lender.loans where borrower_id = %s", (added["borrower_id"],)
    ).one()
    assert (float(loan[0]), loan[1]) == (45000.5, "2026-12-01")
    call = start_call(db, added["borrower_id"], check_dialing=False)
    assert "Acme Microfinance" in call.messages[0]["content"] and "Nudge Finance" not in call.messages[0]["content"]


def test_a_call_for_an_unknown_borrower_is_a_clear_error(pg_engine, db):
    assert client(pg_engine).post("/calls", headers=KEY, json={"borrower_id": "nobody"}).status_code == 503


def test_a_live_ticket_can_be_for_the_borrower_the_company_added(pg_engine, db):
    c = client(pg_engine)
    added = c.post("/borrowers", headers=KEY, json=FORM).json()
    ticket = c.post("/realtime/tickets", headers=KEY, json={"borrower_id": added["borrower_id"]}).json()
    assert ticket["borrower"]["company"] == "Acme Microfinance"
