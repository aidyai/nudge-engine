"""A company tells us who to call: one borrower and one loan, typed into the form."""

import calendar
import re
import uuid
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

PHONE = re.compile(r"^\+?[0-9][0-9 ()-]{6,17}$")


class WhoToCall(BaseModel):
    borrower_id: str | None = Field(default=None, max_length=40)  # none means the built-in demo borrower


class NewBorrower(BaseModel):
    company: str = Field(min_length=2, max_length=60)
    full_name: str = Field(min_length=2, max_length=80)
    phone: str
    birth_day: int = Field(ge=1, le=31)
    birth_month: int = Field(ge=1, le=12)
    amount_due_ngn: Decimal = Field(gt=0, le=100_000_000, decimal_places=2)
    due_date: date

    @field_validator("company", "full_name")
    @classmethod
    def tidy(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("too short")
        return value

    @field_validator("phone")
    @classmethod
    def phone_looks_right(cls, value: str) -> str:
        if not PHONE.match(value.strip()):
            raise ValueError("a phone number, like +234 803 000 0000")
        return value.strip()

    def day_month(self) -> str:
        """'03-11', the way the database keeps it. Raises ValueError for a date that does not exist, like 31 February."""
        date(2024, self.birth_month, self.birth_day)  # 2024 has a 29 February
        return f"{self.birth_day:02d}-{self.birth_month:02d}"


def create_borrower(db, form: NewBorrower) -> str:
    """Saves the borrower and the loan. Returns the borrower id. Nobody is rung: this is only a record."""
    key = uuid.uuid4().hex[:8]
    borrower_id, loan_id = f"C-{key}", f"CL-{key}"
    db.execute(
        text(
            "insert into lender.borrowers (id, full_name, phone, birth_day_month, archetype, company) "
            "values (:id, :name, :phone, :dm, 'custom', :company)"
        ),
        {"id": borrower_id, "name": form.full_name, "phone": form.phone, "dm": form.day_month(), "company": form.company},
    )
    db.execute(
        text("insert into lender.loans (id, borrower_id, amount_due_ngn, due_date) values (:l, :b, :amount, :due)"),
        {"l": loan_id, "b": borrower_id, "amount": form.amount_due_ngn, "due": form.due_date},
    )
    return borrower_id


def pick_borrower(db, borrower_id: str | None, archetype: str):
    """The borrower a call is for: the one the company just entered, or else the built-in demo borrower. None if there is no such person."""
    if borrower_id:
        query, args = "where id = :id and not opted_out", {"id": borrower_id}
    else:
        query, args = "where archetype = :a and not opted_out order by id limit 1", {"a": archetype}
    return db.execute(text(f"select id, full_name, birth_day_month, company from lender.borrowers {query}"), args).first()


def describe(row) -> dict:
    """What the page shows about the person being called."""
    day, month = row.birth_day_month.split("-")
    return {"name": row.full_name, "company": row.company, "birthday": f"{int(day)} {calendar.month_name[int(month)]}"}
