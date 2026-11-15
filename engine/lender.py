"""Everything the agent reads from or writes to the database. Plain SQL, no magic.

`db` is a SQLAlchemy connection in autocommit mode, so every write is visible at once.
"""

import json

from sqlalchemy import text


def one(db, sql, **params):
    row = db.execute(text(sql), params).mappings().first()
    return dict(row) if row else None


def must(db, sql, **params):
    row = one(db, sql, **params)
    if row is None:
        raise LookupError("the database returned no row")
    return row


def get_borrower(db, borrower_id):
    return one(
        db,
        "select id, full_name, birth_day_month, calls_allowed, opted_out from lender.borrowers where id = :id",
        id=borrower_id,
    )


def get_loan(db, borrower_id):
    return one(
        db,
        "select id, amount_due_ngn, due_date, status from lender.loans where borrower_id = :b and status <> 'paid' order by due_date limit 1",
        b=borrower_id,
    )


def recent_payments(db, loan_id):
    rows = db.execute(
        text("select amount_ngn, paid_on from lender.payments where loan_id = :l order by paid_on desc limit 3"),
        {"l": loan_id},
    ).mappings()
    return [dict(r) for r in rows]


def new_call(db, borrower_id, prompt_version):
    row = must(
        db,
        "insert into nudge.calls (borrower_id, scheduled_for, status, started_at, prompt_version, llm_backend) values (:b, now(), 'in_progress', now(), :v, 'modal') returning id",
        b=borrower_id,
        v=prompt_version,
    )
    return str(row["id"])


def end_call(db, call_id, outcome):
    db.execute(
        text("update nudge.calls set status = 'done', outcome = :o, ended_at = now() where id = :id"),
        {"o": outcome, "id": call_id},
    )


def log_event(db, call_id, type, payload, latency_ms=None):
    db.execute(
        text("insert into nudge.events (call_id, type, payload, latency_ms) values (:c, :t, cast(:p as jsonb), :l)"),
        {"c": call_id, "t": type, "p": json.dumps(payload, default=str), "l": latency_ms},
    )


def save_promise(db, call_id, loan_id, option, amount, on, key):
    """True if saved, False if this exact promise was already saved (the idempotency key)."""
    row = one(
        db,
        "insert into nudge.promises (call_id, loan_id, option, amount_ngn, promise_date, idempotency_key) values (:c, :l, :o, :a, :d, :k) on conflict (idempotency_key) do nothing returning id",
        c=call_id,
        l=loan_id,
        o=option,
        a=amount,
        d=on,
        k=key,
    )
    return row is not None


def schedule_callback(db, borrower_id, when):
    db.execute(
        text("insert into nudge.calls (borrower_id, scheduled_for) values (:b, :w)"),
        {"b": borrower_id, "w": when},
    )


def set_opted_out(db, borrower_id):
    db.execute(text("update lender.borrowers set opted_out = true where id = :id"), {"id": borrower_id})


def save_handoff(db, call_id, reason, summary):
    return str(
        must(
            db,
            "insert into nudge.handoffs (call_id, reason, summary) values (:c, :r, :s) returning id",
            c=call_id,
            r=reason,
            s=summary,
        )["id"]
    )
