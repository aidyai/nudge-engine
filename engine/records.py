"""Read calls back out of the database for the website: the list, and one call in full."""

from sqlalchemy import text

from engine.lender import one

LIST = """
select c.id, c.borrower_id, b.full_name as name, c.status, c.outcome, c.started_at, c.llm_backend, l.amount_due_ngn,
       (select percentile_cont(0.5) within group (order by e.latency_ms)
        from nudge.events e where e.call_id = c.id and e.type = 'model') as reply_ms
from nudge.calls c
join lender.borrowers b on b.id = c.borrower_id
left join lateral (
    select amount_due_ngn from lender.loans where borrower_id = b.id and status <> 'paid' order by due_date limit 1
) l on true
where c.started_at is not null
order by c.started_at desc
limit :n
"""

STATS = """
select
    (select count(*) from nudge.calls where started_at is not null) as calls,
    (select count(*) from nudge.promises) as promises,
    (select coalesce(sum(amount_ngn), 0) from nudge.promises) as promised_ngn,
    (select count(*) from nudge.handoffs) as handoffs,
    (select count(*) from nudge.handoffs where status = 'open') as open_handoffs,
    (select percentile_cont(0.5) within group (order by latency_ms) from nudge.events where type = 'model') as reply_ms
"""


def list_calls(db, limit=50):
    rows = db.execute(text(LIST), {"n": limit}).mappings()
    return {"stats": dict(db.execute(text(STATS)).mappings().one()), "calls": [dict(r) for r in rows]}


def mask(phone):
    return f"{phone[:4]} ••• {phone[-4:]}" if len(phone) > 8 else "•••"


def call_detail(db, call_id):
    call = one(
        db,
        """select c.id, c.borrower_id, b.full_name as name, b.phone, c.status, c.outcome, c.started_at, c.ended_at,
                  c.prompt_version, c.llm_backend,
                  (select amount_due_ngn from lender.loans
                   where borrower_id = b.id and status <> 'paid' order by due_date limit 1) as amount_due_ngn
           from nudge.calls c join lender.borrowers b on b.id = c.borrower_id where c.id = cast(:id as uuid)""",
        id=call_id,
    )
    if call is None:
        return None
    call["phone"] = mask(call["phone"])
    events = db.execute(
        text("select type, payload, latency_ms from nudge.events where call_id = cast(:id as uuid) order by id"), {"id": call_id}
    ).all()
    transcript, tools, checks, model_ms, link_sent = [], [], [], 0, False
    for kind, payload, ms in events:
        if kind == "model":
            model_ms += ms or 0  # the time the model took for the reply that comes next
        elif kind == "user":
            transcript.append({"who": "caller", "text": payload["text"]})
        elif kind == "agent":
            transcript.append({"who": "agent", "text": payload["text"], "reply_ms": model_ms or None})
            model_ms = 0
        elif kind == "tool":
            tools.append({"name": payload["name"], "result": payload["result"]})
        elif kind == "policy":
            checks.append({"rule": payload["rule"], "reason": payload["reason"], "allowed": payload["allowed"]})
        elif kind == "output_check":
            checks.append({"rule": "OUTPUT_CHECK", "reason": payload["problem"], "allowed": False})
        elif kind == "payment_link_sent":
            link_sent = True
    promise = one(
        db,
        "select option, amount_ngn, promise_date from nudge.promises where call_id = cast(:id as uuid) order by created_at desc limit 1",
        id=call_id,
    )
    handoff = one(db, "select reason, summary from nudge.handoffs where call_id = cast(:id as uuid) limit 1", id=call_id)
    return {
        **call,
        "transcript": transcript,
        "tools": tools,
        "checks": checks,
        "promise": promise,
        "handoff": handoff,
        "payment_link_sent": link_sent,
    }
