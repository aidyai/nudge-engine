"""Baseline: lender and nudge schemas, seven tables.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

UP = [
    "CREATE SCHEMA lender",
    "CREATE SCHEMA nudge",
    """CREATE TABLE lender.borrowers (
        id text PRIMARY KEY,
        full_name text NOT NULL,
        phone text NOT NULL,
        birth_day_month text NOT NULL,
        calls_allowed boolean NOT NULL DEFAULT true,
        opted_out boolean NOT NULL DEFAULT false,
        timezone text NOT NULL DEFAULT 'Africa/Lagos',
        archetype text NOT NULL
    )""",
    """CREATE TABLE lender.loans (
        id text PRIMARY KEY,
        borrower_id text NOT NULL REFERENCES lender.borrowers(id),
        amount_due_ngn numeric(12,2) NOT NULL CHECK (amount_due_ngn > 0),
        due_date date NOT NULL,
        status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paid', 'disputed'))
    )""",
    """CREATE TABLE lender.payments (
        id bigserial PRIMARY KEY,
        loan_id text NOT NULL REFERENCES lender.loans(id),
        amount_ngn numeric(12,2) NOT NULL CHECK (amount_ngn > 0),
        paid_on date NOT NULL,
        channel text NOT NULL DEFAULT 'transfer'
    )""",
    """CREATE TABLE nudge.calls (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        borrower_id text NOT NULL REFERENCES lender.borrowers(id),
        scheduled_for timestamptz NOT NULL,
        attempt_no int NOT NULL DEFAULT 1,
        status text NOT NULL DEFAULT 'scheduled'
            CHECK (status IN ('scheduled', 'claimed', 'in_progress', 'done', 'failed')),
        outcome text,
        claimed_by text,
        claimed_at timestamptz,
        started_at timestamptz,
        ended_at timestamptz,
        prompt_version text,
        llm_backend text
    )""",
    "CREATE INDEX calls_due_idx ON nudge.calls (scheduled_for) WHERE status = 'scheduled'",
    """CREATE TABLE nudge.events (
        id bigserial PRIMARY KEY,
        call_id uuid REFERENCES nudge.calls(id),
        ts timestamptz NOT NULL DEFAULT now(),
        type text NOT NULL,
        payload jsonb NOT NULL DEFAULT '{}',
        latency_ms int
    )""",
    "CREATE INDEX events_call_idx ON nudge.events (call_id, id)",
    """CREATE TABLE nudge.promises (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        call_id uuid NOT NULL REFERENCES nudge.calls(id),
        loan_id text NOT NULL REFERENCES lender.loans(id),
        option text NOT NULL
            CHECK (option IN ('pay_now', 'part_payment', 'extension', 'promise', 'callback')),
        amount_ngn numeric(12,2),
        promise_date date,
        status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'kept', 'partial', 'broken')),
        idempotency_key text NOT NULL UNIQUE,
        created_at timestamptz NOT NULL DEFAULT now()
    )""",
    """CREATE TABLE nudge.handoffs (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        call_id uuid NOT NULL REFERENCES nudge.calls(id),
        reason text NOT NULL CHECK (reason IN (
            'hardship', 'dispute', 'abuse', 'distress', 'requested_human',
            'repeated_misunderstanding', 'tool_failure', 'policy_conflict')),
        summary text NOT NULL,
        status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'reviewed', 'closed')),
        created_at timestamptz NOT NULL DEFAULT now()
    )""",
]


def upgrade() -> None:
    for statement in UP:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP SCHEMA nudge CASCADE")
    op.execute("DROP SCHEMA lender CASCADE")
