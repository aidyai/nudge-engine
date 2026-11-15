from datetime import date

import pytest
from sqlalchemy import create_engine, text

from engine.seed import ARCHETYPES, build_fixtures, migrate, seed

TODAY = date(2026, 11, 7)


def test_fixtures_are_deterministic():
    assert build_fixtures(TODAY) == build_fixtures(TODAY)


def test_every_archetype_is_present_with_its_count():
    borrowers, loans, _ = build_fixtures(TODAY)
    assert len(borrowers) == len(loans) == sum(a.count for a in ARCHETYPES)
    for arch in ARCHETYPES:
        assert sum(b["archetype"] == arch.name for b in borrowers) == arch.count


def test_names_and_phones_are_unique_and_fake():
    borrowers, _, _ = build_fixtures(TODAY)
    assert len({b["full_name"] for b in borrowers}) == len(borrowers)
    assert len({b["phone"] for b in borrowers}) == len(borrowers)
    assert all(b["phone"].startswith("+234800") and len(b["phone"]) == 14 for b in borrowers)


def test_due_dates_follow_the_archetype():
    borrowers, loans, _ = build_fixtures(TODAY)
    by_id = {b["id"]: b["archetype"] for b in borrowers}
    offsets = {a.name: a.due_in_days for a in ARCHETYPES}
    for loan in loans:
        assert (loan["due_date"] - TODAY).days == offsets[by_id[loan["borrower_id"]]]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    import os

    import pgserver

    server = pgserver.get_server(tmp_path_factory.mktemp("pg"))  # pyright: ignore[reportPrivateImportUsage]
    url = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["DATABASE_URL"] = url
    eng = create_engine(url)
    yield eng
    eng.dispose()
    server.cleanup()


def test_migrations_apply_roll_back_and_reapply(engine):
    from alembic import command
    from alembic.config import Config

    from engine.seed import ALEMBIC_INI

    migrate()
    command.downgrade(Config(str(ALEMBIC_INI)), "base")
    migrate()
    with engine.connect() as conn:
        tables = conn.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema IN ('lender', 'nudge')")).scalar()
    assert tables == 7


def test_seed_is_repeatable(engine):
    for _ in range(2):  # reseeding resets cleanly
        with engine.begin() as conn:
            seed(conn, TODAY)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM lender.borrowers")).scalar() == 45
        assert conn.execute(text("SELECT count(*) FROM lender.loans")).scalar() == 45
        assert conn.execute(text("SELECT count(*) FROM lender.payments")).scalar() == 2
        assert (
            conn.execute(
                text(
                    "SELECT amount_due_ngn FROM lender.loans l "
                    "JOIN lender.borrowers b ON b.id = l.borrower_id "
                    "WHERE b.archetype = 'number_confusion'"
                )
            ).scalar()
            == 10250.50
        )


def test_promise_idempotency_key_is_unique(engine):
    from sqlalchemy.exc import IntegrityError

    with engine.begin() as conn:
        call_id = conn.execute(text("INSERT INTO nudge.calls (borrower_id, scheduled_for) VALUES ('B001', now()) RETURNING id")).scalar()
    ins = text(
        "INSERT INTO nudge.promises (call_id, loan_id, option, amount_ngn, idempotency_key) VALUES (:c, 'L001', 'pay_now', 12500, 'k1')"
    )
    with engine.begin() as conn:
        conn.execute(ins, {"c": call_id})
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(ins, {"c": call_id})


def test_handoff_reason_is_constrained(engine):
    from sqlalchemy.exc import IntegrityError

    with engine.begin() as conn:
        call_id = conn.execute(text("SELECT id FROM nudge.calls LIMIT 1")).scalar()
        conn.execute(
            text("INSERT INTO nudge.handoffs (call_id, reason, summary) VALUES (:c, 'hardship', 'lost job')"),
            {"c": call_id},
        )
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text("INSERT INTO nudge.handoffs (call_id, reason, summary) VALUES (:c, 'because', 'x')"),
            {"c": call_id},
        )
