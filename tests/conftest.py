import os
from datetime import date

import pytest
from hypothesis import settings
from sqlalchemy import create_engine

# No per-example deadline (CI machines are slow); plenty of examples for the property tests.
settings.register_profile("nudge", deadline=None, max_examples=200)
settings.load_profile("nudge")

TODAY = date(2026, 11, 9)  # a Monday. The seeded borrowers are due relative to this date.


@pytest.fixture(scope="session")
def pg_engine(tmp_path_factory):
    """One real Postgres for the whole test run, with the migrations applied."""
    import pgserver

    server = pgserver.get_server(tmp_path_factory.mktemp("pg_agent"))  # pyright: ignore[reportPrivateImportUsage]
    url = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["DATABASE_URL"] = url
    from engine.seed import migrate

    migrate()
    engine = create_engine(url)
    yield engine
    engine.dispose()
    server.cleanup()


@pytest.fixture
def db(pg_engine):
    """A fresh seeded database for every test, as an autocommit connection (like the real agent uses)."""
    from engine.seed import seed

    with pg_engine.begin() as conn:
        seed(conn, TODAY)
    with pg_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        yield conn
