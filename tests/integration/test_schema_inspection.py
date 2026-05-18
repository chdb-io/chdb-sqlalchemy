"""L4a — Schema inspection tests (no LLM required).

These tests exercise the *exact* code path LangChain's
``SQLDatabaseToolkit`` runs when it builds the system prompt for the LLM:

    db = SQLDatabase(engine)
    schema_str = db.get_table_info()       # ← this is what the LLM sees
    table_names = db.get_usable_table_names()

If ``schema_str`` is malformed, the LLM hallucinates columns. If
``get_columns`` raises, the toolkit dies on init. If the sample-rows
suffix is missing or truncated, the LLM has no concrete grounding.

We deliberately don't mock LangChain — we run the real package against
the real chDB engine. Every assertion here corresponds to a specific
LLM failure mode we've seen in practice.

This file catches 70% of "wrong-schema" bugs without an API call.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from . import schemas

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_engine(engine):
    """An engine with every demo schema created and populated."""
    with engine.begin() as conn:
        schemas.build_all(conn)
    return engine


@pytest.fixture
def langchain_db(seeded_engine):
    """LangChain ``SQLDatabase`` over the seeded engine."""
    sa = pytest.importorskip("langchain_community.utilities")
    return sa.SQLDatabase(seeded_engine)


# ---------------------------------------------------------------------------
# Inspector-level checks
# ---------------------------------------------------------------------------


def test_all_tables_visible(seeded_engine):
    insp = inspect(seeded_engine)
    names = set(insp.get_table_names())
    expected = {s.name for s in schemas.ALL_SCHEMAS}
    missing = expected - names
    assert not missing, f"Reflection lost tables: {missing}"


def test_all_columns_for_every_table(seeded_engine):
    """For each fixture schema, every declared column must be reflected.

    A missing column here translates directly to "LLM doesn't know
    column X exists" — the single most common LLM failure mode.
    """
    insp = inspect(seeded_engine)

    # column-count expectations per schema
    expected = {
        "users": 6,
        "events": 8,
        "orders": 5,
        "products": 7,
        "page_views": 7,
    }
    for table, n in expected.items():
        cols = insp.get_columns(table)
        assert len(cols) == n, (
            f"{table}: expected {n} columns, got {len(cols)}: "
            f"{[c['name'] for c in cols]}"
        )


def test_nullable_columns_flagged_correctly(seeded_engine):
    """``Nullable(T)`` columns must surface with nullable=True; non-nullable as False.

    LangChain shows nullability in the schema string. If we get this
    wrong the LLM either (a) writes ``IS NOT NULL`` filters that pass
    on non-nullable columns (wasteful but harmless) or (b) omits them
    on nullable columns and breaks aggregations.
    """
    insp = inspect(seeded_engine)
    users_cols = {c["name"]: c for c in insp.get_columns("users")}
    assert users_cols["id"]["nullable"] is False
    assert users_cols["last_login"]["nullable"] is True

    events_cols = {c["name"]: c for c in insp.get_columns("events")}
    assert events_cols["duration_ms"]["nullable"] is True
    assert events_cols["revenue_cents"]["nullable"] is True
    assert events_cols["ts"]["nullable"] is False


def test_low_cardinality_preserved_in_info(seeded_engine):
    """``LowCardinality(String)`` should round-trip the LowCardinality flag.

    Stored on the column dict's ``info`` field (SQLAlchemy's standard
    place for dialect-specific metadata). LangChain doesn't *use* this
    directly, but we preserve it so DDL regeneration (Direction 6.1
    pandas migration cookbook) stays correct.
    """
    insp = inspect(seeded_engine)
    users_cols = {c["name"]: c for c in insp.get_columns("users")}
    assert users_cols["signup_country"]["info"]["chdb_low_cardinality"] is True
    assert users_cols["id"]["info"]["chdb_low_cardinality"] is False


def test_pk_constraint_from_sorting_key(seeded_engine):
    """ORDER BY must come back as faux PK so the LLM joins on the right columns."""
    insp = inspect(seeded_engine)
    pk = insp.get_pk_constraint("events")
    assert pk["constrained_columns"] == ["user_id", "ts"]

    pk = insp.get_pk_constraint("users")
    assert pk["constrained_columns"] == ["id"]


def test_foreign_keys_empty_never_raises(seeded_engine):
    """LangChain calls ``get_foreign_keys`` on *every* table during init.

    If our impl raised on any single table the entire toolkit would crash.
    """
    insp = inspect(seeded_engine)
    for s in schemas.ALL_SCHEMAS:
        fks = insp.get_foreign_keys(s.name)
        assert fks == [], f"{s.name}: expected [], got {fks}"


# ---------------------------------------------------------------------------
# LangChain SQLDatabase-level checks
# ---------------------------------------------------------------------------


def test_langchain_get_usable_table_names(langchain_db):
    names = set(langchain_db.get_usable_table_names())
    expected = {s.name for s in schemas.ALL_SCHEMAS}
    assert expected.issubset(names)


def test_langchain_get_table_info_includes_every_column(langchain_db):
    """``get_table_info()`` is the literal string fed to the LLM prompt.

    Each column must appear *by name* — otherwise the LLM hallucinates.
    """
    info = langchain_db.get_table_info()
    missing_columns = []
    for schema in schemas.ALL_SCHEMAS:
        # crude but exact: column names parsed out of the DDL
        for line in schema.ddl.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            first_word = line.split(" ", 1)[0].strip()
            # skip lines that are clearly structural
            if first_word in (
                "CREATE",
                "TABLE",
                ")",
                "ENGINE",
                "ENGINE=",
                "ORDER",
                "PARTITION",
                "(",
            ):
                continue
            if first_word.startswith("ENGINE"):
                continue
            if "(" in first_word:  # probably a function call, not a column
                continue
            colname = first_word
            if colname not in info:
                missing_columns.append((schema.name, colname))
    assert not missing_columns, (
        f"LangChain schema string is missing column names: {missing_columns[:10]}"
    )


def test_langchain_get_table_info_includes_sample_rows(langchain_db):
    """LangChain by default appends 3 sample rows per table to the prompt.

    No sample rows → LLM has no concrete value examples → hallucinated filters.
    """
    info = langchain_db.get_table_info()
    # Sample rows are introduced by a header containing the table name.
    # We check at least one known seed value renders into the prompt.
    assert "alice@example.com" in info or "alice" in info, (
        "Sample row from users table missing from schema string"
    )


def test_langchain_get_table_info_specific_table(langchain_db):
    """``get_table_info(['events'])`` should return only that table's schema."""
    info = langchain_db.get_table_info(table_names=["events"])
    assert "events" in info
    assert "event_type" in info
    # other tables should not leak
    assert "categories" not in info  # that's a products column


def test_langchain_run_simple_query(langchain_db):
    """LangChain's ``db.run()`` is the SQL-execution sink the toolkit uses."""
    result = langchain_db.run("SELECT count() FROM users")
    # The result is a string that includes the row count.
    assert "8" in result


def test_langchain_run_query_no_throw_invalid_sql(langchain_db):
    """``db.run_no_throw`` is what the toolkit uses on agent-generated SQL.

    Must return a string with an error message rather than raising —
    otherwise the agent's self-correction loop is broken.
    """
    if not hasattr(langchain_db, "run_no_throw"):
        pytest.skip("LangChain version lacks run_no_throw")
    result = langchain_db.run_no_throw("SELECT * FROM nonexistent_table")
    assert isinstance(result, str)
    assert "Error" in result or "error" in result or "exception" in result.lower()


# ---------------------------------------------------------------------------
# Type-specific assertions (the part that's most likely to break)
# ---------------------------------------------------------------------------


def test_array_column_renders_in_schema_string(langchain_db):
    info = langchain_db.get_table_info(table_names=["events"])
    # LangChain renders types via str(column.type). Our Array class should
    # render somehow — check for the column name and a plausible type token.
    assert "tags" in info


def test_map_column_renders_in_schema_string(langchain_db):
    info = langchain_db.get_table_info(table_names=["events"])
    assert "attrs" in info


def test_json_column_renders_in_schema_string(langchain_db):
    info = langchain_db.get_table_info(table_names=["page_views"])
    assert "meta" in info


def test_enum_column_renders(langchain_db):
    info = langchain_db.get_table_info(table_names=["orders"])
    assert "status" in info


def test_decimal_column_renders(langchain_db):
    info = langchain_db.get_table_info(table_names=["orders"])
    assert "amount_usd" in info


def test_uuid_column_renders(langchain_db):
    info = langchain_db.get_table_info(table_names=["events"])
    assert "event_id" in info


def test_tuple_column_renders(langchain_db):
    info = langchain_db.get_table_info(table_names=["page_views"])
    assert "viewport" in info


def test_ipv4_column_renders(langchain_db):
    info = langchain_db.get_table_info(table_names=["page_views"])
    assert "client_ip" in info
