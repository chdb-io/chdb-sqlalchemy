"""L4d — CrewAI ``NL2SQLTool`` compatibility (live integration).

CrewAI's ``NL2SQLTool`` runs two hardcoded PostgreSQL-style introspection
queries that misbehave against chDB:

1. ``_fetch_available_tables()`` filters by ``table_schema = 'public'``,
   which doesn't exist in chDB → empty result.
2. ``_fetch_all_available_columns(name)`` filters only by ``table_name``,
   not schema → leaks columns from same-named tables in other databases.

Both are rewritten transparently in
``chdb_sqlalchemy/_cursor.py::_apply_crewai_compat_rewrites``.

This file has:

1. **Behavioral** — instantiate the real ``NL2SQLTool`` and verify the
   end-to-end introspection sees the right tables and only the right
   columns (including the cross-database isolation regression).
2. **Canary** — inspect ``crewai_tools`` source and assert the SQL
   shapes the shims target still match. Fails LOUDLY when CrewAI changes
   upstream, with pointers back to the shim file.

The pure unit-level tests for the regex itself live in
``tests/test_crewai_shim_regex.py`` — those run on every PR regardless of
whether ``crewai_tools`` is installed.
"""

from __future__ import annotations

import inspect

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Module-level skip if crewai_tools is not installable in this environment.
# The package is heavyweight (pulls langchain transitively); only the
# integration job installs it.
# ---------------------------------------------------------------------------

crewai_tools = pytest.importorskip(
    "crewai_tools",
    reason="crewai_tools not installed; install with `chdb-sqlalchemy[test-integration]`",
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def crewai_seeded_engine(engine):
    """In-memory chDB engine with three differently-named user tables."""
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE crew_orders ("
            "id Int32, customer String, total Decimal(10,2)"
            ") ENGINE=Memory"
        ))
        conn.execute(text(
            "CREATE TABLE crew_products ("
            "sku String, name String, price Decimal(10,2)"
            ") ENGINE=Memory"
        ))
        conn.execute(text(
            "CREATE TABLE crew_customers ("
            "id Int32, email String, signup_date Date"
            ") ENGINE=Memory"
        ))
    return engine


@pytest.fixture
def crewai_cross_database_engine(engine):
    """Two databases each containing a table called ``crew_orders`` with
    deliberately-disjoint columns.

    Used to exercise the cross-database columns-leak fix: without the
    ``AND table_schema = currentDatabase()`` rewrite, the agent's NL2SQL
    prompt would see *six* columns under "crew_orders" — three from
    ``default`` plus three from ``other_crewai``.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE DATABASE IF NOT EXISTS other_crewai"))
        # default.crew_orders — what CrewAI *should* see.
        conn.execute(text(
            "CREATE TABLE default.crew_orders ("
            "id Int32, customer String, total Decimal(10,2)"
            ") ENGINE=Memory"
        ))
        # other_crewai.crew_orders — distractor that must NOT leak in.
        conn.execute(text(
            "CREATE TABLE other_crewai.crew_orders ("
            "alien Int64, payload String, signature UUID"
            ") ENGINE=Memory"
        ))
    yield engine
    # Cleanup — the in-memory engine is shared across the test session, so
    # leftover state from one test would pollute the next.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS default.crew_orders"))
        conn.execute(text("DROP TABLE IF EXISTS other_crewai.crew_orders"))
        conn.execute(text("DROP DATABASE IF EXISTS other_crewai"))


# ---------------------------------------------------------------------------
# 1. Behavioral — does the shim let CrewAI actually see user tables?
# ---------------------------------------------------------------------------


def test_nl2sqltool_discovers_user_tables_through_dialect_shim(crewai_seeded_engine):
    """End-to-end: ``NL2SQLTool._fetch_available_tables()`` returns the
    user's tables, not an empty list.

    Regression check for ``_rewrite_crewai_public_schema``. If this fails,
    the tables-query shim has stopped firing — check the canary test below
    for the most likely cause (upstream CrewAI query-shape change).
    """
    from crewai_tools import NL2SQLTool

    tool = NL2SQLTool(db_uri=str(crewai_seeded_engine.url))
    fetched = tool._fetch_available_tables()

    assert fetched, (
        "NL2SQLTool saw zero tables. The chDB 'public'→currentDatabase() "
        "rewrite in chdb_sqlalchemy/_cursor.py may have stopped firing — "
        "check test_crewai_query_shape_unchanged for an upstream change."
    )

    table_names = {row["table_name"] for row in fetched}
    assert {"crew_orders", "crew_products", "crew_customers"}.issubset(table_names), (
        f"NL2SQLTool saw tables {table_names!r}, expected all three "
        f"seeded tables to be visible."
    )


def test_nl2sqltool_column_introspection_round_trip(crewai_seeded_engine):
    """End-to-end: ``_fetch_all_available_columns(table_name)`` returns the
    table's columns. Without functioning columns introspection, the
    agent's schema prompt for the LLM is incomplete.
    """
    from crewai_tools import NL2SQLTool

    tool = NL2SQLTool(db_uri=str(crewai_seeded_engine.url))
    columns = tool._fetch_all_available_columns("crew_orders")

    column_names = {row["column_name"] for row in columns}
    assert column_names == {"id", "customer", "total"}, (
        f"Expected columns {{'id', 'customer', 'total'}} on crew_orders, "
        f"got {column_names!r}. The information_schema.columns view may "
        f"have changed shape in chDB, or CrewAI's columns query may have "
        f"changed."
    )


def test_nl2sqltool_columns_scoped_to_current_database(crewai_cross_database_engine):
    """Regression: same-named tables in sibling databases must NOT leak
    their columns into the agent's schema prompt.

    Without the ``AND table_schema = currentDatabase()`` rewrite in
    ``_rewrite_crewai_columns_filter``, CrewAI's ``WHERE table_name =
    'crew_orders'`` filter returns the union of columns from
    ``default.crew_orders`` and ``other_crewai.crew_orders``. The LLM then
    hallucinates SQL referencing alien columns that don't exist in the
    table it's actually addressing.
    """
    from crewai_tools import NL2SQLTool

    tool = NL2SQLTool(db_uri=str(crewai_cross_database_engine.url))
    columns = tool._fetch_all_available_columns("crew_orders")

    column_names = {row["column_name"] for row in columns}

    # We must see exactly the columns from default.crew_orders.
    assert column_names == {"id", "customer", "total"}, (
        f"CrewAI columns query leaked columns across databases: got "
        f"{column_names!r}, expected {{'id', 'customer', 'total'}} from "
        f"default.crew_orders only. The cross-database scope fix in "
        f"chdb_sqlalchemy/_cursor.py::_rewrite_crewai_columns_filter has "
        f"stopped firing — check test_crewai_query_shape_unchanged for "
        f"an upstream CrewAI query-shape change."
    )

    # Defense in depth: explicitly assert no other_crewai columns made it in.
    forbidden = {"alien", "payload", "signature"}
    assert not (column_names & forbidden), (
        f"Columns from sibling database 'other_crewai' leaked through: "
        f"{column_names & forbidden!r}"
    )


# ---------------------------------------------------------------------------
# 2. Canary — does CrewAI still emit the exact SQL the shim targets?
# ---------------------------------------------------------------------------


def test_crewai_query_shape_unchanged():
    """Inspect ``crewai_tools.tools.nl2sql.nl2sql_tool.NL2SQLTool`` source
    and assert the introspection queries still match what the shims target.

    **What this guard buys**: a CrewAI upgrade that changes either query
    shape would otherwise revert chDB users to silent failure modes
    (empty schema, or cross-database column leak) without any test
    breaking — the SQL would still execute, the shim would just silently
    no-op because its regex no longer matched.

    **When this test fails**, do one of:

    1. CrewAI still emits PostgreSQL-style queries but in a new shape —
       update the matching regex in ``chdb_sqlalchemy/_cursor.py``
       (``_CREWAI_PG_PUBLIC_TABLES_QUERY`` for the tables query,
       ``_CREWAI_PG_COLUMNS_QUERY`` for the columns query) and re-run the
       behavioral tests above.

    2. CrewAI switched to SQLAlchemy ``Inspector`` (the upstream-correct
       fix) — delete both shims, this whole test file, the unit-level
       ``tests/test_crewai_shim_regex.py``, and the ``crewai-tools``
       dependency line in ``pyproject.toml``. Update CHANGELOG.
    """
    from crewai_tools.tools.nl2sql.nl2sql_tool import NL2SQLTool  # type: ignore[import-untyped]

    tables_src = inspect.getsource(NL2SQLTool._fetch_available_tables)
    columns_src = inspect.getsource(NL2SQLTool._fetch_all_available_columns)

    # --- Tables query: targeted by _rewrite_crewai_public_schema. ---
    expected_tables_fragments = [
        "information_schema.tables",
        "table_schema",
        "'public'",
        "SELECT table_name",
    ]
    for fragment in expected_tables_fragments:
        assert fragment in tables_src, (
            f"CrewAI's NL2SQLTool._fetch_available_tables no longer contains "
            f"{fragment!r}. The chDB shim "
            f"`chdb_sqlalchemy/_cursor.py::_rewrite_crewai_public_schema` "
            f"is now stale and chDB users will silently get an empty table "
            f"list. Re-inspect:\n\n{tables_src}"
        )

    # --- Columns query: targeted by _rewrite_crewai_columns_filter. ---
    expected_columns_fragments = [
        "information_schema.columns",
        "column_name",
        "data_type",
        "table_name",
    ]
    for fragment in expected_columns_fragments:
        assert fragment in columns_src, (
            f"CrewAI's NL2SQLTool._fetch_all_available_columns no longer "
            f"contains {fragment!r}. The cross-database columns scope shim "
            f"`chdb_sqlalchemy/_cursor.py::_rewrite_crewai_columns_filter` "
            f"is now stale and chDB users will silently get cross-database "
            f"column leakage. Re-inspect:\n\n{columns_src}"
        )

    # Sanity: the columns query MUST still be the "only-table_name-filter"
    # shape. If CrewAI started passing a schema filter themselves
    # (`AND table_schema = ...`), our rewrite would be redundant or
    # actively wrong. Detect that and prompt re-evaluation.
    assert "table_schema" not in columns_src, (
        "CrewAI's _fetch_all_available_columns now filters by table_schema "
        "directly. The chDB columns-scope shim is now redundant (or worse, "
        "doubled). Re-inspect and remove the shim if no longer needed:\n\n"
        f"{columns_src}"
    )
