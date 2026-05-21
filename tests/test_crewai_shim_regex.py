"""Unit tests for the CrewAI compatibility regex rewrites in ``_cursor.py``.

These tests run on every PR — they are deliberately decoupled from whether
``crewai_tools`` is installed, because their purpose is to lock in the
regex shape itself, not the integration behavior. The companion
behavioral + canary tests live in ``tests/integration/test_crewai_compat.py``
and are gated on the ``test-integration`` extra.

If the regex tests below pass but the integration tests fail, the shim
*pattern* is intact but reality has moved (chDB / CrewAI / chdb.dbapi
changed something). Read the integration failure messages.

If the regex tests below fail, somebody changed ``_cursor.py`` in a way
that broke the shim itself — re-validate against the docstrings in
``chdb_sqlalchemy/_cursor.py``.
"""

from __future__ import annotations

import pytest

from chdb_sqlalchemy._cursor import (  # type: ignore[attr-defined]
    _CREWAI_PG_COLUMNS_QUERY,
    _CREWAI_PG_PUBLIC_TABLES_QUERY,
    _apply_crewai_compat_rewrites,
    _rewrite_crewai_columns_filter,
    _rewrite_crewai_public_schema,
)

# ---------------------------------------------------------------------------
# Tables-query rewrite (the original CrewAI shim)
# ---------------------------------------------------------------------------


_CANONICAL_TABLES_QUERY = (
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
)


def test_tables_regex_matches_canonical_query():
    assert _CREWAI_PG_PUBLIC_TABLES_QUERY.match(_CANONICAL_TABLES_QUERY), (
        "Regex no longer matches the canonical CrewAI tables query. If "
        "CrewAI changed shape, update the regex AND the canary in "
        "tests/integration/test_crewai_compat.py to match."
    )


def test_tables_rewrite_replaces_public_with_currentdb():
    rewritten = _rewrite_crewai_public_schema(_CANONICAL_TABLES_QUERY)
    assert "currentDatabase()" in rewritten, "Expected currentDatabase() in result"
    assert "'public'" not in rewritten, "Expected 'public' to be removed"
    # Other parts of the query should remain bit-identical (no accidental edits).
    assert rewritten == _CANONICAL_TABLES_QUERY.replace("'public'", "currentDatabase()")


@pytest.mark.parametrize("sql", [
    # Same shape but different schema literal — must NOT rewrite.
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'system';",
    # Different selected columns — must NOT rewrite.
    "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = 'public';",
    # User-authored query mentioning 'public' in a column literal — must NOT rewrite.
    "SELECT 'public' AS schema_name",
    # Different view (columns, not tables) — must NOT rewrite.
    "SELECT * FROM information_schema.columns WHERE table_schema = 'public'",
])
def test_tables_rewrite_skips_unrelated_queries(sql: str):
    assert _rewrite_crewai_public_schema(sql) == sql, (
        f"Tables-shim incorrectly rewrote a non-CrewAI query: {sql!r}"
    )


# ---------------------------------------------------------------------------
# Columns-query rewrite (the cross-database scoping fix)
# ---------------------------------------------------------------------------


# Two canonical forms — older crewai-tools (<= ~1.14) inlined the table
# name via f-string; newer versions use SA bind params (:table_name) which
# SQLAlchemy compiles down to a ``?`` placeholder at the chdb.dbapi cursor
# layer. The shim must rewrite either form.
_CANONICAL_COLUMNS_QUERY_LITERAL = (
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = 'crew_orders';"
)
_CANONICAL_COLUMNS_QUERY_QMARK = (
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = ?"
)


@pytest.mark.parametrize("sql", [
    _CANONICAL_COLUMNS_QUERY_LITERAL,
    _CANONICAL_COLUMNS_QUERY_QMARK,
])
def test_columns_regex_matches_both_canonical_forms(sql: str):
    assert _CREWAI_PG_COLUMNS_QUERY.match(sql), (
        f"Regex no longer matches CrewAI columns query form: {sql!r}"
    )


def test_columns_rewrite_appends_schema_filter_literal_form():
    rewritten = _rewrite_crewai_columns_filter(_CANONICAL_COLUMNS_QUERY_LITERAL)
    expected = (
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'crew_orders' "
        "AND table_schema = currentDatabase();"
    )
    assert rewritten == expected, (
        f"Columns rewrite (literal form) produced unexpected SQL.\n"
        f"Got:      {rewritten!r}\nExpected: {expected!r}"
    )


def test_columns_rewrite_appends_schema_filter_qmark_form():
    """SA-parameterized form: ``?`` placeholder stays in place, schema
    filter is appended after it. The bind-parameter list is unchanged by
    this rewrite — the new clause uses ``currentDatabase()`` (no params),
    so the existing ``?`` ↔ value mapping remains valid."""
    rewritten = _rewrite_crewai_columns_filter(_CANONICAL_COLUMNS_QUERY_QMARK)
    expected = (
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? "
        "AND table_schema = currentDatabase()"
    )
    assert rewritten == expected, (
        f"Columns rewrite (qmark form) produced unexpected SQL.\n"
        f"Got:      {rewritten!r}\nExpected: {expected!r}"
    )


def test_columns_rewrite_preserves_table_name_literal():
    """For the literal form, chDB allows non-ASCII identifiers — the
    regex must capture whatever is inside the single quotes."""
    sql = (
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = '订单_2026';"
    )
    rewritten = _rewrite_crewai_columns_filter(sql)
    assert "'订单_2026'" in rewritten
    assert "AND table_schema = currentDatabase()" in rewritten


@pytest.mark.parametrize("sql", [
    # Already has a schema filter (e.g. someone wrote it by hand) — must NOT rewrite.
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = 'crew_orders' AND table_schema = 'default';",
    # Different selected columns — must NOT rewrite.
    "SELECT column_name FROM information_schema.columns WHERE table_name = 'crew_orders';",
    # Different filter shape (no table_name filter at all) — must NOT rewrite.
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_schema = 'default';",
    # SELECT from a different table — must NOT rewrite.
    "SELECT column_name, data_type FROM system.columns WHERE table_name = 'crew_orders';",
    # qmark in unrelated position — must NOT rewrite (different shape).
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_schema = ?",
])
def test_columns_rewrite_skips_unrelated_queries(sql: str):
    assert _rewrite_crewai_columns_filter(sql) == sql, (
        f"Columns-shim incorrectly rewrote a non-CrewAI query: {sql!r}"
    )


# ---------------------------------------------------------------------------
# Combined dispatcher
# ---------------------------------------------------------------------------


def test_apply_dispatcher_runs_both_rewrites():
    """Both rewrites compose without interfering with each other."""
    # Tables query → only tables rewrite fires.
    out1 = _apply_crewai_compat_rewrites(_CANONICAL_TABLES_QUERY)
    assert "currentDatabase()" in out1 and "'public'" not in out1

    # Columns query (qmark form, what the installed CrewAI actually emits) →
    # only columns rewrite fires.
    out2 = _apply_crewai_compat_rewrites(_CANONICAL_COLUMNS_QUERY_QMARK)
    assert "AND table_schema = currentDatabase()" in out2

    # Random unrelated SQL → no rewrites fire.
    out3 = _apply_crewai_compat_rewrites("SELECT 1")
    assert out3 == "SELECT 1"
