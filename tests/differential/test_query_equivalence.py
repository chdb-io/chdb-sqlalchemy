"""L5 — Differential testing: chDB vs ClickHouse 26.3.9.8-lts reference.

For each query in :data:`QUERY_CORPUS`, we run it twice:

1. Through our SQLAlchemy dialect against the chDB engine
2. Directly through ``clickhouse local`` (the same v26.3.9.8 build,
   non-embedded)

…then compare the row-string output. Mismatches are classified into
three buckets:

* **Bug** — same query, materially different result → CI red.
* **Known difference** — documented in ``docs/chdb-vs-server-differences.md``;
  test passes after registering the diff there.
* **chDB-not-supported** — chDB raises on a query the reference accepts;
  skip and document in ``docs/known-skips.md``.

The corpus deliberately mixes:

* Our reflection queries (the SQL the dialect emits internally —
  ``system.tables``, ``system.columns``, etc.). If these diverge, every
  reflection-dependent consumer breaks.
* The L4b NL2SQL reference SQL — the *exact* SQL a competent LLM would
  emit against the seeded schemas. If chDB and reference disagree here,
  LangChain agents will produce inconsistent answers depending on
  which chDB build they hit.
* ClickHouse-specific function corner cases — Array/Map/Tuple/JSON
  paths most likely to vary between builds.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.differential


# ---------------------------------------------------------------------------
# QUERY_CORPUS — runs against the seeded fixtures from tests/integration
# ---------------------------------------------------------------------------

QUERY_CORPUS: list[tuple[str, str]] = [
    # (scenario_id, sql)
    # Reflection-flavoured queries — the dialect's own working set
    ("reflect_table_count", "SELECT count() FROM system.tables WHERE database = currentDatabase()"),
    ("reflect_column_count_events", (
        "SELECT count() FROM system.columns "
        "WHERE database = currentDatabase() AND table = 'events'"
    )),
    ("reflect_sorting_key_events", (
        "SELECT sorting_key FROM system.tables "
        "WHERE database = currentDatabase() AND name = 'events'"
    )),

    # L4b NL2SQL reference SQL — what a correctly-prompted LLM would emit
    ("count_users", "SELECT count() FROM users"),
    ("active_user_count", "SELECT count() FROM users WHERE is_active = true"),
    ("nullable_count", "SELECT count() FROM users WHERE last_login IS NULL"),
    ("group_by_country", (
        "SELECT signup_country, count() AS n FROM users "
        "GROUP BY signup_country ORDER BY n DESC, signup_country"
    )),
    ("decimal_sum", (
        "SELECT sum(amount_usd) FROM orders WHERE status = 'paid'"
    )),
    ("enum_filter", "SELECT count() FROM orders WHERE status = 'cancelled'"),
    ("array_join", (
        "SELECT tag, count() AS n FROM events ARRAY JOIN tags AS tag "
        "GROUP BY tag ORDER BY n DESC, tag"
    )),
    ("map_key_access", (
        "SELECT attrs['payment_method'] AS m, count() AS n FROM events "
        "WHERE event_type = 'purchase' GROUP BY m ORDER BY n DESC, m"
    )),
    ("map_has_key", (
        "SELECT count() FROM events WHERE has(mapKeys(attrs), 'referrer')"
    )),
    ("tuple_index", (
        "SELECT DISTINCT viewport.1 AS width FROM page_views ORDER BY width"
    )),
    ("json_extract", (
        "SELECT JSONExtractString(toString(meta), 'browser') AS browser, count() AS n "
        "FROM page_views GROUP BY browser ORDER BY n DESC, browser"
    )),
    ("ipv4_distinct", "SELECT count(DISTINCT client_ip) FROM page_views"),
    ("datetime64_window", (
        "SELECT count() FROM events "
        "WHERE ts >= toDateTime64('2026-05-15 00:00:00', 3, 'UTC') "
        "  AND ts <  toDateTime64('2026-05-19 00:00:00', 3, 'UTC')"
    )),
    ("groupby_date_bucket", (
        "SELECT toDate(ts) AS d, count() AS n FROM events GROUP BY d ORDER BY d"
    )),
    ("avg_array_elem", (
        "SELECT round(avg(w), 4) FROM products ARRAY JOIN weights AS w"
    )),

    # Pure-function corner cases — no fixture state needed, but we run
    # them against the seeded environment for consistency.
    ("simple_arithmetic", "SELECT 1 + 1"),
    ("string_function", "SELECT lower('CHDB')"),
    ("date_truncation", "SELECT toStartOfMonth(toDate('2026-05-18'))"),
    ("array_function", "SELECT arraySort([3, 1, 2])"),
    ("map_function", "SELECT map('a', 1, 'b', 2)"),
    ("tuple_function", "SELECT tuple(1, 'two', 3.14)"),
]


# ---------------------------------------------------------------------------
# Known differences — populated as we discover them.
#
# Format: scenario_id → reason. Tests for these scenarios xfail with the
# attached reason rather than failing outright. Every entry must also
# have a corresponding section in docs/chdb-vs-server-differences.md.
# ---------------------------------------------------------------------------

KNOWN_DIFFERENCES: dict[str, str] = {
    # Populate as L5 surfaces real diffs. Empty for a clean v0.1 baseline.
}


# ---------------------------------------------------------------------------
# Known-not-supported — chDB will raise where reference accepts.
# Skip the test cleanly and require docs/known-skips.md to document.
# ---------------------------------------------------------------------------

KNOWN_NOT_SUPPORTED: dict[str, str] = {
    # Populate as L5 surfaces real gaps. Empty for a clean v0.1 baseline.
}


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


_DATE_RE = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = __import__("re").compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?(\s+[A-Z][\w/+-]*)?$"
)


def _normalize_cell(v: object) -> object:
    """Reduce one cell from either backend to a canonical form for comparison.

    The two backends emit syntactically different representations of the
    same value. We coerce both sides to the same Python types so that
    structural comparison only flags *real* divergences:

    * Lists/Arrays: chDB → ``list``, reference TabSeparated → ``"[1,2,3]"`` string
    * Tuples: chDB → ``tuple``, reference → ``"(1,'two',3.14)"`` string
    * Maps: chDB → ``dict``, reference → ``"{'a':1,'b':2}"`` string
    * Dates: chDB → ``datetime.date``, reference → ``"2026-05-15"`` string
    * Datetimes: chDB → ``datetime.datetime``, reference → ``"2026-05-15 12:00:00"`` string
    * Decimals: chDB → ``Decimal('589.95')``, reference → ``"589.95"`` string
    * NULL: chDB → ``None``, reference → ``"\\N"`` string
    * Booleans: chDB → ``True``/``False``, reference → ``"1"``/``"0"`` string

    Canonical form: numerics → ``float`` (lossy but tolerant), dates →
    ``datetime.date``, datetimes → ``datetime.datetime``, containers →
    ``tuple``-of-normalised elements (so order matters but type doesn't).
    """
    import ast
    import datetime as dt
    import decimal as _decimal

    if v is None or v == "\\N":
        return None

    # Numerics from either side → float for tolerant equality.
    if isinstance(v, _decimal.Decimal):
        return float(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
        return v
    if isinstance(v, dt.datetime):
        # Strip tzinfo for comparison (reference TSV strips it).
        return v.replace(tzinfo=None)
    if isinstance(v, dt.date):
        return v

    if isinstance(v, str):
        stripped = v.strip()
        # Composite literal from TSV
        if stripped and stripped[0] in "([{":
            try:
                return _normalize_cell(ast.literal_eval(stripped))
            except (ValueError, SyntaxError):
                pass
        # Date literal
        if _DATE_RE.match(stripped):
            return dt.date.fromisoformat(stripped)
        # Datetime literal
        if _DATETIME_RE.match(stripped):
            try:
                # Tolerate trailing 'UTC' / timezone token.
                core = stripped.split()
                base = core[0] + (" " + core[1] if len(core) > 1 and ":" in core[1] else "")
                return dt.datetime.fromisoformat(base.replace(" ", "T"))
            except ValueError:
                pass
        # Numeric literal — float catches both ints and decimals.
        try:
            return float(stripped)
        except ValueError:
            return v

    if isinstance(v, (list, tuple)):
        return tuple(_normalize_cell(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((str(k), _normalize_cell(val)) for k, val in v.items()))
    return v


def _normalize_row(row) -> tuple:
    return tuple(_normalize_cell(v) for v in row)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id, sql",
    QUERY_CORPUS,
    ids=[c[0] for c in QUERY_CORPUS],
)
def test_chdb_matches_reference(
    scenario_id,
    sql,
    chdb_seeded_engine,
    clickhouse_local,
    server_seed_sql,
):
    """Both backends must produce equivalent rows for the same SQL.

    The chDB side runs through our SQLAlchemy dialect; the reference side
    runs through ``clickhouse local`` as a subprocess. Tab-separated
    output is the lowest common denominator that survives all type
    quirks.
    """
    if scenario_id in KNOWN_NOT_SUPPORTED:
        pytest.skip(f"chDB-not-supported: {KNOWN_NOT_SUPPORTED[scenario_id]}")

    # 1) chDB through our dialect
    try:
        with chdb_seeded_engine.connect() as conn:
            chdb_rows_raw = conn.execute(text(sql)).fetchall()
    except Exception as e:
        # chDB raised — this is either a "bug" or a "not-supported" finding.
        pytest.fail(
            f"[chDB raised] {scenario_id}: {type(e).__name__}: {e}\n"
            f"sql: {sql}"
        )

    chdb_rows = [_normalize_row(r) for r in chdb_rows_raw]

    # 2) Reference clickhouse local
    full_sql = server_seed_sql + "\n" + sql
    try:
        ref_rows_raw = clickhouse_local.run_rows(full_sql)
    except RuntimeError as e:
        pytest.fail(
            f"[reference raised] {scenario_id}: {e}\n(this is a fixture/SQL issue, not chDB)"
        )
    ref_rows = [_normalize_row(r) for r in ref_rows_raw]

    if scenario_id in KNOWN_DIFFERENCES:
        pytest.xfail(
            f"Documented difference: {KNOWN_DIFFERENCES[scenario_id]}"
        )

    assert chdb_rows == ref_rows, (
        f"[divergence] {scenario_id}\n"
        f"chDB:      {chdb_rows!r}\n"
        f"reference: {ref_rows!r}\n"
        f"sql:       {sql}"
    )


def test_version_strings_are_compatible(clickhouse_local, chdb_seeded_engine):
    """Sanity check: chDB and the reference binary are both 26.3 series.

    A real version mismatch (e.g. reference downloaded as 25.x by accident)
    would invalidate every comparison below. We just check the major.minor
    prefix.
    """
    ref_ver = clickhouse_local.version()
    with chdb_seeded_engine.connect() as conn:
        chdb_ver = conn.execute(text("SELECT version()")).scalar()
    # Both should be on the 26.3 line. We don't require exact patch match
    # — but if they're more than one minor apart, flag it.
    ref_prefix = ".".join(ref_ver.split(".")[:2])
    chdb_prefix = ".".join(str(chdb_ver).split(".")[:2])
    assert ref_prefix == chdb_prefix, (
        f"version skew: chDB={chdb_ver} vs reference={ref_ver}; "
        f"L5 results are not authoritative."
    )
