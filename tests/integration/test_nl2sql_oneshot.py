"""L4b — One-shot NL2SQL tests against the real chDB engine.

Each test pairs:

* A natural-language question an analyst might actually ask
* The SQL a competent LLM with access to ``langchain_db.get_table_info()``
  would most likely emit
* An assertion on the returned rows / shape

Two run modes:

1. **Default**: the SQL is hand-written (or LLM-generated and frozen in this
   file). We *execute* it on the live chDB engine through SQLAlchemy and
   assert the answer. This catches dialect-level bugs that prevent
   plausible LLM-generated SQL from running — without needing an API key.

2. **With ANTHROPIC_API_KEY set**: a parallel test class re-runs each
   scenario by asking Claude to generate the SQL fresh, then executes it.
   Catches schema-prompt regressions (the LLM seeing a wrong type label,
   missing column, malformed sample row, …).

The hand-written SQL is the **reference** — what a perfectly-informed LLM
would produce given a correct schema prompt. If it fails to execute, the
dialect has a bug. If it runs but the LLM-generated version fails, the
``get_table_info`` prompt is misleading.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from . import schemas

pytestmark = pytest.mark.integration

API_KEY_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_engine(engine):
    """Engine with every demo schema created and populated."""
    with engine.begin() as conn:
        schemas.build_all(conn)
    return engine


# ---------------------------------------------------------------------------
# Reference SQL scenarios — what a competent LLM should emit given our
# get_table_info() schema string. Each entry is (id, question, sql, validator).
#
# ``validator`` is a callable(rows) → None that asserts the result shape.
# ---------------------------------------------------------------------------


def _has_rows(min_n: int = 1):
    def check(rows: list) -> None:
        assert len(rows) >= min_n, f"expected ≥{min_n} rows, got {len(rows)}: {rows!r}"

    return check


def _scalar_equals(expected):
    def check(rows: list) -> None:
        assert len(rows) == 1, f"expected exactly 1 row, got {rows!r}"
        assert rows[0][0] == expected, f"expected {expected!r}, got {rows[0][0]!r}"

    return check


def _scalar_in(lo, hi):
    def check(rows: list) -> None:
        assert len(rows) == 1, f"expected exactly 1 row, got {rows!r}"
        v = rows[0][0]
        assert lo <= v <= hi, f"expected {lo}≤v≤{hi}, got {v}"

    return check


SCENARIOS = [
    # ---------------- Plain scalar SELECT ----------------
    (
        "count_users",
        "How many users are in the system?",
        "SELECT count() FROM users",
        _scalar_equals(8),
    ),
    (
        "active_user_count",
        "How many users are currently active?",
        "SELECT count() FROM users WHERE is_active = true",
        _scalar_equals(6),
    ),
    (
        "top_country_by_users",
        "Which country has the most users?",
        """
        SELECT signup_country, count() AS n
        FROM users
        GROUP BY signup_country
        ORDER BY n DESC
        LIMIT 1
        """,
        lambda rows: (
            rows[0][0] == "US" and rows[0][1] == 3,
        ),
    ),
    # ---------------- Nullable handling ----------------
    (
        "users_never_logged_in",
        "How many users have never logged in?",
        "SELECT count() FROM users WHERE last_login IS NULL",
        _scalar_equals(2),
    ),
    (
        "avg_event_duration_nonnull",
        "What is the average event duration?",
        "SELECT avg(duration_ms) FROM events WHERE duration_ms IS NOT NULL",
        _scalar_in(100.0, 600.0),
    ),
    # ---------------- LowCardinality is transparent at SQL level ----------------
    (
        "events_by_type",
        "How many events of each type are there?",
        """
        SELECT event_type, count() AS n
        FROM events
        GROUP BY event_type
        ORDER BY n DESC
        """,
        _has_rows(min_n=3),
    ),
    # ---------------- DateTime / time windows ----------------
    (
        "events_between_specific_dates",
        # Concrete date range — avoid the ambiguity of "last week of May".
        # The first live-LLM run revealed Claude (correctly) interpreted
        # "last week of May 2026" as May 25-31, while our seed data lives
        # in May 15-18. Lesson learned: dialect tests should remove
        # natural-language date ambiguity to keep the focus on the SQL
        # surface.
        "How many events happened between May 15 2026 and May 18 2026 (inclusive)?",
        """
        SELECT count()
        FROM events
        WHERE ts >= toDateTime64('2026-05-15 00:00:00', 3, 'UTC')
          AND ts <  toDateTime64('2026-05-19 00:00:00', 3, 'UTC')
        """,
        _scalar_in(8, 11),
    ),
    (
        "events_per_day",
        "How many events per day in May 2026?",
        """
        SELECT toDate(ts) AS d, count() AS n
        FROM events
        GROUP BY d
        ORDER BY d
        """,
        _has_rows(min_n=3),
    ),
    # ---------------- Aggregations with Decimal ----------------
    (
        "total_revenue_paid_orders",
        "What is the total revenue from paid orders?",
        """
        SELECT sum(amount_usd)
        FROM orders
        WHERE status = 'paid'
        """,
        # 49.99 + 129.99 + 59.99 + 99.99 + 249.99 = 589.95
        # Decimal column → decimal.Decimal in result_processor
        _scalar_equals(__import__("decimal").Decimal("589.95")),
    ),
    # ---------------- JOIN across tables ----------------
    (
        "top_spenders",
        "Which users have spent the most?",
        """
        SELECT u.email, sum(o.amount_usd) AS total
        FROM users u
        INNER JOIN orders o ON o.user_id = u.id
        WHERE o.status IN ('paid', 'shipped')
        GROUP BY u.email
        ORDER BY total DESC
        LIMIT 3
        """,
        _has_rows(min_n=3),
    ),
    # ---------------- Array unnesting ----------------
    (
        "popular_tags",
        "What are the 3 most common event tags?",
        """
        SELECT tag, count() AS n
        FROM events
        ARRAY JOIN tags AS tag
        GROUP BY tag
        ORDER BY n DESC
        LIMIT 3
        """,
        _has_rows(min_n=3),
    ),
    # ---------------- Map key access ----------------
    (
        "events_by_referrer",
        "How many page-view events came from each referrer?",
        """
        SELECT attrs['referrer'] AS ref, count() AS n
        FROM events
        WHERE event_type = 'page_view'
          AND has(mapKeys(attrs), 'referrer')
        GROUP BY ref
        ORDER BY n DESC
        """,
        _has_rows(min_n=1),
    ),
    (
        "purchase_payment_methods",
        "What payment methods do users use for purchases?",
        """
        SELECT attrs['payment_method'] AS m, count() AS n
        FROM events
        WHERE event_type = 'purchase'
        GROUP BY m
        ORDER BY n DESC
        """,
        _has_rows(min_n=2),
    ),
    # ---------------- Array(Float) aggregation across rows ----------------
    (
        "avg_product_weight",
        "What is the average weight of all products across all weight measurements?",
        """
        SELECT avg(w)
        FROM products
        ARRAY JOIN weights AS w
        """,
        _scalar_in(0.0, 5.0),
    ),
    # ---------------- Tuple column access ----------------
    (
        "unique_viewports",
        "How many unique viewport sizes do we see?",
        "SELECT count(DISTINCT viewport) FROM page_views",
        _scalar_in(1, 100),
    ),
    (
        "viewport_widths",
        "What are the viewport widths in use?",
        """
        SELECT DISTINCT viewport.1 AS width
        FROM page_views
        ORDER BY width
        """,
        _has_rows(min_n=3),
    ),
    # ---------------- JSON path access ----------------
    (
        "browser_breakdown",
        "What's the breakdown of browsers in page views?",
        """
        SELECT JSONExtractString(toString(meta), 'browser') AS browser, count() AS n
        FROM page_views
        GROUP BY browser
        ORDER BY n DESC
        """,
        _has_rows(min_n=3),
    ),
    # ---------------- IPv4 / FixedString ----------------
    (
        "unique_clients",
        "How many distinct client IPs viewed pages?",
        "SELECT count(DISTINCT client_ip) FROM page_views",
        _scalar_in(1, 10),
    ),
    # ---------------- Enum filter ----------------
    (
        "cancelled_count",
        "How many orders were cancelled?",
        "SELECT count() FROM orders WHERE status = 'cancelled'",
        _scalar_equals(1),
    ),
    # ---------------- Subquery / EXISTS ----------------
    (
        "users_with_purchases",
        "How many users made at least one purchase?",
        """
        SELECT count(DISTINCT user_id)
        FROM events
        WHERE event_type = 'purchase'
        """,
        _scalar_in(2, 8),
    ),
]


@pytest.mark.parametrize("scenario_id, question, sql, validator", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_reference_sql_executes(seeded_engine, scenario_id, question, sql, validator):
    """Reference SQL (what a correctly-prompted LLM would emit) must run.

    If this fails, our dialect has a bug — either the SQL doesn't reach
    chDB intact, or a result type can't be decoded, or our compiler emits
    something the engine can't parse.
    """
    with seeded_engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    validator(rows)


# ---------------------------------------------------------------------------
# Live-LLM variant — only runs with ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not API_KEY_AVAILABLE,
    reason="ANTHROPIC_API_KEY not set; live LLM NL2SQL test skipped",
)
@pytest.mark.parametrize(
    "scenario_id, question, _reference_sql, validator",
    SCENARIOS,
    ids=[f"live-{s[0]}" for s in SCENARIOS],
)
def test_live_llm_generates_valid_sql(
    seeded_engine, scenario_id, question, _reference_sql, validator
):
    """Ask Claude to generate SQL for the question; execute on chDB.

    The LLM sees only what ``SQLDatabase.get_table_info()`` produces — so
    if the LLM-generated SQL fails to execute, that's evidence the schema
    prompt is misleading. Compare against the reference SQL to localise
    the issue.
    """
    sa = pytest.importorskip("langchain_community.utilities")
    ca = pytest.importorskip("langchain_anthropic")

    db = sa.SQLDatabase(seeded_engine)
    schema_str = db.get_table_info()

    llm = ca.ChatAnthropic(model_name="claude-sonnet-4-6", temperature=0)
    prompt = f"""You write ClickHouse SQL for chDB. Given the schema below, write a single SQL query that answers the question. Return ONLY the SQL, no commentary, no markdown fences.

Schema:
{schema_str}

Question: {question}

SQL:"""
    response = llm.invoke(prompt)
    sql = (
        response.content
        if isinstance(response.content, str)
        else "\n".join(b["text"] for b in response.content if b["type"] == "text")
    )
    # Strip any code fences just in case.
    sql = sql.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()

    # Log the generated SQL — invaluable when this test fails. Pytest
    # captures stdout per test; ``-s`` reveals it inline.
    print(f"\n[{scenario_id}] LLM SQL:\n{sql}\n")

    with seeded_engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    print(f"[{scenario_id}] rows: {rows[:5]}{'...' if len(rows) > 5 else ''}")
    validator(rows)
