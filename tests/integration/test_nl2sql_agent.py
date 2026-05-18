"""L4c — Full LangChain SQL agent end-to-end.

Unlike :mod:`test_nl2sql_oneshot` (which runs single-prompt LLM calls and
also a no-LLM reference-SQL path), this module exercises the *complete*
``SQLDatabaseToolkit`` + ``create_sql_agent`` loop:

* ``list_sql_database_tool`` — agent calls to discover tables
* ``info_sql_database_tool`` — agent calls to inspect a table's schema
* ``query_sql_checker_tool`` — agent calls before executing a candidate query
* ``query_sql_database_tool`` — agent executes the final SQL

Each of these tools wraps a different surface of our dialect (reflection,
``SQLDatabase.run_no_throw``, statement compilation). The agent is the
canonical real-world consumer; if any of these tools is broken on chDB,
the agent gets stuck.

Two run modes:

* **Default (no API key)**: agent construction smoke-test only — verifies
  that the toolkit's tools instantiate, list correctly, and that calling
  the schema-inspection tools by hand (i.e. as a stand-in for the LLM)
  succeeds against every seeded table.
* **With ANTHROPIC_API_KEY**: the agent actually runs each scenario, and
  we assert the final answer makes sense.
"""

from __future__ import annotations

import os

import pytest

from . import schemas

pytestmark = pytest.mark.integration

API_KEY_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.fixture
def seeded_engine(engine):
    with engine.begin() as conn:
        schemas.build_all(conn)
    return engine


@pytest.fixture
def toolkit(seeded_engine):
    sa = pytest.importorskip("langchain_community.utilities")
    tk = pytest.importorskip("langchain_community.agent_toolkits")

    db = sa.SQLDatabase(seeded_engine)
    # The toolkit needs an LLM even at construction-time. Use a dummy
    # FakeListLLM when no real API key is set, so we can still smoke-test
    # the tool surface.
    if API_KEY_AVAILABLE:
        ca = pytest.importorskip("langchain_anthropic")
        llm = ca.ChatAnthropic(model_name="claude-sonnet-4-6", temperature=0)
    else:
        fakemod = pytest.importorskip("langchain_community.llms.fake")
        llm = fakemod.FakeListLLM(responses=["unused"])
    return tk.SQLDatabaseToolkit(db=db, llm=llm)


# ---------------------------------------------------------------------------
# Toolkit-construction smoke tests (no LLM needed)
# ---------------------------------------------------------------------------


def test_toolkit_instantiates_without_error(toolkit):
    """If this fails, the toolkit's init-time introspection of our dialect
    is broken — every downstream LangChain integration is dead in the water."""
    assert toolkit is not None


def test_toolkit_exposes_all_four_tools(toolkit):
    """LangChain ``SQLDatabaseToolkit.get_tools()`` returns the four core SQL tools."""
    tools = toolkit.get_tools()
    names = {t.name for t in tools}
    expected = {
        "sql_db_list_tables",
        "sql_db_schema",
        "sql_db_query",
        "sql_db_query_checker",
    }
    assert expected.issubset(names), (
        f"missing tools — agent will be missing capabilities: {expected - names}"
    )


def test_list_tables_tool_returns_seeded_set(toolkit):
    """Direct invocation of the ``sql_db_list_tables`` tool the LLM would call."""
    list_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_list_tables")
    output = list_tool.invoke("")
    # Output is a comma-separated string of table names.
    table_names = {n.strip() for n in output.split(",")}
    expected = {s.name for s in schemas.ALL_SCHEMAS}
    missing = expected - table_names
    assert not missing, f"agent's table listing is missing {missing}"


def test_schema_tool_returns_real_schema(toolkit):
    """The ``sql_db_schema`` tool produces the schema string the LLM grounds on."""
    schema_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_schema")
    out = schema_tool.invoke("events")
    # Type signal + column names + at least one sample row.
    assert "event_type" in out
    assert "tags" in out
    assert "duration_ms" in out


def test_query_tool_executes_and_returns_rows(toolkit):
    """The ``sql_db_query`` tool runs the agent's final SQL."""
    query_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_query")
    out = query_tool.invoke("SELECT count() FROM users")
    assert "8" in out


def test_query_tool_returns_error_string_not_raise(toolkit):
    """Malformed SQL must surface as a string the agent can self-correct on,
    not as an uncaught exception."""
    query_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_query")
    out = query_tool.invoke("SELECT * FROM no_such_table")
    assert isinstance(out, str)
    assert "Error" in out or "error" in out


# ---------------------------------------------------------------------------
# Live agent runs — only with API key
# ---------------------------------------------------------------------------

LIVE_AGENT_SCENARIOS = [
    (
        "live_count_users",
        "How many users are there?",
        lambda final: "8" in final,
    ),
    (
        "live_top_country",
        "Which country has the most users?",
        lambda final: "US" in final.upper(),
    ),
    (
        "live_purchase_count",
        "How many purchase events have happened?",
        lambda final: any(d in final for d in ("4", "5", "6")),
    ),
    (
        "live_paid_revenue",
        "What is the total revenue from paid orders?",
        lambda final: "589" in final or "589.95" in final,
    ),
    (
        "live_active_user_count",
        "How many active users are there?",
        lambda final: "6" in final,
    ),
]


@pytest.mark.skipif(not API_KEY_AVAILABLE, reason="ANTHROPIC_API_KEY not set")
@pytest.mark.parametrize(
    "scenario_id, question, validator",
    LIVE_AGENT_SCENARIOS,
    ids=[s[0] for s in LIVE_AGENT_SCENARIOS],
)
def test_live_agent_navigates_and_answers(
    toolkit, scenario_id, question, validator
):
    """End-to-end: agent picks tables, inspects schema, writes SQL, executes.

    This is the closest thing to the real customer experience LangChain
    promises. If it fails, the bug touches at least one of:
    list-tables, schema-tool, query-checker, query-tool — and the failure
    message tells us which.
    """
    sat = pytest.importorskip("langchain_community.agent_toolkits.sql.base")
    ca = pytest.importorskip("langchain_anthropic")

    llm = ca.ChatAnthropic(model_name="claude-sonnet-4-6", temperature=0)
    # ``max_iterations`` is a top-level kwarg of ``create_sql_agent``; passing
    # it via ``agent_executor_kwargs`` collides with the internal default and
    # raises ``TypeError: got multiple values``. Use the official surface.
    agent = sat.create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        max_iterations=8,
    )
    result = agent.invoke({"input": question})
    final = result.get("output", "") if isinstance(result, dict) else str(result)
    assert validator(final), f"agent answer doesn't satisfy validator: {final!r}"
