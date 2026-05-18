"""L4 — End-to-end LangChain ``SQLDatabaseToolkit`` tests with a real LLM.

Strategy: use ``vcrpy`` to record real Anthropic API calls into yaml
cassettes the first time the test runs. Subsequent CI runs replay from
cassettes (no API key, no quota usage). To re-record after a prompt or
schema change, run with::

    pytest tests/integration/ --vcr-record=new_episodes

Each test corresponds to a concrete agent capability we promise in the
README: "ask in English, get an answer over a real schema." If any of
these regress, the LangChain anchor story is broken.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

# Scenario list — kept in this file as a single source of truth so adding
# a new agent capability requires adding both the scenario and a cassette.
SCENARIOS = [
    "simple_count_group_by",
    "join_across_two_tables",
    "array_column_array_join",
    "map_column_key_access",
    "time_window_filter",
    "nullable_column_handling",
    "wide_schema_100_columns",
    "variant_column_extract",
    "json_path_access",
    "list_all_tables",
]


@pytest.fixture
def llm():
    """Real Anthropic LLM. Tests are vcr-cassette-backed."""
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model_name="claude-sonnet-4-6", temperature=0)


@pytest.fixture
def toolkit(engine, llm):
    pytest.importorskip("langchain_community")
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit

    db = SQLDatabase(engine)
    return SQLDatabaseToolkit(db=db, llm=llm)


# Per-scenario tests are stubbed below with xfail so the structure is visible
# in CI today. Each will be activated as the corresponding scenario fixture
# (test data + cassette) lands.

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_placeholder(scenario):
    """Placeholder until L4 scenarios are filled in (v0.2 milestone).

    See ``docs/L4-scenarios.md`` (TODO) for the per-scenario fixture spec.
    """
    pytest.xfail(f"L4 scenario {scenario!r} pending v0.2 implementation")
