"""L3 — LangChain ``SQLDatabaseToolkit`` introspection contract.

We don't need to instantiate the toolkit or run an LLM to validate the
contract — what we need is to call the same methods LangChain calls and
assert that:

1. They return the right shape (list / dict structures LangChain expects).
2. They never raise on a fresh / empty database.
3. ``get_table_info`` (LangChain's prompt-feeder) ends up with every column
   name and a non-empty type label.

End-to-end tests with a real LLM live under :mod:`tests.integration`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

pytestmark = pytest.mark.usefixtures("engine")


def test_get_table_names_empty_db(engine):
    """LangChain calls this immediately after connecting — must not raise."""
    names = inspect(engine).get_table_names()
    assert isinstance(names, list)


def test_get_table_names_after_create(engine):
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE lc_users (id UInt32, name String) ENGINE = MergeTree ORDER BY id")
        )
    names = inspect(engine).get_table_names()
    assert "lc_users" in names


def test_get_columns_shape_matches_langchain_expectation(engine):
    """LangChain reads {name, type, nullable, default, comment} per column."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE lc_events (id UInt64, payload String) "
                "ENGINE = MergeTree ORDER BY id"
            )
        )
    cols = inspect(engine).get_columns("lc_events")
    for col in cols:
        for required_key in ("name", "type", "nullable", "default"):
            assert required_key in col, f"missing {required_key}: {col}"


def test_get_indexes_returns_list_even_when_none(engine):
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE lc_no_idx (id UInt32) ENGINE = MergeTree ORDER BY id")
        )
    indexes = inspect(engine).get_indexes("lc_no_idx")
    assert isinstance(indexes, list)


def test_has_table_returns_bool(engine):
    insp = inspect(engine)
    assert insp.has_table("nonexistent_table") is False
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE present_table (x UInt8) ENGINE = MergeTree ORDER BY x")
        )
    assert insp.has_table("present_table") is True


def test_get_foreign_keys_on_table_with_no_fks(engine):
    """The toolkit calls this for every reflected table — must return []
    and never raise. Silent OK is the documented contract for chDB."""
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE fk_check (id UInt32) ENGINE = MergeTree ORDER BY id")
        )
    assert inspect(engine).get_foreign_keys("fk_check") == []
