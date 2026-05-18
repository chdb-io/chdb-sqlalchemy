"""Engine-backed dialect tests — requires chDB installed.

These exercise the dialect end-to-end:
* engine creation
* trivial round-trip query
* DDL + insert + select
* reflection on a real table

Skipped automatically when chDB is not importable (see ``conftest.py``).
"""

from __future__ import annotations

import pytest
from sqlalchemy import MetaData, Table, inspect, text


def test_engine_select_one(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        assert result == 1


def test_engine_select_version(engine):
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
        assert isinstance(version, str)
        assert len(version) > 0


def test_create_table_and_insert(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE tbl_simple (id UInt32, name String) "
                "ENGINE = MergeTree ORDER BY id"
            )
        )
        conn.execute(text("INSERT INTO tbl_simple VALUES (1, 'alpha'), (2, 'beta')"))
        rows = conn.execute(text("SELECT id, name FROM tbl_simple ORDER BY id")).fetchall()
        assert rows == [(1, "alpha"), (2, "beta")]


def test_reflection_lists_tables(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE reflect_test (a UInt8, b String) ENGINE = MergeTree ORDER BY a"
            )
        )
    insp = inspect(engine)
    names = insp.get_table_names()
    assert "reflect_test" in names


def test_reflection_columns_round_trip(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE typed (
                    id UInt64,
                    name LowCardinality(String),
                    tags Array(String),
                    metric Nullable(Float64),
                    ts DateTime64(3, 'UTC')
                ) ENGINE = MergeTree ORDER BY id
                """
            )
        )
    insp = inspect(engine)
    cols = {c["name"]: c for c in insp.get_columns("typed")}
    assert set(cols) == {"id", "name", "tags", "metric", "ts"}
    assert cols["metric"]["nullable"] is True
    # LowCardinality is captured in `info`, not in the type itself
    lc_flag = cols["name"]["info"]["chdb_low_cardinality"]
    assert lc_flag is True


def test_foreign_keys_always_empty(engine):
    """LangChain SQLDatabaseToolkit calls get_foreign_keys; must not raise."""
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE fk_test (id UInt32) ENGINE = MergeTree ORDER BY id")
        )
    insp = inspect(engine)
    assert insp.get_foreign_keys("fk_test") == []


def test_pk_constraint_from_sorting_key(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE pk_test (a UInt32, b UInt32, c String) "
                "ENGINE = MergeTree ORDER BY (a, b)"
            )
        )
    insp = inspect(engine)
    pk = insp.get_pk_constraint("pk_test")
    assert pk["constrained_columns"] == ["a", "b"]
