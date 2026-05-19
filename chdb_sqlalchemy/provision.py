"""SQLAlchemy testing provision hooks for chDB.

SQLAlchemy's official compliance suite (``sqlalchemy.testing.suite``)
uses a small handful of "test databases" / "test schemas" that the
dialect is expected to provision. Postgres/MySQL provision them at
test session start via ``CREATE DATABASE test_schema`` etc.; chDB
needs to do the same for ``CREATE TABLE test_schema.foo`` to work
during ComponentReflectionTest et al.

We hook into ``post_configure_engine`` (fires once per engine at
session start) to ensure the test databases exist. Drop hooks
clean up between test classes.

Loading: SA's ``provision.setup_config`` calls
``dialect.load_provisioning()`` which by convention imports
``<dialect_module>.provision``. Our :mod:`chdb_sqlalchemy.dialect`
exposes that via ``ChdbDialect.load_provisioning``.
"""

from __future__ import annotations

import contextlib
from typing import Any

from sqlalchemy.testing.provision import (
    delete_from_all_tables,
    drop_all_schema_objects_post_tables,
    drop_all_schema_objects_pre_tables,
    post_configure_engine,
    set_default_schema_on_connection,
    temp_table_keyword_args,
)

# These are the schema names SA tests use. See
# sqlalchemy.testing.schema.eq_clause_set_schema and similar.
TEST_SCHEMA = "test_schema"
TEST_SCHEMA_2 = "test_schema_2"


@post_configure_engine.for_db("chdb")
def _post_configure_engine(url: Any, engine: Any, follower_ident: Any) -> None:
    """Create the test schemas (= ClickHouse databases) the SA suite needs.

    Also set ``union_default_mode = 'DISTINCT'`` so bare ``UNION`` (which
    SA emits) becomes ``UNION DISTINCT`` — ClickHouse rejects bare UNION
    otherwise.
    """
    with engine.connect() as conn:
        conn.exec_driver_sql(f"CREATE DATABASE IF NOT EXISTS {TEST_SCHEMA}")
        conn.exec_driver_sql(f"CREATE DATABASE IF NOT EXISTS {TEST_SCHEMA_2}")
        with contextlib.suppress(Exception):
            conn.exec_driver_sql("SET union_default_mode = 'DISTINCT'")
        conn.commit() if hasattr(conn, "commit") else None


@drop_all_schema_objects_pre_tables.for_db("chdb")
def _drop_all_schema_objects_pre_tables(cfg: Any, eng: Any) -> None:
    """Drop tables/views inside test schemas before the suite re-creates them.

    chDB is in-process and process-global — state leaks between test
    classes unless we proactively clean. We DROP every table in each
    test schema then re-create the empty schemas.
    """
    with eng.connect() as conn:
        for db in (TEST_SCHEMA, TEST_SCHEMA_2, "default"):
            # Drop the test schemas entirely then re-create — much
            # simpler than enumerating tables. The 'default' schema is
            # special; for it, only drop our test tables.
            if db == "default":
                # Best-effort: query system.tables for tables in default
                # and drop them. Skip on failure.
                try:
                    rows = conn.exec_driver_sql(
                        "SELECT name FROM system.tables "
                        "WHERE database = currentDatabase() "
                        "AND name NOT LIKE 'system.%'"
                    ).fetchall()
                    for (name,) in rows:
                        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {name}")
                except Exception:
                    pass
                continue
            try:
                conn.exec_driver_sql(f"DROP DATABASE IF EXISTS {db}")
                conn.exec_driver_sql(f"CREATE DATABASE {db}")
            except Exception:
                pass


@drop_all_schema_objects_post_tables.for_db("chdb")
def _drop_all_schema_objects_post_tables(cfg: Any, eng: Any) -> None:
    """Mirror of pre-tables hook, runs after suite teardown."""
    _drop_all_schema_objects_pre_tables(cfg, eng)


@set_default_schema_on_connection.for_db("chdb")
def _set_default_schema(cfg: Any, dbapi_connection: Any, schema: str) -> None:
    """Switch the default schema for a connection.

    ClickHouse: ``USE <db>`` — but chDB's in-process model doesn't strictly
    obey it. The cleanest path is to set the ``database`` query param at
    connect time, but for a *running* connection we issue ``USE`` and hope
    chDB respects it for subsequent statements on that session.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(f"USE {schema}")
    finally:
        cursor.close()


@delete_from_all_tables.for_db("chdb")
def _delete_from_all_tables(connection: Any, cfg: Any, metadata: Any) -> None:
    """Clean rows from every table between SA test cases.

    ClickHouse doesn't accept standalone ``DELETE FROM t``; it parses
    that as ``SELECT DELETE FROM t``. Use ``TRUNCATE TABLE`` which
    chDB supports for MergeTree-family engines, or fall back to
    ``ALTER TABLE ... DELETE WHERE 1=1`` (works on all engines but slower).

    Signature matches SA's default: ``(connection, cfg, metadata)``.
    """
    for table in metadata.tables.values():
        full = (
            f"{table.schema}.{table.name}" if table.schema else table.name
        )
        try:
            connection.exec_driver_sql(f"TRUNCATE TABLE {full}")
        except Exception:
            with contextlib.suppress(Exception):
                connection.exec_driver_sql(f"ALTER TABLE {full} DELETE WHERE 1=1")


@temp_table_keyword_args.for_db("chdb")
def _temp_table_keyword_args(cfg: Any, eng: Any) -> dict[str, Any]:
    """Extra kwargs SA's ``Table()`` should receive for temp tables.

    ClickHouse temp-table semantics differ from generic SQL; the SA
    fixture passes ``prefixes=['TEMPORARY']`` by default. We return an
    empty dict so SA falls back to its own defaults (which our
    Requirements.temporary_tables=closed() already gates).
    """
    return {}
