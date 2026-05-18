"""The ``ChdbDialect`` — SQLAlchemy's view of chDB.

This dialect:

* Reuses SQLAlchemy's default compiler/DDL emission for ClickHouse-flavoured SQL.
  chDB executes ClickHouse SQL verbatim, so query rewriting is rarely needed.
* Delegates DB-API concerns (cursor, paramstyle, escape, base value conversion)
  to ``chdb.dbapi`` and ``chdb.dbapi.converters`` — see Part 0 of the
  implementation plan.
* Implements introspection (``get_table_names`` / ``get_columns`` / etc.) by
  querying ``system.tables`` and ``system.columns`` directly, because
  ``chdb.dbapi``'s ``cursor.description`` collapses ClickHouse composite types
  into MySQL-flavoured FIELD_TYPE codes that lose Array/Map/Variant/Nullable
  information.

For v0.1 we ship the minimum surface that makes ``SQLDatabase.from_uri()`` and
``pandas.read_sql()`` work. v0.2 adds the full LangChain ``SQLDatabaseToolkit``
and CrewAI ``NL2SQLTool`` certification.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.engine import default
from sqlalchemy.engine.url import URL

from .compiler import ChdbDDLCompiler, ChdbTypeCompiler
from .connector import import_dbapi, url_to_connect_args
from .reflection import ChdbReflection


class ChdbDialect(default.DefaultDialect):
    """SQLAlchemy dialect for chDB.

    Used via ``create_engine('chdb:///:memory:')`` or
    ``create_engine('chdb:////absolute/path')``.
    """

    name = "chdb"
    driver = "dbapi"

    # Compiler classes — these turn SQLAlchemy expression trees back into
    # ClickHouse-flavoured SQL. Critical for LangChain's get_table_info(),
    # which rebuilds CREATE TABLE from reflected metadata to feed the LLM.
    type_compiler_cls = ChdbTypeCompiler
    ddl_compiler = ChdbDDLCompiler

    # chdb.dbapi declares paramstyle='format' (i.e. %s placeholders).
    # SQLAlchemy's compiler will emit positional placeholders accordingly.
    paramstyle = "format"

    # chDB executes locally; we don't have a real connection pool in the
    # network-driver sense. SQLAlchemy's StaticPool / NullPool is a better
    # fit, but we leave the default and let dispose() clean up.
    supports_statement_cache = True

    # ClickHouse SQL features chDB supports:
    supports_native_decimal = True
    supports_native_boolean = True
    supports_native_uuid = True

    # ClickHouse SQL features chDB / ClickHouse don't support:
    supports_alter = True            # ALTER TABLE works for most engines
    supports_sequences = False
    supports_native_enum = True
    supports_default_values = False
    supports_empty_insert = False
    supports_multivalues_insert = True
    supports_comments = True
    supports_constraint_comments = False

    # No transactions in chDB.
    supports_sane_rowcount = False
    supports_sane_multi_rowcount = False

    # Case sensitivity matches ClickHouse: identifiers are case-sensitive,
    # keywords are case-insensitive.
    requires_name_normalize = False

    # Server-side cursors aren't a thing in-process.
    supports_server_side_cursors = False

    # Default schema name in chDB is the current database; we resolve dynamically.
    default_schema_name: str | None = None

    # JSON (de)serialisation hooks — SQLAlchemy's JSON type calls these via
    # ``dialect._json_serializer`` / ``dialect._json_deserializer`` when
    # binding / processing JSON-typed columns. Without them SA crashes the
    # moment a JSON column is read. chDB emits JSON values as already-parsed
    # Python objects in most cases, but defensive ``json.loads`` handles
    # the legacy string-backed path.
    _json_serializer = staticmethod(json.dumps)
    _json_deserializer = staticmethod(json.loads)

    @classmethod
    def import_dbapi(cls) -> Any:  # type: ignore[override]
        """Return the DB-API 2.0 module SQLAlchemy should use.

        Called once per dialect class by SQLAlchemy. We return the
        ``chdb.dbapi`` module — see ``connector.import_dbapi``.
        """
        return import_dbapi()

    # Older SQLAlchemy 1.4 entry point — keep for compatibility.
    @classmethod
    def dbapi(cls) -> Any:  # pragma: no cover - SA<2.0 shim
        return cls.import_dbapi()

    def create_connect_args(self, url: URL) -> tuple[list[Any], dict[str, Any]]:
        """Translate ``chdb:///...`` URL into ``chdb.dbapi.connect`` kwargs."""
        return [], url_to_connect_args(url)

    def do_ping(self, dbapi_connection: Any) -> bool:
        """Cheap health check used by connection-pool pre-ping."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        finally:
            cursor.close()
        return True

    def do_execute(  # type: ignore[override]
        self,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any = None,
    ) -> None:
        """Translate chDB's bare ``Exception`` raises into PEP 249 form.

        ``chdb.dbapi.Cursor.execute`` (chdb 4.x) raises a bare ``Exception``
        with a ``"Code: NN. DB::Exception: ..."`` message instead of the
        ``DatabaseError`` PEP 249 mandates. SQLAlchemy's
        ``_handle_dbapi_exception`` only wraps subclasses of ``dbapi.Error``,
        so without translation the error escapes raw and LangChain's
        ``SQLDatabase.run_no_throw`` can't catch it.

        We re-raise as ``chdb.dbapi.err.DatabaseError`` (which our patched
        ``chdb.dbapi.DatabaseError`` alias references) so the rest of the
        SQLAlchemy / LangChain pipeline sees a well-typed exception.
        """
        try:
            cursor.execute(statement, parameters)
        except Exception as e:
            dbapi_mod = self.loaded_dbapi if hasattr(self, "loaded_dbapi") else self.dbapi
            db_error_cls = getattr(dbapi_mod, "DatabaseError", None) or getattr(
                dbapi_mod, "Error", Exception
            )
            if isinstance(e, db_error_cls):
                raise
            # chDB raises ``Exception`` directly — translate.
            raise db_error_cls(str(e)) from e

    # ------------------------------------------------------------------
    # Introspection delegate
    # ------------------------------------------------------------------
    #
    # All ``get_*`` methods live on ChdbReflection so the dialect class
    # stays focused on SQLAlchemy plumbing. We mix the reflection methods
    # in below via __init_subclass__-friendly composition.

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reflection = ChdbReflection(self)

    # The SQLAlchemy reflection contract is method-based: SQLAlchemy calls
    # ``dialect.get_table_names(connection, schema)`` etc. We forward each
    # to the reflection delegate. This keeps reflection unit-testable
    # without needing a live dialect.

    def get_schema_names(self, connection: Any, **kw: Any) -> list[str]:
        return self._reflection.get_schema_names(connection)

    def get_table_names(
        self, connection: Any, schema: str | None = None, **kw: Any
    ) -> list[str]:
        return self._reflection.get_table_names(connection, schema)

    def get_view_names(
        self, connection: Any, schema: str | None = None, **kw: Any
    ) -> list[str]:
        return self._reflection.get_view_names(connection, schema)

    def get_temp_table_names(
        self, connection: Any, schema: str | None = None, **kw: Any
    ) -> list[str]:
        return self._reflection.get_temp_table_names(connection, schema)

    def get_columns(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        return self._reflection.get_columns(connection, table_name, schema)

    def get_pk_constraint(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> dict[str, Any]:
        return self._reflection.get_pk_constraint(connection, table_name, schema)

    def get_foreign_keys(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        # ClickHouse / chDB don't have foreign keys. Returning [] (never raising)
        # is part of the LangChain SQLDatabaseToolkit introspection contract.
        return []

    def get_indexes(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        return self._reflection.get_indexes(connection, table_name, schema)

    def has_table(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> bool:
        return self._reflection.has_table(connection, table_name, schema)
