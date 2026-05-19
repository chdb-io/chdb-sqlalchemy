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
from typing import Any, ClassVar

from sqlalchemy.engine import default
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import sqltypes as _sqltypes

from .compiler import ChdbDDLCompiler, ChdbSQLCompiler, ChdbTypeCompiler
from .connector import import_dbapi, url_to_connect_args
from .reflection import ChdbReflection
from .types import (
    UUID as _ChDUUID,
)
from .types import (
    Boolean as _ChDBoolean,
)
from .types import (
    Date as _ChDDate,
)
from .types import (
    DateTime as _ChDDateTime,
)
from .types import (
    Time as _ChDTime,
)


class ChdbDialect(default.DefaultDialect):
    """SQLAlchemy dialect for chDB.

    Used via ``create_engine('chdb:///:memory:')`` or
    ``create_engine('chdb:////absolute/path')``.
    """

    name = "chdb"
    driver = "dbapi"

    # colspecs — map SA generic types to our chDB-specific implementations.
    # This makes ``Column(Date)`` / ``literal(date.today())`` use our
    # ``Date`` class (which knows ``toDate('...')`` literal rendering),
    # rather than SA's generic Date (which would render bare ``'YYYY-MM-DD'``
    # as a String literal that chDB can't compare to a Date column).
    colspecs: ClassVar[dict] = {
        _sqltypes.Date: _ChDDate,
        _sqltypes.DateTime: _ChDDateTime,
        _sqltypes.Time: _ChDTime,
        _sqltypes.Boolean: _ChDBoolean,
        _sqltypes.Uuid: _ChDUUID,
    }

    # Compiler classes — these turn SQLAlchemy expression trees back into
    # ClickHouse-flavoured SQL. Critical for LangChain's get_table_info(),
    # which rebuilds CREATE TABLE from reflected metadata to feed the LLM.
    type_compiler_cls = ChdbTypeCompiler
    ddl_compiler = ChdbDDLCompiler
    statement_compiler = ChdbSQLCompiler

    # chdb.dbapi supports both 'format' (%s) and 'qmark' (?) paramstyles.
    # We pick 'qmark' so SQLAlchemy's ``render_literal_value`` doesn't
    # double-up literal ``%`` characters (which would otherwise show up
    # in stored comments and column data). chdb.dbapi's _format_query
    # parses both styles transparently.
    paramstyle = "qmark"

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
    # moment a JSON column is read.
    #
    # Our cursor wrapper already parses JSON cells from Python-repr strings
    # to native dict/list (see ``_cursor.py``). SA's JSON.result_processor
    # then calls ``_json_deserializer(value)`` again — which would fail
    # with ``TypeError`` if value is already a dict. We make the
    # deserializer idempotent: pass-through for already-parsed values,
    # fallback to ``json.loads`` for raw strings.
    _json_serializer = staticmethod(json.dumps)

    @staticmethod
    def _json_deserializer(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list, bool, int, float)):
            return value  # cursor wrapper already parsed it
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        return value

    @classmethod
    def import_dbapi(cls) -> Any:
        """Return the DB-API 2.0 module SQLAlchemy should use.

        Called once per dialect class by SQLAlchemy. We return the
        ``chdb.dbapi`` module — see ``connector.import_dbapi``.
        """
        return import_dbapi()

    @classmethod
    def load_provisioning(cls) -> None:
        """Load ``chdb_sqlalchemy.provision`` so SA's testing harness can
        find our ``post_configure_engine`` / ``drop_all_schema_objects_*``
        hooks. Called by SA's :func:`provision.setup_config` during
        session start of the dialect compliance suite.
        """
        __import__("chdb_sqlalchemy.provision")

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

    def do_execute(
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
            db_error_cls: type = getattr(dbapi_mod, "DatabaseError", None) or getattr(
                dbapi_mod, "Error", Exception
            ) or Exception
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

    def get_table_comment(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> dict[str, Any]:
        return self._reflection.get_table_comment(connection, table_name, schema)

    def get_view_definition(
        self, connection: Any, view_name: str, schema: str | None = None, **kw: Any
    ) -> str:
        return self._reflection.get_view_definition(connection, view_name, schema)

    def get_materialized_view_names(
        self, connection: Any, schema: str | None = None, **kw: Any
    ) -> list[str]:
        return self._reflection.get_materialized_view_names(connection, schema)

    def get_temp_view_names(
        self, connection: Any, schema: str | None = None, **kw: Any
    ) -> list[str]:
        return self._reflection.get_temp_view_names(connection, schema)

    def get_check_constraints(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        return self._reflection.get_check_constraints(connection, table_name, schema)

    def get_unique_constraints(
        self, connection: Any, table_name: str, schema: str | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        return self._reflection.get_unique_constraints(connection, table_name, schema)
