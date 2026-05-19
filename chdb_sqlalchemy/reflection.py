"""Reflection (``get_*``) methods backing the ``ChdbDialect``.

Why this module exists separately from :mod:`dialect`:

* All introspection in chDB happens via ``system.tables`` / ``system.columns`` /
  ``system.data_skipping_indices`` SQL queries — there is no native metadata
  protocol. Keeping that code together makes it easier to audit and patch.
* The LangChain / CrewAI introspection contracts can be exercised directly
  against ``ChdbReflection`` without spinning up a SQLAlchemy engine, which
  speeds up unit tests considerably.

Per Part 0 of the plan: we **must not** use ``cursor.description`` for type
recovery — ``chdb.dbapi.FIELD_TYPE`` collapses every ClickHouse composite into
MySQL-flavoured codes. We query ``system.columns.type`` instead and parse the
returned string with :mod:`chdb_sqlalchemy.types.parser`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import exc as sa_exc
from sqlalchemy import text

from .types.parser import parse_column_type

if TYPE_CHECKING:
    from .dialect import ChdbDialect


def _split_sorting_key(sorting_key: str) -> list[str]:
    """Split a ClickHouse ``ORDER BY`` / sorting-key expression list by commas.

    The naive ``split(",")`` breaks on function-call expressions:
    ``cityHash64(u, s), ts`` would yield ``['cityHash64(u', ' s)', ' ts']``.
    Track paren depth and quote state so commas inside ``(...)`` /
    ``'...'`` are not treated as separators.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_squote = False
    in_bquote = False
    i = 0
    while i < len(sorting_key):
        ch = sorting_key[i]
        if in_squote:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(sorting_key):
                buf.append(sorting_key[i + 1])
                i += 2
                continue
            if ch == "'":
                in_squote = False
        elif in_bquote:
            buf.append(ch)
            if ch == "`":
                in_bquote = False
        elif ch == "'":
            in_squote = True
            buf.append(ch)
        elif ch == "`":
            in_bquote = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_check_constraints(create_table_query: str) -> list[dict[str, Any]]:
    """Extract CHECK constraints from a ``CREATE TABLE ...`` text.

    Scans the column-list parenthesis for ``CONSTRAINT <name> CHECK <expr>``
    items, tracking paren depth + quote state so commas inside function
    calls / string literals don't split clauses.

    Returns SA's ``[{'name': str, 'sqltext': str}, ...]`` shape; if no
    CHECK constraints are present (or the column list can't be located)
    returns ``[]``.
    """
    open_idx = create_table_query.find("(")
    if open_idx == -1:
        return []
    depth = 0
    in_squote = False
    in_bquote = False
    end_idx = -1
    for i in range(open_idx, len(create_table_query)):
        ch = create_table_query[i]
        if in_squote:
            if ch == "\\" and i + 1 < len(create_table_query):
                continue
            if ch == "'":
                in_squote = False
        elif in_bquote:
            if ch == "`":
                in_bquote = False
        elif ch == "'":
            in_squote = True
        elif ch == "`":
            in_bquote = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx == -1:
        return []
    body = create_table_query[open_idx + 1 : end_idx]

    # Split at top-level commas only (same logic as _split_sorting_key,
    # generalised over the column-list body).
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_squote = False
    in_bquote = False
    i = 0
    while i < len(body):
        ch = body[i]
        if in_squote:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(body):
                buf.append(body[i + 1])
                i += 2
                continue
            if ch == "'":
                in_squote = False
        elif in_bquote:
            buf.append(ch)
            if ch == "`":
                in_bquote = False
        elif ch == "'":
            in_squote = True
            buf.append(ch)
        elif ch == "`":
            in_bquote = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())

    # Pick out the CONSTRAINT ... CHECK ... clauses.
    out: list[dict[str, Any]] = []
    for part in parts:
        if not part.upper().startswith("CONSTRAINT "):
            continue
        # Form: CONSTRAINT <name> CHECK <expr>
        rest = part[len("CONSTRAINT "):].lstrip()
        # Name: identifier (possibly backtick-quoted) up to whitespace.
        if rest.startswith("`"):
            close = rest.find("`", 1)
            if close == -1:
                continue
            name = rest[1:close]
            rest = rest[close + 1 :].lstrip()
        else:
            tokens = rest.split(None, 1)
            if len(tokens) < 2:
                continue
            name, rest = tokens[0], tokens[1].lstrip()
        if not rest.upper().startswith("CHECK"):
            continue
        sqltext = rest[len("CHECK"):].strip()
        out.append({"name": name, "sqltext": sqltext})
    return out


class ChdbReflection:
    """Pulls schema metadata out of chDB's ``system.*`` tables."""

    def __init__(self, dialect: ChdbDialect) -> None:
        self.dialect = dialect

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _db(schema: str | None) -> str:
        """Return a SQL expression for the target ClickHouse database name.

        When ``schema`` is None we emit the ``currentDatabase()`` function
        call (so reflection works without an explicit schema). When it's
        a real name we emit it as a single-quoted string literal — bare
        identifier would parse as a column reference and break with
        ``UNKNOWN_IDENTIFIER``. Quote-escape any embedded apostrophe
        defensively.
        """
        if schema is None:
            return "currentDatabase()"
        escaped = schema.replace("'", "''")
        return f"'{escaped}'"

    def _exec(self, connection: Any, sql: str, **params: Any) -> list[tuple]:
        result = connection.execute(text(sql), params)
        return list(result.fetchall())

    def _require_table(self, connection: Any, table_name: str, schema: str | None) -> None:
        """Raise ``NoSuchTableError`` if table does not exist.

        SA expects ``get_columns(missing_table)`` etc. to raise. We probe
        ``system.tables`` and raise the standard exception.
        """
        if not self.has_table(connection, table_name, schema):
            raise sa_exc.NoSuchTableError(
                f"{schema}.{table_name}" if schema else table_name
            )

    # ------------------------------------------------------------------
    # schemas / databases
    # ------------------------------------------------------------------

    def get_schema_names(self, connection: Any) -> list[str]:
        rows = self._exec(connection, "SELECT name FROM system.databases ORDER BY name")
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # tables / views
    # ------------------------------------------------------------------

    def get_table_names(self, connection: Any, schema: str | None) -> list[str]:
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"""
            SELECT name FROM system.tables
            WHERE database = {db}
              AND engine NOT LIKE '%View'
              AND is_temporary = 0
            ORDER BY name
            """,
        )
        return [r[0] for r in rows]

    def get_view_names(self, connection: Any, schema: str | None) -> list[str]:
        """Return plain views — *not* materialized views.

        SA's Inspector contract: ``get_view_names`` and
        ``get_materialized_view_names`` are disjoint. We previously used
        ``engine LIKE '%View'`` which matched both ``View`` and
        ``MaterializedView``, double-listing MVs across the two methods.
        """
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"""
            SELECT name FROM system.tables
            WHERE database = {db}
              AND engine = 'View'
            ORDER BY name
            """,
        )
        return [r[0] for r in rows]

    def get_temp_table_names(self, connection: Any, schema: str | None) -> list[str]:
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"""
            SELECT name FROM system.tables
            WHERE database = {db}
              AND is_temporary = 1
            ORDER BY name
            """,
        )
        return [r[0] for r in rows]

    def has_table(self, connection: Any, table_name: str, schema: str | None) -> bool:
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"SELECT count() FROM system.tables WHERE database = {db} AND name = :name",
            name=table_name,
        )
        return bool(rows and rows[0][0])

    # ------------------------------------------------------------------
    # columns
    # ------------------------------------------------------------------

    def get_columns(
        self, connection: Any, table_name: str, schema: str | None
    ) -> list[dict[str, Any]]:
        """Return SQLAlchemy column metadata for ``table_name``."""
        self._require_table(connection, table_name, schema)
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"""
            SELECT
                name,
                type,
                default_kind,
                default_expression,
                comment
            FROM system.columns
            WHERE database = {db}
              AND table = :table
            ORDER BY position
            """,
            table=table_name,
        )

        out: list[dict[str, Any]] = []
        for name, type_str, default_kind, default_expr, comment in rows:
            parsed = parse_column_type(type_str)
            col: dict[str, Any] = {
                "name": name,
                "type": parsed.sa_type,
                "nullable": parsed.nullable,
                "default": default_expr or None,
                "comment": comment or None,
                # SQLAlchemy's `info` dict survives reflection and is the
                # idiomatic place for dialect-specific metadata that the
                # core Column class doesn't know about. Keys are deliberately
                # namespaced under 'chdb_' to avoid collisions with other
                # consumers of `Column.info`.
                "info": {
                    "chdb_low_cardinality": parsed.low_cardinality,
                    "chdb_default_kind": default_kind or None,
                },
            }
            out.append(col)
        return out

    # ------------------------------------------------------------------
    # primary key (de facto: MergeTree ORDER BY)
    # ------------------------------------------------------------------

    def get_pk_constraint(
        self, connection: Any, table_name: str, schema: str | None
    ) -> dict[str, Any]:
        """Return the table's sorting key as a faux PK."""
        self._require_table(connection, table_name, schema)
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"""
            SELECT sorting_key FROM system.tables
            WHERE database = {db} AND name = :table
            """,
            table=table_name,
        )
        if not rows or not rows[0][0]:
            return {"constrained_columns": [], "name": None}
        sorting_key = rows[0][0]
        cols = _split_sorting_key(sorting_key)
        return {"constrained_columns": cols, "name": None}

    # ------------------------------------------------------------------
    # indexes (data-skipping)
    # ------------------------------------------------------------------

    def get_table_comment(
        self, connection: Any, table_name: str, schema: str | None
    ) -> dict[str, Any]:
        """Return the table's comment as ``{'text': str | None}``."""
        self._require_table(connection, table_name, schema)
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"SELECT comment FROM system.tables WHERE database = {db} AND name = :name",
            name=table_name,
        )
        if not rows:
            return {"text": None}
        return {"text": rows[0][0] or None}

    def get_view_definition(
        self, connection: Any, view_name: str, schema: str | None
    ) -> str:
        """Return the SQL for a view."""
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"SELECT create_table_query FROM system.tables "
            f"WHERE database = {db} AND name = :name",
            name=view_name,
        )
        if not rows or not rows[0][0]:
            raise __import__("sqlalchemy").exc.NoSuchTableError(view_name)
        return rows[0][0]

    def get_materialized_view_names(
        self, connection: Any, schema: str | None
    ) -> list[str]:
        """Return MaterializedView names — separate from regular views."""
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"SELECT name FROM system.tables "
            f"WHERE database = {db} AND engine = 'MaterializedView' "
            f"ORDER BY name",
        )
        return [r[0] for r in rows]

    def get_temp_view_names(
        self, connection: Any, schema: str | None
    ) -> list[str]:
        """Temp views — chDB doesn't really do these; return empty."""
        return []

    def get_check_constraints(
        self, connection: Any, table_name: str, schema: str | None
    ) -> list[dict[str, Any]]:
        """Return CHECK constraints declared on the table.

        chDB enforces ``CHECK`` constraints (error code 469
        VIOLATED_CONSTRAINT on insert), but doesn't expose a
        ``system.constraints`` table — the only way to recover them is
        to parse ``system.tables.create_table_query``. The format is::

            CREATE TABLE db.tbl (`x` Int32, CONSTRAINT positive CHECK x > 0, ...)

        We pull the column-list parenthesis group, scan for
        ``CONSTRAINT <name> CHECK <expr>`` items at paren-depth 0, and
        return SA's standard ``{'name': str, 'sqltext': str}`` shape.
        """
        self._require_table(connection, table_name, schema)
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"SELECT create_table_query FROM system.tables WHERE database = {db} AND name = :name",
            name=table_name,
        )
        if not rows or not rows[0][0]:
            return []
        return _parse_check_constraints(rows[0][0])

    def get_unique_constraints(
        self, connection: Any, table_name: str, schema: str | None
    ) -> list[dict[str, Any]]:
        """ClickHouse has no UNIQUE; return empty."""
        return []

    def get_indexes(
        self, connection: Any, table_name: str, schema: str | None
    ) -> list[dict[str, Any]]:
        """Return data-skipping indexes declared on the table."""
        self._require_table(connection, table_name, schema)
        db = self._db(schema)
        rows = self._exec(
            connection,
            f"""
            SELECT name, type, expr
            FROM system.data_skipping_indices
            WHERE database = {db} AND table = :table
            ORDER BY name
            """,
            table=table_name,
        )
        out: list[dict[str, Any]] = []
        for idx_name, idx_type, expr in rows:
            out.append(
                {
                    "name": idx_name,
                    "column_names": [expr],  # raw expression — not always a single column
                    "unique": False,
                    "dialect_options": {"chdb": {"index_type": idx_type}},
                }
            )
        return out
