"""Cursor wrapper that repairs chDB's lossy result-value formatting.

Why this layer exists:

``chdb.dbapi`` (chdb 4.x) returns several categories of values in
non-native forms; the exact set depends on the chdb-core binary:

* ``Array(T)`` / ``Map(K, V)`` / ``Tuple(...)`` / ``JSON`` cells come back
  as **Python repr-style strings** (e.g. ``"['a', 'b']"``, ``"{'k': 'v'}"``)
  on every chdb-core version
* chdb-core 26.5+ (ClickHouse 26.5 baseline) additionally quotes Float /
  Decimal / temporal leaves *inside* those composite strings
  (``"['1.5', '2.5']"`` for ``Array(Float64)``) where 26.3 emitted bare
  numerics
* chdb-core ≤26.3 returned scalar ``Decimal(P, S)`` and
  ``Nullable(<numeric>)`` cells as ``str``; 26.5 returns natives — the
  coercers here are no-ops on native input, so both generations work

SQLAlchemy's per-type ``result_processor`` machinery only fires when the
column type is known at compile time. For raw ``text()`` queries — which
is what LangChain's ``SQLDatabase.run`` uses — SA has no type info, so the
processors never run and the LLM agent gets a string where it expected a
number/list/dict.

We close the gap at the cursor layer: every ``fetch*`` reads
``cursor.description`` to learn each column's ClickHouse type and applies
the appropriate coercion. This is transparent to upstream callers (SA, the
LangChain toolkit, pandas, Django, anything PEP 249) and survives every
query path including ``text()``.

The shim is defensive: when chDB ships a fix upstream and starts returning
native types, the ``isinstance(value, str)`` guard makes coercion a no-op.
"""

from __future__ import annotations

import ast
import decimal as _decimal
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

# Type-string prefix → row converter.
# Order matters: more specific prefixes must come before broader ones.
_COMPOSITE_PREFIXES = ("Array(", "Map(", "Tuple(", "Nested(")


# ----------------------------------------------------------------------
# CrewAI NL2SQLTool compatibility shim
# ----------------------------------------------------------------------
#
# CrewAI's ``NL2SQLTool`` (crewAIInc/crewAI-tools
# ``crewai_tools/tools/nl2sql/nl2sql_tool.py``) emits two hardcoded
# PostgreSQL-style introspection queries that misbehave against chDB:
#
# 1. ``_fetch_available_tables()`` (line 48 at time of writing):
#
#        SELECT table_name FROM information_schema.tables
#        WHERE table_schema = 'public';
#
#    chDB has ``information_schema.tables`` but stores user tables under
#    ``table_schema = currentDatabase()``. The literal ``'public'`` filter
#    returns zero rows → agent silently sees empty schema → useless output.
#    **Fix:** rewrite ``'public'`` → ``currentDatabase()`` so the session's
#    own tables are visible.
#
# 2. ``_fetch_all_available_columns(table_name)`` (line 53):
#
#        SELECT column_name, data_type FROM information_schema.columns
#        WHERE table_name = '<name>';
#
#    The query filters only by table name. If two databases each have a
#    table named e.g. ``crew_orders``, chDB's ``information_schema.columns``
#    returns the union of both column sets — the agent's NL2SQL prompt then
#    sees a phantom schema that mixes columns from a sibling database.
#    **Fix:** append ``AND table_schema = currentDatabase()`` to scope to
#    the session's own database.
#
# Design choices for both rewrites:
#
# * ``currentDatabase()`` (not a hardcoded ``'default'``) so the shim
#   follows the session's actual database — including persistent sessions
#   and explicit ``USE other_db`` statements.
#
# * Anchored, whitespace-tolerant regex matching the exact CrewAI query
#   shape, not a broad substitution. This avoids false-positives on user
#   queries that happen to mention ``information_schema``, ``'public'``,
#   or ``column_name`` for unrelated reasons.
#
# * The shim is intentionally upstream-fragile: if CrewAI changes either
#   query shape (re-formats the SQL, adds columns, switches to SA
#   Inspector, etc.) the regex will no longer match and the original
#   silent-failure modes return. The canary test in
#   ``tests/integration/test_crewai_compat.py::test_crewai_query_shape_unchanged``
#   inspects the upstream CrewAI source and fails LOUDLY with a pointer
#   to this file when the pattern drifts.
#
# When that test fails: re-read the new CrewAI source, decide whether to
# update the regexes below (still PG-style → keep shim, adapt pattern) or
# remove the shim entirely (CrewAI now uses SA Inspector → no shim needed,
# just delete this block and the tests).
_CREWAI_PG_PUBLIC_TABLES_QUERY = re.compile(
    r"^\s*SELECT\s+table_name\s+FROM\s+information_schema\.tables\s+"
    r"WHERE\s+table_schema\s*=\s*'public'\s*;?\s*$",
    re.IGNORECASE,
)


# Group 1 captures everything from SELECT through the closing of the
# table_name comparand; group 2 captures any trailing whitespace + optional
# semicolon. The comparand can be:
#
# * a single-quoted literal (older crewai-tools <= ~1.14 inlined the table
#   name via f-string):
#       ``WHERE table_name = 'crew_orders'``
#
# * a chdb.dbapi qmark placeholder (newer crewai-tools uses SA bind params
#   ``:table_name`` which SA converts to ``?`` at this cursor layer):
#       ``WHERE table_name = ?``
#
# Both forms must be rewritten — the cross-database leak exists in either
# case because neither version filters by schema. We insert the schema
# filter between groups 1 and 2 so the result is
# ``... WHERE table_name = <comparand> AND table_schema = currentDatabase();``.
# The bind-parameter array (passed alongside the SQL) is untouched, so its
# ``?`` ↔ value mapping remains valid.
_CREWAI_PG_COLUMNS_QUERY = re.compile(
    r"^(\s*SELECT\s+column_name\s*,\s*data_type\s+FROM\s+information_schema\.columns\s+"
    r"WHERE\s+table_name\s*=\s*(?:'[^']*'|\?))(\s*;?\s*)$",
    re.IGNORECASE,
)


def _rewrite_crewai_public_schema(sql: str) -> str:
    """Rewrite CrewAI's hardcoded PG-style ``'public'`` filter on
    ``information_schema.tables`` to ``currentDatabase()``.

    Returns the SQL unchanged unless it matches the exact CrewAI
    ``_fetch_available_tables()`` query shape.
    """
    if _CREWAI_PG_PUBLIC_TABLES_QUERY.match(sql):
        return sql.replace("'public'", "currentDatabase()", 1)
    return sql


def _rewrite_crewai_columns_filter(sql: str) -> str:
    """Append ``AND table_schema = currentDatabase()`` to CrewAI's
    ``information_schema.columns`` query so same-named tables in sibling
    databases don't leak their columns into the agent's schema prompt.

    Returns the SQL unchanged unless it matches the exact CrewAI
    ``_fetch_all_available_columns()`` query shape.
    """
    m = _CREWAI_PG_COLUMNS_QUERY.match(sql)
    if m is None:
        return sql
    return f"{m.group(1)} AND table_schema = currentDatabase(){m.group(2)}"


def _apply_crewai_compat_rewrites(sql: str) -> str:
    """Apply every CrewAI compatibility rewrite. No-op for non-matching SQL."""
    sql = _rewrite_crewai_public_schema(sql)
    sql = _rewrite_crewai_columns_filter(sql)
    return sql


def _coerce_int(v: Any) -> Any:
    if v is None or (isinstance(v, int) and not isinstance(v, bool)):
        return v
    if isinstance(v, str):
        return int(v)
    return v


def _coerce_float(v: Any) -> Any:
    if v is None or isinstance(v, float):
        return v
    if isinstance(v, (int, str)):
        return float(v)
    return v


def _coerce_decimal(v: Any) -> Any:
    if v is None or isinstance(v, _decimal.Decimal):
        return v
    if isinstance(v, (str, int, float)):
        return _decimal.Decimal(str(v))
    return v


def _coerce_date(v: Any) -> Any:
    """``Nullable(Date)`` / sometimes ``Date`` comes back as 'YYYY-MM-DD' str."""
    import datetime as _dt
    if v is None or (isinstance(v, _dt.date) and not isinstance(v, _dt.datetime)):
        return v
    if isinstance(v, str):
        try:
            return _dt.date.fromisoformat(v)
        except ValueError:
            return v
    return v


def _coerce_datetime(v: Any) -> Any:
    """``Nullable(DateTime[64])`` comes back as 'YYYY-MM-DD HH:MM:SS[.fff]' str.

    Python's stdlib ``datetime`` only carries microsecond resolution, so
    ``DateTime64(7/8/9)`` cells come back with 7-9 fractional digits that
    must be truncated to 6 before parsing. Python 3.11+ ``fromisoformat``
    silently truncates the extras; 3.10 raises ``ValueError`` and we'd
    return the raw string. Truncate explicitly so behavior matches across
    Python versions.
    """
    import datetime as _dt
    if v is None or isinstance(v, _dt.datetime):
        return v
    if not isinstance(v, str):
        return v
    s = v.replace(" ", "T")
    # Clip sub-microsecond fractional digits — stdlib datetime can't hold them
    # and 3.10 fromisoformat doesn't accept them.
    if "." in s:
        head, _, frac = s.partition(".")
        # ``frac`` may carry a trailing timezone block (``123456789+00:00``);
        # split it off, truncate digits, glue back.
        digits = ""
        tail = ""
        for i, ch in enumerate(frac):
            if ch.isdigit():
                digits += ch
            else:
                tail = frac[i:]
                break
        s = head + "." + digits[:6] + tail
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        return v


def _coerce_time(v: Any) -> Any:
    """``Nullable(Time[64])`` comes back as 'HH:MM:SS[.fff]' str."""
    import datetime as _dt
    if v is None or isinstance(v, _dt.time):
        return v
    if isinstance(v, str):
        try:
            return _dt.time.fromisoformat(v)
        except ValueError:
            return v
    return v


def _coerce_literal(v: Any) -> Any:
    """Parse a Python repr-style string into a real Python object.

    Used for Array / Map / JSON. Pass-through for non-strings or strings
    that fail to parse (defensive — we never want to crash a user's
    fetch on an unexpected format).
    """
    if v is None or not isinstance(v, str):
        return v
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    # Fallback for properly-emitted JSON
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def _peel_wrappers(type_str: str) -> str:
    """Strip outer ``Nullable`` / ``LowCardinality`` wrappers.

    They have no effect on the converter. ClickHouse allows arbitrary
    nesting in either order (``Nullable(LowCardinality(T))`` and
    ``LowCardinality(Nullable(T))`` both legal), so peel in a single
    alternating loop instead of stripping all Nullables first and then
    all LowCardinality.
    """
    inner = type_str
    while True:
        if inner.startswith("Nullable("):
            inner = inner[len("Nullable(") : -1]
        elif inner.startswith("LowCardinality("):
            inner = inner[len("LowCardinality(") : -1]
        else:
            return inner


def _split_type_args(args_str: str) -> list[str]:
    """Split a composite type's argument list at top-level commas.

    ``'Nullable(Int32), String, Map(String, Int32)'`` →
    ``['Nullable(Int32)', 'String', 'Map(String, Int32)']``.

    Paren- and quote-aware so commas inside ``Decimal(18, 2)``,
    ``DateTime64(3, 'UTC')``, ``Enum8('a' = 1, 'b' = 2)`` or quoted
    identifiers never split. Quoted regions honor backslash escapes
    (``Enum8('it\\'s' = 1)``).
    """
    parts: list[str] = []
    depth = 0
    start = 0
    quote: str | None = None
    i = 0
    while i < len(args_str):
        ch = args_str[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", "`"):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(args_str[start:i].strip())
            start = i + 1
        i += 1
    tail = args_str[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_named_field(arg: str) -> tuple[str | None, str]:
    """Split one Tuple/Nested argument into ``(field_name, type_str)``.

    ``'a Float64'`` → ``('a', 'Float64')``; ``'Float64'`` →
    ``(None, 'Float64')``. The field name is everything before the first
    top-level space — unnamed elements never contain one, because spaces
    in types like ``Decimal(18, 2)`` or ``DateTime64(3, 'UTC')`` sit
    inside parentheses. Backquoted names (`` `my field` ``) are unquoted.
    """
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(arg):
        ch = arg[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", "`"):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == " " and depth == 0:
            name = arg[:i].strip().strip("`")
            rest = arg[i + 1 :].strip()
            if rest:
                return name, rest
            return None, arg
        i += 1
    return None, arg


def _scalar_converter(inner: str) -> Callable[[Any], Any] | None:
    """Leaf converter for an (already wrapper-peeled) scalar type string."""
    if inner.startswith(("Decimal(", "Decimal32", "Decimal64", "Decimal128", "Decimal256")):
        return _coerce_decimal
    if inner.startswith("Int") or inner.startswith("UInt"):
        return _coerce_int
    if inner.startswith("Float") or inner.startswith("BFloat"):
        return _coerce_float
    # Date / DateTime / Time — Nullable(date) tends to come back as str
    if inner == "Date" or inner == "Date32":
        return _coerce_date
    if inner.startswith("DateTime"):
        return _coerce_datetime
    if inner == "Time" or inner.startswith("Time64"):
        return _coerce_time
    return None


# Geo types are sugar over composites; the converter walks the equivalent
# composite type. Containers keep their parsed (list) shape — only the
# top-level Point column is re-wrapped as a tuple, in _converter_for.
_GEO_EQUIVALENTS = {
    "Point": "Tuple(Float64, Float64)",
    "Ring": "Array(Point)",
    "LineString": "Array(Point)",
    "MultiLineString": "Array(Ring)",
    "Polygon": "Array(Ring)",
    "MultiPolygon": "Array(Polygon)",
}


def _element_converter(type_str: str) -> Callable[[Any], Any] | None:
    """Converter for an *already-parsed* element of a composite cell.

    chdb-core 26.5 (ClickHouse 26.5 baseline) serialises Float / Decimal /
    temporal leaves inside ``Array`` / ``Tuple`` / ``Map`` / ``Nested``
    cells as quoted strings — ``"['1.5', '2.5']"`` where chdb-core 26.3
    emitted bare numerics. After :func:`_coerce_literal` parses the cell,
    this walker re-coerces those leaves to native types, driven by the
    column's ClickHouse type string. Container shapes are preserved
    exactly as parsed (lists stay lists, dicts stay dicts) so the repair
    is value-only.

    Returns ``None`` when no leaf anywhere in the type needs coercion, so
    identity composites (e.g. ``Array(String)``) skip the walk. Walkers
    are defensive: any shape mismatch returns the value unchanged rather
    than raising mid-fetch.
    """
    inner = _peel_wrappers(type_str)
    inner = _GEO_EQUIVALENTS.get(inner, inner)

    scalar = _scalar_converter(inner)
    if scalar is not None:
        return scalar

    if inner.startswith("Array("):
        elem = _element_converter(inner[len("Array(") : -1])
        if elem is None:
            return None

        def _convert_array(v: Any) -> Any:
            if isinstance(v, list):
                return [elem(x) for x in v]
            return v

        return _convert_array

    if inner.startswith("Map("):
        args = _split_type_args(inner[len("Map(") : -1])
        if len(args) != 2:
            return None
        key_conv = _element_converter(args[0])
        val_conv = _element_converter(args[1])
        if key_conv is None and val_conv is None:
            return None
        kc = key_conv or (lambda x: x)
        vc = val_conv or (lambda x: x)

        def _convert_map(v: Any) -> Any:
            if isinstance(v, dict):
                return {kc(k): vc(x) for k, x in v.items()}
            return v

        return _convert_map

    if inner.startswith("Tuple(") or inner.startswith("Nested("):
        is_nested = inner.startswith("Nested(")
        body = inner[len("Nested(") : -1] if is_nested else inner[len("Tuple(") : -1]
        field_convs: list[Callable[[Any], Any] | None] = []
        named_convs: dict[str, Callable[[Any], Any]] = {}
        for arg in _split_type_args(body):
            name, field_type = _split_named_field(arg)
            conv = _element_converter(field_type)
            field_convs.append(conv)
            if name is not None and conv is not None:
                named_convs[name] = conv
        if not any(c is not None for c in field_convs):
            return None

        def _convert_record(v: Any) -> Any:
            # chdb serialises unnamed Tuple cells with list brackets and
            # named Tuple cells as dicts keyed by field name.
            if isinstance(v, dict):
                return {k: named_convs[k](x) if k in named_convs else x for k, x in v.items()}
            if isinstance(v, (list, tuple)) and len(v) == len(field_convs):
                out = [c(x) if c is not None else x for c, x in zip(field_convs, v)]
                return tuple(out) if isinstance(v, tuple) else out
            return v

        if not is_nested:
            return _convert_record

        # A Nested(...) cell is a list of records (one per nested row).
        def _convert_nested(v: Any) -> Any:
            if isinstance(v, list):
                return [_convert_record(x) for x in v]
            return v

        return _convert_nested

    return None


def _converter_for(type_str: str) -> Callable[[Any], Any] | None:
    """Return the column converter for a ClickHouse type string, or None.

    ``None`` means "no conversion needed" — pass values through. The caller
    avoids the dispatch overhead on identity columns.
    """
    inner = _peel_wrappers(type_str)

    scalar = _scalar_converter(inner)
    if scalar is not None:
        return scalar

    # Tuple cells arrive as repr-style strings; chdb.dbapi serialises
    # unnamed Tuples with list-bracket syntax ('[1, 2, 3]') and named
    # Tuples as dicts. The natural Python counterpart of a fixed-arity
    # heterogeneous Tuple is tuple, so re-wrap the list form. Point is
    # exactly Tuple(Float64, Float64) — same treatment.
    if inner.startswith("Tuple(") or inner == "Point":
        elem = _element_converter(inner)

        def _convert_tuple_cell(v: Any) -> Any:
            parsed = _coerce_literal(v)
            if elem is not None:
                parsed = elem(parsed)
            if isinstance(parsed, list):
                return tuple(parsed)
            return parsed

        return _convert_tuple_cell

    # Other composites and the list-shaped geo aliases: parse the
    # repr-style string, then repair quoted numeric/temporal leaves.
    if (
        any(inner.startswith(p) for p in _COMPOSITE_PREFIXES)
        or inner in _GEO_EQUIVALENTS
    ):
        elem = _element_converter(inner)
        if elem is None:
            return _coerce_literal

        def _convert_composite_cell(v: Any) -> Any:
            return elem(_coerce_literal(v))

        return _convert_composite_cell

    # JSON values are dynamically typed per path — the column type string
    # carries no leaf info, so parse the cell shape and leave leaves alone.
    if inner == "JSON" or inner.startswith("JSON(") or inner.startswith("Object("):
        return _coerce_literal
    return None


def _build_converters(description: Sequence[Sequence[Any]]) -> list[Callable[[Any], Any] | None]:
    """Pre-compute per-column converters from cursor.description.

    ``description`` follows PEP 249: each entry is a 7-tuple whose second
    element is the chdb-supplied ClickHouse type string.
    """
    return [_converter_for(col[1]) if col and col[1] else None for col in description]


class _CursorWrapper:
    """PEP 249 cursor that post-processes rows to repair chDB return formats.

    Delegates everything except ``fetch*`` to the underlying cursor.
    """

    __slots__ = ("_converters", "_cur")

    def __init__(self, cursor: Any) -> None:
        self._cur = cursor
        self._converters: list[Callable[[Any], Any] | None] | None = None

    # ------------------------------------------------------------------
    # PEP 249 cursor interface
    # ------------------------------------------------------------------

    @property
    def description(self) -> Any:
        return self._cur.description

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    @property
    def arraysize(self) -> int:
        return getattr(self._cur, "arraysize", 1)

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        self._cur.arraysize = value

    @property
    def lastrowid(self) -> Any:
        return getattr(self._cur, "lastrowid", None)

    def close(self) -> None:
        self._cur.close()

    def execute(self, statement: str, parameters: Any = None) -> Any:
        self._converters = None  # recompute on next fetch
        statement = _apply_crewai_compat_rewrites(statement)
        if parameters is None:
            return self._cur.execute(statement)
        return self._cur.execute(statement, parameters)

    def executemany(self, statement: str, seq_of_params: Any) -> Any:
        self._converters = None
        statement = _apply_crewai_compat_rewrites(statement)
        return self._cur.executemany(statement, seq_of_params)

    def callproc(self, procname: str, parameters: Any = None) -> Any:
        return self._cur.callproc(procname, parameters)

    def setinputsizes(self, sizes: Any) -> None:
        self._cur.setinputsizes(sizes)

    def setoutputsizes(self, size: Any, column: Any = None) -> None:
        # chdb.dbapi takes (size, column) or just (size) — forward whatever.
        if column is None:
            self._cur.setoutputsizes(size)
        else:
            self._cur.setoutputsizes(size, column)

    def nextset(self) -> Any:
        return self._cur.nextset()

    def mogrify(self, *args: Any, **kw: Any) -> Any:
        return self._cur.mogrify(*args, **kw)

    @property
    def max_stmt_length(self) -> Any:
        return self._cur.max_stmt_length

    # ------------------------------------------------------------------
    # The actual conversion work
    # ------------------------------------------------------------------

    def _ensure_converters(self) -> list[Callable[[Any], Any] | None]:
        if self._converters is None:
            self._converters = _build_converters(self._cur.description or ())
        return self._converters

    def _convert_row(self, row: Any) -> tuple:
        converters = self._ensure_converters()
        if not converters or not any(c is not None for c in converters):
            return tuple(row)
        return tuple(
            (c(v) if c is not None else v) for c, v in zip(converters, row)
        )

    def fetchone(self) -> Any:
        row = self._cur.fetchone()
        return None if row is None else self._convert_row(row)

    def fetchmany(self, size: int | None = None) -> list:
        rows = self._cur.fetchmany(size) if size is not None else self._cur.fetchmany()
        return [self._convert_row(r) for r in rows]

    def fetchall(self) -> list:
        return [self._convert_row(r) for r in self._cur.fetchall()]

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row


class _ConnectionWrapper:
    """Wraps a ``chdb.dbapi`` connection so ``cursor()`` returns ``_CursorWrapper``.

    Everything else delegates to the underlying connection. The wrapper is
    transparent to SQLAlchemy / LangChain / pandas — they all only need a
    PEP 249 ``Connection`` shape.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def cursor(self) -> _CursorWrapper:
        return _CursorWrapper(self._conn.cursor())

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:  # chdb has no transactions, but PEP 249 requires the method
        if hasattr(self._conn, "commit"):
            self._conn.commit()

    def rollback(self) -> None:
        if hasattr(self._conn, "rollback"):
            self._conn.rollback()

    def __getattr__(self, name: str) -> Any:
        # Pass through anything else (e.g. .escape_string, .character_set_name).
        return getattr(self._conn, name)


def wrap_connection(conn: Any) -> _ConnectionWrapper:
    """Wrap a raw ``chdb.dbapi`` connection with the row-coercion cursor."""
    return _ConnectionWrapper(conn)
