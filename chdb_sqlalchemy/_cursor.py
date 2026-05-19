"""Cursor wrapper that repairs chDB's lossy result-value formatting.

Why this layer exists:

``chdb.dbapi`` (chdb 4.x, ClickHouse 26.3) returns several categories of
values in non-native forms:

* ``Decimal(P, S)`` cells come back as ``str`` (always)
* ``Nullable(<numeric>)`` cells come back as ``str`` for non-NULL
* ``Array(T)`` / ``Map(K, V)`` / ``Tuple(...)`` / ``JSON`` cells come back
  as **Python repr-style strings** (e.g. ``"['a', 'b']"``, ``"{'k': 'v'}"``)

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
from collections.abc import Callable, Sequence
from typing import Any

# Type-string prefix → row converter.
# Order matters: more specific prefixes must come before broader ones.
_COMPOSITE_PREFIXES = ("Array(", "Map(", "Tuple(", "Nested(")


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


def _coerce_tuple(v: Any) -> Any:
    """Same as :func:`_coerce_literal` but returns ``tuple`` not ``list``.

    chdb.dbapi (chdb 4.x / CH 26.3) serialises ClickHouse ``Tuple(...)``
    cells using list-bracket syntax — ``'[1, 2, 3]'`` — so
    ``ast.literal_eval`` yields a Python list. That's the wrong native
    mapping: ``Tuple`` is fixed-arity heterogeneous, the natural Python
    counterpart is ``tuple``. Downstream code that unpacks positionally
    via ``(a, b, c) = row['t']`` works on both, but type-annotated
    consumers (Pydantic, dataclasses, mypy users) expect tuple.

    This is the upstream chdb.dbapi gap; the shim survives until that's
    fixed and our minimum-version floor is raised.
    """
    parsed = _coerce_literal(v)
    if isinstance(parsed, list):
        return tuple(parsed)
    return parsed


def _converter_for(type_str: str) -> Callable[[Any], Any] | None:
    """Return the column converter for a ClickHouse type string, or None.

    ``None`` means "no conversion needed" — pass values through. The caller
    avoids the dispatch overhead on identity columns.
    """
    # Strip outer Nullable / LowCardinality wrappers — they have no effect
    # on the converter. ClickHouse allows arbitrary nesting in either order
    # (``Nullable(LowCardinality(T))`` and ``LowCardinality(Nullable(T))``
    # both legal), so peel in a single alternating loop instead of stripping
    # all Nullables first and then all LowCardinality.
    inner = type_str
    while True:
        if inner.startswith("Nullable("):
            inner = inner[len("Nullable(") : -1]
        elif inner.startswith("LowCardinality("):
            inner = inner[len("LowCardinality(") : -1]
        else:
            break

    # Numerics
    if inner.startswith("Decimal(") or inner.startswith("Decimal32") or inner.startswith("Decimal64") or inner.startswith("Decimal128") or inner.startswith("Decimal256"):
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
    # Tuple needs the dedicated tuple-coercer (chdb.dbapi serialises Tuples
    # using list-bracket syntax — see _coerce_tuple). Must come before the
    # generic composite fallback.
    if inner.startswith("Tuple("):
        return _coerce_tuple
    # Point is exactly Tuple(Float64, Float64) — apply the same fix.
    if inner == "Point":
        return _coerce_tuple
    # Other composites — list-like is fine.
    if any(inner.startswith(p) for p in _COMPOSITE_PREFIXES):
        return _coerce_literal
    if inner == "JSON" or inner.startswith("JSON(") or inner.startswith("Object("):
        return _coerce_literal
    # Geo aliases that unwrap to Array-of-X are list-shaped, not tuple-shaped.
    if inner in ("Ring", "Polygon", "MultiPolygon"):
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
        if parameters is None:
            return self._cur.execute(statement)
        return self._cur.execute(statement, parameters)

    def executemany(self, statement: str, seq_of_params: Any) -> Any:
        self._converters = None
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
