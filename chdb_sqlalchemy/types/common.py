"""Basic ClickHouse types — scalars with no further parameterisation by
sub-types.

Composite types (``Array``, ``Tuple``, ``Map``, ``Nullable``, ``LowCardinality``,
``Nested``) live in :mod:`composite`. Modern types (``Variant`` / ``Dynamic`` /
new ``JSON``) live in their own modules. Geo types live in :mod:`geo`.

Per Part 0 of the plan: ``bind_processor`` and ``result_processor`` are
deliberately thin — ``chdb.dbapi.converters`` already handles escape and
basic Python ↔ SQL coercion for the scalar types, so SQLAlchemy's defaults
on top of those converters are correct for nearly every scalar here.

Internal note on ``adapt()``: SQLAlchemy uses ``adapt()`` during query
compilation to translate a column-declaration type into its dialect-impl
form. The default implementation calls ``util.constructor_copy``, which
introspects ``__init__`` parameters and rebuilds via kwargs. That breaks
for composite chDB types that take ``*args`` (``Array``, ``Tuple``, ``Map``,
``Variant``, ``AggregateFunction``…). We override ``adapt()`` on the base
class to use ``copy.copy()`` — yielding a fresh instance (which SQLAlchemy
asserts for) without going through ``__init__``.
"""

from __future__ import annotations

import copy
import decimal as _decimal
from collections.abc import Callable
from typing import Any

from sqlalchemy import types

# ---------------------------------------------------------------------------
# Numeric coercion helpers
# ---------------------------------------------------------------------------
#
# As of chDB 26.3 the underlying ``chdb.dbapi`` driver returns numeric cells
# wrapped in ``Nullable(...)`` as **strings** instead of native ints/floats —
# and plain ``Decimal`` columns are *always* strings regardless of
# nullability. SQLAlchemy's column type processors are the right place to
# repair this: when reflection unwraps ``Nullable(Decimal)`` to ``Decimal``
# with ``nullable=True``, the ``Decimal.result_processor`` we register here
# is what SA invokes on each cell.
#
# The processors are defensive: if chDB fixes its dbapi upstream and starts
# returning native numerics, the ``isinstance(value, str)`` guard makes the
# coercion a no-op.


def _coerce_int(value: Any) -> Any:
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    if isinstance(value, str):
        return int(value)
    return value


def _coerce_float(value: Any) -> Any:
    if value is None or isinstance(value, float):
        return value
    if isinstance(value, (int, str)):
        return float(value)
    return value


def _coerce_decimal(value: Any) -> Any:
    if value is None or isinstance(value, _decimal.Decimal):
        return value
    if isinstance(value, (str, int, float)):
        return _decimal.Decimal(str(value))
    return value


def _make_int_result_processor(self: Any, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
    return _coerce_int


def _make_float_result_processor(self: Any, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
    return _coerce_float


def _make_decimal_result_processor(self: Any, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
    return _coerce_decimal


class _ChdbType(types.TypeEngine):
    """Marker base class so reflection / DDL emission can detect chDB types.

    Overrides ``adapt()`` to short-circuit SQLAlchemy's default
    ``util.constructor_copy``-based adaptation. That default introspects the
    constructor signature and rebuilds the type from kwargs — which breaks
    for composite chDB types whose ``__init__`` uses ``*args`` (Array,
    Tuple, Map, Variant, AggregateFunction, etc.). Since chDB types are
    already the dialect-impl form, ``adapt`` just returns ``self``.
    """

    __abstract__ = True

    def adapt(self, impltype, **kw):
        # Bypass SA's util.constructor_copy (which assumes kwarg-only init)
        # by returning a shallow copy. SA asserts impl is not self, so we
        # must return a different instance.
        if impltype is type(self):
            return copy.copy(self)
        # Subclass-preservation: SA's colspecs maps ``_sqltypes.DateTime →
        # our DateTime``. When the column is already a more-specific subclass
        # (DateTime64, Time64, etc.), the MRO walk in ``adapt_type`` finds
        # the colspecs key and SA wants to "adapt down" to the impl class.
        # Doing so would lose precision/timezone/etc. carried on the subclass.
        # If self is already an instance of impltype, the subclass is already
        # at least as specific — keep its class.
        if isinstance(self, impltype):
            return copy.copy(self)
        # Cross-class adapt — let SA try its default path; if it fails (which
        # it can for our *args composites), fall back to copying self under
        # the requested class.
        try:
            return super().adapt(impltype, **kw)
        except (TypeError, ValueError):
            new = copy.copy(self)
            new.__class__ = impltype
            return new


# -------------------- string family --------------------

class String(types.String, _ChdbType):
    __visit_name__ = "String"


class FixedString(types.CHAR, _ChdbType):
    """``FixedString(N)`` — chDB native fixed-width string. Inherits SA's
    ``CHAR`` so reflection round-trips ``Column(CHAR(N))`` declarations
    pass SA's ``isinstance(reflected, CHAR)`` assertions."""

    __visit_name__ = "FixedString"

    def __init__(self, length: int) -> None:
        if length <= 0:
            raise ValueError("FixedString length must be positive")
        super().__init__(length=length)


def _uuid_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        return f"toUUID('{value}')"
    return process


def _uuid_bind_processor(self: Any, dialect: Any) -> Callable[[Any], Any]:
    """Convert ``uuid.UUID`` bind values to their canonical string form.

    chdb.dbapi's parameter-escape path calls ``.translate()`` on string-ish
    objects; a ``uuid.UUID`` instance has no such method, so the call
    crashes with AttributeError. chDB itself accepts the canonical
    ``'xxxxxxxx-xxxx-...'`` string as a bind for a UUID column (auto-cast
    via VALUES), so str-ifying here is sufficient — no wrapping in
    ``toUUID(?)`` needed.
    """
    import uuid as _uuid

    def process(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, _uuid.UUID):
            return str(value)
        return value
    return process


def _uuid_result_processor(self: Any, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
    import uuid as _uuid
    def process(value: Any) -> Any:
        if value is None or isinstance(value, _uuid.UUID):
            return value
        if isinstance(value, str):
            try:
                return _uuid.UUID(value)
            except (ValueError, AttributeError):
                return value
        return value
    return process


class UUID(types.Uuid, _ChdbType):
    """Inherits SA Uuid for isinstance-compatibility in reflection."""
    __visit_name__ = "UUID"
    literal_processor = _uuid_literal_processor
    bind_processor = _uuid_bind_processor
    result_processor = _uuid_result_processor
    result_processor = _uuid_result_processor


# -------------------- integer family --------------------

class _IntBase(types.Integer, _ChdbType):
    __abstract__ = True
    result_processor = _make_int_result_processor


class Int8(_IntBase):
    __visit_name__ = "Int8"


class Int16(_IntBase):
    __visit_name__ = "Int16"


class Int32(_IntBase):
    __visit_name__ = "Int32"


class Int64(types.BigInteger, _ChdbType):
    __visit_name__ = "Int64"
    result_processor = _make_int_result_processor


class Int128(types.BigInteger, _ChdbType):
    __visit_name__ = "Int128"
    result_processor = _make_int_result_processor


class Int256(types.BigInteger, _ChdbType):
    __visit_name__ = "Int256"
    result_processor = _make_int_result_processor


class UInt8(_IntBase):
    __visit_name__ = "UInt8"


class UInt16(_IntBase):
    __visit_name__ = "UInt16"


class UInt32(_IntBase):
    __visit_name__ = "UInt32"


class UInt64(types.BigInteger, _ChdbType):
    __visit_name__ = "UInt64"
    result_processor = _make_int_result_processor


class UInt128(types.BigInteger, _ChdbType):
    __visit_name__ = "UInt128"
    result_processor = _make_int_result_processor


class UInt256(types.BigInteger, _ChdbType):
    __visit_name__ = "UInt256"
    result_processor = _make_int_result_processor


# -------------------- float family --------------------

class Float32(types.Float, _ChdbType):
    __visit_name__ = "Float32"
    result_processor = _make_float_result_processor


class Float64(types.Float, _ChdbType):
    __visit_name__ = "Float64"
    result_processor = _make_float_result_processor


class BFloat16(types.Float, _ChdbType):
    """16-bit Brain Float — added in ClickHouse 24.6."""

    __visit_name__ = "BFloat16"
    result_processor = _make_float_result_processor


# -------------------- decimal --------------------

class Decimal(types.Numeric, _ChdbType):
    """``Decimal(P, S)`` — fixed-precision decimal.

    chDB always returns Decimal cells as strings; we coerce to
    ``decimal.Decimal`` in ``result_processor`` so callers get the
    SQLAlchemy-promised return type.
    """

    __visit_name__ = "Decimal"

    def __init__(self, precision: int, scale: int) -> None:
        if precision < 1:
            raise ValueError("Decimal precision must be >= 1")
        if scale < 0 or scale > precision:
            raise ValueError("Decimal scale must be in [0, precision]")
        super().__init__(precision=precision, scale=scale, asdecimal=True)

    result_processor = _make_decimal_result_processor


# -------------------- bool --------------------

def _bool_result_processor(self: Any, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
    """Coerce int/str values to native bool — chdb.dbapi sometimes returns
    bool columns as ``'True'``/``'False'`` strings or ``0``/``1`` ints
    depending on storage."""
    def process(value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() not in ("false", "0", "")
        if isinstance(value, int):
            return bool(value)
        return value
    return process


def _bool_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        return "true" if value else "false"
    return process


class Boolean(types.Boolean, _ChdbType):
    __visit_name__ = "Boolean"
    result_processor = _bool_result_processor
    literal_processor = _bool_literal_processor


# -------------------- date/time family --------------------
#
# Why bind_processor matters here:
#
# chdb.dbapi's parameter-escape path doesn't recognise ``datetime.date`` /
# ``datetime.datetime`` / ``datetime.time`` objects. Passing a raw datetime
# object as a bind param ends up as NULL in the table (chdb.dbapi treats
# the un-escapable object as "no value"). We have to ISO-stringify in
# bind_processor before chdb.dbapi sees the value. Server-side functions
# like ``toDateTime(?)`` happily parse the string form.


def _resolve_target_tz(tz_name: str | None):
    """Resolve the column's declared timezone name to a ``tzinfo`` instance.

    chDB doesn't accept ISO 8601 offset suffixes (``+00:00``) in the
    string forms of ``toDateTime`` / ``toDateTime64``; it expects a
    naive ``'YYYY-MM-DD HH:MM:SS[.fff]'`` plus a separate ``'TZ'`` arg.
    To bind a tz-aware Python datetime correctly we have to normalise
    it into the column's declared timezone first and strip ``tzinfo``,
    so the wire form is always parseable.

    ``None`` / ``"UTC"`` returns ``datetime.timezone.utc`` (always
    available without external tzdata, unlike ``zoneinfo.ZoneInfo("UTC")``
    on slim Windows installs). Any other name goes through ``zoneinfo``.
    """
    import datetime as _dt

    if tz_name is None or tz_name.upper() == "UTC":
        return _dt.timezone.utc
    import zoneinfo  # Python 3.9+; chdb-sqlalchemy supports 3.10+.

    return zoneinfo.ZoneInfo(tz_name)


def _normalize_aware(value: Any, tz_name: str | None) -> Any:
    """Convert tz-aware datetime to its naive equivalent in ``tz_name``.

    Why this matters: ``datetime.isoformat(sep=" ")`` on a tz-aware
    value produces ``'2025-06-15 12:30:45+00:00'``. chDB's
    ``toDateTime('...+00:00', 'UTC')`` raises ``CANNOT_PARSE_TEXT``;
    the bind path then NULL-folds (Nullable column) or epochs the cell.
    We must hand chDB the moment **expressed in the column's declared
    timezone, with no offset suffix**.

    Behavior:

    * Naive values pass through unchanged.
    * Aware values are first ``astimezone()``'d into the column's tz
      (defaulting to UTC when the column doesn't declare one), then
      ``replace(tzinfo=None)`` strips the offset so isoformat() emits
      a naive string.
    """
    if value is None:
        return None
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is None:
        return value
    target = _resolve_target_tz(tz_name)
    return value.astimezone(target).replace(tzinfo=None)


def _date_bind_processor(self: Any, dialect: Any) -> Callable[[Any], Any]:
    def process(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
    return process


def _datetime_bind_processor_secondprec(self: Any, dialect: Any) -> Callable[[Any], Any]:
    """Bind processor for the **second-precision** ``DateTime`` type.

    ClickHouse's ``DateTime`` (no precision parameter) is integer-seconds.
    ``toDateTime('2025-06-15 12:30:45.123456')`` raises
    ``Cannot parse string ... as Date`` on the server, which the chdb.dbapi
    VALUES path silently folds to NULL (Nullable column) or epoch
    (non-Nullable). We strip sub-second precision here so the wire form
    is always parseable.

    Aware datetimes are first normalised to the column's tz_name (or UTC
    if the column doesn't declare one) — chDB rejects the
    ``+HH:MM`` offset suffix that ``isoformat`` would otherwise emit.

    Use ``_datetime_bind_processor`` for ``DateTime64`` which carries
    the precision as a column parameter.
    """
    tz_name = getattr(self, "tz_name", None)

    def process(value: Any) -> Any:
        if value is None:
            return None
        value = _normalize_aware(value, tz_name)
        if hasattr(value, "replace") and hasattr(value, "microsecond"):
            value = value.replace(microsecond=0)
        if hasattr(value, "isoformat"):
            return value.isoformat(sep=" ")
        return str(value)
    return process


def _datetime_bind_processor(self: Any, dialect: Any) -> Callable[[Any], Any]:
    """Bind processor for ``DateTime64`` — keeps microseconds.

    ``DateTime64`` columns store sub-second precision based on the column's
    ``precision`` argument; chDB's ``toDateTime64('...', p[, tz])`` accepts
    fractional digits up to that precision.

    Aware datetimes are first normalised to the column's tz_name (or UTC)
    so the emitted string is the naive form chDB's parser expects.
    """
    tz_name = getattr(self, "tz_name", None)

    def process(value: Any) -> Any:
        if value is None:
            return None
        value = _normalize_aware(value, tz_name)
        if hasattr(value, "isoformat"):
            return value.isoformat(sep=" ")
        return str(value)
    return process


def _time_bind_processor_secondprec(self: Any, dialect: Any) -> Callable[[Any], Any]:
    """Bind processor for second-precision ``Time``.

    chDB's ``Time`` is integer-seconds with no timezone concept. If the
    Python ``time`` happens to carry tzinfo we strip it silently —
    ``time`` objects with timezone are rare and chDB can't store the
    offset anyway.
    """
    def process(value: Any) -> Any:
        if value is None:
            return None
        # Time has no TZ in chDB — strip any incidental tzinfo, no offset
        # resolution needed.
        if getattr(value, "tzinfo", None) is not None:
            value = value.replace(tzinfo=None)
        if hasattr(value, "replace") and hasattr(value, "microsecond"):
            value = value.replace(microsecond=0)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
    return process


def _time_bind_processor(self: Any, dialect: Any) -> Callable[[Any], Any]:
    """Bind processor for ``Time64`` — keeps microseconds."""
    def process(value: Any) -> Any:
        if value is None:
            return None
        if getattr(value, "tzinfo", None) is not None:
            value = value.replace(tzinfo=None)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
    return process


# ---------------------------------------------------------------------------
# Literal processors: emit **bare quoted ISO strings**, NOT chDB-cast wraps.
#
# Why this matters: SA's ``literal_binds=True`` compile mode invokes both
# ``literal_processor`` (to render the value) AND ``bind_expression`` (the
# column-level wrap added in ``bind_expression``-equipped types). If the
# literal_processor wraps in ``toDateTime('...')`` *and* the bind_expression
# wraps in another ``toDateTime(?, 'UTC')``, the result is
# ``toDateTime(toDateTime('...'), 'UTC')`` — a double-conversion that loses
# timezone information (the inner call uses session TZ, the outer
# overrides) and breaks Date32 for pre-1970 values (inner toDate maps to
# epoch).
#
# Resolution: literal_processor emits ``'<iso-form>'`` only. The chDB cast
# is applied exactly once by ``bind_expression`` on insert / select. For
# ``Time`` / ``Time64`` which have no ``bind_expression`` (chDB's
# ``toTime`` is a datetime-extractor, not a parser), chDB auto-casts the
# string in INSERT VALUES context, so a bare quoted form is still correct.
# ---------------------------------------------------------------------------


def _date_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        iso = value.isoformat() if hasattr(value, "isoformat") else value
        return f"'{iso}'"
    return process


def _datetime_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    """``DateTime`` literal: bare quoted ISO string, second-precision,
    aware datetimes normalised to the column's tz."""
    tz_name = getattr(self, "tz_name", None)

    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        value = _normalize_aware(value, tz_name)
        # DateTime is second-precision — strip microseconds for the bare
        # literal, same as the bind processor. Otherwise a literal_binds
        # render against a DateTime column with a microsecond datetime
        # would emit a fractional-second string that chDB's toDateTime
        # parser rejects.
        if hasattr(value, "replace") and hasattr(value, "microsecond"):
            value = value.replace(microsecond=0)
        iso = value.isoformat(sep=" ") if hasattr(value, "isoformat") else str(value)
        return f"'{iso}'"
    return process


def _datetime64_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    """``DateTime64`` literal: bare quoted ISO string, sub-second precision
    preserved, aware datetimes normalised to the column's tz."""
    tz_name = getattr(self, "tz_name", None)

    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        value = _normalize_aware(value, tz_name)
        iso = value.isoformat(sep=" ") if hasattr(value, "isoformat") else str(value)
        return f"'{iso}'"
    return process


def _time_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        if getattr(value, "tzinfo", None) is not None:
            value = value.replace(tzinfo=None)
        if hasattr(value, "replace") and hasattr(value, "microsecond"):
            value = value.replace(microsecond=0)
        iso = value.isoformat() if hasattr(value, "isoformat") else value
        return f"'{iso}'"
    return process


def _time64_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        if getattr(value, "tzinfo", None) is not None:
            value = value.replace(tzinfo=None)
        iso = value.isoformat() if hasattr(value, "isoformat") else value
        return f"'{iso}'"
    return process


def _wrap_bind(fn_name: str):
    """Build a ``bind_expression`` that wraps the ``?`` placeholder in a
    chDB conversion function. SA emits ``toDate(?)``, ``toDateTime(?)``,
    ``toUUID(?)`` etc., letting chdb.dbapi pass the raw value as a string
    bind that the function then parses into the right type."""
    from sqlalchemy.sql import elements

    def bind_expression(self, bindvalue):
        return elements.BinaryExpression(
            elements.literal_column(fn_name + "("),
            elements.BinaryExpression(
                bindvalue, elements.literal_column(")"), elements.operators.concat_op
            ),
            elements.operators.concat_op,
        )

    # Simpler: use SA's func()
    from sqlalchemy import func

    def bind_expression_simple(self, bindvalue):
        return getattr(func, fn_name)(bindvalue)

    return bind_expression_simple


class Date(types.Date, _ChdbType):
    __visit_name__ = "Date"
    literal_processor = _date_literal_processor
    bind_processor = _date_bind_processor
    bind_expression = _wrap_bind("toDate")


# Subclass-of-our-class (not SA's generic) so ``_ChdbType.adapt``'s
# ``isinstance(self, impltype)`` check fires and prevents downcast when
# SA's colspecs maps generic ``sqltypes.Date`` to our ``Date``.
class Date32(Date):
    __visit_name__ = "Date32"
    bind_expression = _wrap_bind("toDate32")


class DateTime(types.DateTime, _ChdbType):
    __visit_name__ = "DateTime"
    literal_processor = _datetime_literal_processor
    bind_processor = _datetime_bind_processor_secondprec

    def __init__(self, timezone: str | None = None) -> None:
        super().__init__(timezone=bool(timezone))
        self.tz_name = timezone

    def bind_expression(self, bindvalue):
        """Emit ``toDateTime(?, 'TZ')`` when this column carries a timezone.

        The bare ``toDateTime(?)`` form lets chDB interpret the string in
        the *session* timezone, which silently shifts the stored value by
        whatever offset is set on the host. For ``DateTime('UTC')`` we
        must pass ``'UTC'`` as the second argument so the parsed value is
        anchored to UTC regardless of session TZ.
        """
        from sqlalchemy import func, literal

        if self.tz_name:
            return func.toDateTime(bindvalue, literal(self.tz_name))
        return func.toDateTime(bindvalue)


# Inherits from chDB ``DateTime`` (not ``types.DateTime``) so the
# colspecs-driven adapt() doesn't strip the ``precision`` /
# ``tz_name`` attributes by remapping to the parent class. See
# ``_ChdbType.adapt`` for the isinstance() guard.
class DateTime64(DateTime):
    __visit_name__ = "DateTime64"
    literal_processor = _datetime64_literal_processor
    # Override DateTime's second-precision bind: DateTime64 carries
    # ``precision`` and accepts sub-second digits.
    bind_processor = _datetime_bind_processor

    def __init__(self, precision: int = 3, timezone: str | None = None) -> None:
        if not 0 <= precision <= 9:
            raise ValueError("DateTime64 precision must be in [0, 9]")
        super().__init__(timezone=timezone)
        self.precision = precision

    def bind_expression(self, bindvalue):
        # ``toDateTime64`` is the only date/time CH function that requires
        # extra arguments (precision, optional timezone). Build the call
        # site-specifically rather than going through ``_wrap_bind``.
        from sqlalchemy import func, literal
        if self.tz_name:
            return func.toDateTime64(bindvalue, self.precision, literal(self.tz_name))
        return func.toDateTime64(bindvalue, self.precision)


# ``toTime(?)`` in chDB is a datetime-component extractor, not a string
# parser — it rejects ``toTime('10:11:12')`` with ILLEGAL_TYPE_OF_ARGUMENT.
# chDB's ``Time`` column does auto-cast a 'HH:MM:SS' string bind via the
# VALUES path, so we leave ``bind_expression`` unset and rely on the
# bind_processor's ISO string + chDB's implicit string→Time cast.
class Time(types.Time, _ChdbType):
    __visit_name__ = "Time"
    literal_processor = _time_literal_processor
    bind_processor = _time_bind_processor_secondprec


class Time64(Time):
    __visit_name__ = "Time64"
    literal_processor = _time64_literal_processor
    # Override Time's second-precision bind: Time64 keeps fractional digits.
    bind_processor = _time_bind_processor

    def __init__(self, precision: int = 3) -> None:
        if not 0 <= precision <= 9:
            raise ValueError("Time64 precision must be in [0, 9]")
        super().__init__()
        self.precision = precision


# -------------------- enum --------------------

class Enum(_ChdbType):
    """Generic ``Enum`` — sized variants are :class:`Enum8` / :class:`Enum16`."""

    __visit_name__ = "Enum"

    def __init__(self, members: dict[str, int]) -> None:
        if not members:
            raise ValueError("Enum requires at least one member")
        self.members = dict(members)


class Enum8(Enum):
    __visit_name__ = "Enum8"


class Enum16(Enum):
    __visit_name__ = "Enum16"


# -------------------- network --------------------

class IPv4(_ChdbType):
    __visit_name__ = "IPv4"


class IPv6(_ChdbType):
    __visit_name__ = "IPv6"
