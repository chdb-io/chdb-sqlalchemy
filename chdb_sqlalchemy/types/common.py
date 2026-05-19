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

def _date_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    """Render Python date as ``toDate('YYYY-MM-DD')`` so chDB stores it
    as Date not String."""
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        return f"toDate('{value.isoformat() if hasattr(value, 'isoformat') else value}')"
    return process


def _datetime_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        iso = value.isoformat().replace("T", " ") if hasattr(value, "isoformat") else str(value)
        return f"toDateTime('{iso}')"
    return process


def _datetime64_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        iso = value.isoformat().replace("T", " ") if hasattr(value, "isoformat") else str(value)
        precision = getattr(self, "precision", 3)
        return f"toDateTime64('{iso}', {precision})"
    return process


def _time_literal_processor(self: Any, dialect: Any) -> Callable[[Any], str]:
    def process(value: Any) -> str:
        if value is None:
            return "NULL"
        return f"toTime('{value.isoformat() if hasattr(value, 'isoformat') else value}')"
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
    bind_expression = _wrap_bind("toDate")


class Date32(types.Date, _ChdbType):
    __visit_name__ = "Date32"
    literal_processor = _date_literal_processor
    bind_expression = _wrap_bind("toDate32")


class DateTime(types.DateTime, _ChdbType):
    __visit_name__ = "DateTime"
    literal_processor = _datetime_literal_processor
    bind_expression = _wrap_bind("toDateTime")

    def __init__(self, timezone: str | None = None) -> None:
        super().__init__(timezone=bool(timezone))
        self.tz_name = timezone


class DateTime64(types.DateTime, _ChdbType):
    __visit_name__ = "DateTime64"
    literal_processor = _datetime64_literal_processor
    bind_expression = _wrap_bind("toDateTime64")

    def __init__(self, precision: int = 3, timezone: str | None = None) -> None:
        if not 0 <= precision <= 9:
            raise ValueError("DateTime64 precision must be in [0, 9]")
        super().__init__(timezone=bool(timezone))
        self.precision = precision
        self.tz_name = timezone


class Time(types.Time, _ChdbType):
    __visit_name__ = "Time"
    literal_processor = _time_literal_processor


class Time64(types.Time, _ChdbType):
    __visit_name__ = "Time64"
    literal_processor = _time_literal_processor

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
