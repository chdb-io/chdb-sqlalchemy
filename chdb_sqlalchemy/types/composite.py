"""Composite ClickHouse types that wrap one or more inner types.

These are the types where the recursive parser earns its keep: a real
production column can be ``Array(Nullable(LowCardinality(String)))`` or
``Map(String, Array(Tuple(UInt32, String)))``, and reflection must round-trip
the full tree.

Result-processor note: as of chDB 26.3 the underlying ``chdb.dbapi`` driver
returns composite cells (Array, Map, Tuple, JSON) as **Python repr-style
strings** rather than native Python collections — e.g. ``"['a', 'b']"``
instead of ``['a', 'b']``. We override ``result_processor()`` on each
composite class to ``ast.literal_eval`` the cell back into a real Python
object so downstream consumers (LangChain sample-row rendering, pandas,
Django ORM) see what they expect. The upstream issue should be tracked
against chdb-io/chdb; once chDB emits native collections this shim can
be deleted.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from sqlalchemy import types

from .common import _ChdbType


def _literal_eval_processor(value: Any) -> Any:
    """Convert a Python repr-style string back to a native object.

    Pass-throughs ``None`` and already-native types unchanged so the
    processor stays safe if chDB ever upgrades to native return values.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Defensive: if chDB ever returns an unrecognised string form
        # (e.g. ``inf`` literal that ast can't handle), return the raw
        # string so callers can decide rather than crashing.
        return value


class Nullable(_ChdbType):
    """``Nullable(T)`` — wraps an inner type to allow NULL values.

    ``Nullable`` is a *modifier*, not a regular composite type. Many tools
    expect the inner type to surface at the SQLAlchemy level with
    ``nullable=True`` on the Column; the parser handles that translation in
    :mod:`chdb_sqlalchemy.types.parser`.
    """

    __visit_name__ = "Nullable"

    def __init__(self, inner: types.TypeEngine) -> None:
        self.inner = inner


class LowCardinality(_ChdbType):
    """``LowCardinality(T)`` — dictionary-encoded storage hint, transparent at SQL level."""

    __visit_name__ = "LowCardinality"

    def __init__(self, inner: types.TypeEngine) -> None:
        self.inner = inner


class Array(_ChdbType):
    """``Array(T)``."""

    __visit_name__ = "Array"

    def __init__(self, item_type: types.TypeEngine) -> None:
        self.item_type = item_type

    def result_processor(self, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
        return _literal_eval_processor


class Tuple(_ChdbType):
    """``Tuple(T1, T2, ...)`` — heterogeneous ordered collection."""

    __visit_name__ = "Tuple"

    def __init__(self, *element_types: types.TypeEngine) -> None:
        if not element_types:
            raise ValueError("Tuple requires at least one element type")
        self.element_types = element_types

    def result_processor(self, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
        return _literal_eval_processor


class Map(_ChdbType):
    """``Map(K, V)``."""

    __visit_name__ = "Map"

    def __init__(self, key_type: types.TypeEngine, value_type: types.TypeEngine) -> None:
        self.key_type = key_type
        self.value_type = value_type

    def result_processor(self, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
        return _literal_eval_processor


class Nested(_ChdbType):
    """``Nested(field1 T1, field2 T2, ...)``.

    Semantically a shorthand for parallel arrays. We expose the field
    declarations so DDL can be reproduced; result_processor flattens to
    list-of-dicts.
    """

    __visit_name__ = "Nested"

    def __init__(self, fields: list[tuple[str, types.TypeEngine]]) -> None:
        if not fields:
            raise ValueError("Nested requires at least one field")
        self.fields = list(fields)


class AggregateFunction(_ChdbType):
    """``AggregateFunction(func, T1, T2, ...)`` — intermediate aggregation state.

    Round-trippable as bytes; not directly queryable without ``-Merge`` combinator.
    """

    __visit_name__ = "AggregateFunction"

    def __init__(self, function: str, *arg_types: types.TypeEngine) -> None:
        self.function = function
        self.arg_types = arg_types


class SimpleAggregateFunction(_ChdbType):
    """``SimpleAggregateFunction(func, T)`` — same as above but state == final value."""

    __visit_name__ = "SimpleAggregateFunction"

    def __init__(self, function: str, value_type: types.TypeEngine) -> None:
        self.function = function
        self.value_type = value_type
