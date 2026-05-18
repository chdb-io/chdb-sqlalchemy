"""ClickHouse / chDB type system for SQLAlchemy.

The public surface intentionally mirrors what reflection produces: every
ClickHouse type string parseable by :func:`parser.parse_type` returns one
of the type instances exported here. Importing this module also registers
the type names with SQLAlchemy's compiler dispatch.
"""

from __future__ import annotations

from .common import (
    UUID,
    BFloat16,
    Boolean,
    Date,
    Date32,
    DateTime,
    DateTime64,
    Decimal,
    Enum,
    Enum8,
    Enum16,
    FixedString,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Int256,
    IPv4,
    IPv6,
    String,
    Time,
    Time64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    UInt256,
)
from .composite import (
    AggregateFunction,
    Array,
    LowCardinality,
    Map,
    Nested,
    Nullable,
    SimpleAggregateFunction,
    Tuple,
)
from .dynamic import Dynamic
from .geo import MultiPolygon, Point, Polygon, Ring
from .json_ import JSON, JSONLegacy
from .variant import Variant

__all__ = [
    # Basic
    "BFloat16",
    "Boolean",
    "Date",
    "Date32",
    "DateTime",
    "DateTime64",
    "Decimal",
    "Enum",
    "Enum8",
    "Enum16",
    "FixedString",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "Int256",
    "IPv4",
    "IPv6",
    "String",
    "Time",
    "Time64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt128",
    "UInt256",
    "UUID",
    # Composite
    "AggregateFunction",
    "Array",
    "LowCardinality",
    "Map",
    "Nested",
    "Nullable",
    "SimpleAggregateFunction",
    "Tuple",
    # Modern (CH 24.x+ → 26.3 LTS)
    "Dynamic",
    "JSON",
    "JSONLegacy",
    "Variant",
    # Geo
    "MultiPolygon",
    "Point",
    "Polygon",
    "Ring",
]
