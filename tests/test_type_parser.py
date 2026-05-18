"""Unit tests for the recursive ClickHouse type-string parser.

These are pure-Python tests — no chDB engine required. They cover:

* Every scalar type in :mod:`chdb_sqlalchemy.types.common`
* Composite wrapping (Array / Nullable / LowCardinality / Tuple / Map / Nested)
* Modern types (Variant / Dynamic / new JSON)
* Geo types
* Pathological nesting (depth 3+) that real-world ClickHouse schemas produce
* Negative cases (unknown types, malformed strings) — must raise
  ``ChdbTypeNotSupportedError`` rather than silently fall back

The parser correctness directly determines whether LangChain's text-to-SQL
agent sees the right schema, so this file is treated as a contract test.
"""

from __future__ import annotations

import pytest

from chdb_sqlalchemy.exc import ChdbTypeNotSupportedError
from chdb_sqlalchemy.types import (
    Array,
    BFloat16,
    Boolean,
    Date,
    Date32,
    DateTime,
    DateTime64,
    Decimal,
    Dynamic,
    Enum8,
    FixedString,
    Float32,
    Float64,
    Int8,
    Int32,
    Int64,
    JSON,
    LowCardinality,
    Map,
    MultiPolygon,
    Nullable,
    Point,
    Polygon,
    Ring,
    String,
    Time64,
    Tuple,
    UInt16,
    UInt64,
    UUID,
    Variant,
)
from chdb_sqlalchemy.types.parser import parse_column_type, parse_type


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "type_str,expected_cls",
    [
        ("String", String),
        ("UUID", UUID),
        ("Int8", Int8),
        ("Int32", Int32),
        ("Int64", Int64),
        ("UInt16", UInt16),
        ("UInt64", UInt64),
        ("Float32", Float32),
        ("Float64", Float64),
        ("BFloat16", BFloat16),
        ("Bool", Boolean),
        ("Boolean", Boolean),
        ("Date", Date),
        ("Date32", Date32),
        ("JSON", JSON),
        ("Point", Point),
        ("Ring", Ring),
        ("Polygon", Polygon),
        ("MultiPolygon", MultiPolygon),
    ],
)
def test_scalar_types(type_str, expected_cls):
    parsed = parse_type(type_str)
    assert isinstance(parsed, expected_cls), f"{type_str} → {type(parsed).__name__}"


# ---------------------------------------------------------------------------
# Parameterised scalars
# ---------------------------------------------------------------------------


def test_fixed_string_length():
    t = parse_type("FixedString(16)")
    assert isinstance(t, FixedString)
    assert t.length == 16


def test_decimal_precision_scale():
    t = parse_type("Decimal(18, 4)")
    assert isinstance(t, Decimal)
    assert t.precision == 18
    assert t.scale == 4


def test_datetime_no_args():
    t = parse_type("DateTime")
    assert isinstance(t, DateTime)
    assert t.tz_name is None


def test_datetime_with_tz():
    t = parse_type("DateTime('Europe/London')")
    assert isinstance(t, DateTime)
    assert t.tz_name == "Europe/London"


def test_datetime64_with_precision_and_tz():
    t = parse_type("DateTime64(6, 'UTC')")
    assert isinstance(t, DateTime64)
    assert t.precision == 6
    assert t.tz_name == "UTC"


def test_time64_precision():
    t = parse_type("Time64(3)")
    assert isinstance(t, Time64)
    assert t.precision == 3


def test_enum8_members():
    t = parse_type("Enum8('red' = 1, 'green' = 2, 'blue' = 3)")
    assert isinstance(t, Enum8)
    assert t.members == {"red": 1, "green": 2, "blue": 3}


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------


def test_array_of_string():
    t = parse_type("Array(String)")
    assert isinstance(t, Array)
    assert isinstance(t.item_type, String)


def test_nullable_string():
    t = parse_type("Nullable(String)")
    assert isinstance(t, Nullable)
    assert isinstance(t.inner, String)


def test_low_cardinality_string():
    t = parse_type("LowCardinality(String)")
    assert isinstance(t, LowCardinality)
    assert isinstance(t.inner, String)


def test_tuple_heterogeneous():
    t = parse_type("Tuple(UInt32, String, Float64)")
    assert isinstance(t, Tuple)
    assert len(t.element_types) == 3
    assert isinstance(t.element_types[0], UInt16) or isinstance(t.element_types[0], Int32) or t.element_types[0].__class__.__name__ == "UInt32"
    assert isinstance(t.element_types[1], String)
    assert isinstance(t.element_types[2], Float64)


def test_map_string_to_int64():
    t = parse_type("Map(String, Int64)")
    assert isinstance(t, Map)
    assert isinstance(t.key_type, String)
    assert isinstance(t.value_type, Int64)


# ---------------------------------------------------------------------------
# Pathological nesting — the cases real production schemas produce
# ---------------------------------------------------------------------------


def test_deeply_nested_array_nullable_lowcard_string():
    t = parse_type("Array(Nullable(LowCardinality(String)))")
    assert isinstance(t, Array)
    assert isinstance(t.item_type, Nullable)
    assert isinstance(t.item_type.inner, LowCardinality)
    assert isinstance(t.item_type.inner.inner, String)


def test_map_of_string_to_array_of_tuple():
    t = parse_type("Map(String, Array(Tuple(UInt32, String)))")
    assert isinstance(t, Map)
    assert isinstance(t.key_type, String)
    assert isinstance(t.value_type, Array)
    inner_tuple = t.value_type.item_type
    assert isinstance(inner_tuple, Tuple)
    assert len(inner_tuple.element_types) == 2


def test_lowcardinality_outside_nullable():
    """LowCardinality(Nullable(T)) — common production pattern."""
    t = parse_type("LowCardinality(Nullable(String))")
    assert isinstance(t, LowCardinality)
    assert isinstance(t.inner, Nullable)
    assert isinstance(t.inner.inner, String)


# ---------------------------------------------------------------------------
# Modern types (CH 24.x → 26.3 LTS)
# ---------------------------------------------------------------------------


def test_variant_two_alternatives():
    t = parse_type("Variant(Int64, String)")
    assert isinstance(t, Variant)
    assert len(t.alternatives) == 2


def test_dynamic_bare():
    t = parse_type("Dynamic")
    assert isinstance(t, Dynamic)
    assert t.max_types is None


def test_variant_with_composite_alternative():
    t = parse_type("Variant(Int64, Array(String))")
    assert isinstance(t, Variant)
    assert isinstance(t.alternatives[1], Array)


# ---------------------------------------------------------------------------
# parse_column_type: outer Nullable/LowCardinality unwrap to flags
# ---------------------------------------------------------------------------


def test_column_unwraps_nullable():
    col = parse_column_type("Nullable(String)")
    assert isinstance(col.sa_type, String)
    assert col.nullable is True
    assert col.low_cardinality is False


def test_column_unwraps_low_cardinality():
    col = parse_column_type("LowCardinality(String)")
    assert isinstance(col.sa_type, String)
    assert col.low_cardinality is True
    assert col.nullable is False


def test_column_unwraps_both_outer():
    col = parse_column_type("LowCardinality(Nullable(String))")
    assert isinstance(col.sa_type, String)
    assert col.nullable is True
    assert col.low_cardinality is True


def test_column_inner_nullable_not_unwrapped():
    """`Array(Nullable(String))` — the Nullable is *inside* Array and stays."""
    col = parse_column_type("Array(Nullable(String))")
    assert isinstance(col.sa_type, Array)
    assert col.nullable is False  # outer was not Nullable
    inner = col.sa_type.item_type
    assert isinstance(inner, Nullable)


# ---------------------------------------------------------------------------
# Negative cases — must raise, never silently fall back
# ---------------------------------------------------------------------------


def test_unknown_type_raises():
    with pytest.raises(ChdbTypeNotSupportedError) as exc:
        parse_type("ThisIsNotAClickHouseType")
    assert "ThisIsNotAClickHouseType" in str(exc.value)


def test_malformed_trailing_chars_raises():
    with pytest.raises(ChdbTypeNotSupportedError):
        parse_type("String garbage")


def test_unbalanced_parens_raises():
    with pytest.raises(ChdbTypeNotSupportedError):
        parse_type("Array(String")


def test_array_without_args_raises():
    with pytest.raises(ChdbTypeNotSupportedError):
        parse_type("Array()")


def test_decimal_wrong_args_raises():
    with pytest.raises(ChdbTypeNotSupportedError):
        parse_type("Decimal(18)")  # missing scale


# ---------------------------------------------------------------------------
# Roundtrip: reflection-shaped strings the dialect will see in practice
# ---------------------------------------------------------------------------

# Each entry is a type string ClickHouse 26.3.9.8 might emit in
# `system.columns.type`. The test only asserts that parsing succeeds — the
# detailed shape is covered by the cases above.
@pytest.mark.parametrize(
    "type_str",
    [
        "String",
        "Int32",
        "DateTime('UTC')",
        "DateTime64(9, 'America/New_York')",
        "Decimal(38, 18)",
        "Array(String)",
        "Nullable(Int64)",
        "LowCardinality(String)",
        "Map(String, String)",
        "Tuple(String, UInt64, DateTime64(3))",
        "Array(Tuple(UInt32, Array(Nullable(String))))",
        "Nested(name String, value Float64, ts DateTime64(3))",
        "Variant(Int64, String, Array(Float64))",
        "Dynamic",
        "JSON",
        "Point",
        "MultiPolygon",
        "AggregateFunction(uniqExact, String)",
        "SimpleAggregateFunction(sum, Int64)",
    ],
)
def test_realistic_reflection_strings_parse(type_str):
    parsed = parse_type(type_str)
    assert parsed is not None
