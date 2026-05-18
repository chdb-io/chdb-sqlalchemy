# Type mapping reference

Authoritative mapping from ClickHouse 26.3.9.8-lts types (as emitted in
`system.columns.type`) to the SQLAlchemy types `chdb-sqlalchemy` returns
from reflection.

## Scalars

| ClickHouse | SQLAlchemy class | Notes |
|---|---|---|
| `String` | `chdb_sqlalchemy.types.String` | UTF-8, no length limit |
| `FixedString(N)` | `FixedString(length=N)` | Padded with zero bytes |
| `UUID` | `UUID` | |
| `Int8` … `Int256` | `Int8`, `Int16`, `Int32`, `Int64`, `Int128`, `Int256` | 128/256-bit map to BigInteger |
| `UInt8` … `UInt256` | `UInt8`, `UInt16`, `UInt32`, `UInt64`, `UInt128`, `UInt256` | |
| `Float32`, `Float64` | `Float32`, `Float64` | IEEE 754 |
| `BFloat16` | `BFloat16` | Brain Float — CH 24.6+ |
| `Decimal(P, S)` | `Decimal(precision=P, scale=S)` | `asdecimal=True` by default |
| `Bool` / `Boolean` | `Boolean` | |
| `Date`, `Date32` | `Date`, `Date32` | Date32 extends range |
| `DateTime[(TZ)]` | `DateTime(timezone=TZ)` | TZ kept as string |
| `DateTime64(P[, TZ])` | `DateTime64(precision=P, timezone=TZ)` | P in [0, 9] |
| `Time`, `Time64(P)` | `Time`, `Time64(precision=P)` | |
| `Enum8(...)`, `Enum16(...)` | `Enum8(members={...})`, `Enum16(members={...})` | Members dict preserved |
| `IPv4`, `IPv6` | `IPv4`, `IPv6` | |

## Composites

| ClickHouse | SQLAlchemy class | Notes |
|---|---|---|
| `Nullable(T)` | unwrapped to `nullable=True` on outer Column | Inner `Nullable` stays nested |
| `LowCardinality(T)` | unwrapped to `dialect_options['chdb']['low_cardinality']=True` | DDL preserves wrapper |
| `Array(T)` | `Array(item_type=T)` | |
| `Tuple(T1, T2, ...)` | `Tuple(element_types=(T1, T2, ...))` | |
| `Map(K, V)` | `Map(key_type=K, value_type=V)` | |
| `Nested(f1 T1, ...)` | `Nested(fields=[(f1, T1), ...])` | |
| `AggregateFunction(fn, T...)` | `AggregateFunction(function=fn, arg_types=(T...))` | |
| `SimpleAggregateFunction(fn, T)` | `SimpleAggregateFunction(function=fn, value_type=T)` | |

## Modern (CH 24.x → 26.3 LTS)

| ClickHouse | SQLAlchemy class | Notes |
|---|---|---|
| `Variant(T1, T2, ...)` | `Variant(alternatives=(T1, ...))` | Sum type |
| `Dynamic` / `Dynamic(max_types=N)` | `Dynamic(max_types=N)` | |
| `JSON` (24.10+ semantic) | `JSON` | Maps to SA's `JSON` |
| `Object('json')` (legacy) | `JSONLegacy` | String-backed pre-24.10 |

## Geo

| ClickHouse | SQLAlchemy class | Underlying shape |
|---|---|---|
| `Point` | `Point` | `Tuple(Float64, Float64)` |
| `Ring` | `Ring` | `Array(Point)` |
| `Polygon` | `Polygon` | `Array(Ring)` |
| `MultiPolygon` | `MultiPolygon` | `Array(Polygon)` |

## Not yet supported

Reflection on any of these will raise `ChdbTypeNotSupportedError` — **no
silent fallback**. Add a mapping in `chdb_sqlalchemy.types` and a builder
in `parser.py` to extend coverage.

- `Object(...)` with non-`'json'` argument (deprecated alternative forms)
- Experimental types not yet stabilised in CH 26.3
