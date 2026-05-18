"""DDL / type compiler — turns our SQLAlchemy type tree back into ClickHouse SQL.

Why this exists separately from :mod:`types`: SQLAlchemy's compiler resolves
``visit_<name>`` by ``__visit_name__`` and the compiler must know how to render
*every* chDB-specific type. We can't put the visitor methods on the type
classes themselves — SQLAlchemy expects them on a compiler subclass.

The most important consumer is ``LangChain.SQLDatabase.get_table_info()``,
which builds the system prompt by *reflecting a table and then re-emitting
its CREATE TABLE statement*. If any column type can't be compiled, the entire
prompt fails and the LLM never sees the schema.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.sql import compiler

from . import types as ct
from .types import composite


class ChdbTypeCompiler(compiler.GenericTypeCompiler):
    """Compiles chDB SQLAlchemy types back to ClickHouse DDL fragments."""

    # ------------------------------------------------------------------
    # String family
    # ------------------------------------------------------------------

    def visit_String(self, type_: ct.String, **kw: Any) -> str:
        return "String"

    def visit_FixedString(self, type_: ct.FixedString, **kw: Any) -> str:
        return f"FixedString({type_.length})"

    def visit_UUID(self, type_: ct.UUID, **kw: Any) -> str:  # type: ignore[override]
        return "UUID"

    # ------------------------------------------------------------------
    # Integer / float
    # ------------------------------------------------------------------

    def visit_Int8(self, type_: ct.Int8, **kw: Any) -> str:
        return "Int8"

    def visit_Int16(self, type_: ct.Int16, **kw: Any) -> str:
        return "Int16"

    def visit_Int32(self, type_: ct.Int32, **kw: Any) -> str:
        return "Int32"

    def visit_Int64(self, type_: ct.Int64, **kw: Any) -> str:
        return "Int64"

    def visit_Int128(self, type_: ct.Int128, **kw: Any) -> str:
        return "Int128"

    def visit_Int256(self, type_: ct.Int256, **kw: Any) -> str:
        return "Int256"

    def visit_UInt8(self, type_: ct.UInt8, **kw: Any) -> str:
        return "UInt8"

    def visit_UInt16(self, type_: ct.UInt16, **kw: Any) -> str:
        return "UInt16"

    def visit_UInt32(self, type_: ct.UInt32, **kw: Any) -> str:
        return "UInt32"

    def visit_UInt64(self, type_: ct.UInt64, **kw: Any) -> str:
        return "UInt64"

    def visit_UInt128(self, type_: ct.UInt128, **kw: Any) -> str:
        return "UInt128"

    def visit_UInt256(self, type_: ct.UInt256, **kw: Any) -> str:
        return "UInt256"

    def visit_Float32(self, type_: ct.Float32, **kw: Any) -> str:
        return "Float32"

    def visit_Float64(self, type_: ct.Float64, **kw: Any) -> str:
        return "Float64"

    def visit_BFloat16(self, type_: ct.BFloat16, **kw: Any) -> str:
        return "BFloat16"

    # ------------------------------------------------------------------
    # Decimal / Boolean
    # ------------------------------------------------------------------

    def visit_Decimal(self, type_: ct.Decimal, **kw: Any) -> str:
        return f"Decimal({type_.precision}, {type_.scale})"

    def visit_Boolean(self, type_: ct.Boolean, **kw: Any) -> str:  # type: ignore[override]
        return "Bool"

    # ------------------------------------------------------------------
    # Date / Time
    # ------------------------------------------------------------------

    def visit_Date(self, type_: ct.Date, **kw: Any) -> str:  # type: ignore[override]
        return "Date"

    def visit_Date32(self, type_: ct.Date32, **kw: Any) -> str:
        return "Date32"

    def visit_DateTime(self, type_: ct.DateTime, **kw: Any) -> str:  # type: ignore[override]
        if type_.tz_name:
            return f"DateTime('{type_.tz_name}')"
        return "DateTime"

    def visit_DateTime64(self, type_: ct.DateTime64, **kw: Any) -> str:
        if type_.tz_name:
            return f"DateTime64({type_.precision}, '{type_.tz_name}')"
        return f"DateTime64({type_.precision})"

    def visit_Time(self, type_: ct.Time, **kw: Any) -> str:  # type: ignore[override]
        return "Time"

    def visit_Time64(self, type_: ct.Time64, **kw: Any) -> str:
        return f"Time64({type_.precision})"

    # ------------------------------------------------------------------
    # Enum
    # ------------------------------------------------------------------

    def _enum_members_clause(self, members: dict[str, int]) -> str:
        return ", ".join(f"'{name}' = {value}" for name, value in members.items())

    def visit_Enum(self, type_: ct.Enum, **kw: Any) -> str:  # type: ignore[override]
        return f"Enum({self._enum_members_clause(type_.members)})"

    def visit_Enum8(self, type_: ct.Enum8, **kw: Any) -> str:
        return f"Enum8({self._enum_members_clause(type_.members)})"

    def visit_Enum16(self, type_: ct.Enum16, **kw: Any) -> str:
        return f"Enum16({self._enum_members_clause(type_.members)})"

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def visit_IPv4(self, type_: ct.IPv4, **kw: Any) -> str:
        return "IPv4"

    def visit_IPv6(self, type_: ct.IPv6, **kw: Any) -> str:
        return "IPv6"

    # ------------------------------------------------------------------
    # Modifiers / composites
    # ------------------------------------------------------------------

    def _process_inner(self, type_: Any, **kw: Any) -> str:
        """Compile a nested type using the same compiler dispatch."""
        return self.process(type_, **kw)

    def visit_Nullable(self, type_: composite.Nullable, **kw: Any) -> str:
        return f"Nullable({self._process_inner(type_.inner, **kw)})"

    def visit_LowCardinality(self, type_: composite.LowCardinality, **kw: Any) -> str:
        return f"LowCardinality({self._process_inner(type_.inner, **kw)})"

    def visit_Array(self, type_: composite.Array, **kw: Any) -> str:  # type: ignore[override]
        return f"Array({self._process_inner(type_.item_type, **kw)})"

    def visit_Tuple(self, type_: composite.Tuple, **kw: Any) -> str:
        inner = ", ".join(self._process_inner(t, **kw) for t in type_.element_types)
        return f"Tuple({inner})"

    def visit_Map(self, type_: composite.Map, **kw: Any) -> str:
        k = self._process_inner(type_.key_type, **kw)
        v = self._process_inner(type_.value_type, **kw)
        return f"Map({k}, {v})"

    def visit_Nested(self, type_: composite.Nested, **kw: Any) -> str:
        fields = ", ".join(
            f"{name} {self._process_inner(t, **kw)}" for name, t in type_.fields
        )
        return f"Nested({fields})"

    def visit_AggregateFunction(
        self, type_: composite.AggregateFunction, **kw: Any
    ) -> str:
        types_str = ", ".join(self._process_inner(t, **kw) for t in type_.arg_types)
        return f"AggregateFunction({type_.function}, {types_str})"

    def visit_SimpleAggregateFunction(
        self, type_: composite.SimpleAggregateFunction, **kw: Any
    ) -> str:
        return (
            f"SimpleAggregateFunction({type_.function}, "
            f"{self._process_inner(type_.value_type, **kw)})"
        )

    # ------------------------------------------------------------------
    # Modern (CH 24.x+)
    # ------------------------------------------------------------------

    def visit_Variant(self, type_: Any, **kw: Any) -> str:
        inner = ", ".join(self._process_inner(t, **kw) for t in type_.alternatives)
        return f"Variant({inner})"

    def visit_Dynamic(self, type_: Any, **kw: Any) -> str:
        if type_.max_types is not None:
            return f"Dynamic(max_types={type_.max_types})"
        return "Dynamic"

    def visit_JSON(self, type_: Any, **kw: Any) -> str:  # type: ignore[override]
        return "JSON"

    def visit_JSONLegacy(self, type_: Any, **kw: Any) -> str:
        return "Object('json')"

    # ------------------------------------------------------------------
    # Geo
    # ------------------------------------------------------------------

    def visit_Point(self, type_: Any, **kw: Any) -> str:
        return "Point"

    def visit_Ring(self, type_: Any, **kw: Any) -> str:
        return "Ring"

    def visit_Polygon(self, type_: Any, **kw: Any) -> str:
        return "Polygon"

    def visit_MultiPolygon(self, type_: Any, **kw: Any) -> str:
        return "MultiPolygon"

    # ------------------------------------------------------------------
    # SQLAlchemy generic types — needed by the SA dialect compliance
    # suite (L2), which declares ``Column(Integer)`` / ``Column(ARRAY(T))``
    # / ``Column(NUMERIC(p, s))`` and expects the dialect to translate
    # them to its native syntax. The chDB-specific classes in
    # :mod:`chdb_sqlalchemy.types` are the canonical surface; these
    # visitors map the generic SA classes onto the same chDB SQL output.
    # ------------------------------------------------------------------

    def visit_INTEGER(self, type_: Any, **kw: Any) -> str:
        return "Int32"

    def visit_SMALLINT(self, type_: Any, **kw: Any) -> str:
        return "Int16"

    def visit_BIGINT(self, type_: Any, **kw: Any) -> str:
        return "Int64"

    def visit_VARCHAR(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_NVARCHAR(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_CHAR(self, type_: Any, **kw: Any) -> str:
        # CHAR(N) → FixedString(N) is the closest mapping.
        length = getattr(type_, "length", None)
        return f"FixedString({length})" if length else "String"

    def visit_NCHAR(self, type_: Any, **kw: Any) -> str:
        return self.visit_CHAR(type_, **kw)

    def visit_TEXT(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_CLOB(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_BLOB(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_BOOLEAN(self, type_: Any, **kw: Any) -> str:
        return "Bool"

    def visit_DATE(self, type_: Any, **kw: Any) -> str:
        return "Date"

    def visit_DATETIME(self, type_: Any, **kw: Any) -> str:
        return "DateTime"

    def visit_TIMESTAMP(self, type_: Any, **kw: Any) -> str:
        return "DateTime"

    def visit_TIME(self, type_: Any, **kw: Any) -> str:
        return "Time"

    def visit_FLOAT(self, type_: Any, **kw: Any) -> str:
        return "Float64"

    def visit_REAL(self, type_: Any, **kw: Any) -> str:
        return "Float32"

    def visit_DOUBLE(self, type_: Any, **kw: Any) -> str:
        return "Float64"

    def visit_DOUBLE_PRECISION(self, type_: Any, **kw: Any) -> str:
        return "Float64"

    def visit_NUMERIC(self, type_: Any, **kw: Any) -> str:
        p = getattr(type_, "precision", None)
        s = getattr(type_, "scale", None)
        if p is not None:
            return f"Decimal({p}, {s if s is not None else 0})"
        # Unparameterised NUMERIC — use a sensible default.
        return "Decimal(38, 9)"

    def visit_DECIMAL(self, type_: Any, **kw: Any) -> str:
        return self.visit_NUMERIC(type_, **kw)

    def visit_UUID(self, type_: Any, **kw: Any) -> str:
        return "UUID"

    def visit_JSON(self, type_: Any, **kw: Any) -> str:
        return "JSON"

    def visit_ARRAY(self, type_: Any, **kw: Any) -> str:
        # SA generic ``ARRAY(item_type)`` → chDB ``Array(T)``.
        inner = self.process(type_.item_type, **kw) if type_.item_type is not None else "String"
        return f"Array({inner})"

    def visit_null(self, type_: Any, **kw: Any) -> str:
        return "Nullable(String)"

    def visit_string(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_unicode(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_unicode_text(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_text(self, type_: Any, **kw: Any) -> str:
        return "String"

    def visit_integer(self, type_: Any, **kw: Any) -> str:
        return "Int32"

    def visit_big_integer(self, type_: Any, **kw: Any) -> str:
        return "Int64"

    def visit_small_integer(self, type_: Any, **kw: Any) -> str:
        return "Int16"

    def visit_float(self, type_: Any, **kw: Any) -> str:
        return "Float64"

    def visit_numeric(self, type_: Any, **kw: Any) -> str:
        return self.visit_NUMERIC(type_, **kw)

    def visit_boolean(self, type_: Any, **kw: Any) -> str:
        return "Bool"

    def visit_date(self, type_: Any, **kw: Any) -> str:
        return "Date"

    def visit_datetime(self, type_: Any, **kw: Any) -> str:
        return "DateTime"

    def visit_time(self, type_: Any, **kw: Any) -> str:
        return "Time"

    def visit_uuid(self, type_: Any, **kw: Any) -> str:
        return "UUID"

    def visit_enum(self, type_: Any, **kw: Any) -> str:
        # SA generic Enum — caller declares the members via Python Enum class
        # or string list. Map to Enum8 with sequential integer codes; v0.3
        # may add Enum16 promotion for >127 members.
        enums = getattr(type_, "enums", None) or []
        if not enums:
            return "String"  # graceful fallback
        members = ", ".join(f"'{name}' = {i + 1}" for i, name in enumerate(enums))
        return f"Enum8({members})"

    def visit_large_binary(self, type_: Any, **kw: Any) -> str:
        return "String"


class ChdbDDLCompiler(compiler.DDLCompiler):
    """DDL compiler — emits chDB-flavoured CREATE TABLE / engine clauses.

    Our v0.1 strategy is conservative: we let SQLAlchemy's GenericDDLCompiler
    do most of the work, and only override the bits where ClickHouse SQL
    diverges from the generic SQL it would emit. The big one is the
    ``ENGINE = MergeTree ORDER BY (...)`` suffix, which has no standard
    SQL equivalent.

    For v0.1 reflection we don't *regenerate* DDL with engine specs; LangChain
    just needs the column-list portion to render. The post_create_table_clause
    hook below stays empty until v0.3.
    """

    def post_create_table(self, table: Any) -> str:
        """Append ENGINE clause. v0.1: stay quiet; v0.3 reads from reflection."""
        return ""

    def visit_primary_key_constraint(self, constraint: Any, **kw: Any) -> str:
        # ClickHouse PKs are not enforced; they map to the MergeTree ORDER BY.
        # Emitting "PRIMARY KEY (...)" would be valid syntax but redundant
        # with the engine clause. We suppress the generic emission.
        return ""

    def visit_foreign_key_constraint(self, constraint: Any, **kw: Any) -> str:
        # No foreign keys in ClickHouse / chDB.
        return ""
