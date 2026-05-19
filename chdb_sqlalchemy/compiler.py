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

from typing import Any, ClassVar

from sqlalchemy.sql import compiler
from sqlalchemy.sql import sqltypes as _sqltypes

from . import types as ct
from .types import composite

# Literal-render helper — SA's render_literal_value() needs a type instance.
_SAStringType = _sqltypes.String()


class ChdbSQLCompiler(compiler.SQLCompiler):
    """chDB-flavoured SQL statement compiler.

    Overrides:
    * ``compound_keywords`` — emit ``UNION DISTINCT`` for bare UNION
      (ClickHouse rejects bare UNION as ambiguous)
    * ``limit_clause`` — handle OFFSET-without-LIMIT (SA emits
      ``LIMIT -1 OFFSET m``; chDB rejects negative). Substitute the
      UInt64 sentinel.
    """

    compound_keywords: ClassVar[dict] = dict(compiler.SQLCompiler.compound_keywords)
    # mypy thinks ``CompoundSelect.UNION`` doesn't exist on the generic
    # alias type, but it does at runtime (UNION/INTERSECT/EXCEPT are
    # class attributes on CompoundSelect). Index assignment is fine.
    compound_keywords[
        compiler.selectable.CompoundSelect.UNION  # type: ignore[attr-defined]
    ] = "UNION DISTINCT"

    # UInt64 max value chDB happily ignores when paired with OFFSET.
    _LIMIT_UNLIMITED = 18446744073709551615  # 2**64 - 1

    def visit_delete(self, delete_stmt: Any, **kw: Any) -> str:
        """ClickHouse uses ``ALTER TABLE t DELETE WHERE ...`` not ``DELETE FROM t``.

        Build the equivalent ALTER TABLE form. SA's generic visit_delete
        produces ``DELETE FROM t [WHERE ...]`` which chDB parses as
        ``SELECT DELETE FROM t`` and errors with UNKNOWN_IDENTIFIER.
        """
        # Pull the target table + the WHERE clause.
        table = delete_stmt.table
        full = self.preparer.format_table(table)
        where_clause = delete_stmt._where_criteria
        if where_clause:
            # Combine multiple criteria with AND, compile each
            from sqlalchemy.sql.elements import BooleanClauseList
            if len(where_clause) > 1:
                where = BooleanClauseList._construct_raw(
                    __import__("sqlalchemy").and_, where_clause
                )
            else:
                where = where_clause[0]
            where_sql = self.process(where, **kw)
            return f"ALTER TABLE {full} DELETE WHERE {where_sql}"
        # No WHERE: delete everything
        return f"ALTER TABLE {full} DELETE WHERE 1=1"

    def limit_clause(self, select: Any, **kw: Any) -> str:
        # SA's default uses ``LIMIT %s OFFSET %s`` or ``LIMIT %s`` /
        # ``OFFSET %s``. With OFFSET-only, SA emits ``LIMIT -1 OFFSET N``
        # which chDB rejects.
        text = ""
        if select._limit_clause is not None:
            limit_val = self.process(select._limit_clause, **kw)
            text += f"\n LIMIT {limit_val}"
        elif select._offset_clause is not None:
            # OFFSET without LIMIT — chDB requires both
            text += f"\n LIMIT {self._LIMIT_UNLIMITED}"
        if select._offset_clause is not None:
            text += f" OFFSET {self.process(select._offset_clause, **kw)}"
        return text


class ChdbTypeCompiler(compiler.GenericTypeCompiler):
    """Compiles chDB SQLAlchemy types back to ClickHouse DDL fragments."""

    # ------------------------------------------------------------------
    # String family
    # ------------------------------------------------------------------

    def visit_String(self, type_: ct.String, **kw: Any) -> str:
        return "String"

    def visit_FixedString(self, type_: ct.FixedString, **kw: Any) -> str:
        return f"FixedString({type_.length})"

    def visit_UUID(self, type_: ct.UUID, **kw: Any) -> str:
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

    def visit_Boolean(self, type_: ct.Boolean, **kw: Any) -> str:
        return "Bool"

    # ------------------------------------------------------------------
    # Date / Time
    # ------------------------------------------------------------------

    def visit_Date(self, type_: ct.Date, **kw: Any) -> str:
        return "Date"

    def visit_Date32(self, type_: ct.Date32, **kw: Any) -> str:
        return "Date32"

    def visit_DateTime(self, type_: ct.DateTime, **kw: Any) -> str:
        if type_.tz_name:
            return f"DateTime('{type_.tz_name}')"
        return "DateTime"

    def visit_DateTime64(self, type_: ct.DateTime64, **kw: Any) -> str:
        if type_.tz_name:
            return f"DateTime64({type_.precision}, '{type_.tz_name}')"
        return f"DateTime64({type_.precision})"

    def visit_Time(self, type_: ct.Time, **kw: Any) -> str:
        return "Time"

    def visit_Time64(self, type_: ct.Time64, **kw: Any) -> str:
        return f"Time64({type_.precision})"

    # ------------------------------------------------------------------
    # Enum
    # ------------------------------------------------------------------

    def _enum_members_clause(self, members: dict[str, int]) -> str:
        return ", ".join(f"'{name}' = {value}" for name, value in members.items())

    def visit_Enum(self, type_: ct.Enum, **kw: Any) -> str:
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

    def visit_Array(self, type_: composite.Array, **kw: Any) -> str:
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

    def visit_JSON(self, type_: Any, **kw: Any) -> str:
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

    # Note: ``visit_UUID`` and ``visit_JSON`` are defined earlier in this
    # class for our typed ``ct.UUID``/``ct.JSON`` classes; the late
    # duplicates have been removed to avoid mypy [no-redef] noise. They
    # also handle SA generic ``UUID``/``JSON`` types because those are
    # bound via ``colspecs`` in the dialect.

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


def _column_is_chdb_nullable_already(column: Any) -> bool:
    """True if the column's type is already wrapped in ``Nullable(...)``
    or ``LowCardinality(Nullable(...))`` — avoids double-wrapping."""
    from .types.composite import LowCardinality, Nullable

    t = column.type
    if isinstance(t, Nullable):
        return True
    return bool(isinstance(t, LowCardinality) and isinstance(t.inner, Nullable))


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
        """Append the ``ENGINE = ... ORDER BY (...)`` suffix every chDB
        ``CREATE TABLE`` needs.

        SA-generic CREATE TABLE produces just the column list. ClickHouse /
        chDB requires every table to declare an engine — at minimum
        ``ENGINE = MergeTree`` with an ``ORDER BY`` clause (or
        ``ORDER BY tuple()`` to opt out of sorting).

        We pick ``MergeTree`` and use the primary-key columns as the
        sorting key when available — this both satisfies chDB's
        constraint and gives reflection a faithful PK round-trip via
        ``system.tables.sorting_key``. For tables without a PK we fall
        back to ``ORDER BY tuple()`` (no sort key).
        """
        pk_cols = [c.name for c in table.primary_key.columns]
        if pk_cols:
            order_by = ", ".join(self.preparer.quote(c) for c in pk_cols)
            return f"\nENGINE = MergeTree\nORDER BY ({order_by})"
        return "\nENGINE = MergeTree\nORDER BY tuple()"

    def get_column_specification(self, column: Any, **kwargs: Any) -> str:
        """Emit ``<name> Nullable(<type>)`` when SA Column has ``nullable=True``.

        ClickHouse defaults to NOT NULL. SA's generic SQL default is NULL.
        Our reflection unwraps ``Nullable(T)`` to ``nullable=True`` on the
        Column; this method does the reverse — wraps ``Nullable(...)``
        around the rendered type spec when needed for round-trip DDL.

        We also suppress the ``NOT NULL`` suffix SA's default appends —
        chDB doesn't need it because non-Nullable types are already
        not-null at the type level.
        """
        col_name = self.preparer.format_column(column)
        type_spec = self.dialect.type_compiler_instance.process(
            column.type, type_expression=column
        )
        # Wrap nullable columns in Nullable() if not already wrapped.
        if column.nullable and not _column_is_chdb_nullable_already(column):
            type_spec = f"Nullable({type_spec})"

        colspec = f"{col_name} {type_spec}"

        default = self.get_column_default_string(column)
        if default is not None:
            colspec += " DEFAULT " + default

        # chDB doesn't accept NOT NULL — non-Nullable types are already
        # not-null at the type level.
        return colspec

    def create_table_constraints(
        self, table: Any, _include_foreign_key_constraints: Any = None, **kw: Any
    ) -> str:
        """Emit only constraints chDB supports inside CREATE TABLE.

        SA's default emits PK / FK / CHECK / UNIQUE. chDB enforces none of
        them — PK is just MergeTree ORDER BY, FK / CHECK / UNIQUE don't
        exist as enforced constraints. Filter them at the listing level
        (returning empty strings from ``visit_*`` still leaves stray commas
        in the DDL). We reimplement SA's join loop so the filter is local.
        """
        from sqlalchemy.schema import (
            CheckConstraint,
            ForeignKeyConstraint,
            PrimaryKeyConstraint,
            UniqueConstraint,
        )

        skip_types = (
            PrimaryKeyConstraint,
            ForeignKeyConstraint,
            CheckConstraint,
            UniqueConstraint,
        )

        constraints = [
            c
            for c in table._sorted_constraints
            if not isinstance(c, skip_types)
        ]
        return ", \n\t".join(
            p
            for p in (self.process(c) for c in constraints if c._should_create_for_compiler(self))
            if p is not None and p != ""
        )

    def visit_primary_key_constraint(self, constraint: Any, **kw: Any) -> str:
        # Belt + suspenders — should be filtered by create_table_constraints,
        # but ALTER TABLE ADD CONSTRAINT path can also call this directly.
        return ""

    def visit_foreign_key_constraint(self, constraint: Any, **kw: Any) -> str:
        return ""

    def visit_check_constraint(self, constraint: Any, **kw: Any) -> str:
        return ""

    def visit_column_check_constraint(self, constraint: Any, **kw: Any) -> str:
        return ""

    def visit_unique_constraint(self, constraint: Any, **kw: Any) -> str:
        return ""

    def visit_create_index(self, create: Any, **kw: Any) -> str:
        """Suppress generic ``CREATE INDEX`` DDL.

        ClickHouse uses ``ALTER TABLE t ADD INDEX name expr TYPE …`` for
        its data-skipping indexes, not the standard ``CREATE INDEX`` SQL.
        Returning empty here makes SA's ``MetaData.create_all()`` (which
        emits each index as a separate DDL statement) skip indexes
        entirely — they'd need to be applied via ALTER post-create. v0.3
        will add native ALTER TABLE ADD INDEX support.
        """
        return ""

    def visit_drop_index(self, drop: Any, **kw: Any) -> str:
        # Same reasoning — ClickHouse uses ALTER TABLE DROP INDEX.
        return ""

    # ------------------------------------------------------------------
    # COMMENT — ClickHouse syntax differs from standard SQL
    # ------------------------------------------------------------------
    # Standard:   COMMENT ON TABLE foo IS 'x'
    # ClickHouse: ALTER TABLE foo MODIFY COMMENT 'x'
    #             (or inline in CREATE TABLE: COMMENT 'x')
    # SA still declares supports_comments = True (it's true that chDB
    # *has* comment support), but standalone COMMENT-set DDL needs the
    # chDB syntax. The SA fixture factory uses set_table_comment for
    # roundtrip tests, so we translate.

    def visit_set_table_comment(self, create: Any, **kw: Any) -> str:
        table = create.element
        full_name = self.preparer.format_table(table)
        return (
            f"ALTER TABLE {full_name} MODIFY COMMENT "
            f"{self.sql_compiler.render_literal_value(table.comment, _SAStringType)}"
        )

    def visit_drop_table_comment(self, drop: Any, **kw: Any) -> str:
        table = drop.element
        full_name = self.preparer.format_table(table)
        return f"ALTER TABLE {full_name} MODIFY COMMENT ''"

    def visit_set_column_comment(self, create: Any, **kw: Any) -> str:
        column = create.element
        table = column.table
        full_name = self.preparer.format_table(table)
        col_name = self.preparer.format_column(column)
        return (
            f"ALTER TABLE {full_name} COMMENT COLUMN {col_name} "
            f"{self.sql_compiler.render_literal_value(column.comment, _SAStringType)}"
        )

    def visit_drop_column_comment(self, drop: Any, **kw: Any) -> str:
        column = drop.element
        table = column.table
        full_name = self.preparer.format_table(table)
        col_name = self.preparer.format_column(column)
        return f"ALTER TABLE {full_name} COMMENT COLUMN {col_name} ''"

    def visit_set_constraint_comment(self, create: Any, **kw: Any) -> str:
        # ClickHouse has no constraint comments.
        return ""

    def visit_drop_constraint_comment(self, drop: Any, **kw: Any) -> str:
        return ""
