"""Recursive parser for ClickHouse type strings.

The ``system.columns.type`` virtual column gives the canonical type as a
string like::

    Array(Nullable(LowCardinality(String)))
    Map(String, Array(Tuple(UInt32, String)))
    DateTime64(6, 'UTC')
    Decimal(18, 2)
    Variant(String, Int64)
    Nested(name String, value Float64)

We hand-roll the parser rather than pulling in a grammar dependency:

* The grammar is small (only function-call-like ``Name(args)`` syntax plus
  quoted-string literals, numeric literals, and identifiers).
* The parser runs on every column of every reflected table, so import time
  and per-call cost matter.
* The output (a SQLAlchemy ``TypeEngine`` plus a ``nullable`` flag) needs
  custom logic anyway — a generic AST would still need the same mapping pass.

**Public surface**: only :func:`parse_type` and :func:`parse_column_type`.
Everything else is implementation detail.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import types as sa_types

from ..exc import ChdbTypeNotSupportedError
from . import common, composite, dynamic, geo, json_, variant

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ParsedColumn:
    """Result of parsing a ``system.columns.type`` string.

    Attributes:
        sa_type: The SQLAlchemy type instance (already unwrapped from
            ``Nullable`` so SQLAlchemy can apply ``nullable=True`` on the
            Column).
        nullable: True if the column was wrapped in ``Nullable(...)`` at
            the outermost level.
        low_cardinality: True if the column was wrapped in
            ``LowCardinality(...)`` at the outermost level. Most callers
            ignore this — kept for round-trip DDL fidelity.
    """

    sa_type: sa_types.TypeEngine
    nullable: bool = False
    low_cardinality: bool = False


def parse_type(type_str: str) -> sa_types.TypeEngine:
    """Parse a raw ClickHouse type string into a SQLAlchemy type.

    Wrappers like ``Nullable`` and ``LowCardinality`` are preserved in the
    returned type tree (use :func:`parse_column_type` if you want them
    pulled out into separate flags as SQLAlchemy expects on a Column).

    :raises ChdbTypeNotSupportedError: if the type cannot be mapped.
    """
    parser = _Parser(type_str)
    result = parser.parse_type()
    parser.expect_eof()
    return result


def parse_column_type(type_str: str) -> ParsedColumn:
    """Parse a type string and split out the outer ``Nullable`` / ``LowCardinality``.

    SQLAlchemy expresses nullability on the Column, not on the type, so we
    unwrap ``Nullable(T)`` at the outer level. ``LowCardinality`` is a storage
    hint with no SQLAlchemy equivalent; we record it separately so DDL
    emission can reproduce it.
    """
    sa_type = parse_type(type_str)
    nullable = False
    low_cardinality = False

    # Peel modifiers — at most one of each, outer first.
    # Order seen in real schemas: LowCardinality(Nullable(T)) is common.
    while isinstance(sa_type, (composite.LowCardinality, composite.Nullable)):
        if isinstance(sa_type, composite.LowCardinality):
            if low_cardinality:
                break  # don't unwrap twice
            low_cardinality = True
            sa_type = sa_type.inner
        else:  # Nullable
            if nullable:
                break
            nullable = True
            sa_type = sa_type.inner

    return ParsedColumn(sa_type=sa_type, nullable=nullable, low_cardinality=low_cardinality)


# ---------------------------------------------------------------------------
# Internal parser
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INT_RE = re.compile(r"-?\d+")


class _Parser:
    """Hand-rolled recursive descent parser for ClickHouse type strings.

    The grammar (informal)::

        type      := name ( '(' args ')' )?
        args      := arg ( ',' arg )*
        arg       := type
                   | int_literal
                   | string_literal
                   | field_decl                    (Nested only)
                   | enum_member                   (Enum only)
        field_decl    := identifier type
        enum_member   := string_literal '=' int_literal
    """

    def __init__(self, s: str) -> None:
        self.s = s
        self.i = 0
        self.n = len(s)

    # -- low-level lexer --------------------------------------------------

    def _skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i].isspace():
            self.i += 1

    def _peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def _consume(self, ch: str) -> None:
        self._skip_ws()
        if self._peek() != ch:
            raise ChdbTypeNotSupportedError(
                self.s, f"Expected {ch!r} at position {self.i}, got {self._peek()!r}"
            )
        self.i += 1

    def _consume_if(self, ch: str) -> bool:
        self._skip_ws()
        if self._peek() == ch:
            self.i += 1
            return True
        return False

    def _read_ident(self) -> str:
        self._skip_ws()
        m = _IDENT_RE.match(self.s, self.i)
        if not m:
            raise ChdbTypeNotSupportedError(
                self.s, f"Expected identifier at position {self.i}"
            )
        self.i = m.end()
        return m.group(0)

    def _read_int(self) -> int:
        self._skip_ws()
        m = _INT_RE.match(self.s, self.i)
        if not m:
            raise ChdbTypeNotSupportedError(
                self.s, f"Expected integer at position {self.i}"
            )
        self.i = m.end()
        return int(m.group(0))

    def _read_string(self) -> str:
        """Read a single-quoted string literal with backslash escapes."""
        self._skip_ws()
        if self._peek() != "'":
            raise ChdbTypeNotSupportedError(
                self.s, f"Expected quoted string at position {self.i}"
            )
        self.i += 1
        out: list[str] = []
        while self.i < self.n:
            c = self.s[self.i]
            if c == "\\" and self.i + 1 < self.n:
                # Escape sequence — keep simple, just take the next char verbatim.
                out.append(self.s[self.i + 1])
                self.i += 2
            elif c == "'":
                self.i += 1
                return "".join(out)
            else:
                out.append(c)
                self.i += 1
        raise ChdbTypeNotSupportedError(self.s, "Unterminated string literal")

    # -- structural ------------------------------------------------------

    def expect_eof(self) -> None:
        self._skip_ws()
        if self.i != self.n:
            raise ChdbTypeNotSupportedError(
                self.s, f"Trailing characters at position {self.i}: {self.s[self.i:]!r}"
            )

    def parse_type(self) -> sa_types.TypeEngine:
        name = self._read_ident()
        # Some types are always parameterless (String, Boolean, UUID, ...).
        # Some are conditional (DateTime / DateTime64).
        # If we see '(' we parse args, otherwise we hand the builder an empty list.
        args: list[object] = []
        if self._consume_if("("):
            args = self._read_args(name)
            self._consume(")")

        builder = _BUILDERS.get(name)
        if builder is None:
            raise ChdbTypeNotSupportedError(self.s, f"Unknown ClickHouse type: {name}")
        return builder(self, args)

    def _read_args(self, parent_name: str) -> list[object]:
        """Parse a comma-separated arg list inside parentheses.

        The interpretation of each arg depends on the parent type — Nested
        wants ``ident type`` declarations, Enum wants ``'name' = N``, most
        others just want a chain of types or integers. We dispatch by
        parent name to keep the grammar precise.
        """
        if parent_name in ("Nested",):
            return self._read_nested_fields()
        if parent_name in ("Enum", "Enum8", "Enum16"):
            return self._read_enum_members()
        if parent_name in ("AggregateFunction", "SimpleAggregateFunction"):
            return self._read_agg_args()
        # Default: each arg is either an int, a quoted string, or a nested type.
        return self._read_generic_args()

    def _read_generic_args(self) -> list[object]:
        out: list[object] = []
        while True:
            self._skip_ws()
            c = self._peek()
            if c == "":
                raise ChdbTypeNotSupportedError(self.s, "Unexpected end of input")
            if c == ")":
                break
            if c == "'":
                out.append(self._read_string())
            elif c == "-" or c.isdigit():
                out.append(self._read_int())
            else:
                out.append(self.parse_type())
            if not self._consume_if(","):
                break
        return out

    def _read_nested_fields(self) -> list[object]:
        fields: list[tuple[str, sa_types.TypeEngine]] = []
        while True:
            self._skip_ws()
            if self._peek() == ")":
                break
            name = self._read_ident()
            ty = self.parse_type()
            fields.append((name, ty))
            if not self._consume_if(","):
                break
        return [fields]  # wrap so builder sees a single arg

    def _read_enum_members(self) -> list[object]:
        members: dict[str, int] = {}
        while True:
            self._skip_ws()
            if self._peek() == ")":
                break
            name = self._read_string()
            self._consume("=")
            value = self._read_int()
            members[name] = value
            if not self._consume_if(","):
                break
        return [members]

    def _read_agg_args(self) -> list[object]:
        """AggregateFunction(func_name, T1, T2, ...).

        First arg is the function name (identifier or quoted function-call-like
        form e.g. 'quantilesTimingWeighted(0.5, 0.9)'). Remaining are types.
        """
        out: list[object] = []
        # Function name — accept ident, possibly followed by a parenthesised
        # parameter list which we slurp verbatim.
        self._skip_ws()
        fname_start = self.i
        # Identifier
        self._read_ident()
        # Optional parameter group: depth-tracked slurp
        if self._consume_if("("):
            depth = 1
            while self.i < self.n and depth > 0:
                c = self.s[self.i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                self.i += 1
            # self.i is now one past the closing ')'
        fname_end = self.i
        out.append(self.s[fname_start:fname_end])
        # Now zero-or-more types
        while self._consume_if(","):
            out.append(self.parse_type())
        return out


# ---------------------------------------------------------------------------
# Builders: name → (parser, args) → SQLAlchemy type instance
# ---------------------------------------------------------------------------


def _scalar(cls: type[sa_types.TypeEngine]) -> Callable[[_Parser, list], sa_types.TypeEngine]:
    def build(parser: _Parser, args: list) -> sa_types.TypeEngine:
        if args:
            raise ChdbTypeNotSupportedError(
                parser.s, f"Type {cls.__name__} takes no arguments, got {args!r}"
            )
        return cls()

    return build


def _fixed_string(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if len(args) != 1 or not isinstance(args[0], int):
        raise ChdbTypeNotSupportedError(parser.s, "FixedString requires one integer arg")
    return common.FixedString(args[0])


def _decimal(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if len(args) != 2 or not all(isinstance(a, int) for a in args):
        raise ChdbTypeNotSupportedError(parser.s, "Decimal requires two integer args")
    return common.Decimal(args[0], args[1])


def _datetime(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if not args:
        return common.DateTime()
    if len(args) == 1 and isinstance(args[0], str):
        return common.DateTime(timezone=args[0])
    raise ChdbTypeNotSupportedError(parser.s, "DateTime takes 0 or 1 timezone arg")


def _datetime64(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if not args or not isinstance(args[0], int):
        raise ChdbTypeNotSupportedError(parser.s, "DateTime64 requires integer precision")
    precision = args[0]
    tz: str | None = None
    if len(args) >= 2 and isinstance(args[1], str):
        tz = args[1]
    return common.DateTime64(precision=precision, timezone=tz)


def _time64(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if not args or not isinstance(args[0], int):
        raise ChdbTypeNotSupportedError(parser.s, "Time64 requires integer precision")
    return common.Time64(precision=args[0])


def _enum(cls: type[common.Enum]) -> Callable[[_Parser, list], sa_types.TypeEngine]:
    def build(parser: _Parser, args: list) -> sa_types.TypeEngine:
        if len(args) != 1 or not isinstance(args[0], dict):
            raise ChdbTypeNotSupportedError(parser.s, f"{cls.__name__} requires members")
        return cls(args[0])

    return build


def _nullable(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if len(args) != 1 or not isinstance(args[0], sa_types.TypeEngine):
        raise ChdbTypeNotSupportedError(parser.s, "Nullable requires one inner type")
    return composite.Nullable(args[0])


def _low_cardinality(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if len(args) != 1 or not isinstance(args[0], sa_types.TypeEngine):
        raise ChdbTypeNotSupportedError(parser.s, "LowCardinality requires one inner type")
    return composite.LowCardinality(args[0])


def _array(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if len(args) != 1 or not isinstance(args[0], sa_types.TypeEngine):
        raise ChdbTypeNotSupportedError(parser.s, "Array requires one inner type")
    return composite.Array(args[0])


def _tuple(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if not args or not all(isinstance(a, sa_types.TypeEngine) for a in args):
        raise ChdbTypeNotSupportedError(parser.s, "Tuple requires >=1 inner types")
    return composite.Tuple(*args)


def _map(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if (
        len(args) != 2
        or not isinstance(args[0], sa_types.TypeEngine)
        or not isinstance(args[1], sa_types.TypeEngine)
    ):
        raise ChdbTypeNotSupportedError(parser.s, "Map requires key and value types")
    return composite.Map(args[0], args[1])


def _nested(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if len(args) != 1 or not isinstance(args[0], list):
        raise ChdbTypeNotSupportedError(parser.s, "Nested requires field declarations")
    return composite.Nested(args[0])


def _variant(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if not args or not all(isinstance(a, sa_types.TypeEngine) for a in args):
        raise ChdbTypeNotSupportedError(parser.s, "Variant requires >=1 alternative types")
    return variant.Variant(*args)


def _dynamic(parser: _Parser, args: list) -> sa_types.TypeEngine:
    if not args:
        return dynamic.Dynamic()
    # `Dynamic(max_types=N)` — args[0] could be a string carrying the kwarg, or int.
    # ClickHouse output form is `Dynamic(max_types=32)`; we don't currently parse
    # kwarg=value syntax — accept the bare int form as future-proof.
    if len(args) == 1 and isinstance(args[0], int):
        return dynamic.Dynamic(max_types=args[0])
    raise ChdbTypeNotSupportedError(parser.s, "Unrecognised Dynamic argument form")


def _object_legacy(parser: _Parser, args: list) -> sa_types.TypeEngine:
    """Builder for legacy ``Object(...)`` types.

    ClickHouse pre-24.10 emitted JSON-like columns as ``Object('json')``.
    The argument is a string literal naming the schema family — only
    ``'json'`` is supported (everything else was deprecated long before
    chDB's 26.3 baseline). Accept and ignore the ``'json'`` arg; reject
    anything else explicitly so the user knows the parser saw an unsupported
    form rather than silently mapping it to ``JSONLegacy``.
    """
    if not args:
        return json_.JSONLegacy()
    if len(args) == 1 and isinstance(args[0], str) and args[0].lower() == "json":
        return json_.JSONLegacy()
    raise ChdbTypeNotSupportedError(
        parser.s, f"Object(...) only supports 'json'; got {args!r}"
    )


def _agg(cls):
    def build(parser: _Parser, args: list) -> sa_types.TypeEngine:
        if not args or not isinstance(args[0], str):
            raise ChdbTypeNotSupportedError(parser.s, f"{cls.__name__} requires function name")
        fname = args[0]
        types_args = args[1:]
        if not all(isinstance(t, sa_types.TypeEngine) for t in types_args):
            raise ChdbTypeNotSupportedError(parser.s, f"{cls.__name__} arg types invalid")
        if cls is composite.SimpleAggregateFunction:
            if len(types_args) != 1:
                raise ChdbTypeNotSupportedError(
                    parser.s, "SimpleAggregateFunction takes exactly one type arg"
                )
            return cls(fname, types_args[0])
        return cls(fname, *types_args)

    return build


# Single registry — name → builder.
_BUILDERS: dict[str, Callable[[_Parser, list], sa_types.TypeEngine]] = {
    # Strings
    "String": _scalar(common.String),
    "FixedString": _fixed_string,
    "UUID": _scalar(common.UUID),
    # Ints
    "Int8": _scalar(common.Int8),
    "Int16": _scalar(common.Int16),
    "Int32": _scalar(common.Int32),
    "Int64": _scalar(common.Int64),
    "Int128": _scalar(common.Int128),
    "Int256": _scalar(common.Int256),
    "UInt8": _scalar(common.UInt8),
    "UInt16": _scalar(common.UInt16),
    "UInt32": _scalar(common.UInt32),
    "UInt64": _scalar(common.UInt64),
    "UInt128": _scalar(common.UInt128),
    "UInt256": _scalar(common.UInt256),
    # Floats
    "Float32": _scalar(common.Float32),
    "Float64": _scalar(common.Float64),
    "BFloat16": _scalar(common.BFloat16),
    # Decimal
    "Decimal": _decimal,
    # Bool
    "Bool": _scalar(common.Boolean),
    "Boolean": _scalar(common.Boolean),
    # Date/time
    "Date": _scalar(common.Date),
    "Date32": _scalar(common.Date32),
    "DateTime": _datetime,
    "DateTime64": _datetime64,
    "Time": _scalar(common.Time),
    "Time64": _time64,
    # Enums
    "Enum": _enum(common.Enum),
    "Enum8": _enum(common.Enum8),
    "Enum16": _enum(common.Enum16),
    # Network
    "IPv4": _scalar(common.IPv4),
    "IPv6": _scalar(common.IPv6),
    # Modifiers / composites
    "Nullable": _nullable,
    "LowCardinality": _low_cardinality,
    "Array": _array,
    "Tuple": _tuple,
    "Map": _map,
    "Nested": _nested,
    "AggregateFunction": _agg(composite.AggregateFunction),
    "SimpleAggregateFunction": _agg(composite.SimpleAggregateFunction),
    # Modern (24.x+)
    "Variant": _variant,
    "Dynamic": _dynamic,
    "JSON": _scalar(json_.JSON),
    "Object": _object_legacy,  # legacy `Object('json')` — accepts the 'json' arg
    # Geo
    "Point": _scalar(geo.Point),
    "Ring": _scalar(geo.Ring),
    "Polygon": _scalar(geo.Polygon),
    "MultiPolygon": _scalar(geo.MultiPolygon),
}
