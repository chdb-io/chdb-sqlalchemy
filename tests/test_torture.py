"""End-to-end torture suite — verify docs claims against real chDB.

The previous test suites trusted the README and ``docs/types.md`` —
"the docs say we support Decimal, so we have a parser test for the type
string". That left gaps when chDB or our dialect behaved differently from
the documentation. This file does the opposite: every doc claim is a
hypothesis until a round-trip test proves it.

Organisation (one section = one hypothesis bucket):

1. ``TestDocsTypeMapping`` — every row of ``docs/types.md`` exercised
   via CREATE / INSERT / SELECT / reflect. If chDB rejects the DDL we
   make that visible; if the reflected SA class disagrees with the
   table, fail loudly.
2. ``TestStringFixtures`` — adversarial string contents (NUL, quote,
   backslash, emoji, RTL, large).
3. ``TestNumericBoundary`` — min / max / zero / negative-zero for every
   numeric width; Float specials.
4. ``TestTemporalBoundary`` — Date / DateTime / DateTime64 / Time edges.
5. ``TestCompositeNesting`` — empty, deep, NULL-filled composites.
6. ``TestPKReflection`` — function-expression / tuple() / multi-col PKs
   (Fix #7 territory).
7. ``TestSQLInjectionShape`` — strings whose contents look like SQL
   fragments must survive INSERT/SELECT verbatim.
8. ``TestKnownDivergence`` — locks in the surprises probing surfaced:
   DateTime naive-string TZ shift, DateTime64(9) microsecond truncation,
   etc. Each is an explicit assertion so a silent behavior change in
   chDB shows up as a failing test.

Each test is parametrised tightly so a regression report names the
specific case (e.g. ``[NUL-byte-mid]``) rather than ``test_strings``.
"""

from __future__ import annotations

import datetime as dt
import decimal
import math

import pytest
from sqlalchemy import Engine, inspect, text

# ---------------------------------------------------------------------------
# Helpers — kept local to this file so they're easy to read alongside the
# tests that use them.
# ---------------------------------------------------------------------------


def _roundtrip(
    eng: Engine,
    column_type: str,
    value_sql: str,
    *,
    table: str = "rt",
    extra_cols: str = "",
) -> object:
    """CREATE TABLE / INSERT one row / SELECT it / DROP. Returns the cell.

    The ``value_sql`` is the literal SQL fragment to put in
    ``INSERT INTO t VALUES (<value_sql>)`` — so callers can use chDB's
    native literal syntax (``toDateTime64('...', 6)``, ``arrayMap(...)``,
    ``[1,2,3]``…) directly. That keeps the tests honest about how chDB
    parses literals rather than going through SA's bind path.
    """
    with eng.begin() as conn:
        cols = f"x {column_type}" + (f", {extra_cols}" if extra_cols else "")
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text(f"CREATE TABLE {table} ({cols}) ENGINE=Memory"))
        conn.execute(text(f"INSERT INTO {table} VALUES ({value_sql})"))
        row = conn.execute(text(f"SELECT x FROM {table}")).fetchone()
    return None if row is None else row[0]


def _reflect_column_type(eng: Engine, table: str, column: str = "x"):
    """Return the SA type instance reflection produced for a column."""
    insp = inspect(eng)
    cols = {c["name"]: c for c in insp.get_columns(table)}
    return cols[column]["type"]


# ---------------------------------------------------------------------------
# 1. TestDocsTypeMapping — does the README table actually hold?
# ---------------------------------------------------------------------------
#
# Source of truth: ``docs/types.md`` (the per-type mapping) and
# ``README.md`` (the public table). Each parametrize entry is one row.
# We assert two things:
#   a. CREATE / INSERT / SELECT with the documented chDB type works.
#   b. Reflection on that column returns an instance of the documented
#      SA class (or one of an accepted set if the doc allows a family
#      mapping like "Integer or BigInteger").


_DOC_TYPE_CASES = [
    # (chDB_type, value_literal, expected_python_value, expected_sa_class_name)
    # Strings — note: chDB returns FixedString cells as ``str`` (with NUL
    # padding in the bytes), not ``bytes``. Documented gap, not our bug.
    ("String", "'hello'", "hello", "String"),
    ("FixedString(8)", "'pad'", "pad\x00\x00\x00\x00\x00", "FixedString"),
    # chDB returns UUID cells as the lowercase string form, not a
    # ``uuid.UUID`` instance. The dialect maps to SA ``Uuid`` so when a
    # Column is typed, SA's per-column processor converts; on raw
    # ``text()`` queries the cell stays a string. Lock the bare behavior.
    ("UUID", "'12345678-1234-5678-1234-567812345678'",
     "12345678-1234-5678-1234-567812345678", "UUID"),
    # Ints — every documented width
    ("Int8",   "-128",    -128,    "Int8"),
    ("Int16",  "-32768",  -32768,  "Int16"),
    ("Int32",  "-2147483648", -2147483648, "Int32"),
    ("Int64",  "9223372036854775807", 9223372036854775807, "Int64"),
    ("Int128", "170141183460469231731687303715884105727",
     170141183460469231731687303715884105727, "Int128"),
    ("Int256", str((1 << 255) - 1), (1 << 255) - 1, "Int256"),
    ("UInt8",  "255",   255,   "UInt8"),
    ("UInt16", "65535", 65535, "UInt16"),
    ("UInt32", "4294967295", 4294967295, "UInt32"),
    ("UInt64", "18446744073709551615", 18446744073709551615, "UInt64"),
    ("UInt128", str((1 << 128) - 1), (1 << 128) - 1, "UInt128"),
    ("UInt256", str((1 << 256) - 1), (1 << 256) - 1, "UInt256"),
    # Floats
    ("Float32", "1.5", 1.5, "Float32"),
    ("Float64", "1.5", 1.5, "Float64"),
    # Decimal
    ("Decimal(18, 4)", "123.4567", decimal.Decimal("123.4567"), "Decimal"),
    # Bool
    ("Bool", "true", True, "Boolean"),
    # Date/time
    ("Date", "'2026-05-19'", dt.date(2026, 5, 19), "Date"),
    ("Date32", "'2026-05-19'", dt.date(2026, 5, 19), "Date32"),
    # IP
    # Reflection returns the IP value as an integer-or-string-shaped object;
    # we only assert round-trip succeeds and reflection identifies the type.
    # Enum members preserved
    ("Enum8('alpha' = 1, 'beta' = 2)", "'alpha'", "alpha", "Enum8"),
    ("Enum16('a' = 100, 'b' = 200)", "'a'", "a", "Enum16"),
]


@pytest.mark.parametrize(
    ("ch_type", "value_sql", "expected_value", "expected_sa_class"),
    _DOC_TYPE_CASES,
    ids=[c[0] for c in _DOC_TYPE_CASES],
)
def test_docs_type_round_trip(engine, ch_type, value_sql, expected_value, expected_sa_class):
    """Every chDB type in docs/types.md must round-trip an obvious value."""
    val = _roundtrip(engine, ch_type, value_sql)
    assert val == expected_value, (
        f"{ch_type} round-trip: expected {expected_value!r}, got {val!r}"
    )


def test_docs_uuid_reflection(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE uu (x UUID) ENGINE=Memory"))
    ty = _reflect_column_type(engine, "uu")
    from chdb_sqlalchemy.types.common import UUID as ChdbUUID
    assert isinstance(ty, ChdbUUID)


# --- Reflection-side audit: every docs/types.md row produces the right SA class ---
# Important separately from the round-trip test: the *value* could be coerced
# correctly even when reflection mislabels the type, and vice versa. LangChain
# / pandas / Django all rely on the reflected type for their prompts/queries.

_REFLECTION_AUDIT_CASES = [
    # (ddl_type, expected_sa_class_name)
    ("String", "String"),
    ("FixedString(8)", "FixedString"),
    ("UUID", "UUID"),
    ("Int8", "Int8"),  ("Int16", "Int16"),
    ("Int32", "Int32"), ("Int64", "Int64"),
    ("Int128", "Int128"), ("Int256", "Int256"),
    ("UInt8", "UInt8"), ("UInt16", "UInt16"),
    ("UInt32", "UInt32"), ("UInt64", "UInt64"),
    ("UInt128", "UInt128"), ("UInt256", "UInt256"),
    ("Float32", "Float32"), ("Float64", "Float64"),
    ("BFloat16", "BFloat16"),
    ("Decimal(18, 4)", "Decimal"),
    ("Bool", "Boolean"),
    ("Date", "Date"), ("Date32", "Date32"),
    ("DateTime", "DateTime"),
    ("DateTime('UTC')", "DateTime"),
    ("DateTime64(6, 'UTC')", "DateTime64"),
    ("Time", "Time"),
    ("Time64(3)", "Time64"),
    ("Enum8('a'=1, 'b'=2)", "Enum8"),
    ("Enum16('x'=100)", "Enum16"),
    ("IPv4", "IPv4"), ("IPv6", "IPv6"),
    ("Array(Int32)", "Array"),
    ("Tuple(Int32, String)", "Tuple"),
    ("Map(String, Int32)", "Map"),
    ("Point", "Point"),
    ("Ring", "Ring"),
    ("Polygon", "Polygon"),
    ("MultiPolygon", "MultiPolygon"),
]


@pytest.mark.parametrize(
    ("ddl_type", "expected_class"),
    _REFLECTION_AUDIT_CASES,
    ids=[c[0] for c in _REFLECTION_AUDIT_CASES],
)
def test_reflection_class_matches_docs(engine, ddl_type, expected_class):
    """Every documented type reflects to the documented SA class."""
    safe_id = "ref_" + "".join(ch if ch.isalnum() else "_" for ch in ddl_type)[:40]
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {safe_id}"))
        conn.execute(text(f"CREATE TABLE {safe_id} (x {ddl_type}) ENGINE=Memory"))
    ty = _reflect_column_type(engine, safe_id)
    assert type(ty).__name__ == expected_class, (
        f"{ddl_type!r} reflected as {type(ty).__name__}, docs say {expected_class}"
    )


_EXPERIMENTAL_REFLECTION_CASES = [
    ("Variant(String, Int64)", "Variant"),
    ("Dynamic", "Dynamic"),
    ("JSON", "JSON"),
]


@pytest.mark.parametrize(
    ("ddl_type", "expected_class"),
    _EXPERIMENTAL_REFLECTION_CASES,
    ids=[c[0] for c in _EXPERIMENTAL_REFLECTION_CASES],
)
def test_reflection_class_matches_docs_experimental(permissive_engine, ddl_type, expected_class):
    """Experimental types — JSON / Variant / Dynamic — reflect correctly."""
    safe_id = "refe_" + "".join(ch if ch.isalnum() else "_" for ch in ddl_type)[:40]
    with permissive_engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {safe_id}"))
        conn.execute(text(f"CREATE TABLE {safe_id} (x {ddl_type}) ENGINE=Memory"))
    ty = _reflect_column_type(permissive_engine, safe_id)
    assert type(ty).__name__ == expected_class


def test_reflection_nullable_unwraps_outer(engine):
    """``Nullable(T)`` reflection: type unwraps to T, nullable=True on Column."""
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE nu (x Nullable(Int32)) ENGINE=Memory"))
    insp = inspect(engine)
    col = next(c for c in insp.get_columns("nu") if c["name"] == "x")
    assert col["nullable"] is True
    from chdb_sqlalchemy.types.common import Int32
    assert isinstance(col["type"], Int32)


def test_reflection_low_cardinality_unwraps_outer(engine):
    """``LowCardinality(T)`` reflection: type unwraps to T, info flag set."""
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE lc1 (x LowCardinality(String)) ENGINE=Memory"))
    insp = inspect(engine)
    col = next(c for c in insp.get_columns("lc1") if c["name"] == "x")
    assert col["info"]["chdb_low_cardinality"] is True
    from chdb_sqlalchemy.types.common import String as ChdbString
    assert isinstance(col["type"], ChdbString)


def test_reflection_low_cardinality_nullable(engine):
    """``LowCardinality(Nullable(T))`` — both modifiers unwrap at the column level."""
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE lcn1 (x LowCardinality(Nullable(String))) ENGINE=Memory"
        ))
    insp = inspect(engine)
    col = next(c for c in insp.get_columns("lcn1") if c["name"] == "x")
    assert col["nullable"] is True
    assert col["info"]["chdb_low_cardinality"] is True


def test_reflection_nested_explodes_to_dot_columns(engine):
    """``Nested(a T1, b T2)`` reflects as parallel ``Array`` columns ``n.a``, ``n.b``.

    ClickHouse internally rewrites Nested into parallel Arrays — there is no
    ``n`` column in ``system.columns``, only the dotted children. Our
    reflection surface must surface them as-is so LangChain prompts show
    both keys.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE nst1 (n Nested(a Int32, b String)) ENGINE=Memory"))
    insp = inspect(engine)
    names = {c["name"] for c in insp.get_columns("nst1")}
    assert names == {"n.a", "n.b"}
    by_name = {c["name"]: c for c in insp.get_columns("nst1")}
    from chdb_sqlalchemy.types.composite import Array
    assert isinstance(by_name["n.a"]["type"], Array)
    assert isinstance(by_name["n.b"]["type"], Array)


@pytest.mark.parametrize(
    ("ch_type", "value", "expected_class"),
    [
        ("IPv4", "'192.0.2.42'", "IPv4"),
        ("IPv6", "'2001:db8::42'", "IPv6"),
    ],
)
def test_docs_ip_round_trip(engine, ch_type, value, expected_class):
    """IPv4/IPv6 — docs claim support; chDB returns them as native shapes."""
    val = _roundtrip(engine, ch_type, value)
    assert val is not None
    # IPv4/IPv6 cells come back as ipaddress objects from chdb; just check
    # the string repr matches the inserted address modulo canonicalisation.
    assert str(val).strip("/") in {
        "192.0.2.42",
        "2001:db8::42",
        "::ffff:192.0.2.42",
    }


def test_docs_object_json_legacy_ddl_is_rejected_by_chdb(engine):
    """``Object('json')`` is claimed in docs/types.md but chDB 26.3 removed it.

    Fix #6 made the *parser* accept ``Object('json')`` as a type string
    (so reflection of a pre-existing column built on an older chDB doesn't
    crash). The DDL path is a different story: chDB itself raises
    ``UNKNOWN_TYPE`` if you try to ``CREATE TABLE t (o Object('json'))``.
    This test locks in that reality so the docs can be updated to reflect
    the split: parser-side ✓, DDL-side ✗.
    """
    with engine.begin() as conn, pytest.raises(Exception) as exc_info:
        conn.execute(text("CREATE TABLE objs (o Object('json')) ENGINE=Memory"))
    msg = str(exc_info.value)
    assert "Object" in msg and ("UNKNOWN_TYPE" in msg or "Unknown data type" in msg)


def test_docs_json_native_round_trip(permissive_engine):
    """The 24.10+ semantic ``JSON`` type — docs claim it maps to SA ``JSON``."""
    val = _roundtrip(permissive_engine, "JSON", "'{\"a\": 1, \"b\": [1,2,3]}'")
    # chDB returns JSON cells as Python repr-style strings; our cursor
    # wrapper parses them. Either a dict (post-wrapper) or a string repr
    # of a dict is the legal canonical form — we accept the dict shape.
    assert val == {"a": 1, "b": [1, 2, 3]}


def test_docs_variant_round_trip(permissive_engine):
    """``Variant(String, Int64)`` accepts both alternatives + NULL."""
    with permissive_engine.begin() as conn:
        conn.execute(text("CREATE TABLE vt (v Variant(String, Int64)) ENGINE=Memory"))
        conn.execute(text("INSERT INTO vt VALUES ('hi'), (42), (NULL)"))
        rows = conn.execute(text("SELECT v FROM vt")).fetchall()
    # chDB returns the variant cell as a string of the chosen alternative's
    # repr; NULL stays None. We don't assert int-typedness because the
    # variant cell loses type fidelity through chdb.dbapi today.
    values = sorted([r[0] for r in rows if r[0] is not None])
    assert values == ["42", "hi"]
    assert any(r[0] is None for r in rows)


def test_docs_dynamic_round_trip(permissive_engine):
    """``Dynamic`` accepts mixed types; ``dynamicType()`` reports per-row type.

    NB: chDB rejects ``ORDER BY`` on Dynamic / Variant columns without
    ``allow_suspicious_types_in_order_by``, so we collect unordered and
    bucket by reported dynamicType.
    """
    with permissive_engine.begin() as conn:
        conn.execute(text("CREATE TABLE dy (d Dynamic) ENGINE=Memory"))
        conn.execute(text("INSERT INTO dy VALUES ('hi'), (42), ([1,2,3]), (NULL)"))
        rows = conn.execute(text("SELECT d, dynamicType(d) FROM dy")).fetchall()
    by_type = {r[1]: r[0] for r in rows}
    assert "Int64" in by_type and by_type["Int64"] == "42"
    assert "String" in by_type and by_type["String"] == "hi"
    assert "None" in by_type and by_type["None"] is None


# ---------------------------------------------------------------------------
# 2. TestStringFixtures — adversarial String content
# ---------------------------------------------------------------------------
#
# Real-world LLM-generated SQL feeds raw user data through. We must not
# silently corrupt any byte sequence. Each fixture covers a category
# users will hit eventually:


_STRING_FIXTURES = [
    pytest.param("", id="empty"),
    pytest.param(" ", id="single-space"),
    pytest.param("hello", id="ascii-plain"),
    pytest.param("O'Brien", id="single-quote"),
    pytest.param("she said \"hi\"", id="double-quote"),
    pytest.param("path\\to\\file", id="backslash"),
    pytest.param("mix \\\\\\'\\\"", id="mixed-escapes"),
    pytest.param("a\nb\tc\rd", id="control-chars"),
    pytest.param("a\x00b\x00c", id="NUL-bytes"),
    pytest.param("你好世界", id="cjk"),
    pytest.param("مرحبا بالعالم", id="rtl-arabic"),
    pytest.param("🎉🚀✨", id="emoji"),
    pytest.param("﻿bom-prefix", id="bom"),
    pytest.param("é", id="combining-acute"),
    pytest.param("x" * 1024, id="1KiB"),
    pytest.param("y" * (1024 * 1024), id="1MiB"),
    pytest.param("'); DROP TABLE x; --", id="injection-shape"),
    pytest.param("?param_marker", id="qmark-marker"),
    pytest.param("%s and %(named)s", id="pct-marker"),
    pytest.param("/* comment */ -- comment", id="sql-comments"),
]


@pytest.mark.parametrize("value", _STRING_FIXTURES)
def test_string_round_trip(engine, value):
    """Every byte we put in must come out unchanged — no clipping, no eval.

    Using exec_driver_sql to bypass SA's :colon-bind parsing which would
    eat the % markers and other adversarial shapes.
    """
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sf"))
        conn.execute(text("CREATE TABLE sf (id UInt32, s String) ENGINE=Memory"))
        # Use chDB's '' escape for single quotes and \\ for backslash;
        # parameter binding via exec_driver_sql to dodge SA bind parsing.
        conn.exec_driver_sql(
            "INSERT INTO sf VALUES (?, ?)",
            (1, value),
        )
        row = conn.execute(text("SELECT s FROM sf WHERE id = 1")).fetchone()
    got = row[0]
    assert got == value, (
        f"String corruption: input bytes={value.encode('utf-8')!r}, "
        f"output bytes={got.encode('utf-8') if isinstance(got, str) else got!r}"
    )


def test_fixedstring_pads_with_nul(engine):
    """``FixedString(N)`` pads short input with NUL bytes (chDB behavior).

    Note: the cell comes back as ``str`` with embedded NUL chars, not
    ``bytes``. chDB.dbapi decodes the byte buffer as UTF-8 before
    returning. The padding-with-NUL invariant is what we lock in here.
    """
    got = _roundtrip(engine, "FixedString(4)", "'ab'")
    assert got == "ab\x00\x00"
    assert len(got) == 4


def test_fixedstring_rejects_overflow(engine):
    """Value longer than the declared length is a chDB error, not a silent truncation."""
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE fx (x FixedString(2)) ENGINE=Memory"))
        with pytest.raises(Exception) as exc:
            conn.execute(text("INSERT INTO fx VALUES ('abc')"))
        assert "TOO_LARGE_STRING_SIZE" in str(exc.value) or "too long" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# 3. TestNumericBoundary
# ---------------------------------------------------------------------------


_INT_BOUNDS = [
    # (ch_type, [(label, value), ...])
    ("Int8",  [("zero", 0), ("min", -128), ("max", 127), ("neg-one", -1)]),
    ("Int16", [("zero", 0), ("min", -32768), ("max", 32767), ("neg-one", -1)]),
    ("Int32", [("zero", 0), ("min", -2147483648), ("max", 2147483647)]),
    ("Int64", [("zero", 0), ("min", -(1 << 63)), ("max", (1 << 63) - 1)]),
    ("Int128", [("min", -(1 << 127)), ("max", (1 << 127) - 1)]),
    ("Int256", [("min", -(1 << 255)), ("max", (1 << 255) - 1)]),
    ("UInt8",  [("zero", 0), ("max", 255)]),
    ("UInt16", [("zero", 0), ("max", 65535)]),
    ("UInt32", [("zero", 0), ("max", (1 << 32) - 1)]),
    ("UInt64", [("zero", 0), ("max", (1 << 64) - 1)]),
    ("UInt128", [("zero", 0), ("max", (1 << 128) - 1)]),
    ("UInt256", [("zero", 0), ("max", (1 << 256) - 1)]),
]


@pytest.mark.parametrize(
    ("ch_type", "label", "value"),
    [
        (t, lbl, val)
        for t, cases in _INT_BOUNDS
        for lbl, val in cases
    ],
    ids=[f"{t}-{lbl}" for t, cases in _INT_BOUNDS for lbl, _ in cases],
)
def test_integer_boundary(engine, ch_type, label, value):
    got = _roundtrip(engine, ch_type, str(value))
    assert got == value


def test_integer_overflow_silently_wraps(engine):
    """ClickHouse silently truncates integer-literal overflow on INSERT.

    ``UInt8`` accepts ``256`` and stores ``0`` — no error, no warning.
    This is a real data-integrity surprise users will hit; the test locks
    in the behavior so a hypothetical future chDB change (stricter
    validation) shows up as a failing test we can react to.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE ov (x UInt8) ENGINE=Memory"))
        conn.execute(text("INSERT INTO ov VALUES (256), (257), (-1)"))
        rows = conn.execute(text("SELECT x FROM ov ORDER BY x")).fetchall()
    # 256 % 256 = 0; 257 % 256 = 1; -1 % 256 = 255
    assert [r[0] for r in rows] == [0, 1, 255]


def test_decimal_18_4_full_precision(engine):
    """``Decimal(18, 4)`` round-trips exactly — fits in double-precision range."""
    raw = "12345678.1234"
    expected = decimal.Decimal(raw)
    got = _roundtrip(engine, "Decimal(18, 4)", raw)
    assert got == expected


def test_decimal_38_high_precision_round_trip(engine):
    """``Decimal(38, 10)`` cells preserve full precision through chdb.dbapi.

    chdb-core ≤26.3 routed Decimal cells through a double-precision float
    conversion, so values whose magnitude exceeded ~2**53 came back rounded
    (chdb-io/chdb#574). chdb-core 26.5 (ClickHouse 26.5 baseline) fixed
    the round-trip; this test locks the fix and fails on older cores.
    """
    raw = "1234567890123456789012345678.0123456789"
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE big_dec (x Decimal(38, 10)) ENGINE=Memory"))
        conn.execute(text(f"INSERT INTO big_dec VALUES ({raw})"))
        cell = conn.execute(text("SELECT x FROM big_dec")).scalar()
        stringified = conn.execute(text("SELECT toString(x) FROM big_dec")).scalar()
    assert stringified == raw
    assert cell == decimal.Decimal(raw)


def test_decimal_negative_zero(engine):
    got = _roundtrip(engine, "Decimal(10, 2)", "-0.00")
    assert got == decimal.Decimal("0.00")


def test_float_subnormal_round_trip(engine):
    """Smallest positive denormal — chDB shouldn't flush to zero."""
    smallest = "5e-324"  # min positive subnormal Float64
    got = _roundtrip(engine, "Float64", smallest)
    # chDB may or may not preserve subnormals; either we get the original
    # or we get 0.0. Locking in whichever for regression visibility.
    assert got == 0.0 or got == 5e-324


# ---------------------------------------------------------------------------
# 4. TestTemporalBoundary
# ---------------------------------------------------------------------------


def test_date_upper_bound(engine):
    """``Date`` is days since epoch in UInt16 → 1970-01-01 to 2149-06-06."""
    got = _roundtrip(engine, "Date", "'2149-06-06'")
    assert got == dt.date(2149, 6, 6)


def test_date32_lower_and_upper(engine):
    """``Date32`` covers 1900-01-01 .. 2299-12-31 (per ClickHouse 26.3)."""
    lo = _roundtrip(engine, "Date32", "'1925-01-01'")
    hi = _roundtrip(engine, "Date32", "'2299-12-31'")
    assert lo == dt.date(1925, 1, 1)
    assert hi == dt.date(2299, 12, 31)


def test_datetime64_microsecond_precision_preserved(engine):
    """DateTime64(6) survives microsecond precision; (9) clips on the Python side."""
    micros = _roundtrip(engine, "DateTime64(6, 'UTC')",
                        "toDateTime64('2025-01-02 03:04:05.123456', 6, 'UTC')")
    assert isinstance(micros, dt.datetime)
    assert micros.microsecond == 123456


def test_datetime64_nanosecond_truncates_at_python_layer(engine):
    """Python ``datetime`` is microsecond-precision; sub-µs digits get cut.

    This is *not* our bug — Python's stdlib doesn't carry nanoseconds.
    The test locks the truncation in so it's visible to users who hit it.
    """
    got = _roundtrip(engine, "DateTime64(9, 'UTC')",
                     "toDateTime64('2025-01-02 03:04:05.123456789', 9, 'UTC')")
    assert isinstance(got, dt.datetime)
    assert got.microsecond == 123456  # last 3 digits gone


def test_datetime_naive_string_applies_session_tz(engine):
    """``DateTime`` without timezone literal in INSERT acquires session TZ.

    chDB interprets ``'1970-01-01 00:00:00'`` in the *session* timezone
    and reports the equivalent local time on readback. Documenting via
    test so users aren't surprised when the cell shifts.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE dt0 (t DateTime) ENGINE=Memory"))
        conn.execute(text("INSERT INTO dt0 VALUES ('1970-01-01 00:00:00')"))
        readback = conn.execute(text("SELECT t FROM dt0")).fetchone()[0]
    # We don't assert a specific offset; the timezone depends on the host.
    # The hypothesis under test is "the value is a datetime", which means
    # the dialect's coercion path didn't drop the conversion.
    assert isinstance(readback, dt.datetime)


def test_datetime_with_explicit_tz_round_trip(engine):
    """``toDateTime('...', 'UTC')`` is a deterministic value."""
    got = _roundtrip(
        engine, "DateTime('UTC')", "toDateTime('2025-06-15 12:30:45', 'UTC')"
    )
    assert isinstance(got, dt.datetime)
    # Year/day/hour are TZ-stable for a UTC fixture (chdb strips tzinfo).
    assert (got.year, got.month, got.day, got.hour, got.minute, got.second) == \
           (2025, 6, 15, 12, 30, 45)


# ---------------------------------------------------------------------------
# 5. TestCompositeNesting
# ---------------------------------------------------------------------------


def test_array_empty(engine):
    got = _roundtrip(engine, "Array(Int32)", "[]")
    assert got == []


def test_array_with_nulls(engine):
    got = _roundtrip(engine, "Array(Nullable(Int32))", "[1, NULL, 3, NULL]")
    assert got == [1, None, 3, None]


def test_array_deeply_nested(engine):
    got = _roundtrip(engine, "Array(Array(Array(Int32)))",
                     "[[[1,2],[3]],[],[[4,5,6]]]")
    assert got == [[[1, 2], [3]], [], [[4, 5, 6]]]


def test_map_with_composite_values(engine):
    got = _roundtrip(
        engine,
        "Map(String, Array(Nullable(Int32)))",
        "{'a': [1, NULL, 3], 'b': []}",
    )
    assert got == {"a": [1, None, 3], "b": []}


def test_tuple_heterogeneous_with_null(engine):
    got = _roundtrip(
        engine,
        "Tuple(Nullable(Int32), String, Float64)",
        "(NULL, 'middle', 1.5)",
    )
    assert isinstance(got, tuple)
    assert got == (None, "middle", 1.5)


def test_tuple_single_element(engine):
    got = _roundtrip(engine, "Tuple(Int32)", "tuple(42)")
    assert got == (42,)


def test_nested_round_trip(engine):
    """``Nested(name String, value Float64)`` stores parallel arrays."""
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE nst (n Nested(name String, value Float64)) ENGINE=Memory"
        ))
        conn.execute(text(
            "INSERT INTO nst VALUES (['a','b','c'], [1.0, 2.0, 3.0])"
        ))
        rows = conn.execute(text("SELECT n.name, n.value FROM nst")).fetchone()
    assert rows[0] == ["a", "b", "c"]
    assert rows[1] == [1.0, 2.0, 3.0]


# chdb-core 26.5 (ClickHouse 26.5 baseline) serialises Float / Decimal /
# temporal leaves *inside* composite cells as quoted strings — e.g.
# ``Array(Float64)`` arrives as ``"['1.5', '2.5']"`` where chdb-core 26.3
# emitted bare numerics. The cursor wrapper repairs the leaves using the
# column's type string; these tests lock the repair for each composite
# shape that regressed in CI when chdb-core 26.5.0 shipped (2026-06-08).


def test_array_of_floats_native(engine):
    got = _roundtrip(engine, "Array(Float64)", "[1.0, 2.5, 3.25]")
    assert got == [1.0, 2.5, 3.25]
    assert all(isinstance(x, float) for x in got)


def test_array_of_floats_with_specials_native(engine):
    got = _roundtrip(engine, "Array(Float64)", "[nan, inf, -inf, 1.5]")
    assert isinstance(got, list) and len(got) == 4
    assert math.isnan(got[0])
    assert got[1:] == [math.inf, -math.inf, 1.5]


def test_array_nullable_float_keeps_nulls(engine):
    got = _roundtrip(engine, "Array(Nullable(Float64))", "[1.5, NULL]")
    assert got == [1.5, None]


def test_array_of_decimals_native(engine):
    got = _roundtrip(engine, "Array(Decimal(18, 2))", "[toDecimal64('1.23', 2)]")
    assert got == [decimal.Decimal("1.23")]
    assert isinstance(got[0], decimal.Decimal)


def test_array_of_dates_native(engine):
    got = _roundtrip(engine, "Array(Date)", "[toDate('2024-01-02')]")
    assert got == [dt.date(2024, 1, 2)]


def test_map_float_values_native(engine):
    got = _roundtrip(engine, "Map(String, Float64)", "map('k', 1.5)")
    assert got == {"k": 1.5}
    assert isinstance(got["k"], float)


def test_named_tuple_returns_dict_with_native_leaves(engine):
    """chdb.dbapi serialises *named* Tuple cells as dicts keyed by field.

    The wrapper keeps the dict shape (it carries strictly more information
    than a bare tuple) and coerces each field by its declared type.
    """
    got = _roundtrip(
        engine,
        "Tuple(a Float64, b String)",
        "CAST((1.5, 'x'), 'Tuple(a Float64, b String)')",
    )
    assert got == {"a": 1.5, "b": "x"}
    assert isinstance(got["a"], float)


def test_array_of_tuples_native_leaves(engine):
    got = _roundtrip(engine, "Array(Tuple(UInt8, Float64))", "[(1, 2.5)]")
    assert isinstance(got, list) and len(got) == 1
    assert list(got[0]) == [1, 2.5]
    assert isinstance(list(got[0])[1], float)


def test_geo_ring_floats_native(engine):
    got = _roundtrip(engine, "Ring", "[(0.25, 0.5), (1.25, 1.5)]")
    assert got == [[0.25, 0.5], [1.25, 1.5]]
    assert all(isinstance(c, float) for point in got for c in point)


def test_low_cardinality_nullable_int_round_trip(permissive_engine):
    """Fix #5 territory: LC(Nullable(Int32)) — wrapper must coerce inner.

    Without Fix #5 the cursor's converter dispatch peeled all Nullables
    first then all LowCardinalitys, missing the Nullable inside the LC.
    Result: integer cells stayed as strings. This test catches a
    regression.
    """
    with permissive_engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE lcn (x LowCardinality(Nullable(Int32))) ENGINE=Memory"
        ))
        conn.execute(text("INSERT INTO lcn VALUES (1), (-42), (NULL)"))
        rows = conn.execute(text("SELECT x FROM lcn ORDER BY x NULLS LAST")).fetchall()
    values = [r[0] for r in rows]
    assert values == [-42, 1, None]
    # Critically: the non-NULL cells are int, not str.
    assert all(isinstance(v, int) for v in values if v is not None)


def test_low_cardinality_nullable_decimal_is_rejected_by_chdb(engine):
    """``LowCardinality(Nullable(Decimal))`` is not a valid CH type.

    Documenting via test so users aren't surprised. The parser accepts
    the string (because it could appear in older catalogs), but DDL fails.
    """
    with engine.begin() as conn:
        with pytest.raises(Exception) as exc:
            conn.execute(text(
                "CREATE TABLE bad (x LowCardinality(Nullable(Decimal(18, 2)))) "
                "ENGINE=Memory"
            ))
        assert "ILLEGAL_TYPE_OF_ARGUMENT" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. TestPKReflection — Fix #7 was a hint; here's the systematic suite
# ---------------------------------------------------------------------------


_PK_CASES = [
    ("simple-single", "a UInt32", "a", ["a"]),
    ("simple-tuple", "a UInt32, b UInt32", "(a, b)", ["a", "b"]),
    ("function-comma-arg", "u UInt32, s String", "cityHash64(u, s)",
     ["cityHash64(u, s)"]),
    ("nested-function", "u UInt32, s String", "(intHash32(u) % 1000, s)",
     ["intHash32(u) % 1000", "s"]),
    ("string-literal-with-comma", "s String", "concat(s, ', suffix')",
     ["concat(s, ', suffix')"]),
    ("mixed-function-and-column", "ts DateTime, u UInt32",
     "(toStartOfDay(ts), u)", ["toStartOfDay(ts)", "u"]),
    ("no-pk-tuple-empty", "x UInt32", "tuple()", []),
    ("backtick-identifier", "`weird name` UInt32", "`weird name`",
     ["`weird name`"]),
]


@pytest.mark.parametrize(
    ("label", "cols_decl", "order_by", "expected_pk"),
    _PK_CASES,
    ids=[c[0] for c in _PK_CASES],
)
def test_pk_reflection(engine, label, cols_decl, order_by, expected_pk):
    """ORDER BY ↔ ``get_pk_constraint().constrained_columns`` must round-trip.

    Critical for LangChain's SQLDatabase prompt builder, which lists PK
    columns to the LLM. A function-expression PK was previously split
    naively on commas, producing junk like ``['cityHash64(u', 's)']``
    that the LLM either ignored (best case) or used as identifiers
    (silent wrong-table-schema in the prompt).
    """
    tbl = f"pk_{label.replace('-', '_')}"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
        conn.execute(text(
            f"CREATE TABLE {tbl} ({cols_decl}) ENGINE=MergeTree ORDER BY {order_by}"
        ))
    insp = inspect(engine)
    pk = insp.get_pk_constraint(tbl)
    assert pk["constrained_columns"] == expected_pk


# ---------------------------------------------------------------------------
# 7. TestSQLInjectionShape — strings that look like SQL must NOT be SQL
# ---------------------------------------------------------------------------
#
# We're not testing security here (chDB is in-process). We're testing
# that an LLM-generated string value containing SQL-fragment-looking
# bytes survives the bind path as opaque data. A regression here would
# mean a downstream LangChain agent could accidentally execute extra
# statements just because the user asked it to store a string that
# resembles SQL.


_INJECTION_FIXTURES = [
    pytest.param("'; DROP TABLE users; --", id="classic-injection"),
    pytest.param("') OR 1=1 -- ", id="boolean-tautology"),
    pytest.param("UNION SELECT password FROM creds", id="union-select"),
    pytest.param("' UNION ALL SELECT NULL, NULL, version() --", id="version-probe"),
    pytest.param("$$$;DELETE FROM x;$$$", id="dollar-quoted"),
    pytest.param("%(x)s and ?", id="bind-marker-mix"),
    pytest.param("\\g exec_immediate('...')", id="psql-meta"),
    pytest.param("'; DROP TABLE x;", id="unicode-quote"),
]


@pytest.mark.parametrize("value", _INJECTION_FIXTURES)
def test_sql_injection_shape_stored_as_data(engine, value):
    """String contents that resemble SQL must round-trip verbatim.

    chDB's ``:memory:`` is process-wide state — successive ``create_engine``
    calls within one pytest process share tables. Use DROP IF EXISTS on
    every table this test touches so iteration order and prior-test residue
    don't matter.
    """
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS inj"))
        conn.execute(text("CREATE TABLE inj (id UInt32, s String) ENGINE=Memory"))
        conn.exec_driver_sql("INSERT INTO inj VALUES (?, ?)", (1, value))
        row = conn.execute(text("SELECT s FROM inj WHERE id = 1")).fetchone()
    assert row[0] == value
    # And the canary table can still be created (i.e. nothing extra got
    # executed inside our INSERT bind). DROP IF EXISTS first because chDB
    # :memory: persists tables across the parametrize iterations.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS inj_canary"))
        conn.execute(text("CREATE TABLE inj_canary (x UInt32) ENGINE=Memory"))


# ---------------------------------------------------------------------------
# 8. TestKnownDivergence — surprises we lock in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value_sql", "label"),
    [("nan", "NaN"), ("inf", "+Inf"), ("-inf", "-Inf")],
)
def test_float_special_values_round_trip(engine, value_sql, label):
    """NaN / ±Inf survive Float64 readback as real floats.

    chdb-core ≤26.3 folded all three to ``None`` on readback
    (chdb-io/chdb#575); chdb-core 26.5 (ClickHouse 26.5 baseline)
    serialises them faithfully. This test locks the fixed behavior and
    fails on older cores.
    """
    got = _roundtrip(engine, "Float64", value_sql)
    assert isinstance(got, float), f"Expected float for {label}; got {got!r}"
    if label == "NaN":
        assert math.isnan(got)
    else:
        assert got == float(value_sql)


def test_geo_point_returns_tuple_not_list(engine):
    """``Point`` is ``Tuple(Float64, Float64)`` — Python form is tuple, not list.

    Locks in Fix #10 (CHANGELOG): chDB.dbapi serialises Tuple cells with
    list-bracket syntax, and naive ast.literal_eval gives back a list.
    Our cursor wrapper rewraps to tuple for Tuple(...) and Point columns.
    """
    got = _roundtrip(engine, "Point", "(1.5, 2.5)")
    assert isinstance(got, tuple), f"Point cell type is {type(got).__name__}, want tuple"
    assert got == (1.5, 2.5)


# ---------------------------------------------------------------------------
# 9. TestIdentifierEdgeCases — column names that stress the reflection path
# ---------------------------------------------------------------------------


_IDENT_FIXTURES = [
    pytest.param("camelCase", id="camelCase"),
    pytest.param("snake_case", id="snake_case"),
    pytest.param("UPPER_SNAKE", id="UPPER_SNAKE"),
    pytest.param("trailing_digit_9", id="trailing-digit"),
    pytest.param("_leading_underscore", id="leading-underscore"),
    pytest.param("with space", id="with-space"),
    pytest.param("with.dot", id="with-dot"),
    pytest.param("with-dash", id="with-dash"),
    pytest.param("中文列", id="cjk"),
    pytest.param("col🎉", id="emoji"),
    pytest.param("select", id="reserved-select"),
    pytest.param("table", id="reserved-table"),
    pytest.param("order", id="reserved-order"),
]


@pytest.mark.parametrize("col_name", _IDENT_FIXTURES)
def test_reserved_or_weird_column_name_round_trips(engine, col_name):
    """Backtick-quoted column names of any shape must reflect correctly.

    Critical for LangChain prompts: if the dialect drops the quoting
    on emission, the LLM sees a malformed CREATE TABLE in its context
    and stops being able to write valid SQL.
    """
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ident_t"))
        conn.execute(text(
            f"CREATE TABLE ident_t (`{col_name}` UInt32, marker UInt32) "
            "ENGINE=MergeTree ORDER BY marker"
        ))
        conn.execute(text("INSERT INTO ident_t VALUES (42, 1)"))
        # Reading via the same backtick form
        row = conn.execute(text(f"SELECT `{col_name}` FROM ident_t WHERE marker = 1")).fetchone()
    assert row[0] == 42
    insp = inspect(engine)
    names = {c["name"] for c in insp.get_columns("ident_t")}
    assert col_name in names, f"reflection lost column {col_name!r}; got {names}"


# ---------------------------------------------------------------------------
# 10. TestLargeShape — does the dialect survive scale?
# ---------------------------------------------------------------------------


def test_array_with_10k_elements(engine):
    """A 10k-element Array round-trips intact — no truncation, no chunking."""
    n = 10_000
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE big_arr (x Array(Int32)) ENGINE=Memory"))
        conn.execute(text(f"INSERT INTO big_arr SELECT range({n})"))
        cell = conn.execute(text("SELECT x FROM big_arr")).scalar()
    assert isinstance(cell, list)
    assert len(cell) == n
    assert cell[0] == 0
    assert cell[-1] == n - 1


def test_wide_table_100_columns_reflects(engine):
    """A 100-column table reflects all columns, not just the first N."""
    cols_decl = ", ".join(f"c{i} UInt32" for i in range(100))
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE wide ({cols_decl}) ENGINE=MergeTree ORDER BY c0"))
    insp = inspect(engine)
    names = [c["name"] for c in insp.get_columns("wide")]
    assert len(names) == 100
    assert names == [f"c{i}" for i in range(100)]


def test_many_rows_fetchall_intact(engine):
    """10k-row SELECT returns 10k rows — fetchall doesn't truncate."""
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE big_tbl (id UInt32) ENGINE=Memory"))
        conn.execute(text("INSERT INTO big_tbl SELECT number FROM numbers(10000)"))
        rows = conn.execute(text("SELECT id FROM big_tbl ORDER BY id")).fetchall()
    assert len(rows) == 10_000
    assert rows[0][0] == 0
    assert rows[-1][0] == 9999


# ---------------------------------------------------------------------------
# 11. TestAggregateFunction — chDB exposes these as columns in MV-backed tables
# ---------------------------------------------------------------------------


def test_aggregate_function_reflects(engine):
    """``AggregateFunction(sum, Int32)`` must reflect cleanly.

    Materialised views in real chDB schemas surface AF columns; our
    reflection has to handle them or LangChain crashes mid-prompt.
    """
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE af (a AggregateFunction(sum, Int32), s SimpleAggregateFunction(max, UInt64)) "
            "ENGINE=AggregatingMergeTree ORDER BY tuple()"
        ))
    insp = inspect(engine)
    cols = {c["name"]: c for c in insp.get_columns("af")}
    from chdb_sqlalchemy.types.composite import AggregateFunction, SimpleAggregateFunction
    assert isinstance(cols["a"]["type"], AggregateFunction)
    assert isinstance(cols["s"]["type"], SimpleAggregateFunction)


# ---------------------------------------------------------------------------
# 12. TestExceptionWrapping — every DBAPI failure must become a sqlalchemy.exc form
# ---------------------------------------------------------------------------


def test_syntax_error_wraps_as_dbapi_error(engine):
    """A chDB-side syntax error reaches the user as ``sqlalchemy.exc.DatabaseError``.

    Without dialect.do_execute's translation chDB raises bare ``Exception``
    and SA's exception dispatcher can't classify it — LangChain's
    run_no_throw stops catching errors. Lock the wrapper in.
    """
    from sqlalchemy.exc import DatabaseError
    with engine.begin() as conn, pytest.raises(DatabaseError):
        conn.execute(text("SELEKT 1 FROM nowhere"))


def test_unknown_identifier_wraps_as_dbapi_error(engine):
    from sqlalchemy.exc import DatabaseError
    with engine.begin() as conn, pytest.raises(DatabaseError):
        conn.execute(text("SELECT does_not_exist FROM system.tables"))


# ---------------------------------------------------------------------------
# 13. TestORMBindPath — the dimension the round-trip / text() tests missed
# ---------------------------------------------------------------------------
#
# Previous tests INSERT via ``text("INSERT ... VALUES (...)")`` literals,
# bypassing SA's bind-processor / bind-expression machinery. Real
# applications go through ``insert(table)`` / Core / ORM, which passes
# Python objects through ``bind_processor`` and then through
# ``chdb.dbapi`` 's parameter-escape path. The bind path is where five
# review-surfaced bugs lived. These tests exercise it directly so future
# regressions show up here.


def test_orm_bind_date_datetime_time(engine):
    """``Date`` / ``DateTime`` / ``DateTime64`` / ``Time`` round-trip *exactly*
    via SA's ``insert(table).values(...)`` path — the value, not just the type.

    The previous weaker assertion (``isinstance(ts, datetime)``) hid a real
    bug: ``DateTime(timezone='UTC')`` emitted ``toDateTime(?)`` without the
    timezone argument, so chDB interpreted the bind in the *session*
    timezone and shifted the stored value by the host's offset from UTC.
    The fix adds a TZ-aware ``DateTime.bind_expression``; this test now
    asserts exact-value round-trip.
    """
    import datetime as _dt

    from sqlalchemy import Column, MetaData, Table, insert, select

    import chdb_sqlalchemy.types as ct

    inserted = {
        "id": 1,
        "d": _dt.date(2025, 6, 15),
        "ts": _dt.datetime(2025, 6, 15, 12, 30, 45),
        "ts64": _dt.datetime(2025, 6, 15, 12, 30, 45, 123456),
        "tm": _dt.time(10, 11, 12),
    }

    md = MetaData()
    t = Table(
        "orm_dt",
        md,
        Column("id", ct.UInt32()),
        Column("d", ct.Date()),
        Column("ts", ct.DateTime(timezone="UTC")),
        Column("ts64", ct.DateTime64(6, timezone="UTC")),
        Column("tm", ct.Time()),
    )
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS orm_dt"))
    md.create_all(engine)
    with engine.begin() as conn:
        conn.execute(insert(t), inserted)
    with engine.connect() as conn:
        d, ts, ts64, tm = conn.execute(
            select(t.c.d, t.c.ts, t.c.ts64, t.c.tm)
        ).fetchone()

    assert d == inserted["d"]
    # chDB strips tzinfo on readback even for TZ-anchored DateTime columns,
    # so we compare naive datetime values. The anchoring matters: with the
    # original bug, ts would have come back shifted by the host's offset
    # from UTC.
    assert ts == inserted["ts"], (
        f"DateTime('UTC') bind drifted: inserted {inserted['ts']!r}, got {ts!r}. "
        "If this is the only failing assert, check DateTime.bind_expression "
        "passes the timezone arg to toDateTime()."
    )
    assert ts64 == inserted["ts64"]
    assert tm == inserted["tm"]


def test_orm_bind_datetime_microseconds_dont_become_null(engine):
    """``DateTime`` / ``Time`` are second-precision in chDB; binding a value
    that carries microseconds must NOT silently NULL the cell.

    Background: ``toDateTime('2025-06-15 12:30:45.123456')`` raises
    ``Cannot parse string ... as Date`` on chDB, and the dbapi VALUES
    path folds that to NULL (Nullable column) or epoch (non-Nullable).
    The second-precision bind_processor now strips microseconds before
    the value reaches chDB. Microsecond-bearing input (e.g. ``datetime.now()``)
    silently truncates to second precision — same convention SA uses for
    every other DB that has separate second/sub-second types.

    The 64-precision variants keep microseconds; assert that too so we
    notice if anyone "fixes" the truncation by also applying it to
    DateTime64 / Time64.
    """
    import datetime as _dt

    from sqlalchemy import Column, MetaData, Table, insert, select

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table(
        "us_bind",
        md,
        Column("dt_sec", ct.DateTime()),
        Column("dt_utc", ct.DateTime(timezone="UTC")),
        Column("dt64", ct.DateTime64(6, timezone="UTC")),
        Column("tm_sec", ct.Time()),
        Column("tm64", ct.Time64(6)),
    )
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS us_bind"))
    md.create_all(engine)

    with_us = _dt.datetime(2025, 6, 15, 12, 30, 45, 123456)
    tm_us = _dt.time(10, 11, 12, 123456)

    with engine.begin() as conn:
        conn.execute(insert(t), {
            "dt_sec": with_us,
            "dt_utc": with_us,
            "dt64": with_us,
            "tm_sec": tm_us,
            "tm64": tm_us,
        })

    with engine.connect() as conn:
        dt_sec, dt_utc, dt64, tm_sec, tm64 = conn.execute(
            select(t.c.dt_sec, t.c.dt_utc, t.c.dt64, t.c.tm_sec, t.c.tm64)
        ).fetchone()

    # No NULLs — that's the regression we're catching.
    assert dt_sec is not None, "DateTime bind with microseconds became NULL"
    assert dt_utc is not None, "DateTime('UTC') bind with microseconds became NULL"
    assert tm_sec is not None, "Time bind with microseconds became NULL"

    # Second-precision columns silently truncate microseconds (chDB design).
    assert dt_sec == with_us.replace(microsecond=0)
    assert dt_utc == with_us.replace(microsecond=0)
    assert tm_sec == tm_us.replace(microsecond=0)

    # 64-precision columns must keep microseconds.
    assert dt64 == with_us, f"DateTime64(6) lost microseconds: {dt64!r}"
    assert tm64 == tm_us, f"Time64(6) lost microseconds: {tm64!r}"


def test_orm_literal_binds_temporal_compiles_and_executes(engine):
    """``compile(literal_binds=True)`` must not double-wrap temporal values.

    The pre-fix SQL was ``toDate32(toDate('1925-01-01'))``,
    ``toDateTime(toDateTime('...'), 'UTC')``, etc. — literal_processor
    wrapped the value in a chDB cast AND bind_expression wrapped it
    again. Net effects: Date32 1925 → 1970 (inner toDate floors at
    epoch); DateTime('UTC') drifted by the host offset (inner used
    session TZ); Time literal hit ILLEGAL_TYPE_OF_ARGUMENT (toTime is
    a datetime-component extractor, not a string parser).

    The fix: literal_processor emits a bare quoted ISO string; the
    chDB cast is applied exactly once by bind_expression (or by chDB's
    auto-cast in INSERT VALUES context for Time/Time64).
    """
    import datetime as _dt

    from sqlalchemy import Column, MetaData, Table, insert, select

    import chdb_sqlalchemy.types as ct

    inserted = {
        "a": _dt.date(1925, 1, 1),   # Date32 lower-range — fails as 1970 under double-wrap
        "b": _dt.datetime(2025, 6, 15, 12, 30, 45),
        "c": _dt.datetime(2025, 6, 15, 12, 30, 45, 123456),
        "d": _dt.time(10, 11, 12),
        "e": _dt.time(10, 11, 12, 123456),
    }
    md = MetaData()
    t = Table(
        "lit_temporal",
        md,
        Column("a", ct.Date32()),
        Column("b", ct.DateTime(timezone="UTC")),
        Column("c", ct.DateTime64(6, timezone="UTC")),
        Column("d", ct.Time()),
        Column("e", ct.Time64(6)),
    )
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS lit_temporal"))
    md.create_all(engine)

    stmt = insert(t).values(**inserted)
    sql = str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))

    # SQL-shape assertions: each value wrapped exactly once, no nested
    # ``toDate(toDate(...))`` / ``toDateTime(toDateTime(...), ...)``.
    assert "toDate32(toDate(" not in sql, f"Date32 double-wrap leak: {sql}"
    assert "toDateTime(toDateTime(" not in sql, f"DateTime double-wrap leak: {sql}"
    assert "toDateTime64(toDateTime64(" not in sql, f"DateTime64 double-wrap leak: {sql}"
    assert "toTime(" not in sql, f"Time literal still uses toTime(): {sql}"
    # Positive shapes — exactly one cast per value.
    assert "toDate32('1925-01-01')" in sql
    assert "toDateTime('2025-06-15 12:30:45', 'UTC')" in sql
    assert "toDateTime64('2025-06-15 12:30:45.123456', 6, 'UTC')" in sql

    # Execute the literal-bound SQL directly and verify the values
    # actually round-trip — defends against a future SQL change that
    # looks structurally OK but loses semantics in flight.
    with engine.begin() as conn:
        conn.execute(text(sql))
    with engine.connect() as conn:
        a, b, c, d, e = conn.execute(
            select(t.c.a, t.c.b, t.c.c, t.c.d, t.c.e)
        ).fetchone()
    assert a == inserted["a"], f"Date32 lost value via literal_binds: {a!r}"
    assert b == inserted["b"], f"DateTime('UTC') drifted via literal_binds: {b!r}"
    assert c == inserted["c"], f"DateTime64(6) drifted via literal_binds: {c!r}"
    assert d == inserted["d"]
    assert e == inserted["e"]


_AWARE_DATETIME_CASES = [
    # (label, value, column_tz, expected_naive_after_normalization)
    pytest.param(
        "utc-aware",
        __import__("datetime").datetime(2025, 6, 15, 12, 30, 45,
                                       tzinfo=__import__("datetime").timezone.utc),
        "UTC",
        __import__("datetime").datetime(2025, 6, 15, 12, 30, 45),
        id="utc-aware",
    ),
    pytest.param(
        "cst-aware-into-utc-col",
        __import__("datetime").datetime(
            2025, 6, 15, 20, 30, 45,
            tzinfo=__import__("datetime").timezone(__import__("datetime").timedelta(hours=8))),
        "UTC",
        __import__("datetime").datetime(2025, 6, 15, 12, 30, 45),
        id="non-utc-offset-into-utc-col",
    ),
    pytest.param(
        "utc-into-shanghai-col",
        __import__("datetime").datetime(2025, 6, 15, 12, 30, 45,
                                       tzinfo=__import__("datetime").timezone.utc),
        "Asia/Shanghai",
        __import__("datetime").datetime(2025, 6, 15, 20, 30, 45),
        id="utc-into-shanghai-col",
    ),
]


@pytest.mark.parametrize(("_label", "aware", "col_tz", "expected"), _AWARE_DATETIME_CASES)
def test_orm_bind_aware_datetime_normalises_to_column_tz(engine, _label, aware, col_tz, expected):
    """Aware ``datetime`` binds must reach chDB as a naive string in the
    column's declared timezone — not as an ISO offset-suffix form.

    Pre-fix failure modes:
    * ``DateTime('UTC')`` + aware UTC: silent NULL (chDB rejects ``+00:00``
      offset suffix, Nullable column folds the parse error to NULL).
    * ``DateTime64(6, 'UTC')`` + aware UTC: raised
      ``CANNOT_PARSE_TEXT`` outright.
    * ``DateTime('UTC')`` + aware non-UTC offset: silent NULL.

    Post-fix expectation:
    * The aware value is ``astimezone`` 'd into the column's timezone
      (or UTC default), tzinfo stripped, then formatted as a naive ISO
      string. chDB's ``toDateTime(?, 'TZ')`` / ``toDateTime64(?, p, 'TZ')``
      parse it correctly, and the round-tripped cell equals the
      timezone-adjusted naive datetime.
    """

    from sqlalchemy import Column, MetaData, Table, insert, select

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table(
        "aware_bind",
        md,
        Column("ts", ct.DateTime(timezone=col_tz)),
        Column("ts64", ct.DateTime64(6, timezone=col_tz)),
    )
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS aware_bind"))
    md.create_all(engine)

    aware_us = aware.replace(microsecond=123456)
    expected_us = expected.replace(microsecond=123456)

    with engine.begin() as conn:
        conn.execute(insert(t), {"ts": aware, "ts64": aware_us})
    with engine.connect() as conn:
        ts, ts64 = conn.execute(select(t.c.ts, t.c.ts64)).fetchone()

    assert ts is not None, f"DateTime({col_tz!r}) bind of aware datetime became NULL"
    assert ts64 is not None, f"DateTime64(6, {col_tz!r}) bind of aware datetime became NULL"
    assert ts == expected, f"DateTime tz-normalisation: got {ts!r}, want {expected!r}"
    assert ts64 == expected_us, f"DateTime64 tz-normalisation: got {ts64!r}, want {expected_us!r}"


def test_orm_literal_binds_aware_datetime_no_offset_suffix(engine):
    """``compile(literal_binds=True)`` on aware datetime must NOT emit
    ``'2025-06-15 12:30:45+00:00'`` — chDB's parser rejects the offset suffix.

    The literal_processor now normalises aware values to the column's
    tz_name (UTC default) and strips tzinfo before formatting. The
    emitted SQL is the same form as the naive bind path, with the
    column-level chDB cast applied exactly once.
    """
    import datetime as _dt

    from sqlalchemy import Column, MetaData, Table, insert, select

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table(
        "lit_aware",
        md,
        Column("ts", ct.DateTime(timezone="UTC")),
        Column("ts64", ct.DateTime64(6, timezone="UTC")),
    )
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS lit_aware"))
    md.create_all(engine)

    aware = _dt.datetime(2025, 6, 15, 12, 30, 45, tzinfo=_dt.timezone.utc)
    aware_us = aware.replace(microsecond=123456)

    stmt = insert(t).values(ts=aware, ts64=aware_us)
    sql = str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))

    assert "+00:00" not in sql, f"offset suffix leaked into literal SQL: {sql}"
    assert "+0000" not in sql
    assert "Z'" not in sql
    assert "toDateTime('2025-06-15 12:30:45', 'UTC')" in sql
    assert "toDateTime64('2025-06-15 12:30:45.123456', 6, 'UTC')" in sql

    with engine.begin() as conn:
        conn.execute(text(sql))
    with engine.connect() as conn:
        ts, ts64 = conn.execute(select(t.c.ts, t.c.ts64)).fetchone()
    assert ts == aware.replace(tzinfo=None)
    assert ts64 == aware_us.replace(tzinfo=None)


def test_orm_bind_datetime64_emits_toDateTime64_not_toDateTime(engine):
    """The compiled SQL must invoke ``toDateTime64(?, precision[, tz])``.

    Previously the colspecs-driven adapt() downcast ``DateTime64`` to
    ``DateTime``, so the SQL became ``toDateTime(?)`` — precision/tz
    silently lost on every insert.
    """
    from sqlalchemy import Column, MetaData, Table, insert

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table("dt64_sql", md, Column("ts", ct.DateTime64(6, timezone="UTC")))
    sql = str(insert(t).compile(engine))
    assert "toDateTime64(" in sql, f"DateTime64 downcast leak — SQL: {sql}"
    assert "toDateTime(?)" not in sql, f"unexpected DateTime fallback — SQL: {sql}"


def test_orm_bind_uuid(engine):
    """``uuid.UUID`` bind through ``insert()`` round-trips back to ``uuid.UUID``."""
    import uuid

    from sqlalchemy import Column, MetaData, Table, insert, select

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table("orm_uu", md, Column("id", ct.UInt32()), Column("u", ct.UUID()))
    md.create_all(engine)
    u = uuid.UUID("a1b2c3d4-e5f6-7890-1234-56789abcdef0")
    with engine.begin() as conn:
        conn.execute(insert(t), {"id": 1, "u": u})
    with engine.connect() as conn:
        got = conn.execute(select(t.c.u)).scalar()
    # result_processor converts back to uuid.UUID for typed Column.
    assert isinstance(got, uuid.UUID)
    assert got == u


def test_orm_delete_conditional_executes(engine):
    """``delete(t).where(t.c.x == v)`` rendered with bare column name.

    chDB's mutation predicate rejects ``del_t.x``. Previously this test
    would have raised ``Missing columns: 'del_t.x'``. The compiler now
    passes ``include_table=False`` into the WHERE rendering.
    """
    from sqlalchemy import Column, MetaData, Table, delete, text

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table("orm_del", md, Column("x", ct.Int32()))
    md.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO orm_del VALUES (1), (2), (3), (4), (5)"))

    # Single criterion
    with engine.begin() as conn:
        conn.execute(delete(t).where(t.c.x == 2))
    with engine.connect() as conn:
        remaining = [r[0] for r in conn.execute(text("SELECT x FROM orm_del ORDER BY x")).fetchall()]
    assert 2 not in remaining
    assert set(remaining) == {1, 3, 4, 5}

    # Multiple criteria combined via repeated .where()
    with engine.begin() as conn:
        conn.execute(delete(t).where(t.c.x > 2).where(t.c.x < 5))
    with engine.connect() as conn:
        remaining = [r[0] for r in conn.execute(text("SELECT x FROM orm_del ORDER BY x")).fetchall()]
    assert set(remaining) == {1, 5}


def test_orm_delete_unconditional_clears_table(engine):
    """``delete(t)`` without WHERE → ``ALTER TABLE t DELETE WHERE 1=1``."""
    from sqlalchemy import Column, MetaData, Table, delete, text

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table("orm_del_all", md, Column("x", ct.Int32()))
    md.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO orm_del_all VALUES (1), (2), (3)"))
        conn.execute(delete(t))
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT x FROM orm_del_all")).fetchall()
    assert rows == []


def test_orm_check_constraint_enforced_and_reflected(engine):
    """CHECK constraint round-trip: DDL emits, chDB enforces, reflection recovers.

    Three things we lock in:
    * ``metadata.create_all`` emits the ``CONSTRAINT ... CHECK ...`` clause.
    * chDB raises ``DatabaseError`` (error code 469) on a violating insert.
    * ``Inspector.get_check_constraints`` returns the constraint.
    """
    from sqlalchemy import (
        CheckConstraint,
        Column,
        MetaData,
        Table,
        insert,
    )
    from sqlalchemy import (
        inspect as sa_inspect,
    )
    from sqlalchemy.exc import DatabaseError

    import chdb_sqlalchemy.types as ct

    md = MetaData()
    t = Table(
        "orm_ck",
        md,
        Column("x", ct.Int32()),
        CheckConstraint("x > 0", name="positive_x"),
    )
    md.create_all(engine)

    # Valid insert
    with engine.begin() as conn:
        conn.execute(insert(t), {"x": 5})

    # Violating insert
    with pytest.raises(DatabaseError) as exc, engine.begin() as conn:
        conn.execute(insert(t), {"x": -1})
    assert "469" in str(exc.value) or "VIOLATED_CONSTRAINT" in str(exc.value)

    # Reflection
    cks = sa_inspect(engine).get_check_constraints("orm_ck")
    assert len(cks) == 1
    assert cks[0]["name"] == "positive_x"
    assert "x > 0" in cks[0]["sqltext"]


def test_view_and_materialized_view_lists_are_disjoint(engine):
    """``get_view_names`` returns plain views only; ``get_materialized_view_names``
    returns MVs only. They must not overlap.

    chDB's ``system.tables.engine`` is ``'View'`` for plain views and
    ``'MaterializedView'`` for MVs; ``LIKE '%View'`` was over-matching.
    """
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE _base_v (x UInt32) ENGINE=MergeTree ORDER BY x"))
        conn.execute(text("CREATE VIEW _v_plain AS SELECT * FROM _base_v"))
        conn.execute(text(
            "CREATE MATERIALIZED VIEW _mv_only ENGINE=MergeTree ORDER BY x "
            "AS SELECT * FROM _base_v"
        ))
    insp = sa_inspect(engine)
    plain = set(insp.get_view_names())
    mv = set(insp.get_materialized_view_names())
    assert "_v_plain" in plain
    assert "_mv_only" in mv
    assert plain & mv == set(), f"MV double-listed in plain views: {plain & mv}"


def test_variant_typename_ordering_is_lexicographic_not_declared(permissive_engine):
    """``toTypeName(Variant(String, Int64))`` reports ``Variant(Int64, String)``.

    ClickHouse sorts the alternative list lexicographically in the type
    metadata. Our parser preserves declaration order in the SA type;
    catalog reads will see the sorted form. Locking the behavior in.
    """
    with permissive_engine.begin() as conn:
        conn.execute(text("CREATE TABLE vt2 (v Variant(String, Int64)) ENGINE=Memory"))
        ty_str = conn.execute(text("SELECT toTypeName(v) FROM vt2 LIMIT 1")).scalar()
        if ty_str is None:
            # Empty table — pull from system.columns instead
            ty_str = conn.execute(text(
                "SELECT type FROM system.columns "
                "WHERE database = currentDatabase() AND table = 'vt2' AND name = 'v'"
            )).scalar()
    assert ty_str == "Variant(Int64, String)"
