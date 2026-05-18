"""L7 — Property-based fuzz tests for the ClickHouse type-string parser.

Goal: the parser should **never** raise an exception other than
``ChdbTypeNotSupportedError``. Any ``ValueError`` / ``IndexError`` /
``AttributeError`` etc. counts as a parser bug — we'd rather raise a clean
"unknown type" error than crash with a stack trace, because in production
this code path runs inside ``Inspector.get_columns()`` deep under the
LangChain toolkit.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fuzz

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chdb_sqlalchemy.exc import ChdbTypeNotSupportedError
from chdb_sqlalchemy.types.parser import parse_type

# Realistic base type vocabulary the parser must accept.
_BASE_TYPES = [
    "String",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float32",
    "Float64",
    "Bool",
    "Date",
    "Date32",
    "UUID",
    "JSON",
    "Point",
]


base_strategy = st.sampled_from(_BASE_TYPES)


def _wrap_nullable(inner):
    return st.builds(lambda t: f"Nullable({t})", inner)


def _wrap_low_cardinality(inner):
    return st.builds(lambda t: f"LowCardinality({t})", inner)


def _wrap_array(inner):
    return st.builds(lambda t: f"Array({t})", inner)


def _wrap_map(inner):
    return st.builds(lambda k, v: f"Map({k}, {v})", inner, inner)


def _wrap_tuple(inner):
    return st.builds(
        lambda ts: "Tuple(" + ", ".join(ts) + ")",
        st.lists(inner, min_size=1, max_size=4),
    )


nested_type_strategy = st.recursive(
    base_strategy,
    lambda children: st.one_of(
        _wrap_nullable(children),
        _wrap_low_cardinality(children),
        _wrap_array(children),
        _wrap_map(children),
        _wrap_tuple(children),
    ),
    max_leaves=8,
)


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=200)
@given(nested_type_strategy)
def test_valid_nested_strings_always_parse(type_str):
    """For any combination of well-formed ClickHouse types, parsing succeeds."""
    parsed = parse_type(type_str)
    assert parsed is not None


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=200)
@given(st.text(min_size=0, max_size=80))
def test_arbitrary_text_does_not_crash(text):
    """Arbitrary input must either parse or raise ``ChdbTypeNotSupportedError``.

    Anything else — KeyError, IndexError, RecursionError, AttributeError —
    is a parser bug: ``Inspector.get_columns()`` would crash unhandled in
    production with a stack trace.
    """
    try:
        parse_type(text)
    except ChdbTypeNotSupportedError:
        pass  # expected for malformed input
    except Exception as e:  # pragma: no cover - we expect this never to fire
        pytest.fail(f"Parser raised unexpected {type(e).__name__}: {e!r}")
