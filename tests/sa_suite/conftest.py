"""Conftest for the SQLAlchemy official dialect compliance suite.

Isolated from the rest of the test tree because loading
``sqlalchemy.testing.plugin.pytestplugin`` for *every* pytest invocation
breaks collection of our other tests (the plugin's fixtures intercept
the no-dburi path). Scoping the plugin to this subdirectory keeps the
suite available but inert outside of it.

Run with::

    pytest tests/sa_suite/ --dburi chdb:///:memory:
"""

from __future__ import annotations

import pytest

# Transitive plugin load — only active for tests in this subdirectory.
pytest_plugins = ["sqlalchemy.testing.plugin.pytestplugin"]

# Re-register the dialect here too because the parent conftest's
# registration may not have fired yet by the time the SA plugin's
# session-start hook tries to resolve the URL.
from sqlalchemy.dialects import registry

registry.register("chdb", "chdb_sqlalchemy.dialect", "ChdbDialect")
registry.register("chdb.dbapi", "chdb_sqlalchemy.dialect", "ChdbDialect")


# Pre-register markers that SA's testing plugin attaches dynamically — some at
# import time via ``config.add_to_marker.<name>`` (mypy / *_intensive), others
# during collection via ``test_class.add_marker(...)`` (backend variants).
# Under pytest 9 with ``--strict-markers`` (set in the root pyproject.toml
# addopts), the lookup fails before SA's session-start hook can register them
# itself, breaking the whole suite.
def pytest_configure(config: pytest.Config) -> None:
    for marker in (
        "mypy",
        "memory_intensive",
        "timing_intensive",
        "backend",
        "sparse_backend",
        "sparse_driver_backend",
    ):
        config.addinivalue_line("markers", f"{marker}: SQLAlchemy testing marker")


# ---------------------------------------------------------------------------
# Skip list — SA tests whose generic assumptions fundamentally clash with
# chDB / chdb.dbapi behaviour and that can't be gated via the
# ``Requirements`` class (because the underlying SA test method has no
# ``@testing.requires`` decorator we can hook). Each entry is recorded in
# ``docs/known-skips.md`` with v0.3 follow-up notes.
# ---------------------------------------------------------------------------

SKIP_TESTS = {
    # chDB Decimal storage normalises trailing zeros — Decimal(5,3)
    # storing 40.020 reads back as 40.02. Not a dialect bug, a chDB
    # storage choice.
    "NumericTest::test_decimal_coerce_round_trip": "chDB Decimal drops trailing zeros",
    "NumericTest::test_numeric_as_decimal": "chDB Decimal drops trailing zeros",
    "NumericTest::test_numeric_null_as_decimal": "chDB Decimal drops trailing zeros",
    # chDB integer division ``intDiv()`` doesn't match Python ``//`` for
    # negative operands, and float division semantics differ slightly.
    "TrueDivTest::test_floordiv_integer": "chDB intDiv semantics differ",
    "TrueDivTest::test_floordiv_integer_bound": "chDB intDiv semantics differ",
    "TrueDivTest::test_truediv_numeric": "chDB float division semantics differ",
    # ClickHouse database identifiers must match ``[a-zA-Z_][a-zA-Z0-9_]*``.
    "UnicodeSchemaTest::test_insert": "chDB DB identifiers must be ASCII",
    "UnicodeSchemaTest::test_reflect": "chDB DB identifiers must be ASCII",
    # chDB String type is variable-length — no fixed length reflection.
    "ComponentReflectionTestExtra::test_string_length_reflection": "chDB String is unbounded",
    # SA-generic LEFT JOIN ON FALSE produces ``WHERE 0`` which chDB
    # doesn't treat as a join falsy.
    "JoinTest::test_outer_join_false": "chDB LEFT JOIN ON FALSE semantics differ",
    # chDB raises bare ``Exception``, not a specific IntegrityError type —
    # because it has no FK/UNIQUE constraints to violate.
    "ExceptionTest::test_integrity_error": "chDB has no integrity constraints",
    # ClickHouse identifier length limit. Test creates a 200+ char name.
    "LongNameBlowoutTest::test_long_convention_name": "chDB identifier max-length differs",
    # SA's RETURNING clause — chDB doesn't support.
    "ReturningGuardsTest::test_delete_many": "chDB has no DELETE RETURNING",
    # SA backslash escaping vs chDB literal escaping differs.
    "StringTest::test_literal_backslashes": "chDB literal backslash differs from SA",
    "TextTest::test_literal_backslashes": "chDB literal backslash differs from SA",
    # UUID literal via SA's text-style binding (toString-then-cast).
    "UuidTest::test_uuid_round_trip": "chDB UUID bind round-trip via string",
    "UuidTest::test_uuid_text_round_trip": "chDB UUID bind round-trip via string",
    "UuidTest::test_literal_text": "chDB UUID text-literal round-trip",
    "UuidTest::test_literal_nonnative_text": "chDB UUID text-literal round-trip",
    # SA inspector caching test — chDB schema may not be fully stable
    # across multiple Inspector() instances in same process.
    "HasTableTest::test_has_table_cache": "chDB schema cache behaviour differs",
    # SA fixture creates a scalar subselect chDB doesn't accept verbatim.
    "RowFetchTest::test_row_w_scalar_select": "chDB scalar subselect syntax differs",
    # SA UNION suite uses parenthesised subselects with ORDER BY/LIMIT —
    # chDB handles them but result order is non-deterministic.
    "DeprecatedCompoundSelectTest::test_plain_union": "chDB UNION result order non-deterministic",
    "DeprecatedCompoundSelectTest::test_limit_offset_aliased_selectable_in_unions": "chDB UNION ordering",
    "CompoundSelectTest::test_plain_union": "chDB UNION result order non-deterministic",
    "CompoundSelectTest::test_limit_offset_aliased_selectable_in_unions": "chDB UNION ordering",
    "DeprecatedCompoundSelectTest::test_distinct_selectable_in_unions": "chDB UNION result order non-deterministic",
    "CompoundSelectTest::test_distinct_selectable_in_unions": "chDB UNION result order non-deterministic",
    # chdb.dbapi list param binding for INSERT into Array column sends
    # ``(1,2,3)`` (Tuple syntax) instead of ``[1,2,3]`` (Array syntax).
    "ArrayTest::test_array_roundtrip": "chdb.dbapi list->Tuple bind syntax",
    "ArrayTest::test_literal_simple": "chdb.dbapi array literal binding",
    "ArrayTest::test_literal_complex": "chdb.dbapi array literal binding",
    # Reflection: chDB DROP VIEW for nonexistent view doesn't raise the
    # same exception type SA expects.
    "ComponentReflectionTest::test_get_view_definition_does_not_exist": "chDB DROP VIEW error differs",
}


def pytest_collection_modifyitems(config, items):
    """Mark tests in :data:`SKIP_TESTS` as skipped.

    The keys are ``ClassName::method_name`` strings. We match against the
    pytest nodeid which has the form
    ``tests/sa_suite/test_suite.py::ClassName_chdb+dbapi::method_name[params]``
    after the SA testing plugin parametrises each test method per backend.
    """
    for item in items:
        for key, reason in SKIP_TESTS.items():
            cls, _, meth = key.partition("::")
            # Class name in nodeid is suffixed by '_chdb+dbapi'.
            target = f"::{cls}_chdb+dbapi::{meth}"
            if target in item.nodeid:
                item.add_marker(
                    pytest.mark.skip(reason=f"chDB-not-supported: {reason}")
                )
                break
