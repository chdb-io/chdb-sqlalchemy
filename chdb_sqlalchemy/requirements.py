"""Declares chDB's feature support to SQLAlchemy's official test suite.

The ``sqlalchemy.testing.suite`` battery is opinionated: each test is
gated on a requirement flag. If a dialect doesn't support a feature, the
matching requirement should return ``exclusions.closed()`` and the test
is **skipped** rather than failed.

We aim for: any failing test in the SA suite reflects either a real
dialect bug or an unhandled requirement gap — never "chDB doesn't
support transactions, so test_savepoints fails", which would be
a noise failure that hides real signal.

Run the suite via::

    pytest tests/test_suite.py --dburi chdb:///:memory: -ra
"""

from __future__ import annotations

from sqlalchemy.testing import exclusions
from sqlalchemy.testing.requirements import SuiteRequirements


def _open() -> exclusions.compound:
    return exclusions.open()


def _closed() -> exclusions.compound:
    return exclusions.closed()


class Requirements(SuiteRequirements):
    """chDB's answers to every feature flag the SA test suite asks about.

    Convention: ``@property`` returning ``_open()`` means "we support it";
    ``_closed()`` means "skip the tests gated on this". The bulk of the
    inherited defaults from :class:`SuiteRequirements` happen to be
    correct for chDB; we only override the cells where chDB diverges
    from a generic relational backend.
    """

    # ------------------------------------------------------------------
    # Transactions — chDB has none. Everything is autocommit.
    # ------------------------------------------------------------------

    @property
    def transactional_ddl(self) -> exclusions.compound:
        return _closed()

    @property
    def two_phase_transactions(self) -> exclusions.compound:
        return _closed()

    @property
    def savepoints(self) -> exclusions.compound:
        return _closed()

    @property
    def dialect_level_isolation_level_param(self) -> exclusions.compound:
        return _closed()

    @property
    def implicit_default_schema(self) -> exclusions.compound:
        return _open()

    # ------------------------------------------------------------------
    # Identifier / catalogue features chDB doesn't have
    # ------------------------------------------------------------------

    @property
    def sequences(self) -> exclusions.compound:
        # chDB has no sequence objects; generateUUIDv4 / now64 are the
        # idiomatic surrogates.
        return _closed()

    @property
    def sequences_optional(self) -> exclusions.compound:
        return _closed()

    @property
    def autoincrement_insert(self) -> exclusions.compound:
        return _closed()

    @property
    def autoincrement_without_sequence(self) -> exclusions.compound:
        return _closed()

    @property
    def emulated_lastrowid(self) -> exclusions.compound:
        return _closed()

    @property
    def dbapi_lastrowid(self) -> exclusions.compound:
        return _closed()

    # ------------------------------------------------------------------
    # Foreign keys — ClickHouse does not enforce them
    # ------------------------------------------------------------------

    @property
    def foreign_keys(self) -> exclusions.compound:
        return _closed()

    @property
    def foreign_key_constraint_reflection(self) -> exclusions.compound:
        return _closed()

    @property
    def foreign_key_constraint_name_reflection(self) -> exclusions.compound:
        return _closed()

    @property
    def foreign_key_constraint_option_reflection_ondelete(self) -> exclusions.compound:
        return _closed()

    @property
    def foreign_key_constraint_option_reflection_onupdate(self) -> exclusions.compound:
        return _closed()

    @property
    def cross_schema_fk_reflection(self) -> exclusions.compound:
        return _closed()

    @property
    def self_referential_foreign_keys(self) -> exclusions.compound:
        return _closed()

    @property
    def fk_constraint_option_reflection_ondelete_noaction(self) -> exclusions.compound:
        return _closed()

    @property
    def fk_constraint_option_reflection_ondelete_restrict(self) -> exclusions.compound:
        return _closed()

    @property
    def fk_constraint_option_reflection_onupdate_restrict(self) -> exclusions.compound:
        return _closed()

    @property
    def deferrable_fks(self) -> exclusions.compound:
        return _closed()

    # ------------------------------------------------------------------
    # Constraints and DDL features chDB doesn't have or only partially has
    # ------------------------------------------------------------------

    @property
    def check_constraints(self) -> exclusions.compound:
        # chDB has CHECK only on INSERT (constraints clause on table) — not
        # at the column level, no UPDATE enforcement.
        return _closed()

    @property
    def check_constraint_reflection(self) -> exclusions.compound:
        return _closed()

    @property
    def index_reflects_included_columns(self) -> exclusions.compound:
        return _closed()

    @property
    def unique_constraint_reflection(self) -> exclusions.compound:
        # Unique constraints are a hint, not enforced.
        return _closed()

    @property
    def temporary_tables(self) -> exclusions.compound:
        # ClickHouse temporary tables exist but with caveats; v0.2 focuses
        # on the headline path. Skip for now.
        return _closed()

    @property
    def temp_table_reflection(self) -> exclusions.compound:
        return _closed()

    @property
    def temp_table_names(self) -> exclusions.compound:
        return _closed()

    @property
    def views(self) -> exclusions.compound:
        return _open()

    # ------------------------------------------------------------------
    # DML / SELECT features
    # ------------------------------------------------------------------

    @property
    def insert_returning(self) -> exclusions.compound:
        return _closed()

    @property
    def update_returning(self) -> exclusions.compound:
        return _closed()

    @property
    def delete_returning(self) -> exclusions.compound:
        return _closed()

    @property
    def empty_inserts(self) -> exclusions.compound:
        # ClickHouse rejects ``INSERT INTO t VALUES ()`` without values.
        return _closed()

    @property
    def empty_inserts_executemany(self) -> exclusions.compound:
        return _closed()

    @property
    def update_from(self) -> exclusions.compound:
        return _closed()

    @property
    def delete_from(self) -> exclusions.compound:
        # ALTER TABLE ... DELETE WHERE exists but isn't generic DELETE FROM.
        return _closed()

    @property
    def returning(self) -> exclusions.compound:
        return _closed()

    @property
    def ctes(self) -> exclusions.compound:
        return _open()

    @property
    def ctes_with_update_delete(self) -> exclusions.compound:
        return _closed()

    @property
    def ctes_on_dml(self) -> exclusions.compound:
        return _closed()

    @property
    def ctes_with_values(self) -> exclusions.compound:
        return _open()

    # ------------------------------------------------------------------
    # Schemas
    # ------------------------------------------------------------------

    @property
    def schemas(self) -> exclusions.compound:
        # ClickHouse calls them "databases"; reflection works via system.databases.
        return _open()

    @property
    def schema_create_delete(self) -> exclusions.compound:
        return _open()

    @property
    def default_schema_name_switch(self) -> exclusions.compound:
        # ``USE`` exists but our dialect doesn't currently use it.
        return _closed()

    # ------------------------------------------------------------------
    # Types — chDB has more than the generic SQL surface, but the SA suite
    # specifically tests "do you handle these correctly". We say yes to
    # everything we've covered in tests/test_type_parser.py.
    # ------------------------------------------------------------------

    @property
    def json_type(self) -> exclusions.compound:
        return _open()

    @property
    def json_array_indexes(self) -> exclusions.compound:
        # ClickHouse JSON indexing via path expressions exists but doesn't
        # match SA's generic JSON[i] surface. Skip the generic test.
        return _closed()

    @property
    def array_type(self) -> exclusions.compound:
        return _open()

    @property
    def datetime_microseconds(self) -> exclusions.compound:
        # DateTime64(6, ...) supports microseconds; default precision is 3.
        return _open()

    @property
    def datetime_timezone(self) -> exclusions.compound:
        return _open()

    @property
    def date_implicit_bound(self) -> exclusions.compound:
        return _open()

    @property
    def datetime_implicit_bound(self) -> exclusions.compound:
        return _open()

    @property
    def precision_generic_float_type(self) -> exclusions.compound:
        return _open()

    @property
    def floats_to_four_decimals(self) -> exclusions.compound:
        return _open()

    @property
    def precision_numerics_general(self) -> exclusions.compound:
        return _open()

    @property
    def precision_numerics_enotation_large(self) -> exclusions.compound:
        return _open()

    @property
    def precision_numerics_many_significant_digits(self) -> exclusions.compound:
        return _open()

    @property
    def precision_numerics_retains_significant_digits(self) -> exclusions.compound:
        # Decimal round-trip preserves significant digits.
        return _open()

    @property
    def cast_precision_numerics_many_significant_digits(self) -> exclusions.compound:
        return _open()

    @property
    def fetch_null_from_numeric(self) -> exclusions.compound:
        return _open()

    @property
    def implicit_decimal_binds(self) -> exclusions.compound:
        return _open()

    @property
    def nullable_booleans(self) -> exclusions.compound:
        return _open()

    @property
    def boolean_col_expressions(self) -> exclusions.compound:
        return _open()

    @property
    def binary_comparisons(self) -> exclusions.compound:
        return _open()

    @property
    def binary_literals(self) -> exclusions.compound:
        return _closed()

    @property
    def unicode_data(self) -> exclusions.compound:
        return _open()

    @property
    def unicode_ddl(self) -> exclusions.compound:
        return _open()

    @property
    def computed_columns(self) -> exclusions.compound:
        # ClickHouse has MATERIALIZED / DEFAULT / ALIAS expressions but the
        # generic computed-column surface doesn't map 1:1. Skip for v0.2.
        return _closed()

    @property
    def computed_columns_stored(self) -> exclusions.compound:
        return _closed()

    @property
    def computed_columns_virtual(self) -> exclusions.compound:
        return _closed()

    @property
    def computed_columns_default_persisted(self) -> exclusions.compound:
        return _closed()

    @property
    def computed_columns_reflect_persisted(self) -> exclusions.compound:
        return _closed()

    @property
    def identity_columns(self) -> exclusions.compound:
        return _closed()

    @property
    def identity_columns_standard(self) -> exclusions.compound:
        return _closed()

    # ------------------------------------------------------------------
    # Server defaults / expressions
    # ------------------------------------------------------------------

    @property
    def server_defaults(self) -> exclusions.compound:
        return _open()

    @property
    def expression_server_defaults(self) -> exclusions.compound:
        return _open()

    @property
    def datetime_literals(self) -> exclusions.compound:
        return _open()

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    @property
    def index_reflection(self) -> exclusions.compound:
        # We reflect data-skipping indexes from system.data_skipping_indices.
        return _open()

    @property
    def indexes_with_expressions(self) -> exclusions.compound:
        # Skip indexes ARE expressions; full SA semantics differs slightly.
        return _closed()

    @property
    def indexes_with_ascdesc(self) -> exclusions.compound:
        return _closed()

    @property
    def reflects_indexes_column_sorting(self) -> exclusions.compound:
        return _closed()

    # ------------------------------------------------------------------
    # Misc reflection
    # ------------------------------------------------------------------

    @property
    def comment_reflection(self) -> exclusions.compound:
        return _open()

    @property
    def comment_reflection_full_unicode(self) -> exclusions.compound:
        return _open()

    @property
    def constraint_comment_reflection(self) -> exclusions.compound:
        return _closed()

    @property
    def primary_key_constraint_reflection(self) -> exclusions.compound:
        # We surface the MergeTree ORDER BY as a PK constraint.
        return _open()

    @property
    def view_reflection(self) -> exclusions.compound:
        return _open()

    @property
    def view_column_reflection(self) -> exclusions.compound:
        return _open()

    @property
    def schema_reflection(self) -> exclusions.compound:
        return _open()
