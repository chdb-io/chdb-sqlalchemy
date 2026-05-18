# SQLAlchemy test suite — known skips

When running the SQLAlchemy `dialect_test_suite` against
`chdb:///:memory:`, the following test cases are skipped because chDB /
ClickHouse does not implement the underlying feature. Each entry must
have a one-line reason.

Format: `TestClass::test_method` — *reason*

## v0.1 baseline (declared via `chdb_sqlalchemy.requirements.Requirements`)

The following 199 test cases skip cleanly because their requirement
flag returns ``exclusions.closed()``:

- **Transactions / isolation**: `transactional_ddl`, `two_phase_transactions`,
  `savepoints`, `dialect_level_isolation_level_param`
- **Sequences / autoincrement**: `sequences`, `sequences_optional`,
  `autoincrement_insert`, `autoincrement_without_sequence`,
  `emulated_lastrowid`, `dbapi_lastrowid`
- **Foreign keys**: `foreign_keys`, `foreign_key_constraint_reflection`,
  `foreign_key_constraint_name_reflection`,
  `foreign_key_constraint_option_reflection_*`, `cross_schema_fk_reflection`,
  `self_referential_foreign_keys`, `deferrable_fks`
- **Constraints**: `check_constraints`, `check_constraint_reflection`,
  `unique_constraint_reflection`, `constraint_comment_reflection`
- **Temp tables**: `temporary_tables`, `temp_table_reflection`, `temp_table_names`
- **DML returning**: `insert_returning`, `update_returning`, `delete_returning`,
  `returning`, `update_from`, `delete_from`
- **Empty inserts**: `empty_inserts`, `empty_inserts_executemany`
- **Computed columns**: `computed_columns_*`, `identity_columns_*`
- **Indexes**: `indexes_with_expressions`, `indexes_with_ascdesc`,
  `reflects_indexes_column_sorting`, `index_reflects_included_columns`
- **Misc**: `default_schema_name_switch`, `binary_literals`, `json_array_indexes`,
  `ctes_with_update_delete`, `ctes_on_dml`

## v0.2 triage queue (215 failing classes — out of scope for v0.1)

The buckets below need targeted fixes before they pass cleanly. They
*don't* block v0.1 because none of them are in the LangChain agent /
text-to-SQL path that v0.1 is shipped for; they're SQLAlchemy's
exhaustive compliance battery touching corners that real chDB users
don't hit until the dialect grows broader adoption.

- **`ComponentReflectionTest` (~750 cases)**: SA's reflection
  exhaustiveness. We pass the basic cases (used by LangChain
  `SQLDatabaseToolkit`); the long tail is `CHECK CONSTRAINT` DDL
  suppression we haven't added, computed-column reflection we haven't
  implemented, schema-name-with-spaces handling, and reserved-keyword
  identifier quoting. Each is small but cumulatively several days.
- **`JSONTest` (~130 cases)**: SA's generic JSON bind path differs
  subtly from chDB native JSON literal binding. Needs an
  `info=` shim plus a generic JSON adapter.
- **`FetchLimitOffsetTest` (~45)**: compiler needs to emit
  ``LIMIT n OFFSET m`` form (chDB-style), not ``OFFSET m FETCH NEXT
  n ROWS`` (generic SQL).
- **`ExpandingBoundInTest`, `InsertBehaviorTest`, `CompoundSelectTest`,
  `CTETest`** (~15-40 each): chDB-specific param expansion and DML
  syntax. Mostly compiler overrides.
- **`UuidTest`, `NumericTest`, `EnumTest`, `DateTimeTZTest`,
  `TimeTest`** (~10-15 each): literal-form coverage. Most fail in
  `test_literal_*` paths where the value is inlined into SQL rather
  than bind-parameterised. Need `literal_processor` overrides per type.

## Categorically not supported by chDB (won't be implemented)

These need entries here so the v0.2 work knows to skip rather than
attempt to fix:

- Foreign-key enforcement (ClickHouse design)
- Transactions, savepoints, isolation levels (ClickHouse design)
- Sequences (ClickHouse design — use `generateUUIDv4()` / `now64()`)
- Server-side cursors (chDB is in-process; not applicable)
- DDL transactions (no transactions, period)
