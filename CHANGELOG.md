# Changelog

All notable changes to `chdb-sqlalchemy` are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-05-21

### Added

- **CrewAI `NL2SQLTool` compatibility** (#9) — finally delivers the
  CrewAI half of the v0.2 milestone. CrewAI's `NL2SQLTool` emits two
  PostgreSQL-style introspection queries that silently misbehaved
  against chDB in 0.2.0:
  - `_fetch_available_tables()` filtered by `table_schema = 'public'`,
    which doesn't exist in chDB → agent saw an empty schema → useless
    LLM output, no error.
  - `_fetch_all_available_columns(table_name)` filtered only by
    `table_name`, so same-named tables in sibling databases leaked
    their columns into the agent's NL2SQL prompt → LLM hallucinated
    SQL referencing nonexistent columns.

  Both are now rewritten transparently in `_CursorWrapper.execute`:
  `'public'` → `currentDatabase()` for the tables query, and
  `AND table_schema = currentDatabase()` appended to the columns query
  (matches both the legacy literal form and the SA bind-param `?`
  form used by current crewai-tools 1.14+).

  The rewrites are intentionally upstream-fragile (narrow anchored
  regexes targeting the exact CrewAI query shapes) and paired with a
  canary test that inspects `crewai_tools` source and fails loudly
  with a pointer back to the shim file when CrewAI changes upstream.

- `crewai-tools` added to `test-integration` extras for end-to-end
  coverage of the above.

### Infrastructure

- **Tag-triggered PyPI publish workflow** (#10) — pushing a `v*` tag
  to `chdb-io/chdb-sqlalchemy` now automatically builds, validates,
  and publishes to PyPI + creates the GitHub Release with attached
  wheel + sdist. Modeled on chdb-io/chdb's wheel-publish workflow,
  simplified for pure-Python single-wheel distribution. Includes a
  pre-build cross-check that the tag's version equals the version in
  `pyproject.toml` — refuses to upload on mismatch.

### Fixed

- README banner (#8) — corrected from the stale `v0.1.0a1 alpha is
  live on TestPyPI` notice (left over from the 0.1.0a1 scaffolding)
  to point at the v0.2.0 production PyPI release. The PyPI
  description for 0.2.0 is a frozen snapshot and cannot be edited
  retroactively; 0.2.1's PyPI description tab will reflect this fix.

## [0.2.0] — 2026-05-21

### Fixed (bug surfaced by L5 differential vs ClickHouse 26.3.9 (LTS))

10. **`Tuple(...)` returned as `list` not `tuple`** — chdb.dbapi serialises
    Tuple cells using list-bracket syntax (``'[1, 2, 3]'``); our cursor
    wrapper ran `ast.literal_eval` and got a Python list, losing the
    fixed-arity heterogeneous semantics. Added dedicated `_coerce_tuple`
    dispatched on `Tuple(` and `Point` type strings.

### Fixed (bugs surfaced by L4 integration suite)

Nine real defects found by running real LangChain workflows against the dialect.
Each would have broken a category of LLM use cases silently or noisily:

1. **PEP 249 exception re-exports** — `chdb.dbapi` defines `Error` /
   `DatabaseError` / etc. inside `chdb.dbapi.err` but doesn't re-export them
   at the package level. SQLAlchemy's exception dispatcher crashed with
   `AttributeError`. Patched via shim in `import_dbapi`.
2. **`text()` mis-parsing time literals** — switched seed-INSERT path to
   `exec_driver_sql` so `'10:00:00'` time strings aren't read as `:bind` params.
3. **`dialect_options` shape rejected by SQLAlchemy** — replaced our
   `dialect_options={"chdb": {...}}` reflection output with the standard
   `info={"chdb_*": ...}` form which `Column` accepts.
4. **No type compiler for chDB types** — added `ChdbTypeCompiler` so
   `CreateTable(...).compile()` (used by LangChain `get_table_info`)
   produces valid DDL for every type. Without this the agent never saw
   any schema.
5. **`adapt()` broken for composite types** — composite types use
   `*args` `__init__`, which breaks SQLAlchemy's default
   `util.constructor_copy`. Override `adapt()` on `_ChdbType` to use
   `copy.copy()`.
6. **JSON deserializer hooks missing on the dialect** — SQLAlchemy
   crashes the moment a JSON column is read if the dialect doesn't expose
   `_json_serializer` / `_json_deserializer`. Wired up to `json.dumps` /
   `json.loads`.
7. **chDB returns `Array` / `Map` / `Tuple` / `JSON` as Python
   repr-style strings** — added a cursor wrapper that post-processes
   every `fetch*` row using `ast.literal_eval`. This is the upstream
   `chdb.dbapi` lossy-format gap; the shim survives queries through
   `text()` (which LangChain uses) where per-type result processors
   don't fire.
8. **chDB raises bare `Exception` from `cursor.execute`** —
   SQLAlchemy's `_handle_dbapi_exception` only wraps subclasses of
   `dbapi.Error`. Translate to `DatabaseError` in `dialect.do_execute`.
9. **`Decimal` / `Nullable(<numeric>)` cells come back as strings** —
   added numeric coercion in the cursor wrapper, dispatched by
   `cursor.description` type strings. Critical for `sum`/`avg` results
   that the LLM agent uses as numeric arguments for follow-up queries.

### Added

- Initial dialect scaffolding (`ChdbDialect`)
- DB-API adapter on top of existing `chdb.dbapi` (no PEP 249 re-implementation)
- DDL / type compilers (`ChdbDDLCompiler`, `ChdbTypeCompiler`) — required
  by LangChain `SQLDatabase.get_table_info()`
- Cursor wrapper (`_cursor.py`) that repairs chDB's lossy return-value
  formatting transparently across all query paths
- L4a schema-inspection test suite (20 cases): asserts every column / type
  in five realistic schemas reflects correctly and renders in the
  LangChain prompt string
- L4b NL2SQL test suite (20 scenarios × 2 modes): hand-written reference
  SQL + optional live-LLM variant gated on `ANTHROPIC_API_KEY`
- L4c LangChain agent toolkit smoke tests (6 cases for tool construction
  and direct tool invocation) + 5 live-agent scenarios gated on API key
- L6 perf regression suite — 4 cases on smoke/large scale knob:
  reflect-1000-tables, 10k-serial-queries, 4-concurrent-engines, and
  LangChain prompt-build budget
- Recursive parser for ClickHouse type strings — supports nested
  `Array` / `Nullable` / `LowCardinality` / `Tuple` / `Map` / `Nested`
  to arbitrary depth
- Type mappings for the full CH 26.3.9 (LTS) scalar set, including
  `BFloat16`, `Time` / `Time64`, `Variant`, `Dynamic`, semantic `JSON`,
  `Point` / `Ring` / `Polygon` / `MultiPolygon`
- Reflection methods backing `Inspector.get_*` for LangChain
  `SQLDatabaseToolkit` compatibility:
  `get_schema_names`, `get_table_names`, `get_view_names`,
  `get_temp_table_names`, `get_columns`, `get_pk_constraint`,
  `get_foreign_keys` (always `[]`), `get_indexes`, `has_table`
- URI parser supporting `chdb:///:memory:`, `chdb:////absolute/path`,
  `chdb:///./relative/path`, plus `?readonly=` and `?settings=k=v` knobs
- Test pyramid scaffolding: L1 unit, L2 SA suite conformance,
  L3 LangChain contract, L4 LangChain end-to-end (cassettes),
  L5 differential vs ClickHouse server, L6 perf, L7 fuzz
- GitHub Actions CI matrix: Python 3.10/3.11/3.12/3.13 × Ubuntu/macOS
- Documentation: `docs/types.md`, `docs/types-baseline.md`,
  `docs/known-skips.md`, `docs/chdb-vs-server-differences.md`

