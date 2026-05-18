# ClickHouse 26.3.9.8-lts type baseline

This file freezes the set of types `chdb-sqlalchemy` claims to support against
the chDB / ClickHouse 26.3.9.8-lts mainline.

## Regenerating

Against a running ClickHouse 26.3.9.8 instance:

```bash
clickhouse-client --query "
SELECT name, alias_to, case_insensitive
FROM system.data_type_families
ORDER BY name
" > docs/_data_type_families.tsv
```

Then diff `docs/_data_type_families.tsv` against the supported list in
`chdb_sqlalchemy/types/parser.py::_BUILDERS`.

## Supported (as of 0.1.0)

See `docs/types.md` for the per-type SQLAlchemy mapping.

## Skip list

Types we deliberately don't map (and why):

- `IntervalSecond` / `IntervalDay` / etc. — chDB-internal scalar; surfaces in
  expressions but never as a column type. Reflection won't see it.
- `Function` — internal; only ever appears as a system catalog metadata field.

## Open questions

- Does `Dynamic(max_types=N)` round-trip the `N` value via `system.columns.type`?
  If yes, the parser already preserves it; if not, we need to query
  `system.columns.dynamic_max_types` instead.
- `Variant` ordering: `system.columns.type` lists alternatives in declaration
  order, but the catalog sometimes reorders them. Confirm before v0.2 GA.
