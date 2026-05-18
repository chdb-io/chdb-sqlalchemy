# SQLAlchemy dialect compliance suite (L2)

This directory pulls in the full ``sqlalchemy.testing.suite`` battery —
~1400 cases SQLAlchemy publishes as the canonical compliance test for
third-party dialects.

Run with::

    pytest tests/sa_suite/ --dburi chdb:///:memory: -ra

This invocation is excluded from the default ``pytest`` run because the
SA testing plugin's session-start hook requires a ``--dburi`` and would
break collection of our other test directories.

## Status (v0.1 baseline)

| Bucket | Count |
|---|---|
| Passing | 188 |
| Skipped via Requirements (chDB-not-supported) | 199 |
| Failing — chDB upstream / dialect gaps to close in v0.2 | 215 |
| Errors (mostly teardown cascades from earlier failures) | 1104 |

The 215 failures concentrate in the following buckets, by SA test class:

* `ComponentReflectionTest` (~750 cases) — exhaustive reflection battery,
  most fail on ``CHECK CONSTRAINT`` DDL we don't suppress, computed-column
  reflection we don't implement, schema-name-with-special-chars handling.
  v0.2: triage and close each via Requirements declarations.
* `JSONTest` (~130 cases) — SA's generic JSON binding tests use SA's
  `JSON` API surface; chDB native JSON behaves slightly differently for
  literal binding. v0.2: add a generic-JSON adapter shim.
* `FetchLimitOffsetTest` (~45 cases) — chDB syntax for OFFSET-FETCH is
  ``LIMIT n OFFSET m``, generic SQL is ``OFFSET m FETCH NEXT n ROWS``.
  v0.2: compiler override.
* `ExpandingBoundInTest` — chDB-specific param style for IN clauses.
* `UuidTest`, `NumericTest`, `DateTimeTZTest`, etc. — narrower categories.

## What "passing" means at this stage

The suite *runs* end-to-end, the Requirements declarations correctly
skip the categories we don't claim to support, and 188 cases pass
cleanly. This is enough to call L2 "wired and baseline-captured" for
the v0.1 alpha release. Full pass requires multi-week triage that
parallels how `psycopg2` and `mysqlclient` reached their current
compliance — out of scope for v0.1.

## Triage workflow for v0.2

1. Pick a failing class with high count
2. Read a single failure: `pytest tests/sa_suite/ -k <classname> --tb=short`
3. Decide: dialect bug (fix), missing Requirements (declare closed),
   chDB upstream gap (file + skip via `__only_on__`)
4. Re-run, move on

`docs/known-skips.md` should record each class's category before
shipping v0.2.
