# chDB vs ClickHouse server — documented semantic differences

Diffs we know about between `chdb:///` and a `clickhouse-server:26.3.9-lts`
running the same query. **Bugs are not listed here** — bugs go to the issue
tracker and CI must turn red. This file is for *intentional* deltas only.

## Format

Each entry has:

- **Query** — the SQL form
- **Server behaviour** — what `clickhouse-server` does
- **chDB behaviour** — what chDB does
- **Why** — the technical reason
- **Workaround** — what users should do

## Entries

### Formatting: composite types in TabSeparated vs Python repr

- **Query forms** — anything returning Array, Map, Tuple, JSON values
- **Server behaviour** — TabSeparated format emits CH-native syntax with
  no internal whitespace: `[1,2,3]`, `{'a':1,'b':2}`, `(1,'two',3.14)`
- **chDB behaviour** — `chdb.dbapi` returns the same values as Python
  repr-style strings *with* whitespace: `'[1, 2, 3]'`, `"{'a': 1, 'b': 2}"`
- **Why** — chDB's Python binding uses Python repr() for composite
  serialisation; the server's TabSeparated output uses ClickHouse's
  native format
- **Workaround** — Both forms parse to the same logical structure
  (`ast.literal_eval` round-trips). Our cursor wrapper converts to
  native Python types before they reach SQLAlchemy; consumers don't
  observe the string-form difference. The differential test normaliser
  compares parsed structures, not raw strings.

### Tuple cells: not actually a difference any more

Initially L5 flagged Tuple cells as `list` from chDB vs `tuple` from
server. Fixed in v0.1.0 by adding a `Tuple`-specific cursor coercer.
Documented in CHANGELOG as bug fix #10, listed here for traceability.
