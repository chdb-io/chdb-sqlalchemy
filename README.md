# chdb-sqlalchemy

[SQLAlchemy](https://www.sqlalchemy.org) dialect for [chDB](https://github.com/chdb-io/chdb) — the in-process OLAP SQL engine powered by ClickHouse.

`chdb-sqlalchemy` lets you use chDB as a SQLAlchemy backend, which in turn unlocks every Python data stack that already speaks SQLAlchemy: Django ORM, Flask-SQLAlchemy, `pandas.read_sql()`, Apache Superset, LangChain's `SQLDatabaseToolkit`, CrewAI's `NL2SQLTool`, and more.

> Status: pre-launch placeholder. The initial release is coming soon. This dialect is the gating dependency for the LangChain and CrewAI integrations.

## What this is

chDB already supports [DB-API 2.0](https://peps.python.org/pep-0249/) through `chdb.dbapi`. `chdb-sqlalchemy` adds the layer above that: a dialect that handles connection URIs, table reflection, type mapping, and the introspection contract SQLAlchemy expects.

The dialect is a thin wrapper — chDB's SQL surface is ClickHouse SQL, so most of the dialect's job is type mapping and reflection, not query rewriting.

## Install

```bash
pip install chdb-sqlalchemy
```

## Usage

### Basic connection

```python
from sqlalchemy import create_engine, text

# In-memory
engine = create_engine("chdb:///:memory:")

# Persistent
engine = create_engine("chdb:////tmp/my_chdb")

with engine.connect() as conn:
    result = conn.execute(text("SELECT version()"))
    print(result.scalar())
```

### With pandas

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("chdb:////tmp/my_chdb")
df = pd.read_sql("SELECT * FROM file('data.parquet') LIMIT 100", engine)
```

### With LangChain `SQLDatabaseToolkit`

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

db = SQLDatabase.from_uri("chdb:////tmp/my_chdb")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
```

### With CrewAI `NL2SQLTool`

```python
from crewai_tools import NL2SQLTool

nl2sql = NL2SQLTool(db_uri="chdb:////tmp/my_chdb")
```

## URI format

```
chdb:///:memory:                 # in-memory session
chdb:////absolute/path/to/dir    # persistent session at that directory
chdb:///./relative/path          # persistent session at a relative path
```

There are no host, port, username, or password components — chDB runs in-process. Authentication is delegated to the surrounding application.

## Type mapping

| ClickHouse type | SQLAlchemy type |
|---|---|
| `String`, `FixedString` | `String` |
| `UInt8` … `UInt64`, `Int8` … `Int64` | `Integer`, `BigInteger` |
| `Float32`, `Float64` | `Float` |
| `Decimal(P, S)` | `Numeric` |
| `Date`, `Date32` | `Date` |
| `DateTime`, `DateTime64` | `DateTime` |
| `UUID` | `Uuid` |
| `Array(T)` | `ARRAY` |
| `Tuple(...)`, `Map(K, V)` | `JSON` |
| `JSON` (native) | `JSON` |

## Introspection support

Following the [LangChain `SQLDatabaseToolkit` introspection contract](https://python.langchain.com/docs/integrations/toolkits/sql_database), the dialect implements:

- `get_table_names()` — list user tables visible to the session.
- `get_columns()` — name, type, nullability, default for each column.
- `get_pk_constraint()` — primary key columns (when a `MergeTree` ORDER BY is the de facto PK).
- `get_foreign_keys()` — empty list; chDB does not enforce foreign keys.
- `get_indexes()` — primary and secondary indexes from `system.data_skipping_indices`.

## Roadmap

- **v0.1** — dialect registration, connection URI, basic type mapping, table reflection for `SQLDatabase.from_uri()` and `pandas.read_sql()`.
- **v0.2** — full LangChain `SQLDatabaseToolkit` and CrewAI `NL2SQLTool` certification — both rely on introspection that the v0.1 surface does not yet cover.
- **v0.3** — `remoteSecure()` federated table support exposed as SQLAlchemy `Table` objects.

Milestones land incrementally; check back here or follow [@chdb_io](https://twitter.com/chdb_io) for releases.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Related

- Main chDB repository: https://github.com/chdb-io/chdb
- chDB documentation: https://clickhouse.com/docs/chdb
- chDB DB-API 2.0 module: `chdb.dbapi` in the main repository.
- LLM-friendly index: https://clickhouse.com/docs/chdb/llms.txt
- LangChain integration: https://github.com/chdb-io/langchain-chdb
- Community: https://discord.gg/D2Daa2fM5K
