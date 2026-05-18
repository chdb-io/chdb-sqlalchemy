"""Fixtures for L5 differential testing.

We compare chDB (in-process via our dialect) against the official
``clickhouse`` binary running in ``local`` mode (single-process,
embedded, same query engine — the canonical reference build).

Why ``clickhouse local`` rather than ``clickhouse server``:

* No daemon to start/stop/cleanup
* No network port to negotiate
* No clickhouse-driver dependency
* Same query engine code path, so any divergence is purely "chDB build
  vs upstream build"

The trade-off: we don't exercise the native TCP wire protocol. That's a
separate axis worth testing later (and would require Docker), but the
v0.1 priority is "does our chDB-flavoured SQL surface match upstream".

The binary is expected at ``chdb-sqlalchemy/clickhouse``. If absent,
all differential tests skip cleanly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLICKHOUSE_BIN = REPO_ROOT / "clickhouse"


def _have_clickhouse_binary() -> bool:
    return CLICKHOUSE_BIN.is_file() and CLICKHOUSE_BIN.stat().st_size > 0


@pytest.fixture(scope="session")
def clickhouse_local() -> "ClickHouseLocal":
    """Subprocess-backed reference ClickHouse 26.3.9.8-lts.

    Returns a callable wrapper that runs ``clickhouse local --query``
    with a session-persistent in-memory database. Skipped automatically
    when the binary isn't downloaded yet.
    """
    if not _have_clickhouse_binary():
        pytest.skip(
            f"reference clickhouse binary missing at {CLICKHOUSE_BIN} — run "
            f"`curl -fsSL <release-url> -o clickhouse && chmod +x clickhouse` "
            f"from the project root to enable L5 differential testing."
        )
    return ClickHouseLocal(str(CLICKHOUSE_BIN))


class ClickHouseLocal:
    """Tiny wrapper around ``clickhouse local --query`` subprocess calls.

    Each ``run(sql)`` call spins up a fresh process. State doesn't persist
    between calls — fine for differential tests of single queries.

    For multi-statement workflows (CREATE + INSERT + SELECT against the
    same dataset), pass all statements as a single semicolon-separated
    SQL block to ``run`` so they execute in one process.
    """

    def __init__(self, binary_path: str) -> None:
        self.binary = binary_path

    def version(self) -> str:
        return self.run("SELECT version()", fmt="TabSeparated").strip()

    def run(
        self,
        sql: str,
        fmt: str = "TabSeparated",
        path: str | None = None,
    ) -> str:
        """Execute the SQL block and return the raw output string.

        :param fmt: One of ClickHouse's output formats — ``TabSeparated``,
            ``JSON``, ``CSV``, etc. ``TabSeparated`` is the format the
            differential comparator parses.
        :param path: Optional persistent ``--path`` directory. By default
            ``clickhouse local`` uses an ephemeral temp dir per process.
        """
        args = [self.binary, "local", "--query", sql, "--format", fmt]
        if path is not None:
            args += ["--path", path]
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"clickhouse local exited {proc.returncode}\n"
                f"sql: {sql!r}\n"
                f"stderr: {proc.stderr}"
            )
        return proc.stdout

    def run_rows(self, sql: str) -> list[list[str]]:
        """Run a SELECT and return the result as a list-of-string-rows.

        Output is uniformly ``str`` — comparison against chDB rows must
        therefore stringify the chDB side. That's intentional: ``str``
        is the lowest common denominator that side-steps Decimal-as-str
        vs Decimal-as-Decimal display flicker, leaves number formatting
        differences visible (good!) and avoids hiding bugs behind clever
        coercion.
        """
        raw = self.run(sql, fmt="TabSeparated")
        if not raw:
            return []
        return [line.split("\t") for line in raw.rstrip("\n").split("\n")]


# ---------------------------------------------------------------------------
# A session-shared chDB engine seeded identically. We share for speed —
# differential tests are read-only against the seed, so cross-test
# contamination is impossible.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chdb_seeded_engine():
    """Read-only seeded chDB engine for differential queries."""
    pytest.importorskip("chdb")
    from sqlalchemy import create_engine

    from tests.integration import schemas as schemas_mod

    eng = create_engine("chdb:///:memory:")
    with eng.begin() as conn:
        schemas_mod.build_all(conn)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def server_seed_sql() -> str:
    """The exact same DDL+seed SQL we feed chDB, as one big block.

    ``clickhouse local`` runs in a fresh process per ``run`` call; to get
    the same data as our chDB seeded engine, we prepend this block to
    every query.
    """
    from tests.integration import schemas as schemas_mod

    parts: list[str] = []
    for s in schemas_mod.ALL_SCHEMAS:
        parts.append(s.ddl.strip().rstrip(";") + ";")
        for stmt in s.seed:
            parts.append(stmt.strip().rstrip(";") + ";")
    return "\n".join(parts)
