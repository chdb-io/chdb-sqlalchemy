"""Shared pytest fixtures for the chdb-sqlalchemy test suite.

Engine fixtures fall back gracefully when ``chdb`` is not installed (e.g.
running just the parser unit tests in a slim CI image). Tests that need an
actual chDB engine declare the ``engine`` fixture explicitly and are skipped
when chDB import fails.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# SQLAlchemy's testing plugin gets loaded conditionally — only by
# tests/sa_suite/conftest.py — because plugin-side fixtures interfere
# with our own conftest discovery when no --dburi is supplied. See
# tests/sa_suite/README.md for how to run the suite.

# Load a project-local .env (if present) so integration tests gated on
# ANTHROPIC_API_KEY pick it up without needing the user to export the var
# in their shell. Silent no-op when no .env exists. Doesn't override
# variables that are already in os.environ.
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.is_file():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)

# Register the dialect before SQLAlchemy is asked for it. In a published
# package this happens via entry-points; during in-tree tests we wire it up
# explicitly so `create_engine('chdb:///...')` resolves to our class.
#
# SQLAlchemy's testing-provision machinery rewrites ``chdb://...`` URLs to
# ``chdb+dbapi://...`` (backend+driver) and re-resolves the dialect. We
# therefore also register the explicit-driver form ``chdb.dbapi`` so
# the SA test suite's URL synthesis round-trips correctly.
from sqlalchemy.dialects import registry

registry.register("chdb", "chdb_sqlalchemy.dialect", "ChdbDialect")
registry.register("chdb.dbapi", "chdb_sqlalchemy.dialect", "ChdbDialect")


def _have_chdb() -> bool:
    try:
        import chdb  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session")
def chdb_available() -> bool:
    return _have_chdb()


@pytest.fixture
def engine(chdb_available):
    """An in-memory chDB SQLAlchemy engine.

    Skipped when chDB is not importable — most unit tests don't need it.
    """
    if not chdb_available:
        pytest.skip("chDB not installed; skipping engine-backed test")
    from sqlalchemy import create_engine

    eng = create_engine("chdb:///:memory:")
    yield eng
    eng.dispose()


@pytest.fixture
def persistent_engine(chdb_available) -> Iterator:
    """A chDB engine with a temp persistent directory.

    Used for tests that need to verify state survives ``engine.dispose()``.
    """
    if not chdb_available:
        pytest.skip("chDB not installed; skipping engine-backed test")
    from sqlalchemy import create_engine

    tmpdir = tempfile.mkdtemp(prefix="chdb-test-")
    try:
        eng = create_engine(f"chdb:///{tmpdir}")
        yield eng
        eng.dispose()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def parser_only():
    """Parser unit tests don't need chDB at all — they're pure-Python."""
    from chdb_sqlalchemy.types.parser import parse_column_type, parse_type

    return parse_type, parse_column_type
