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

# Transitive plugin load — only active for tests in this subdirectory.
pytest_plugins = ["sqlalchemy.testing.plugin.pytestplugin"]

# Re-register the dialect here too because the parent conftest's
# registration may not have fired yet by the time the SA plugin's
# session-start hook tries to resolve the URL.
from sqlalchemy.dialects import registry

registry.register("chdb", "chdb_sqlalchemy.dialect", "ChdbDialect")
registry.register("chdb.dbapi", "chdb_sqlalchemy.dialect", "ChdbDialect")
