"""L2 — SQLAlchemy's official dialect compliance test suite.

We pull in :mod:`sqlalchemy.testing.suite` wholesale. Each contained test
is gated on a requirement flag declared in
:class:`chdb_sqlalchemy.requirements.Requirements`. Tests that need a
feature chDB doesn't support skip cleanly; tests that exercise features
we *do* claim to support are the real signal — a failure there is a
dialect bug.

Run with::

    pytest tests/test_suite.py --dburi chdb:///:memory: -ra

Configuration of the requirement class is in ``setup.cfg`` under the
``[sqla_testing]`` section (the SQLAlchemy pytest plugin reads from that
exact file/section by convention).
"""

from __future__ import annotations

from sqlalchemy.dialects import registry

# Ensure our dialect is loadable before the SA testing plugin walks the URL.
registry.register("chdb", "chdb_sqlalchemy.dialect", "ChdbDialect")

# Pull in every test class defined in the SA testing suite. These will be
# collected by pytest exactly as if they were defined here.
from sqlalchemy.testing.suite import *  # noqa: F401, F403, E402
