"""SQLAlchemy dialect for chDB.

The dialect is registered with SQLAlchemy via the
``sqlalchemy.dialects`` entry point declared in ``pyproject.toml``::

    [project.entry-points."sqlalchemy.dialects"]
    chdb = "chdb_sqlalchemy.dialect:ChdbDialect"

so importing this package is not necessary for ``create_engine('chdb:///...')``
to work — SQLAlchemy discovers the dialect lazily.
"""

from __future__ import annotations

from .dialect import ChdbDialect
from .exc import (
    ChdbSqlAlchemyError,
    ChdbTypeNotSupportedError,
    ChdbUriError,
)

__version__ = "0.2.1"

__all__ = [
    "ChdbDialect",
    "ChdbSqlAlchemyError",
    "ChdbTypeNotSupportedError",
    "ChdbUriError",
    "__version__",
]
