"""Exceptions raised by chdb-sqlalchemy.

Kept separate from SQLAlchemy's exception hierarchy so callers can distinguish
chdb-sqlalchemy-specific failures (e.g. an unsupported ClickHouse type encountered
during reflection) from generic DB-API or SQLAlchemy errors.
"""

from __future__ import annotations


class ChdbSqlAlchemyError(Exception):
    """Base class for all chdb-sqlalchemy errors."""


class ChdbTypeNotSupportedError(ChdbSqlAlchemyError):
    """Raised when reflection encounters a ClickHouse type the dialect does not map.

    We deliberately raise instead of silently falling back to ``String``:
    silent fallback would let an LLM see an incorrect column type and emit SQL
    that fails at execution. Raising forces the user (or us) to add explicit
    mapping before that schema is exposed to a downstream tool.
    """

    def __init__(self, type_str: str, message: str | None = None) -> None:
        self.type_str = type_str
        super().__init__(
            message
            or (
                f"ClickHouse type {type_str!r} is not mapped to a SQLAlchemy type. "
                "Add a mapping in chdb_sqlalchemy.types or open an issue."
            )
        )


class ChdbUriError(ChdbSqlAlchemyError):
    """Raised when a connection URI cannot be parsed into chDB connect args."""
