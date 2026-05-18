"""``Dynamic`` — added in ClickHouse 24.5, GA by 24.10.

Holds values of arbitrary type without a declared alternative list (unlike
``Variant``). At reflection time we have no further sub-type info, so this
maps to a bare ``JSON``-like opaque cell.
"""

from __future__ import annotations

from .common import _ChdbType


class Dynamic(_ChdbType):
    __visit_name__ = "Dynamic"

    def __init__(self, max_types: int | None = None) -> None:
        # CH allows `Dynamic(max_types=32)`; carry it through for DDL fidelity.
        self.max_types = max_types
