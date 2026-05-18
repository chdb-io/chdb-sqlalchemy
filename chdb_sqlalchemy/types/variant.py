"""``Variant(T1, T2, ...)`` — added in ClickHouse 24.4, GA by 24.10.

A column of this type can hold any one of the declared inner types per row.
Maps to SQLAlchemy ``JSON`` at the abstract level (since SQLAlchemy has no
sum-type) while preserving the declared alternatives for DDL/reflection.
"""

from __future__ import annotations

from sqlalchemy import types

from .common import _ChdbType


class Variant(_ChdbType):
    __visit_name__ = "Variant"

    def __init__(self, *alternatives: types.TypeEngine) -> None:
        if not alternatives:
            raise ValueError("Variant requires at least one alternative type")
        self.alternatives = alternatives
