"""ClickHouse geo types.

ClickHouse exposes geometric primitives as type aliases over tuples and arrays::

    Point         := Tuple(Float64, Float64)
    Ring          := Array(Point)
    Polygon       := Array(Ring)
    MultiPolygon  := Array(Polygon)

For SQLAlchemy we expose them as first-class classes so reflection round-trips
faithfully ("the column is a Point, not just a Tuple(Float64, Float64)").
"""

from __future__ import annotations

from .common import _ChdbType


class Point(_ChdbType):
    __visit_name__ = "Point"


class Ring(_ChdbType):
    __visit_name__ = "Ring"


class Polygon(_ChdbType):
    __visit_name__ = "Polygon"


class MultiPolygon(_ChdbType):
    __visit_name__ = "MultiPolygon"
