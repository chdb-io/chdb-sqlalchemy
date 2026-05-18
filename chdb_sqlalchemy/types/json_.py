"""ClickHouse JSON types.

ClickHouse 24.10 shipped a new *semantic* ``JSON`` type that stores values
as a typed sub-column tree, replacing the old string-backed JSON. We keep
two distinct classes:

* :class:`JSON` — the new (24.10+) semantic JSON, default for `JSON` in
  reflected schemas.
* :class:`JSONLegacy` — the pre-24.10 string-backed JSON, exposed for
  forward compat in environments still emitting `Object('json')`.

Result-processor note: chDB returns JSON cells as Python repr-style strings
(``"{'a': 1}"``), not valid JSON. We override the SQLAlchemy ``JSON.result_processor``
to use ``ast.literal_eval`` first, falling back to ``json.loads`` for any
properly-emitted JSON values. See :mod:`composite` for the broader context.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Callable

from sqlalchemy import types

from .common import _ChdbType


def _json_processor(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    # Try ast.literal_eval first — chDB's repr-style output is its native form.
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    # Fallback for any properly-emitted JSON (future chDB versions or
    # values that happened to round-trip through real JSON).
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


class JSON(types.JSON, _ChdbType):
    __visit_name__ = "JSON"

    def result_processor(  # type: ignore[override]
        self, dialect: Any, coltype: Any
    ) -> Callable[[Any], Any]:
        return _json_processor


class JSONLegacy(types.JSON, _ChdbType):
    """Pre-24.10 string-backed JSON / ``Object('json')``."""

    __visit_name__ = "JSONLegacy"

    def result_processor(  # type: ignore[override]
        self, dialect: Any, coltype: Any
    ) -> Callable[[Any], Any]:
        return _json_processor
