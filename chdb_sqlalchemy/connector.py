"""Adapter between SQLAlchemy's URL machinery and ``chdb.dbapi``.

``chdb.dbapi`` is already a PEP 249 driver (apilevel=2.0, paramstyle='format').
This module does the SQLAlchemy-side work only:

1. Parse ``chdb:///...`` URIs into the kwargs ``chdb.dbapi.connect()`` expects.
2. Forward query-string knobs (``readonly``, ``settings``) to the chDB session.
3. Coordinate connection lifecycle so that multiple ``engine.connect()`` calls
   against the same path don't trip chDB's single-process Session limit.

The connector returns the ``chdb.dbapi`` module itself for SQLAlchemy's
``dialect_cls.dbapi()`` hook — SQLAlchemy then uses ``module.connect(**kwargs)``
to materialise individual connections.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import unquote

from sqlalchemy.engine.url import URL

from .exc import ChdbUriError

# Sentinel path that chdb.dbapi recognises for an ephemeral in-memory session.
IN_MEMORY = ":memory:"

# Private kwarg name used to pass post-connect ``SET`` commands through
# SA's ``create_connect_args`` machinery. The shim pops it before calling
# the underlying ``chdb.dbapi.connect``, so it never reaches the driver.
_POST_CONNECT_SETTINGS_KEY = "_chdb_post_connect_settings"


_PEP249_EXCEPTIONS = (
    "Warning",
    "Error",
    "InterfaceError",
    "DatabaseError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
)


class _DbapiShim:
    """Wraps the ``chdb.dbapi`` module so ``connect()`` returns our wrapped
    Connection (which in turn returns our wrapped Cursor for result-value
    coercion). Forwards every other attribute access to the underlying
    module so callers can't tell the shim is there.

    Also overrides ``paramstyle`` from upstream's ``"format"`` (``%s``) to
    ``"qmark"`` (``?``). The underlying ``chdb.dbapi`` accepts both styles
    transparently. We pick qmark because SA's ``IdentifierPreparer.__init__``
    sets ``_double_percents = True`` only for the format/pyformat styles —
    that flag causes ``%`` literals in user data (column comments, string
    values) to be doubled to ``%%`` during ``render_literal_value``, which
    chDB then stores literally. qmark sidesteps this entirely.
    """

    __slots__ = ("_mod",)

    paramstyle = "qmark"  # overrides chdb.dbapi.paramstyle

    def __init__(self, mod: Any) -> None:
        self._mod = mod

    def connect(self, *args: Any, **kwargs: Any) -> Any:
        """Open a chDB connection. ``chdb.dbapi.Connection.__init__`` only
        accepts ``path=``; ``readonly`` and ``settings`` from the URI are
        applied **post-connect** via ``SET`` cursor statements (chdb.dbapi
        rejects them as ``__init__`` kwargs)."""
        from ._cursor import wrap_connection

        post_connect_settings = kwargs.pop(_POST_CONNECT_SETTINGS_KEY, None)

        raw = self._mod.connect(*args, **kwargs)

        if post_connect_settings:
            cur = raw.cursor()
            try:
                for key, value in post_connect_settings.items():
                    # SET key = value — chDB Session settings (max_threads,
                    # readonly, max_memory_usage, …). Boolean-like values
                    # are emitted as bare ints; everything else quoted.
                    if isinstance(value, bool):
                        v_sql = "1" if value else "0"
                    elif isinstance(value, (int, float)):
                        v_sql = str(value)
                    else:
                        escaped = str(value).replace("'", "''")
                        v_sql = f"'{escaped}'"
                    cur.execute(f"SET {key} = {v_sql}")
            finally:
                cur.close()

        return wrap_connection(raw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mod, name)


def import_dbapi() -> Any:
    """Return a PEP-249-compliant view of ``chdb.dbapi``.

    Two compatibility shims are applied:

    1. **Exception re-export**: upstream ``chdb.dbapi`` defines
       ``Warning``/``Error``/``DatabaseError``/etc. inside
       ``chdb.dbapi.err`` but doesn't re-export them at the package top
       level. SQLAlchemy's exception dispatcher
       (``_handle_dbapi_exception``) reads ``dbapi.Error`` directly and
       crashes with ``AttributeError`` if it's missing. We monkey-patch
       the missing names onto the module before handing it back.

    2. **Connection / cursor wrapping**: chDB returns ``Decimal`` /
       ``Nullable(<numeric>)`` cells as strings and ``Array`` / ``Map`` /
       ``Tuple`` / ``JSON`` cells as Python repr-style strings — neither
       form survives downstream consumers that expect native Python types.
       We wrap ``connect()`` so its returned Connection produces a cursor
       that post-processes ``fetch*`` results into the native forms.

    Both shims can be deleted once chDB ships fixes upstream and our
    minimum-version floor is raised.
    """
    import chdb.dbapi as dbapi
    from chdb.dbapi import err as _err

    for name in _PEP249_EXCEPTIONS:
        if not hasattr(dbapi, name) and hasattr(_err, name):
            setattr(dbapi, name, getattr(_err, name))
    return _DbapiShim(dbapi)


def url_to_connect_args(url: URL) -> dict[str, Any]:
    """Translate a SQLAlchemy :class:`URL` into kwargs for ``chdb.dbapi.connect``.

    Accepted URI shapes::

        chdb:///:memory:                  → in-memory session
        chdb:////absolute/path            → persistent session at that directory
        chdb:///./relative/path           → CWD-relative persistent session
        chdb:///relative/path?readonly=1  → relative path + readonly mode

    chDB has no host/port/user/password — they must be absent (or empty).

    :raises ChdbUriError: if the URI carries host/auth components, or if the
        path cannot be resolved.
    """
    if url.host:
        raise ChdbUriError(
            f"chDB runs in-process; URI must not include a host (got {url.host!r}). "
            "Use chdb:///:memory: or chdb:////absolute/path."
        )
    if url.username or url.password:
        raise ChdbUriError(
            "chDB runs in-process; URI must not include credentials. "
            "Authentication is the surrounding application's responsibility."
        )
    if url.port:
        raise ChdbUriError(
            f"chDB runs in-process; URI must not include a port (got {url.port!r})."
        )

    raw_path = (url.database or "").strip()
    if not raw_path:
        raise ChdbUriError(
            "Empty path. Use chdb:///:memory: for an ephemeral session "
            "or chdb:////absolute/path for a persistent one."
        )

    if raw_path == IN_MEMORY:
        path = IN_MEMORY
    else:
        # SQLAlchemy URL.database strips the leading '/' for absolute paths
        # written as chdb:////abs/path, leaving us with 'abs/path'. We need
        # to reconstruct: if the original URL had four slashes, treat as
        # absolute; otherwise relative. URL.query/host parsing doesn't
        # preserve that distinction reliably, so we accept both 'abs/path'
        # forms and resolve based on existence + leading marker.
        decoded = unquote(raw_path)
        if decoded.startswith("/"):
            path = decoded
        elif decoded.startswith("./") or decoded.startswith("../"):
            path = os.path.abspath(decoded)
        else:
            # chdb:////tmp/foo lands here as 'tmp/foo' — treat as absolute.
            # This mirrors how sqlite handles its 4-slash form.
            path = "/" + decoded

    kwargs: dict[str, Any] = {"path": path}

    # ``?readonly=1`` and ``?settings=k=v&settings=k=v`` are applied
    # post-connect via ``SET`` statements, because
    # ``chdb.dbapi.Connection.__init__`` only accepts ``path=`` and
    # rejects every other kwarg with TypeError. We gather them in
    # ``_POST_CONNECT_SETTINGS_KEY`` and the shim's ``connect()`` pops
    # the value before calling the real driver.
    query = dict(url.query)
    post_connect: dict[str, str] = {}
    if "readonly" in query:
        readonly_raw = query.pop("readonly")
        post_connect["readonly"] = "1" if readonly_raw in ("1", "true", "True") else "0"
    # 'settings' is a multi-value field: settings=max_memory_usage=10G&settings=...
    settings = query.pop("settings", None)
    if settings is not None:
        settings_list = [settings] if isinstance(settings, str) else list(settings)
        post_connect.update(_parse_settings_list(settings_list))
    if post_connect:
        kwargs[_POST_CONNECT_SETTINGS_KEY] = post_connect
    # Anything else left in the URI query string is not a documented knob —
    # raise rather than silently forwarding to chdb.dbapi.connect (which
    # would TypeError anyway, since it only accepts ``path``).
    if query:
        unknown = ", ".join(sorted(query))
        raise ChdbUriError(
            f"Unknown URI query parameter(s): {unknown}. "
            "Supported: readonly, settings."
        )
    return kwargs


def _parse_settings_list(items: list[str]) -> dict[str, str]:
    """Parse ``['max_memory_usage=10G', 'max_threads=4']`` → ``{...}``."""
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ChdbUriError(
                f"Malformed setting {item!r} — expected key=value form."
            )
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out
