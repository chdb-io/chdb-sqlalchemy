"""Unit tests for URI parsing in :mod:`chdb_sqlalchemy.connector`.

These exercise the SQLAlchemy ``URL`` → ``chdb.dbapi.connect`` kwargs
translation without touching chDB. Engine-backed tests live in
:mod:`tests.test_dialect`.
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine.url import make_url

from chdb_sqlalchemy.connector import IN_MEMORY, url_to_connect_args
from chdb_sqlalchemy.exc import ChdbUriError


def test_in_memory():
    args = url_to_connect_args(make_url("chdb:///:memory:"))
    assert args == {"path": IN_MEMORY}


def test_absolute_path():
    args = url_to_connect_args(make_url("chdb:////tmp/foo"))
    # 4-slash form encodes absolute path.
    assert args["path"] == "/tmp/foo"


def test_relative_path():
    args = url_to_connect_args(make_url("chdb:///./relative/path"))
    assert args["path"].endswith("/relative/path")
    assert args["path"].startswith("/")  # resolved to abs path


def test_readonly_query_param():
    args = url_to_connect_args(make_url("chdb:////tmp/foo?readonly=1"))
    assert args["readonly"] is True


def test_settings_query_param():
    args = url_to_connect_args(
        make_url("chdb:////tmp/foo?settings=max_memory_usage%3D10G&settings=max_threads%3D4")
    )
    assert args["settings"] == {"max_memory_usage": "10G", "max_threads": "4"}


def test_host_rejected():
    with pytest.raises(ChdbUriError, match="host"):
        url_to_connect_args(make_url("chdb://somehost/:memory:"))


def test_credentials_rejected():
    with pytest.raises(ChdbUriError, match="credentials"):
        url_to_connect_args(make_url("chdb://user:pw@/path"))


def test_empty_path_rejected():
    with pytest.raises(ChdbUriError, match="Empty path"):
        url_to_connect_args(make_url("chdb://"))


def test_malformed_settings_rejected():
    with pytest.raises(ChdbUriError, match="Malformed setting"):
        url_to_connect_args(make_url("chdb:////tmp/foo?settings=nobreak"))
