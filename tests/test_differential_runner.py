"""Unit tests for the ClickHouse differential-test subprocess wrapper."""

from __future__ import annotations

import subprocess

from tests.differential import conftest as differential_conftest


def test_clickhouse_local_sets_only_the_output_format(monkeypatch):
    """The output format must not override how inline VALUES are parsed."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="3\n", stderr="")

    monkeypatch.setattr(differential_conftest.subprocess, "run", fake_run)

    sql = (
        "CREATE TABLE t (x UInt8) ENGINE=Memory; "
        "INSERT INTO t VALUES (1), (2); "
        "SELECT sum(x) FROM t"
    )
    runner = differential_conftest.ClickHouseLocal("/tmp/clickhouse")

    assert runner.run(sql) == "3\n"
    assert calls == [
        (
            [
                "/tmp/clickhouse",
                "local",
                "--query",
                f"{sql} FORMAT TabSeparated",
            ],
            {
                "capture_output": True,
                "text": True,
                "timeout": 60,
            },
        )
    ]
