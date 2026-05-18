"""L6 — Performance regression tests.

We have a real chDB engine locally — there's no excuse for these to stay
as skip placeholders. Thresholds are set against the v0.1 baseline.

Each test is parameterised on a scale knob so the same case can run as a
fast smoke check in CI and a slow regression check overnight. CI defaults
to the smoke scales; the nightly differential job (see ``ci.yml``)
overrides ``CHDB_PERF_SCALE=large`` to exercise the production thresholds.
"""

from __future__ import annotations

import gc
import os
import threading
import time

import pytest
from sqlalchemy import inspect, text

pytestmark = pytest.mark.perf


def _scale(env_value: str | None = None) -> dict[str, int]:
    """Resolve the perf scale knobs from env, default ``smoke``."""
    s = (env_value or os.environ.get("CHDB_PERF_SCALE", "smoke")).lower()
    if s == "large":
        return {"tables": 1000, "cols": 50, "queries": 10000, "concurrent": 4}
    # smoke
    return {"tables": 100, "cols": 20, "queries": 1000, "concurrent": 4}


def test_reflect_many_tables_under_threshold(engine):
    """Reflection on a wide-schema database must stay sub-linear in table count.

    Threshold (large): 1000 tables × 50 cols under 5 s.
    Threshold (smoke):  100 tables × 20 cols under 0.8 s (10× smaller scale,
    10× tighter budget).
    """
    s = _scale()
    n_tables = s["tables"]
    n_cols = s["cols"]

    with engine.begin() as conn:
        for i in range(n_tables):
            cols = ", ".join(f"c{j} String" for j in range(n_cols))
            conn.exec_driver_sql(
                f"CREATE TABLE wide_{i} (id UInt32, {cols}) "
                f"ENGINE = MergeTree ORDER BY id"
            )

    t0 = time.perf_counter()
    names = inspect(engine).get_table_names()
    for name in names:
        inspect(engine).get_columns(name)
    elapsed = time.perf_counter() - t0

    budget = 5.0 if s["tables"] == 1000 else 1.5
    assert elapsed < budget, (
        f"reflect {n_tables} tables × {n_cols} cols took {elapsed:.2f}s "
        f"(threshold: {budget}s)"
    )


def test_serial_queries_memory_stable(engine):
    """Many serial trivial queries must not leak.

    We snapshot ``tracemalloc`` peak and assert growth < 50 MB across the
    full sweep — generous for smoke runs, tight for the large config.
    """
    import tracemalloc

    s = _scale()
    n = s["queries"]

    tracemalloc.start()
    gc.collect()
    _, peak_before = tracemalloc.get_traced_memory()

    with engine.connect() as conn:
        for _ in range(n):
            conn.execute(text("SELECT 1")).fetchone()

    gc.collect()
    _, peak_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    growth_mb = (peak_after - peak_before) / (1024 * 1024)
    assert growth_mb < 50, (
        f"memory grew by {growth_mb:.1f} MB after {n} serial queries "
        f"(threshold: 50 MB)"
    )


def test_concurrent_engines_no_deadlock(engine):
    """Multiple SQLAlchemy connections on the same engine must coexist.

    chDB is single-process; if our dialect doesn't coordinate Session
    use, 4 parallel queries hang. The test passes if all threads complete
    within a generous wall-clock budget.
    """
    s = _scale()
    n_threads = s["concurrent"]
    queries_per_thread = 50

    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            with engine.connect() as conn:
                for _ in range(queries_per_thread):
                    conn.execute(text("SELECT 1")).fetchone()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    elapsed = time.perf_counter() - t0

    assert all(not t.is_alive() for t in threads), (
        f"deadlock — threads still running after {elapsed:.1f}s"
    )
    assert not errors, f"thread errors: {errors!r}"
    assert elapsed < 20, f"too slow: {n_threads}×{queries_per_thread} took {elapsed:.1f}s"


def test_get_table_info_scales_for_langchain_prompt(engine):
    """LangChain calls ``get_table_info`` on every reflected table during
    toolkit init. With many tables, a quadratic implementation would tank
    agent boot time. Budget: 50 tables produces a prompt in <0.5 s.
    """
    sa = pytest.importorskip("langchain_community.utilities")

    with engine.begin() as conn:
        for i in range(50):
            conn.exec_driver_sql(
                f"CREATE TABLE lc_{i} (id UInt32, name String, ts DateTime) "
                f"ENGINE = MergeTree ORDER BY id"
            )

    db = sa.SQLDatabase(engine)
    t0 = time.perf_counter()
    info = db.get_table_info()
    elapsed = time.perf_counter() - t0
    assert info, "empty prompt — LangChain has nothing to feed the LLM"
    assert elapsed < 1.5, (
        f"get_table_info() over 50 tables took {elapsed:.2f}s "
        f"(threshold: 1.5s — LangChain calls this at agent boot)"
    )
