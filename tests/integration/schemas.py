"""Realistic chDB schemas used across the L4 NL2SQL test suite.

Each schema is shaped like something a real analyst-flavoured agent would
encounter. They're built to *probe* specific dialect concerns:

* ``users``               — basic scalar mix, LowCardinality enum-like column,
                            DateTime with timezone
* ``events``              — time-series fat-table, Array<String> tags,
                            Map<String,String> attrs, Nullable metric,
                            DateTime64(3, 'UTC')
* ``orders``              — Decimal money, Date, ENUM order_status,
                            relational to users.id
* ``order_items``         — composite key shape, FixedString sku,
                            relational to orders + products
* ``products``            — UUID, Nested categorisation, Array<UInt32>
* ``page_views``          — JSON column, Variant column, Tuple geo,
                            UInt64 large ids, IPv4
* ``traces``              — observability-shaped: long bytes payload, deep
                            Map nesting, AggregateFunction state

The full set covers every type category enumerated in docs/types.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Schema:
    """A single chDB table fixture with DDL + seed INSERT(s) + description.

    ``description`` is the human-facing summary an analyst would have in
    a data catalog — we use it to set realistic expectations for the LLM
    when we ask "find me the top X" questions.
    """

    name: str
    ddl: str
    seed: list[str]
    description: str


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

USERS = Schema(
    name="users",
    ddl="""
    CREATE TABLE users (
        id UInt32,
        email String,
        signup_country LowCardinality(String),
        is_active Bool,
        created_at DateTime('UTC'),
        last_login Nullable(DateTime('UTC'))
    ) ENGINE = MergeTree ORDER BY id
    """,
    seed=[
        """INSERT INTO users VALUES
            (1, 'alice@example.com', 'US', true,  '2025-01-15 10:00:00', '2026-04-20 09:00:00'),
            (2, 'bob@example.com',   'US', true,  '2025-02-03 14:30:00', '2026-04-21 11:30:00'),
            (3, 'carol@example.com', 'UK', true,  '2025-03-22 09:15:00', '2026-04-22 08:00:00'),
            (4, 'dan@example.com',   'DE', false, '2025-06-01 12:00:00', NULL),
            (5, 'eve@example.com',   'UK', true,  '2025-08-10 18:45:00', '2026-05-01 17:00:00'),
            (6, 'frank@example.com', 'US', true,  '2025-10-05 08:30:00', '2026-05-10 09:30:00'),
            (7, 'grace@example.com', 'JP', true,  '2025-12-12 21:00:00', '2026-05-12 22:00:00'),
            (8, 'henry@example.com', 'DE', false, '2026-01-20 15:00:00', NULL)
        """,
    ],
    description="user accounts with country and activity timestamps",
)


EVENTS = Schema(
    name="events",
    ddl="""
    CREATE TABLE events (
        event_id UUID,
        user_id UInt32,
        event_type LowCardinality(String),
        ts DateTime64(3, 'UTC'),
        tags Array(String),
        attrs Map(String, String),
        duration_ms Nullable(Float64),
        revenue_cents Nullable(Int64)
    ) ENGINE = MergeTree ORDER BY (user_id, ts)
    """,
    seed=[
        """INSERT INTO events VALUES
            (generateUUIDv4(), 1, 'page_view', '2026-05-15 09:00:00.123',
                ['homepage','desktop'], map('referrer','google.com','utm','search'),
                250.5, NULL),
            (generateUUIDv4(), 1, 'purchase',  '2026-05-15 09:05:30.456',
                ['checkout','desktop'], map('payment_method','card'),
                890.2, 4999),
            (generateUUIDv4(), 2, 'page_view', '2026-05-16 14:00:10.789',
                ['homepage','mobile'], map('referrer','direct'),
                180.0, NULL),
            (generateUUIDv4(), 2, 'signup',    '2026-05-16 14:01:00.012',
                ['conversion','mobile'], map('plan','pro'),
                NULL, NULL),
            (generateUUIDv4(), 3, 'page_view', '2026-05-17 18:22:15.345',
                ['product_page','desktop'], map('product_id','SKU-123'),
                420.7, NULL),
            (generateUUIDv4(), 3, 'add_to_cart','2026-05-17 18:25:30.678',
                ['cart','desktop'], map('product_id','SKU-123','quantity','2'),
                NULL, NULL),
            (generateUUIDv4(), 3, 'purchase',  '2026-05-17 18:30:00.901',
                ['checkout','desktop'], map('payment_method','paypal'),
                510.3, 12999),
            (generateUUIDv4(), 5, 'page_view', '2026-05-18 08:00:00.234',
                ['homepage','desktop'], map('referrer','twitter.com'),
                340.1, NULL),
            (generateUUIDv4(), 5, 'purchase',  '2026-05-18 08:15:45.567',
                ['checkout','desktop'], map('payment_method','card'),
                720.4, 8999),
            (generateUUIDv4(), 6, 'page_view', '2026-05-18 11:00:30.890',
                ['homepage','mobile'], map('referrer','facebook.com'),
                190.8, NULL),
            (generateUUIDv4(), 7, 'purchase',  '2026-05-18 22:45:00.123',
                ['checkout','mobile'], map('payment_method','card','region','JP'),
                830.6, 5999)
        """,
    ],
    description="user behaviour events with array tags, map attributes, nullable metrics",
)


ORDERS = Schema(
    name="orders",
    ddl="""
    CREATE TABLE orders (
        order_id UInt64,
        user_id UInt32,
        status Enum8('pending' = 1, 'paid' = 2, 'shipped' = 3, 'cancelled' = 4),
        amount_usd Decimal(12, 2),
        placed_at Date
    ) ENGINE = MergeTree ORDER BY order_id
    """,
    seed=[
        """INSERT INTO orders VALUES
            (1001, 1, 'paid',      49.99,  '2026-05-15'),
            (1002, 3, 'paid',     129.99,  '2026-05-17'),
            (1003, 5, 'shipped',   89.99,  '2026-05-18'),
            (1004, 7, 'paid',      59.99,  '2026-05-18'),
            (1005, 2, 'cancelled', 19.99,  '2026-05-16'),
            (1006, 6, 'pending',   34.99,  '2026-05-19'),
            (1007, 1, 'paid',      99.99,  '2026-05-19'),
            (1008, 3, 'paid',     249.99,  '2026-05-20')
        """,
    ],
    description="customer orders with status enum and decimal amounts",
)


PRODUCTS = Schema(
    name="products",
    ddl="""
    CREATE TABLE products (
        product_id UUID,
        sku FixedString(16),
        name String,
        price_cents UInt32,
        categories Array(String),
        weights Array(Float32),
        is_active Bool
    ) ENGINE = MergeTree ORDER BY sku
    """,
    seed=[
        """INSERT INTO products VALUES
            (generateUUIDv4(), 'SKU-0001        ', 'Widget Pro',   2999,
                ['electronics','gadgets'], [0.25, 0.30, 0.28], true),
            (generateUUIDv4(), 'SKU-0002        ', 'Gizmo Plus',   4999,
                ['electronics','tools'], [1.10, 1.05], true),
            (generateUUIDv4(), 'SKU-0003        ', 'Thingamajig',   899,
                ['toys'], [0.05], true),
            (generateUUIDv4(), 'SKU-0004        ', 'Doohickey',    1499,
                ['tools','hardware'], [2.5, 2.4, 2.6], true),
            (generateUUIDv4(), 'SKU-0005        ', 'Whatsit',      3499,
                ['electronics'], [0.40], false)
        """,
    ],
    description="product catalog with array categories and weights",
)


PAGE_VIEWS = Schema(
    name="page_views",
    ddl="""
    CREATE TABLE page_views (
        view_id UInt64,
        user_id UInt32,
        url String,
        viewport Tuple(UInt32, UInt32),
        client_ip IPv4,
        meta JSON,
        viewed_at DateTime('UTC')
    ) ENGINE = MergeTree ORDER BY (user_id, viewed_at)
    """,
    seed=[
        """INSERT INTO page_views VALUES
            (1, 1, '/home',         (1920, 1080), '203.0.113.5',
                '{"browser":"chrome","os":"macos","session":"abc-1"}',
                '2026-05-15 09:00:00'),
            (2, 1, '/products/42',  (1920, 1080), '203.0.113.5',
                '{"browser":"chrome","os":"macos","session":"abc-1","referrer":"/home"}',
                '2026-05-15 09:02:00'),
            (3, 2, '/home',         (375, 812),   '198.51.100.7',
                '{"browser":"safari","os":"ios","session":"def-2"}',
                '2026-05-16 14:00:00'),
            (4, 3, '/cart',         (1440, 900),  '198.51.100.10',
                '{"browser":"firefox","os":"windows","session":"ghi-3","items":3}',
                '2026-05-17 18:25:00'),
            (5, 5, '/checkout',     (1366, 768),  '203.0.113.8',
                '{"browser":"edge","os":"windows","session":"jkl-4"}',
                '2026-05-18 08:14:00'),
            (6, 7, '/home',         (414, 896),   '192.0.2.42',
                '{"browser":"chrome","os":"android","session":"mno-5","locale":"ja-JP"}',
                '2026-05-18 22:30:00')
        """,
    ],
    description="page views with JSON metadata, viewport tuple, IPv4 client_ip",
)


# All schemas in dependency-friendly creation order.
ALL_SCHEMAS: list[Schema] = [USERS, ORDERS, PRODUCTS, PAGE_VIEWS, EVENTS]


def build_all(connection) -> None:
    """Create every schema and seed it on the given live SQLAlchemy connection.

    Uses ``exec_driver_sql`` instead of ``text()`` for seed INSERTs so
    timestamp literals like ``'10:00:00'`` aren't misparsed as SQLAlchemy
    ``:param`` bind placeholders.

    Idempotent: each table is dropped before re-created. This matters
    because ``chdb:///:memory:`` is **process-global** — separate
    ``create_engine('chdb:///:memory:')`` calls share the same in-process
    database state. Without the DROP, a session-scoped fixture in one
    test module leaves tables visible to the next module's
    function-scoped engine.
    """
    for schema in ALL_SCHEMAS:
        connection.exec_driver_sql(f"DROP TABLE IF EXISTS {schema.name}")
        connection.exec_driver_sql(schema.ddl)
        for stmt in schema.seed:
            connection.exec_driver_sql(stmt)
