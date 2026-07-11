"""SQLite persistence for product foundation records."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from .config import DB_PATH
from .models import InventoryStatus, Product


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    photos_json TEXT NOT NULL DEFAULT '[]',
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    short_description TEXT NOT NULL DEFAULT '',
    cost REAL,
    price REAL,
    shelf TEXT NOT NULL DEFAULT '',
    shopify_product_id TEXT,
    marketplace_url TEXT,
    inventory_status TEXT NOT NULL,
    identify_confidence REAL,
    identify_notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_status ON products(inventory_status);
CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);
"""


def _row_to_product(row: sqlite3.Row) -> Product:
    return Product(
        id=row["id"],
        sku=row["sku"],
        photos=json.loads(row["photos_json"] or "[]"),
        title=row["title"] or "",
        description=row["description"] or "",
        short_description=row["short_description"] or "",
        cost=row["cost"],
        price=row["price"],
        shelf=row["shelf"] or "",
        shopify_product_id=row["shopify_product_id"],
        marketplace_url=row["marketplace_url"],
        inventory_status=InventoryStatus(row["inventory_status"]),
        identify_confidence=row["identify_confidence"],
        identify_notes=row["identify_notes"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ProductStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def save(self, product: Product) -> Product:
        product.touch()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO products (
                    id, sku, photos_json, title, description, short_description,
                    cost, price, shelf, shopify_product_id, marketplace_url,
                    inventory_status, identify_confidence, identify_notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    sku=excluded.sku,
                    photos_json=excluded.photos_json,
                    title=excluded.title,
                    description=excluded.description,
                    short_description=excluded.short_description,
                    cost=excluded.cost,
                    price=excluded.price,
                    shelf=excluded.shelf,
                    shopify_product_id=excluded.shopify_product_id,
                    marketplace_url=excluded.marketplace_url,
                    inventory_status=excluded.inventory_status,
                    identify_confidence=excluded.identify_confidence,
                    identify_notes=excluded.identify_notes,
                    updated_at=excluded.updated_at
                """,
                (
                    product.id,
                    product.sku,
                    json.dumps(product.photos),
                    product.title,
                    product.description,
                    product.short_description,
                    product.cost,
                    product.price,
                    product.shelf,
                    product.shopify_product_id,
                    product.marketplace_url,
                    product.inventory_status.value,
                    product.identify_confidence,
                    product.identify_notes,
                    product.created_at,
                    product.updated_at,
                ),
            )
        return product

    def get(self, product_id: str) -> Optional[Product]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM products WHERE id = ?", (product_id,)
            ).fetchone()
        return _row_to_product(row) if row else None

    def get_by_sku(self, sku: str) -> Optional[Product]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM products WHERE sku = ?", (sku,)
            ).fetchone()
        return _row_to_product(row) if row else None

    def list_all(self) -> list[Product]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM products ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_product(r) for r in rows]

    def delete(self, product_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
            return cur.rowcount > 0
