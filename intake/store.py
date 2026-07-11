"""SQLite persistence for product foundation records."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from .config import DB_PATH
from .models import (
    IdentificationStatus,
    InventoryStatus,
    MarketplaceStatus,
    Product,
    ShopifyStatus,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    photos_json TEXT NOT NULL DEFAULT '[]',
    main_photo_index INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL DEFAULT '',
    short_summary TEXT NOT NULL DEFAULT '',
    marketplace_description TEXT NOT NULL DEFAULT '',
    shopify_description TEXT NOT NULL DEFAULT '',
    short_description TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    brand TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    product_type TEXT NOT NULL DEFAULT '',
    condition TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '',
    material TEXT NOT NULL DEFAULT '',
    dimensions TEXT NOT NULL DEFAULT '',
    quantity TEXT NOT NULL DEFAULT '',
    included_items TEXT NOT NULL DEFAULT '',
    cost REAL,
    price REAL,
    suggested_price REAL,
    suggested_price_low REAL,
    suggested_price_high REAL,
    price_suggestion_source TEXT NOT NULL DEFAULT '',
    cost_user_entered INTEGER NOT NULL DEFAULT 0,
    price_user_entered INTEGER NOT NULL DEFAULT 0,
    shelf TEXT NOT NULL DEFAULT '',
    operator_hint TEXT NOT NULL DEFAULT '',
    identify_confidence REAL,
    identify_notes TEXT NOT NULL DEFAULT '',
    possible_categories_json TEXT NOT NULL DEFAULT '[]',
    uncertain_fields_json TEXT NOT NULL DEFAULT '[]',
    hint_conflict INTEGER NOT NULL DEFAULT 0,
    multi_product_warning INTEGER NOT NULL DEFAULT 0,
    identify_source TEXT NOT NULL DEFAULT '',
    identification_status TEXT NOT NULL DEFAULT 'needs_review',
    shopify_status TEXT NOT NULL DEFAULT 'not_created',
    marketplace_status TEXT NOT NULL DEFAULT 'not_posted',
    inventory_status TEXT NOT NULL DEFAULT 'pending',
    shopify_product_id TEXT,
    marketplace_url TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_status ON products(inventory_status);
CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);
CREATE TABLE IF NOT EXISTS sku_counters (
    day_key TEXT NOT NULL,
    abbr TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day_key, abbr)
);
"""

# Columns added after the first schema — migrated on init.
EXTRA_COLUMNS: list[tuple[str, str]] = [
    ("main_photo_index", "INTEGER NOT NULL DEFAULT 0"),
    ("short_summary", "TEXT NOT NULL DEFAULT ''"),
    ("marketplace_description", "TEXT NOT NULL DEFAULT ''"),
    ("shopify_description", "TEXT NOT NULL DEFAULT ''"),
    ("brand", "TEXT NOT NULL DEFAULT ''"),
    ("model", "TEXT NOT NULL DEFAULT ''"),
    ("category", "TEXT NOT NULL DEFAULT ''"),
    ("product_type", "TEXT NOT NULL DEFAULT ''"),
    ("condition", "TEXT NOT NULL DEFAULT ''"),
    ("color", "TEXT NOT NULL DEFAULT ''"),
    ("material", "TEXT NOT NULL DEFAULT ''"),
    ("dimensions", "TEXT NOT NULL DEFAULT ''"),
    ("quantity", "TEXT NOT NULL DEFAULT ''"),
    ("included_items", "TEXT NOT NULL DEFAULT ''"),
    ("suggested_price", "REAL"),
    ("suggested_price_low", "REAL"),
    ("suggested_price_high", "REAL"),
    ("price_suggestion_source", "TEXT NOT NULL DEFAULT ''"),
    ("cost_user_entered", "INTEGER NOT NULL DEFAULT 0"),
    ("price_user_entered", "INTEGER NOT NULL DEFAULT 0"),
    ("operator_hint", "TEXT NOT NULL DEFAULT ''"),
    ("possible_categories_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("uncertain_fields_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("hint_conflict", "INTEGER NOT NULL DEFAULT 0"),
    ("multi_product_warning", "INTEGER NOT NULL DEFAULT 0"),
    ("identify_source", "TEXT NOT NULL DEFAULT ''"),
    ("identification_status", "TEXT NOT NULL DEFAULT 'needs_review'"),
    ("shopify_status", "TEXT NOT NULL DEFAULT 'not_created'"),
    ("marketplace_status", "TEXT NOT NULL DEFAULT 'not_posted'"),
]


def _safe_enum(enum_cls, value, default):
    try:
        return enum_cls(value)
    except Exception:  # noqa: BLE001
        return default


def _row_get(row: sqlite3.Row, key: str, default=None):
    try:
        val = row[key]
        return default if val is None else val
    except (IndexError, KeyError):
        return default


def _row_to_product(row: sqlite3.Row) -> Product:
    short_summary = _row_get(row, "short_summary", "") or _row_get(row, "short_description", "") or ""
    shopify_description = (
        _row_get(row, "shopify_description", "") or _row_get(row, "description", "") or ""
    )
    return Product(
        id=row["id"],
        sku=row["sku"],
        photos=json.loads(_row_get(row, "photos_json", "[]") or "[]"),
        main_photo_index=int(_row_get(row, "main_photo_index", 0) or 0),
        title=_row_get(row, "title", "") or "",
        short_summary=short_summary,
        marketplace_description=_row_get(row, "marketplace_description", "") or "",
        shopify_description=shopify_description,
        short_description=short_summary,
        description=shopify_description,
        brand=_row_get(row, "brand", "") or "",
        model=_row_get(row, "model", "") or "",
        category=_row_get(row, "category", "") or "",
        product_type=_row_get(row, "product_type", "") or "",
        condition=_row_get(row, "condition", "") or "",
        color=_row_get(row, "color", "") or "",
        material=_row_get(row, "material", "") or "",
        dimensions=_row_get(row, "dimensions", "") or "",
        quantity=_row_get(row, "quantity", "") or "",
        included_items=_row_get(row, "included_items", "") or "",
        cost=_row_get(row, "cost"),
        price=_row_get(row, "price"),
        suggested_price=_row_get(row, "suggested_price"),
        suggested_price_low=_row_get(row, "suggested_price_low"),
        suggested_price_high=_row_get(row, "suggested_price_high"),
        price_suggestion_source=_row_get(row, "price_suggestion_source", "") or "",
        cost_user_entered=bool(_row_get(row, "cost_user_entered", 0)),
        price_user_entered=bool(_row_get(row, "price_user_entered", 0)),
        shelf=_row_get(row, "shelf", "") or "",
        operator_hint=_row_get(row, "operator_hint", "") or "",
        identify_confidence=_row_get(row, "identify_confidence"),
        identify_notes=_row_get(row, "identify_notes", "") or "",
        possible_categories=json.loads(
            _row_get(row, "possible_categories_json", "[]") or "[]"
        ),
        uncertain_fields=json.loads(
            _row_get(row, "uncertain_fields_json", "[]") or "[]"
        ),
        hint_conflict=bool(_row_get(row, "hint_conflict", 0)),
        multi_product_warning=bool(_row_get(row, "multi_product_warning", 0)),
        identify_source=_row_get(row, "identify_source", "") or "",
        identification_status=_safe_enum(
            IdentificationStatus,
            _row_get(row, "identification_status", "needs_review"),
            IdentificationStatus.NEEDS_REVIEW,
        ),
        shopify_status=_safe_enum(
            ShopifyStatus,
            _row_get(row, "shopify_status", "not_created"),
            ShopifyStatus.NOT_CREATED,
        ),
        marketplace_status=_safe_enum(
            MarketplaceStatus,
            _row_get(row, "marketplace_status", "not_posted"),
            MarketplaceStatus.NOT_POSTED,
        ),
        inventory_status=_safe_enum(
            InventoryStatus,
            _row_get(row, "inventory_status", "pending"),
            InventoryStatus.PENDING,
        ),
        shopify_product_id=_row_get(row, "shopify_product_id"),
        marketplace_url=_row_get(row, "marketplace_url"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def category_abbr(category: str = "", product_type: str = "") -> str:
    text = f"{category} {product_type}".upper()
    mapping = [
        ("VANITY", "VAN"),
        ("CHAIR", "CHR"),
        ("MIRROR", "MIR"),
        ("CABLE", "CBL"),
        ("CASE", "CSE"),
        ("BULB", "BLB"),
        ("NOTEBOOK", "NTB"),
        ("BOTTLE", "BTL"),
        ("CHARGER", "CHG"),
        ("FURNITURE", "FUR"),
        ("ELECTRONICS", "ELC"),
    ]
    for needle, abbr in mapping:
        if needle in text:
            return abbr
    letters = re.sub(r"[^A-Z]", "", (product_type or category or "GEN").upper())
    return (letters[:3] or "GEN").ljust(3, "X")[:3]


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
            existing = {
                r[1]
                for r in conn.execute("PRAGMA table_info(products)").fetchall()
            }
            for name, decl in EXTRA_COLUMNS:
                if name not in existing:
                    conn.execute(f"ALTER TABLE products ADD COLUMN {name} {decl}")

    def next_sku(self, category: str = "", product_type: str = "") -> str:
        abbr = category_abbr(category, product_type)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT seq FROM sku_counters WHERE day_key=? AND abbr=?",
                (day, abbr),
            ).fetchone()
            seq = int(row["seq"]) + 1 if row else 1
            conn.execute(
                """
                INSERT INTO sku_counters (day_key, abbr, seq) VALUES (?, ?, ?)
                ON CONFLICT(day_key, abbr) DO UPDATE SET seq=excluded.seq
                """,
                (day, abbr, seq),
            )
            sku = f"{abbr}-{day}-{seq:03d}"
            # Extremely defensive uniqueness check
            while conn.execute(
                "SELECT 1 FROM products WHERE sku=?", (sku,)
            ).fetchone():
                seq += 1
                conn.execute(
                    "UPDATE sku_counters SET seq=? WHERE day_key=? AND abbr=?",
                    (seq, day, abbr),
                )
                sku = f"{abbr}-{day}-{seq:03d}"
            return sku

    def save(self, product: Product) -> Product:
        product.sync_legacy_aliases()
        product.touch()
        if not product.sku:
            product.sku = self.next_sku(product.category, product.product_type)
        with self.connect() as conn:
            # Ensure SKU uniqueness on conflict with another id
            other = conn.execute(
                "SELECT id FROM products WHERE sku=? AND id!=?",
                (product.sku, product.id),
            ).fetchone()
            if other:
                product.sku = self.next_sku(product.category, product.product_type)
            conn.execute(
                """
                INSERT INTO products (
                    id, sku, photos_json, main_photo_index, title,
                    short_summary, marketplace_description, shopify_description,
                    short_description, description,
                    brand, model, category, product_type, condition, color,
                    material, dimensions, quantity, included_items,
                    cost, price, suggested_price, suggested_price_low,
                    suggested_price_high, price_suggestion_source,
                    cost_user_entered, price_user_entered,
                    shelf, operator_hint,
                    identify_confidence, identify_notes,
                    possible_categories_json, uncertain_fields_json,
                    hint_conflict, multi_product_warning, identify_source,
                    identification_status, shopify_status, marketplace_status,
                    inventory_status, shopify_product_id, marketplace_url,
                    created_at, updated_at
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?
                )
                ON CONFLICT(id) DO UPDATE SET
                    sku=excluded.sku,
                    photos_json=excluded.photos_json,
                    main_photo_index=excluded.main_photo_index,
                    title=excluded.title,
                    short_summary=excluded.short_summary,
                    marketplace_description=excluded.marketplace_description,
                    shopify_description=excluded.shopify_description,
                    short_description=excluded.short_description,
                    description=excluded.description,
                    brand=excluded.brand,
                    model=excluded.model,
                    category=excluded.category,
                    product_type=excluded.product_type,
                    condition=excluded.condition,
                    color=excluded.color,
                    material=excluded.material,
                    dimensions=excluded.dimensions,
                    quantity=excluded.quantity,
                    included_items=excluded.included_items,
                    cost=excluded.cost,
                    price=excluded.price,
                    suggested_price=excluded.suggested_price,
                    suggested_price_low=excluded.suggested_price_low,
                    suggested_price_high=excluded.suggested_price_high,
                    price_suggestion_source=excluded.price_suggestion_source,
                    cost_user_entered=excluded.cost_user_entered,
                    price_user_entered=excluded.price_user_entered,
                    shelf=excluded.shelf,
                    operator_hint=excluded.operator_hint,
                    identify_confidence=excluded.identify_confidence,
                    identify_notes=excluded.identify_notes,
                    possible_categories_json=excluded.possible_categories_json,
                    uncertain_fields_json=excluded.uncertain_fields_json,
                    hint_conflict=excluded.hint_conflict,
                    multi_product_warning=excluded.multi_product_warning,
                    identify_source=excluded.identify_source,
                    identification_status=excluded.identification_status,
                    shopify_status=excluded.shopify_status,
                    marketplace_status=excluded.marketplace_status,
                    inventory_status=excluded.inventory_status,
                    shopify_product_id=excluded.shopify_product_id,
                    marketplace_url=excluded.marketplace_url,
                    updated_at=excluded.updated_at
                """,
                (
                    product.id,
                    product.sku,
                    json.dumps(product.photos),
                    product.main_photo_index,
                    product.title,
                    product.short_summary,
                    product.marketplace_description,
                    product.shopify_description,
                    product.short_summary,
                    product.shopify_description,
                    product.brand,
                    product.model,
                    product.category,
                    product.product_type,
                    product.condition,
                    product.color,
                    product.material,
                    product.dimensions,
                    product.quantity,
                    product.included_items,
                    product.cost,
                    product.price,
                    product.suggested_price,
                    product.suggested_price_low,
                    product.suggested_price_high,
                    product.price_suggestion_source,
                    int(product.cost_user_entered),
                    int(product.price_user_entered),
                    product.shelf,
                    product.operator_hint,
                    product.identify_confidence,
                    product.identify_notes,
                    json.dumps(product.possible_categories),
                    json.dumps(product.uncertain_fields),
                    int(product.hint_conflict),
                    int(product.multi_product_warning),
                    product.identify_source,
                    product.identification_status.value,
                    product.shopify_status.value,
                    product.marketplace_status.value,
                    product.inventory_status.value,
                    product.shopify_product_id,
                    product.marketplace_url,
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
