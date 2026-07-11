"""End-to-end tests: five cheap products through Shopify draft."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from intake.identify import SAMPLE_CATALOG
from intake.models import InventoryStatus
from intake.store import ProductStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_products.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    monkeypatch.setenv("SHOPIFY_MOCK", "true")
    monkeypatch.setenv("SHOPIFY_STORE_DOMAIN", "")
    monkeypatch.setenv("SHOPIFY_ACCESS_TOKEN", "")

    import intake.config as config
    import intake.app as app_mod
    import intake.shopify_client as shopify_mod

    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(config, "SHOPIFY_MOCK", True)
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_mod, "SHOPIFY_MOCK", True)

    store = ProductStore(db_path=db_path)
    monkeypatch.setattr(app_mod, "store", store)
    monkeypatch.setattr(app_mod, "shopify", shopify_mod.ShopifyClient(mock=True))

    with TestClient(app_mod.app) as c:
        yield c, store, upload_dir


def _jpeg_bytes(color: tuple[int, int, int], label: str) -> bytes:
    img = Image.new("RGB", (240, 240), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


FIVE_PRODUCTS = [
    {
        "hint": "usb cable",
        "filename": "usb_cable_front.jpg",
        "shelf": "A-01",
        "color": (20, 20, 20),
        "expect_title_substr": "USB",
        "cost": 1.25,
        "price": 6.99,
    },
    {
        "hint": "phone case",
        "filename": "clear_phone_case.jpg",
        "shelf": "A-02",
        "color": (200, 220, 230),
        "expect_title_substr": "Case",
        "cost": 1.80,
        "price": 9.99,
    },
    {
        "hint": "led bulb",
        "filename": "led_bulb_box.jpg",
        "shelf": "B-03",
        "color": (240, 230, 180),
        "expect_title_substr": "LED",
        "cost": 1.10,
        "price": 4.99,
    },
    {
        "hint": "notebook",
        "filename": "spiral_notebook.jpg",
        "shelf": "C-04",
        "color": (80, 120, 180),
        "expect_title_substr": "Notebook",
        "cost": 0.85,
        "price": 3.49,
    },
    {
        "hint": "water bottle",
        "filename": "steel_water_bottle.jpg",
        "shelf": "D-05",
        "color": (160, 160, 165),
        "expect_title_substr": "Bottle",
        "cost": 3.50,
        "price": 14.99,
    },
]


def test_health(client):
    c, _, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["scope"] == "shopify_draft_only"
    assert body["marketplace_handoff"] == "deferred"


def test_five_products_identify_save_shopify_draft(client):
    """The validation gate: five cheap products through draft creation."""
    c, store, upload_dir = client
    created_ids = []

    for spec in FIVE_PRODUCTS:
        files = {
            "photos": (
                spec["filename"],
                _jpeg_bytes(spec["color"], spec["hint"]),
                "image/jpeg",
            )
        }
        data = {
            "hint": spec["hint"],
            "shelf": spec["shelf"],
            "cost": str(spec["cost"]),
            "price": str(spec["price"]),
        }
        resp = c.post("/api/products", data=data, files=files, follow_redirects=False)
        assert resp.status_code == 303, resp.text
        loc = resp.headers["location"]
        assert "/identify" in loc
        product_id = loc.split("/")[2]
        created_ids.append(product_id)

        # Identify correctly
        product = store.get(product_id)
        assert product is not None
        assert spec["expect_title_substr"].lower() in product.title.lower()
        assert product.inventory_status == InventoryStatus.IDENTIFIED
        assert len(product.photos) == 1
        assert product.sku.startswith("SKU-")
        assert product.shelf == spec["shelf"]
        assert product.cost == spec["cost"]
        assert product.price == spec["price"]
        assert product.description
        assert product.marketplace_url is None  # handoff deferred

        # Photo saved on disk
        photo_name = Path(product.photos[0]).name
        assert (upload_dir / photo_name).exists()

        # Identify screen renders
        page = c.get(f"/products/{product_id}/identify")
        assert page.status_code == 200
        assert product.title in page.text

        # Review / save correctly
        review = c.post(
            f"/products/{product_id}/review",
            data={
                "title": product.title,
                "description": product.description,
                "short_description": product.short_description,
                "sku": product.sku,
                "shelf": product.shelf,
                "cost": str(product.cost),
                "price": str(product.price),
            },
            follow_redirects=False,
        )
        assert review.status_code == 303
        assert "/shopify" in review.headers["location"]
        product = store.get(product_id)
        assert product.inventory_status == InventoryStatus.REVIEWED

        # Shopify draft
        draft = c.post(f"/products/{product_id}/shopify", follow_redirects=False)
        # HTML response with result (200)
        assert draft.status_code == 200
        product = store.get(product_id)
        assert product.inventory_status == InventoryStatus.SHOPIFY_DRAFT
        assert product.shopify_product_id
        assert product.marketplace_url is None
        assert "Draft simulated" in draft.text or product.shopify_product_id in draft.text

    assert len(created_ids) == 5
    assert len(store.list_all()) == 5

    # API list exposes foundation fields
    listing = c.get("/api/products")
    assert listing.status_code == 200
    rows = listing.json()
    assert len(rows) == 5
    for row in rows:
        for key in (
            "sku",
            "photos",
            "title",
            "description",
            "cost",
            "price",
            "shelf",
            "shopify_product_id",
            "marketplace_url",
            "inventory_status",
        ):
            assert key in row
        assert row["inventory_status"] == "shopify_draft"
        assert row["shopify_product_id"]
        assert row["marketplace_url"] is None


def test_sample_catalog_covers_five_products():
    assert len(SAMPLE_CATALOG) == 5
    keys = {item["key"] for item in SAMPLE_CATALOG}
    assert keys == {"usb_cable", "phone_case", "led_bulb", "notebook", "water_bottle"}


def test_api_shopify_draft_endpoint(client):
    c, store, _ = client
    files = {
        "photos": ("notebook_test.jpg", _jpeg_bytes((10, 40, 90), "notebook"), "image/jpeg")
    }
    resp = c.post(
        "/api/products",
        data={"hint": "notebook", "shelf": "Z-9"},
        files=files,
        follow_redirects=False,
    )
    product_id = resp.headers["location"].split("/")[2]
    api = c.post(f"/api/products/{product_id}/shopify-draft")
    assert api.status_code == 200
    body = api.json()
    assert body["product"]["shopify_product_id"]
    assert body["shopify"]["status"] == "draft"
    assert body["shopify"]["mock"] is True
