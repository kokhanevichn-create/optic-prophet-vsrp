"""Legacy five-product smoke tests updated for no-sample identify."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from intake.models import IdentificationStatus, ShopifyStatus
from intake.store import ProductStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_products.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    monkeypatch.setenv("SHOPIFY_MOCK", "true")
    monkeypatch.setenv("SHOPIFY_STORE_DOMAIN", "")
    monkeypatch.setenv("SHOPIFY_ACCESS_TOKEN", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import intake.config as config
    import intake.app as app_mod
    import intake.shopify_client as shopify_mod
    import intake.identify as identify_mod

    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(config, "SHOPIFY_MOCK", True)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(identify_mod, "OPENAI_API_KEY", "")
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_mod, "SHOPIFY_MOCK", True)

    store = ProductStore(db_path=db_path)
    monkeypatch.setattr(app_mod, "store", store)
    monkeypatch.setattr(app_mod, "shopify", shopify_mod.ShopifyClient(mock=True))

    with TestClient(app_mod.app) as c:
        yield c, store, upload_dir


def _jpeg(color, name):
    img = Image.new("RGB", (200, 200), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return name, buf.getvalue(), "image/jpeg"


CASES = [
    ("usb cable", "usb.jpg", "cable"),
    ("phone case", "case.jpg", "case"),
    ("led bulb", "bulb.jpg", "bulb"),
    ("notebook", "notebook.jpg", "notebook"),
    ("water bottle", "bottle.jpg", "bottle"),
]


def test_five_hinted_products_through_shopify_draft(client):
    c, store, upload_dir = client
    for hint, filename, expect in CASES:
        files = {"photos": _jpeg((40, 40, 40), filename)}
        resp = c.post(
            "/api/products",
            data={"hint": hint, "shelf": "T-1", "cost": "2", "price": "8"},
            files=files,
            follow_redirects=False,
        )
        assert resp.status_code == 303
        pid = resp.headers["location"].split("/")[2]
        product = store.get(pid)
        assert expect in product.product_type.lower() or expect in product.title.lower()
        assert product.cost == 2
        assert product.price == 8
        assert product.identification_status != IdentificationStatus.MANUAL_ENTRY_REQUIRED
        assert (upload_dir / product.photos[0].split("/")[-1]).exists()

        c.post(
            f"/products/{pid}/review",
            data={
                "title": product.title or hint.title(),
                "short_summary": product.short_summary or hint,
                "marketplace_description": product.marketplace_description or "x",
                "shopify_description": product.shopify_description or "x",
                "brand": "",
                "model": "",
                "category": product.category,
                "product_type": product.product_type,
                "condition": product.condition,
                "color": "",
                "material": "",
                "dimensions": "",
                "quantity": "1",
                "included_items": "",
                "sku": product.sku,
                "shelf": "T-1",
                "cost": "2",
                "price": "8",
                "main_photo_index": "0",
                "identify_notes": "",
            },
            follow_redirects=False,
        )
        draft = c.post(f"/api/products/{pid}/shopify-draft")
        assert draft.status_code == 200
        assert draft.json()["shopify"]["mock"] is True
        assert store.get(pid).shopify_status == ShopifyStatus.DRAFT_CREATED

    assert len(store.list_all()) == 5
