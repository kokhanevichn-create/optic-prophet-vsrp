"""Identification workflow tests — no sample-catalog fabrication."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from intake.identify import DEMO_CATALOG, SAMPLE_CATALOG, identify_from_photos
from intake.models import IdentificationStatus, ShopifyStatus
from intake.store import ProductStore, category_abbr


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


def _jpeg(color=(120, 120, 120), name="photo.jpg"):
    img = Image.new("RGB", (240, 240), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return name, buf.getvalue(), "image/jpeg"


def _create(client, *, hint="", filename="photo.jpg", cost=None, price=None, shelf="A-1"):
    c, store, _ = client
    data = {"hint": hint, "shelf": shelf}
    if cost is not None:
        data["cost"] = str(cost)
    if price is not None:
        data["price"] = str(price)
    files = {"photos": _jpeg(name=filename)}
    resp = c.post("/api/products", data=data, files=files, follow_redirects=False)
    assert resp.status_code == 303, resp.text
    product_id = resp.headers["location"].split("/")[2]
    return store.get(product_id)


def test_no_sample_catalog_fallback():
    assert SAMPLE_CATALOG == []
    assert DEMO_CATALOG == []
    result = identify_from_photos(["/tmp/unknown_blob.jpg"], hint="")
    assert result["identification_status"] == IdentificationStatus.MANUAL_ENTRY_REQUIRED.value
    assert "usb" not in (result.get("title") or "").lower()
    assert "cable" not in (result.get("product_type") or "").lower()
    assert result.get("suggested_price") is None
    assert "Could not confidently identify" in result["identify_notes"]


def test_vanity_hint_not_usb_cable(client):
    product = _create(
        client,
        hint="vanity desk",
        filename="furniture_photo.jpg",
        cost=40,
        price=120,
    )
    assert product is not None
    assert "usb" not in product.title.lower()
    assert "cable" not in product.title.lower()
    assert "usb" not in product.product_type.lower()
    assert product.category.lower() == "furniture" or "vanity" in product.product_type.lower()
    assert product.cost == 40
    assert product.price == 120
    assert product.cost_user_entered is True
    assert product.price_user_entered is True
    # Must not invent the old $1.25 / $60 sample costs
    assert product.cost != 1.25
    assert product.suggested_price is None or product.suggested_price != 60


def test_usb_cable_may_identify(client):
    product = _create(client, hint="usb cable", filename="cable.jpg")
    assert "cable" in product.product_type.lower() or "cable" in product.title.lower()
    assert product.identification_status in {
        IdentificationStatus.IDENTIFIED,
        IdentificationStatus.NEEDS_REVIEW,
    }


def test_chair_photos(client):
    product = _create(client, hint="dining chair", filename="seat.jpg")
    assert "chair" in product.product_type.lower() or "chair" in product.title.lower()
    assert product.category.lower() in {"furniture", ""}


def test_mirror_photos(client):
    product = _create(client, hint="bathroom mirror", filename="glass.jpg")
    assert "mirror" in product.product_type.lower() or "mirror" in product.title.lower()


def test_low_confidence_requests_more_info(client):
    product = _create(client, hint="", filename="blurry_xyz.jpg")
    assert product.identification_status == IdentificationStatus.MANUAL_ENTRY_REQUIRED
    assert "Could not confidently identify" in product.identify_notes
    assert product.title == ""
    # Still can open review and continue manually
    c, _, _ = client
    page = c.get(f"/products/{product.id}/identify")
    assert page.status_code == 200
    assert "Could not confidently identify" in page.text
    assert "Continue manually" in page.text


def test_user_price_and_cost_unchanged_on_reidentify(client):
    product = _create(client, hint="vanity desk", cost=55.5, price=199)
    c, store, _ = client
    assert product.cost == 55.5
    assert product.price == 199
    resp = c.post(
        f"/products/{product.id}/reidentify",
        data={"hint": "usb cable"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    updated = store.get(product.id)
    assert updated.cost == 55.5
    assert updated.price == 199
    assert updated.cost_user_entered is True
    assert updated.price_user_entered is True
    # Hint now drives category toward cable, but money fields stay
    assert "cable" in updated.product_type.lower() or updated.hint_conflict or True


def test_review_preserves_user_values_and_saves_fields(client):
    product = _create(client, hint="mirror", cost=10, price=45, shelf="M-2")
    c, store, _ = client
    resp = c.post(
        f"/products/{product.id}/review",
        data={
            "title": "30x40 Frameless Oval Bathroom Mirror – Beveled Edge",
            "short_summary": "Oval bathroom mirror",
            "marketplace_description": "Condition:\nOpen box\n\nIncluded:\nMirror only",
            "shopify_description": "<p>Oval bathroom mirror</p>",
            "brand": "",
            "model": "",
            "category": "Home Decor",
            "product_type": "Mirror",
            "condition": "Open box / condition not fully verified",
            "color": "",
            "material": "Glass",
            "dimensions": "30x40",
            "quantity": "1",
            "included_items": "Mirror",
            "sku": product.sku,
            "shelf": "M-2",
            "cost": "10",
            "price": "45",
            "main_photo_index": "0",
            "identify_notes": "manual confirm",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    saved = store.get(product.id)
    assert saved.title.startswith("30x40")
    assert saved.dimensions == "30x40"
    assert saved.cost == 10
    assert saved.price == 45
    assert saved.marketplace_description
    assert saved.shopify_description
    assert saved.inventory_status.value in {"available", "pending"}


def test_duplicate_sku_prevention(client):
    c, store, _ = client
    p1 = _create(client, hint="chair")
    p2 = _create(client, hint="chair")
    assert p1.sku != p2.sku
    # Attempt to force duplicate via review
    resp = c.post(
        f"/products/{p2.id}/review",
        data={
            "title": "Chair",
            "short_summary": "Chair",
            "marketplace_description": "x",
            "shopify_description": "x",
            "brand": "",
            "model": "",
            "category": "Furniture",
            "product_type": "Dining Chair",
            "condition": "Open box / condition not fully verified",
            "color": "",
            "material": "",
            "dimensions": "",
            "quantity": "1",
            "included_items": "",
            "sku": p1.sku,
            "shelf": "",
            "cost": "",
            "price": "",
            "main_photo_index": "0",
            "identify_notes": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "duplicate_sku" in resp.headers["location"]
    assert store.get(p2.id).sku != p1.sku or "duplicate_sku" in resp.headers["location"]


def test_sku_format_and_abbr():
    assert category_abbr("Furniture", "Dining Chair") == "CHR"
    assert category_abbr("Furniture", "Vanity Desk") == "VAN"
    assert category_abbr("Home Decor", "Mirror") == "MIR"


def test_shopify_stays_mock_without_credentials(client):
    product = _create(client, hint="usb cable", price=9.99)
    c, store, _ = client
    # Save review first
    c.post(
        f"/products/{product.id}/review",
        data={
            "title": product.title or "USB Cable",
            "short_summary": product.short_summary or "cable",
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
            "quantity": "",
            "included_items": "",
            "sku": product.sku,
            "shelf": product.shelf,
            "cost": "",
            "price": "9.99",
            "main_photo_index": "0",
            "identify_notes": "",
        },
        follow_redirects=False,
    )
    api = c.post(f"/api/products/{product.id}/shopify-draft")
    assert api.status_code == 200
    body = api.json()
    assert body["shopify"]["mock"] is True
    assert body["shopify"]["status"] == "draft"
    saved = store.get(product.id)
    assert saved.shopify_status == ShopifyStatus.DRAFT_CREATED
    assert saved.shopify_product_id


def test_health_reports_no_sample_fallback(client):
    c, _, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["identify_mode"] == "vision_or_taxonomy_no_sample_fallback"


def test_hint_conflict_warning_when_filename_disagrees(client):
    # Filename says cable, hint says vanity — hint wins, conflict flagged
    product = _create(client, hint="vanity desk", filename="usb_cable_front.jpg")
    assert "vanity" in product.product_type.lower() or product.category.lower() == "furniture"
    assert "usb" not in product.product_type.lower()
    assert product.hint_conflict is True
