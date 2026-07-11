"""Product Intake — four screens through Shopify draft creation."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import SHOPIFY_MOCK, UPLOAD_DIR
from .identify import identify_from_photos
from .models import (
    IdentificationStatus,
    InventoryStatus,
    MarketplaceStatus,
    Product,
    ProductUpdate,
    ShopifyStatus,
)
from .shopify_client import ShopifyClient, ShopifyError
from .store import ProductStore

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Product Intake", version="0.2.0")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

store = ProductStore()
shopify = ShopifyClient()

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

SCREENS = [
    {"id": "capture", "label": "1 · Capture", "path": "/"},
    {"id": "identify", "label": "2 · Identify", "path": None},
    {"id": "review", "label": "3 · Review", "path": None},
    {"id": "shopify", "label": "4 · Shopify Draft", "path": None},
]


def _screen_context(active: str, product: Optional[Product] = None) -> dict:
    screens = []
    for s in SCREENS:
        item = dict(s)
        if product and s["id"] == "identify":
            item["path"] = f"/products/{product.id}/identify"
        elif product and s["id"] == "review":
            item["path"] = f"/products/{product.id}/review"
        elif product and s["id"] == "shopify":
            item["path"] = f"/products/{product.id}/shopify"
        item["active"] = s["id"] == active
        screens.append(item)
    return {"screens": screens, "active_screen": active, "shopify_mock": SHOPIFY_MOCK}


def _apply_identify_result(
    product: Product,
    result: dict,
    *,
    preserve_user_cost: bool = True,
    preserve_user_price: bool = True,
) -> Product:
    product.title = result.get("title") or ""
    product.short_summary = result.get("short_summary") or ""
    product.marketplace_description = result.get("marketplace_description") or ""
    product.shopify_description = result.get("shopify_description") or ""
    product.short_description = product.short_summary
    product.description = product.shopify_description

    product.brand = result.get("brand") or ""
    product.model = result.get("model") or ""
    product.category = result.get("category") or ""
    product.product_type = result.get("product_type") or ""
    product.condition = result.get("condition") or "Open box / condition not fully verified"
    product.color = result.get("color") or ""
    product.material = result.get("material") or ""
    product.dimensions = result.get("dimensions") or ""
    product.quantity = result.get("quantity") or ""
    product.included_items = result.get("included_items") or ""

    product.identify_confidence = result.get("confidence")
    product.identify_notes = result.get("identify_notes") or ""
    product.possible_categories = result.get("possible_categories") or []
    product.uncertain_fields = result.get("uncertain_fields") or []
    product.hint_conflict = bool(result.get("hint_conflict"))
    product.multi_product_warning = bool(result.get("multi_product_warning"))
    product.identify_source = result.get("source") or ""

    # Never invent cost. Never overwrite user-entered cost/price.
    if not (preserve_user_cost and product.cost_user_entered):
        # AI must not set cost from sample data — leave user cost alone / blank
        pass
    if preserve_user_price and product.price_user_entered:
        pass
    else:
        product.suggested_price = result.get("suggested_price")
        product.suggested_price_low = result.get("suggested_price_low")
        product.suggested_price_high = result.get("suggested_price_high")
        product.price_suggestion_source = result.get("price_suggestion_source") or ""
        if product.price is None and result.get("suggested_price") is not None:
            # Suggestion only — do not mark as user-entered
            product.price = None  # keep blank; show suggestion separately

    status_val = result.get("identification_status") or IdentificationStatus.NEEDS_REVIEW.value
    try:
        product.identification_status = IdentificationStatus(status_val)
    except ValueError:
        product.identification_status = IdentificationStatus.NEEDS_REVIEW

    if product.identification_status == IdentificationStatus.MANUAL_ENTRY_REQUIRED:
        product.inventory_status = InventoryStatus.PENDING
    elif product.identification_status == IdentificationStatus.IDENTIFIED:
        product.inventory_status = InventoryStatus.IDENTIFIED
    else:
        product.inventory_status = InventoryStatus.PENDING

    # Assign SKU once we have category/type (or GEN)
    if not product.sku:
        product.sku = store.next_sku(product.category, product.product_type)
    return product


@app.get("/", response_class=HTMLResponse)
async def screen_capture(request: Request):
    products = store.list_all()
    return templates.TemplateResponse(
        request,
        "capture.html",
        {
            **_screen_context("capture"),
            "products": products,
            "error": request.query_params.get("error"),
        },
    )


@app.post("/api/products")
async def create_product(
    request: Request,
    photos: list[UploadFile] = File(default=[]),
    hint: str = Form(default=""),
    shelf: str = Form(default=""),
    cost: Optional[str] = Form(default=None),
    price: Optional[str] = Form(default=None),
):
    try:
        if not photos:
            return RedirectResponse(url="/?error=no_photo", status_code=303)

        cost_val: Optional[float] = None
        price_val: Optional[float] = None
        cost_entered = False
        price_entered = False
        if cost not in (None, ""):
            cost_val = float(cost)
            cost_entered = True
        if price not in (None, ""):
            price_val = float(price)
            price_entered = True

        product = Product(
            shelf=shelf or "",
            cost=cost_val,
            price=price_val,
            cost_user_entered=cost_entered,
            price_user_entered=price_entered,
            operator_hint=(hint or "").strip(),
            inventory_status=InventoryStatus.PENDING,
            identification_status=IdentificationStatus.NEEDS_REVIEW,
            shopify_status=ShopifyStatus.NOT_CREATED,
            marketplace_status=MarketplaceStatus.NOT_POSTED,
        )

        saved_paths: list[str] = []
        rejected = False
        for upload in photos:
            if not upload.filename:
                continue
            ext = Path(upload.filename).suffix.lower() or ".jpg"
            if ext not in ALLOWED_EXTS:
                rejected = True
                continue
            safe_hint = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in (hint or "")[:40]
            )
            orig = "".join(
                c if c.isalnum() or c in "-_" else "_"
                for c in Path(upload.filename).stem[:40]
            )
            name = (
                f"{product.id}_{safe_hint or 'photo'}_{orig or 'img'}"
                f"_{uuid.uuid4().hex[:6]}{ext}"
            )
            dest = UPLOAD_DIR / name
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(f"/static/uploads/{name}")

        if not saved_paths:
            msg = "unsupported_image" if rejected else "no_photo"
            return RedirectResponse(url=f"/?error={msg}", status_code=303)

        product.photos = saved_paths
        product.main_photo_index = 0
        # Persist photos first without locking a GEN SKU forever
        store.save(product)

        try:
            result = identify_from_photos(product.photos, hint=hint or "")
        except Exception as exc:  # noqa: BLE001
            result = {
                "title": "",
                "short_summary": "",
                "marketplace_description": "",
                "shopify_description": "",
                "confidence": 0.0,
                "identify_notes": f"Image analysis failure: {exc}",
                "identification_status": IdentificationStatus.MANUAL_ENTRY_REQUIRED.value,
                "possible_categories": [],
                "uncertain_fields": ["title", "category", "product_type"],
                "condition": "Open box / condition not fully verified",
                "source": "error",
            }

        _apply_identify_result(product, result)
        # Prefer category-based SKU once identify has a type
        if (not product.sku) or product.sku.startswith("GEN-"):
            product.sku = store.next_sku(product.category, product.product_type)
        store.save(product)
        return RedirectResponse(url=f"/products/{product.id}/identify", status_code=303)
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse(
            url=f"/?error=save_failed&detail={str(exc)[:80]}",
            status_code=303,
        )


@app.get("/products/{product_id}/identify", response_class=HTMLResponse)
async def screen_identify(request: Request, product_id: str):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return templates.TemplateResponse(
        request,
        "identify.html",
        {**_screen_context("identify", product), "product": product},
    )


@app.post("/products/{product_id}/reidentify")
async def reidentify(product_id: str, hint: str = Form(default="")):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    product.operator_hint = (hint or "").strip() or product.operator_hint
    try:
        result = identify_from_photos(product.photos, hint=product.operator_hint)
    except Exception as exc:  # noqa: BLE001
        result = {
            "title": product.title,
            "confidence": 0.0,
            "identify_notes": f"Image analysis failure: {exc}",
            "identification_status": IdentificationStatus.MANUAL_ENTRY_REQUIRED.value,
            "condition": product.condition
            or "Open box / condition not fully verified",
            "source": "error",
        }
    # Preserve user-entered cost/price on re-identify
    _apply_identify_result(product, result)
    # If SKU was GEN and we now have a better category, refresh SKU only if still generic
    if product.sku.startswith("GEN-") and product.category:
        product.sku = store.next_sku(product.category, product.product_type)
    store.save(product)
    return RedirectResponse(url=f"/products/{product.id}/identify", status_code=303)


@app.get("/products/{product_id}/review", response_class=HTMLResponse)
async def screen_review(request: Request, product_id: str):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            **_screen_context("review", product),
            "product": product,
            "error": request.query_params.get("error"),
        },
    )


@app.post("/products/{product_id}/review")
async def save_review(
    product_id: str,
    title: str = Form(default=""),
    short_summary: str = Form(default=""),
    marketplace_description: str = Form(default=""),
    shopify_description: str = Form(default=""),
    brand: str = Form(default=""),
    model: str = Form(default=""),
    category: str = Form(default=""),
    product_type: str = Form(default=""),
    condition: str = Form(default=""),
    color: str = Form(default=""),
    material: str = Form(default=""),
    dimensions: str = Form(default=""),
    quantity: str = Form(default=""),
    included_items: str = Form(default=""),
    sku: str = Form(default=""),
    shelf: str = Form(default=""),
    cost: Optional[str] = Form(default=None),
    price: Optional[str] = Form(default=None),
    main_photo_index: int = Form(default=0),
    identify_notes: str = Form(default=""),
):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    if not title.strip() and product.identification_status != IdentificationStatus.MANUAL_ENTRY_REQUIRED:
        return RedirectResponse(
            url=f"/products/{product_id}/review?error=missing_title",
            status_code=303,
        )

    product.title = title.strip()
    product.short_summary = short_summary.strip()
    product.marketplace_description = marketplace_description.strip()
    product.shopify_description = shopify_description.strip()
    product.short_description = product.short_summary
    product.description = product.shopify_description
    product.brand = brand.strip()
    product.model = model.strip()
    product.category = category.strip()
    product.product_type = product_type.strip()
    product.condition = condition.strip() or "Open box / condition not fully verified"
    product.color = color.strip()
    product.material = material.strip()
    product.dimensions = dimensions.strip()
    product.quantity = quantity.strip()
    product.included_items = included_items.strip()
    product.shelf = shelf.strip()
    product.identify_notes = identify_notes.strip() or product.identify_notes
    product.main_photo_index = max(0, min(main_photo_index, max(len(product.photos) - 1, 0)))

    if sku.strip():
        # Reject duplicate SKU for another product
        existing = store.get_by_sku(sku.strip())
        if existing and existing.id != product.id:
            return RedirectResponse(
                url=f"/products/{product_id}/review?error=duplicate_sku",
                status_code=303,
            )
        product.sku = sku.strip()

    if cost not in (None, ""):
        product.cost = float(cost)
        product.cost_user_entered = True
    if price not in (None, ""):
        product.price = float(price)
        product.price_user_entered = True

    if product.title:
        product.identification_status = IdentificationStatus.IDENTIFIED
        product.inventory_status = InventoryStatus.AVAILABLE
        product.marketplace_status = MarketplaceStatus.READY_TO_COPY
    else:
        product.identification_status = IdentificationStatus.MANUAL_ENTRY_REQUIRED
        product.inventory_status = InventoryStatus.PENDING

    try:
        store.save(product)
    except Exception:  # noqa: BLE001
        return RedirectResponse(
            url=f"/products/{product_id}/review?error=save_failed",
            status_code=303,
        )
    return RedirectResponse(url=f"/products/{product.id}/shopify", status_code=303)


@app.get("/products/{product_id}/shopify", response_class=HTMLResponse)
async def screen_shopify(request: Request, product_id: str):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return templates.TemplateResponse(
        request,
        "shopify.html",
        {
            **_screen_context("shopify", product),
            "product": product,
            "result": None,
            "error": None,
        },
    )


@app.post("/products/{product_id}/shopify")
async def push_shopify_draft(request: Request, product_id: str):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    if not product.title:
        return templates.TemplateResponse(
            request,
            "shopify.html",
            {
                **_screen_context("shopify", product),
                "product": product,
                "result": None,
                "error": "Title required before creating a Shopify draft. Go back to Review.",
            },
            status_code=400,
        )

    error = None
    result = None
    try:
        result = shopify.create_draft_product(product)
        product.shopify_product_id = str(result["id"])
        product.shopify_status = ShopifyStatus.DRAFT_CREATED
        product.inventory_status = InventoryStatus.AVAILABLE
        store.save(product)
    except ShopifyError as exc:
        error = str(exc)
        product.shopify_status = ShopifyStatus.ERROR
        store.save(product)

    return templates.TemplateResponse(
        request,
        "shopify.html",
        {
            **_screen_context("shopify", product),
            "product": product,
            "result": result,
            "error": error,
        },
    )


@app.get("/api/products")
async def api_list_products():
    return [p.model_dump() for p in store.list_all()]


@app.get("/api/products/{product_id}")
async def api_get_product(product_id: str):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product.model_dump()


@app.patch("/api/products/{product_id}")
async def api_patch_product(product_id: str, update: ProductUpdate):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    data = update.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(product, key, value)
        if key == "cost":
            product.cost_user_entered = True
        if key == "price":
            product.price_user_entered = True
    product.sync_legacy_aliases()
    store.save(product)
    return product.model_dump()


@app.post("/api/products/{product_id}/shopify-draft")
async def api_shopify_draft(product_id: str):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    try:
        result = shopify.create_draft_product(product)
    except ShopifyError as exc:
        product.shopify_status = ShopifyStatus.ERROR
        store.save(product)
        return JSONResponse(status_code=502, content={"error": str(exc)})
    product.shopify_product_id = str(result["id"])
    product.shopify_status = ShopifyStatus.DRAFT_CREATED
    store.save(product)
    return {"product": product.model_dump(), "shopify": result}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "shopify_mock": SHOPIFY_MOCK,
        "products": len(store.list_all()),
        "scope": "shopify_draft_only",
        "marketplace_handoff": "deferred",
        "identify_mode": "vision_or_taxonomy_no_sample_fallback",
    }
