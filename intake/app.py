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
from .identify import SAMPLE_CATALOG, identify_from_photos
from .models import InventoryStatus, Product, ProductUpdate
from .shopify_client import ShopifyClient, ShopifyError
from .store import ProductStore

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Product Intake", version="0.1.0")
# Uploads may live under /tmp on Vercel; mount that path before package static/.
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


@app.get("/", response_class=HTMLResponse)
async def screen_capture(request: Request):
    products = store.list_all()
    return templates.TemplateResponse(
        request,
        "capture.html",
        {
            **_screen_context("capture"),
            "products": products,
            "sample_catalog": SAMPLE_CATALOG,
        },
    )


@app.post("/api/products")
async def create_product(
    photos: list[UploadFile] = File(default=[]),
    hint: str = Form(default=""),
    shelf: str = Form(default=""),
    cost: Optional[float] = Form(default=None),
    price: Optional[float] = Form(default=None),
):
    if not photos:
        raise HTTPException(status_code=400, detail="At least one photo is required.")

    product = Product(shelf=shelf or "", cost=cost, price=price)
    saved_paths: list[str] = []
    for upload in photos:
        if not upload.filename:
            continue
        ext = Path(upload.filename).suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            ext = ".jpg"
        # Embed hint keywords in filename so offline identify can match.
        safe_hint = "".join(c if c.isalnum() or c in "-_" else "_" for c in (hint or "")[:40])
        name = f"{product.id}_{safe_hint or 'photo'}_{uuid.uuid4().hex[:6]}{ext}"
        dest = UPLOAD_DIR / name
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(f"/static/uploads/{name}")

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid photos uploaded.")

    product.photos = saved_paths
    product.inventory_status = InventoryStatus.CAPTURED
    store.save(product)

    # Auto-identify immediately so screen 2 is ready.
    result = identify_from_photos(product.photos, hint=hint)
    product.title = result["title"]
    product.short_description = result.get("short_description") or ""
    product.description = result.get("description") or ""
    if product.cost is None and result.get("suggested_cost") is not None:
        product.cost = float(result["suggested_cost"])
    if product.price is None and result.get("suggested_price") is not None:
        product.price = float(result["suggested_price"])
    product.identify_confidence = result.get("confidence")
    product.identify_notes = result.get("identify_notes") or ""
    product.inventory_status = InventoryStatus.IDENTIFIED
    store.save(product)

    return RedirectResponse(url=f"/products/{product.id}/identify", status_code=303)


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
    result = identify_from_photos(product.photos, hint=hint)
    product.title = result["title"]
    product.short_description = result.get("short_description") or ""
    product.description = result.get("description") or ""
    if result.get("suggested_cost") is not None:
        product.cost = float(result["suggested_cost"])
    if result.get("suggested_price") is not None:
        product.price = float(result["suggested_price"])
    product.identify_confidence = result.get("confidence")
    product.identify_notes = result.get("identify_notes") or ""
    product.inventory_status = InventoryStatus.IDENTIFIED
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
        {**_screen_context("review", product), "product": product},
    )


@app.post("/products/{product_id}/review")
async def save_review(
    product_id: str,
    title: str = Form(...),
    description: str = Form(default=""),
    short_description: str = Form(default=""),
    sku: str = Form(default=""),
    shelf: str = Form(default=""),
    cost: Optional[float] = Form(default=None),
    price: Optional[float] = Form(default=None),
):
    product = store.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    product.title = title.strip()
    product.description = description.strip()
    product.short_description = short_description.strip()
    if sku.strip():
        product.sku = sku.strip()
    product.shelf = shelf.strip()
    product.cost = cost
    product.price = price
    product.inventory_status = InventoryStatus.REVIEWED
    store.save(product)
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
        raise HTTPException(status_code=400, detail="Title required before Shopify draft")

    error = None
    result = None
    try:
        result = shopify.create_draft_product(product)
        product.shopify_product_id = str(result["id"])
        product.inventory_status = InventoryStatus.SHOPIFY_DRAFT
        # marketplace_url intentionally left null — handoff comes after five-product validation
        store.save(product)
    except ShopifyError as exc:
        error = str(exc)

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
        return JSONResponse(status_code=502, content={"error": str(exc)})
    product.shopify_product_id = str(result["id"])
    product.inventory_status = InventoryStatus.SHOPIFY_DRAFT
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
    }
