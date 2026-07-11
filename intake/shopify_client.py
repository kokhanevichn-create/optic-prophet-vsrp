"""Shopify Admin API client — draft product creation only."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import (
    ROOT,
    SHOPIFY_ACCESS_TOKEN,
    SHOPIFY_API_VERSION,
    SHOPIFY_MOCK,
    SHOPIFY_STORE_DOMAIN,
    UPLOAD_DIR,
)
from .models import Product


class ShopifyError(RuntimeError):
    pass


class ShopifyClient:
    def __init__(
        self,
        store_domain: str = SHOPIFY_STORE_DOMAIN,
        access_token: str = SHOPIFY_ACCESS_TOKEN,
        api_version: str = SHOPIFY_API_VERSION,
        mock: bool = SHOPIFY_MOCK,
    ):
        self.store_domain = store_domain.rstrip("/")
        self.access_token = access_token
        self.api_version = api_version
        self.mock = mock

    @property
    def base_url(self) -> str:
        return f"https://{self.store_domain}/admin/api/{self.api_version}"

    def _headers(self) -> dict[str, str]:
        return {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def create_draft_product(self, product: Product) -> dict[str, Any]:
        """Create a Shopify product in draft status with images."""
        if self.mock:
            return self._mock_create(product)

        # Prefer main photo first
        photos = list(product.photos)
        if photos and 0 <= product.main_photo_index < len(photos):
            main = photos.pop(product.main_photo_index)
            photos = [main] + photos

        body = {
            "product": {
                "title": product.title or product.sku,
                "body_html": product.shopify_description
                or product.description
                or product.short_summary
                or product.short_description
                or "",
                "vendor": product.brand or "Intake Tool",
                "product_type": product.product_type or product.category or "General",
                "status": "draft",
                "tags": (
                    f"intake,sku:{product.sku},shelf:{product.shelf or 'unset'}"
                    f",condition:{(product.condition or 'unset')[:40]}"
                ),
                "variants": [
                    {
                        "sku": product.sku,
                        "price": f"{(product.price or 0):.2f}",
                        "inventory_management": None,
                    }
                ],
            }
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.base_url}/products.json",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code >= 400:
                raise ShopifyError(f"Create product failed: {resp.status_code} {resp.text}")
            created = resp.json().get("product") or {}
            product_id = str(created.get("id"))

            for photo in photos:
                self._attach_image(client, product_id, photo, product.title)

            # Re-fetch to return final state
            get = client.get(
                f"{self.base_url}/products/{product_id}.json",
                headers=self._headers(),
            )
            if get.status_code >= 400:
                return {
                    "id": product_id,
                    "admin_url": self._admin_url(product_id),
                    "status": "draft",
                    "mock": False,
                }
            prod = get.json().get("product") or created
            return {
                "id": str(prod.get("id")),
                "title": prod.get("title"),
                "status": prod.get("status"),
                "admin_url": self._admin_url(str(prod.get("id"))),
                "images": [img.get("src") for img in (prod.get("images") or [])],
                "mock": False,
            }

    def _attach_image(
        self, client: httpx.Client, product_id: str, photo_path: str, alt: str
    ) -> None:
        path = self._resolve_photo(photo_path)
        if path is None or not path.exists():
            return
        import base64

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "image": {
                "attachment": encoded,
                "filename": path.name,
                "alt": alt or path.stem,
            }
        }
        resp = client.post(
            f"{self.base_url}/products/{product_id}/images.json",
            headers=self._headers(),
            json=payload,
        )
        if resp.status_code >= 400:
            raise ShopifyError(f"Image upload failed: {resp.status_code} {resp.text}")

    def _resolve_photo(self, photo_path: str) -> Optional[Path]:
        path = Path(photo_path)
        if path.is_absolute() and path.exists():
            return path
        rel = photo_path.lstrip("/")
        candidate = ROOT / rel
        if candidate.exists():
            return candidate
        candidate = UPLOAD_DIR / Path(photo_path).name
        return candidate if candidate.exists() else None

    def _admin_url(self, product_id: str) -> str:
        domain = self.store_domain.replace(".myshopify.com", "")
        return f"https://admin.shopify.com/store/{domain}/products/{product_id}"

    def _mock_create(self, product: Product) -> dict[str, Any]:
        fake_id = str(9000000000000 + int(uuid.uuid4().int % 1_000_000_000))
        return {
            "id": fake_id,
            "title": product.title or product.sku,
            "status": "draft",
            "admin_url": f"https://admin.shopify.com/store/mock/products/{fake_id}",
            "images": list(product.photos),
            "mock": True,
            "message": (
                "Shopify credentials not configured — draft simulated. "
                "Set SHOPIFY_STORE_DOMAIN and SHOPIFY_ACCESS_TOKEN to push for real."
            ),
        }
