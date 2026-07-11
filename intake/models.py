"""Product data model — foundation fields for Inventory OS."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class IdentificationStatus(str, Enum):
    NEEDS_REVIEW = "needs_review"
    IDENTIFIED = "identified"
    MANUAL_ENTRY_REQUIRED = "manual_entry_required"


class ShopifyStatus(str, Enum):
    NOT_CREATED = "not_created"
    DRAFT_CREATED = "draft_created"
    PUBLISHED = "published"
    ERROR = "error"


class MarketplaceStatus(str, Enum):
    NOT_POSTED = "not_posted"
    READY_TO_COPY = "ready_to_copy"
    POSTED = "posted"
    SOLD = "sold"


class InventoryStatus(str, Enum):
    AVAILABLE = "available"
    PENDING = "pending"
    SOLD = "sold"
    ARCHIVED = "archived"
    BIN = "bin"
    AUCTION = "auction"
    # Legacy aliases kept for older rows / transitions
    CAPTURED = "captured"
    IDENTIFIED = "identified"
    REVIEWED = "reviewed"
    SHOPIFY_DRAFT = "shopify_draft"


PICKUP_BLOCK = (
    "Pickup:\n"
    "📍 Pickup: 141 Randall Ave, Girard, PA 16417\n"
    "🚗 Erie meetup also available.\n"
    "Follow my group RustBeltResale20 for first dibs on new inventory + better deals 👍"
)


class Product(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    sku: str = ""
    photos: list[str] = Field(default_factory=list)
    main_photo_index: int = 0

    title: str = ""
    short_summary: str = ""
    marketplace_description: str = ""
    shopify_description: str = ""

    # Backward-compatible aliases used by older templates/tests
    short_description: str = ""
    description: str = ""

    brand: str = ""
    model: str = ""
    category: str = ""
    product_type: str = ""
    condition: str = ""
    color: str = ""
    material: str = ""
    dimensions: str = ""
    quantity: str = ""
    included_items: str = ""

    cost: Optional[float] = None
    price: Optional[float] = None
    suggested_price: Optional[float] = None
    suggested_price_low: Optional[float] = None
    suggested_price_high: Optional[float] = None
    price_suggestion_source: str = ""
    cost_user_entered: bool = False
    price_user_entered: bool = False

    shelf: str = ""
    operator_hint: str = ""

    identify_confidence: Optional[float] = None
    identify_notes: str = ""
    possible_categories: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    hint_conflict: bool = False
    multi_product_warning: bool = False
    identify_source: str = ""

    identification_status: IdentificationStatus = IdentificationStatus.NEEDS_REVIEW
    shopify_status: ShopifyStatus = ShopifyStatus.NOT_CREATED
    marketplace_status: MarketplaceStatus = MarketplaceStatus.NOT_POSTED
    inventory_status: InventoryStatus = InventoryStatus.PENDING

    shopify_product_id: Optional[str] = None
    marketplace_url: Optional[str] = None

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def sync_legacy_aliases(self) -> None:
        """Keep short_description/description in sync for older callers."""
        if self.short_summary and not self.short_description:
            self.short_description = self.short_summary
        elif self.short_description and not self.short_summary:
            self.short_summary = self.short_description
        if self.shopify_description and not self.description:
            self.description = self.shopify_description
        elif self.description and not self.shopify_description:
            self.shopify_description = self.description


class ProductUpdate(BaseModel):
    title: Optional[str] = None
    short_summary: Optional[str] = None
    marketplace_description: Optional[str] = None
    shopify_description: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    category: Optional[str] = None
    product_type: Optional[str] = None
    condition: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None
    quantity: Optional[str] = None
    included_items: Optional[str] = None
    cost: Optional[float] = None
    price: Optional[float] = None
    shelf: Optional[str] = None
    sku: Optional[str] = None
    main_photo_index: Optional[int] = None


def empty_identify_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "",
        "short_summary": "",
        "marketplace_description": "",
        "shopify_description": "",
        "brand": "",
        "model": "",
        "category": "",
        "product_type": "",
        "condition": "Open box / condition not fully verified",
        "color": "",
        "material": "",
        "dimensions": "",
        "quantity": "",
        "included_items": "",
        "suggested_price": None,
        "suggested_price_low": None,
        "suggested_price_high": None,
        "price_suggestion_source": "",
        "confidence": 0.0,
        "possible_categories": [],
        "uncertain_fields": [],
        "hint_conflict": False,
        "multi_product_warning": False,
        "identify_notes": "",
        "source": "none",
        "identification_status": IdentificationStatus.MANUAL_ENTRY_REQUIRED.value,
    }
    base.update(overrides)
    return base
