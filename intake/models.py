"""Product data model — foundation fields for Inventory OS later."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class InventoryStatus(str, Enum):
    CAPTURED = "captured"
    IDENTIFIED = "identified"
    REVIEWED = "reviewed"
    SHOPIFY_DRAFT = "shopify_draft"
    MARKETPLACE_PENDING = "marketplace_pending"  # reserved — not built yet
    LIVE = "live"


def new_sku() -> str:
    stamp = datetime.now(timezone.utc).strftime("%y%m%d")
    return f"SKU-{stamp}-{uuid4().hex[:6].upper()}"


class Product(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    sku: str = Field(default_factory=new_sku)
    photos: list[str] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    short_description: str = ""
    cost: Optional[float] = None
    price: Optional[float] = None
    shelf: str = ""
    shopify_product_id: Optional[str] = None
    marketplace_url: Optional[str] = None  # reserved for Marketplace handoff
    inventory_status: InventoryStatus = InventoryStatus.CAPTURED
    identify_confidence: Optional[float] = None
    identify_notes: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


class ProductUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    short_description: Optional[str] = None
    cost: Optional[float] = None
    price: Optional[float] = None
    shelf: Optional[str] = None
    sku: Optional[str] = None
