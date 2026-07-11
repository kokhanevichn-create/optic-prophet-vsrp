"""Product identification from photos — no sample-catalog fabrication."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import OPENAI_API_KEY, ROOT, UPLOAD_DIR
from .models import PICKUP_BLOCK, IdentificationStatus, empty_identify_result

# Demo-only reference. NEVER used as a silent fallback for real uploads.
DEMO_CATALOG: list[dict[str, Any]] = []

# Hint / filename taxonomy — generates drafts from attributes, not fake SKUs.
CATEGORY_TAXONOMY: list[dict[str, Any]] = [
    {
        "category": "Furniture",
        "product_type": "Vanity Desk",
        "abbr": "VAN",
        "keywords": [
            "vanity",
            "vanity desk",
            "makeup vanity",
            "makeup desk",
            "dressing table",
        ],
        "features": ["mirror", "stool"],
    },
    {
        "category": "Furniture",
        "product_type": "Dining Chair",
        "abbr": "CHR",
        "keywords": ["dining chair", "chair", "side chair", "accent chair"],
        "features": ["upholstered", "wood legs"],
    },
    {
        "category": "Home Decor",
        "product_type": "Mirror",
        "abbr": "MIR",
        "keywords": ["mirror", "bathroom mirror", "wall mirror", "frameless"],
        "features": ["beveled", "oval", "rectangular"],
    },
    {
        "category": "Electronics",
        "product_type": "USB Cable",
        "abbr": "CBL",
        "keywords": ["usb cable", "usb-c", "usb c", "type-c cable", "charging cable"],
        "features": ["usb-a", "data"],
    },
    {
        "category": "Electronics",
        "product_type": "Phone Case",
        "abbr": "CSE",
        "keywords": ["phone case", "tpu case", "clear case"],
        "features": ["protective"],
    },
    {
        "category": "Lighting",
        "product_type": "LED Bulb",
        "abbr": "BLB",
        "keywords": ["led bulb", "light bulb", "a19"],
        "features": ["soft white"],
    },
    {
        "category": "Office",
        "product_type": "Notebook",
        "abbr": "NTB",
        "keywords": ["notebook", "spiral notebook", "journal"],
        "features": ["college ruled"],
    },
    {
        "category": "Drinkware",
        "product_type": "Water Bottle",
        "abbr": "BTL",
        "keywords": ["water bottle", "steel bottle", "insulated bottle"],
        "features": ["stainless"],
    },
    {
        "category": "Tools",
        "product_type": "Charger",
        "abbr": "CHG",
        "keywords": ["charger", "fast charger", "20v charger"],
        "features": ["lcd"],
    },
]


def _read_image_bytes(photo_path: str) -> bytes:
    path = Path(photo_path)
    if not path.is_absolute():
        rel = photo_path.lstrip("/")
        path = ROOT / rel
        if not path.exists():
            path = UPLOAD_DIR / Path(photo_path).name
    return path.read_bytes()


def _match_taxonomy(text: str) -> list[dict[str, Any]]:
    text_l = (text or "").lower().strip()
    text_l = text_l.replace("_", " ").replace("-", " ")
    if not text_l:
        return []
    hits: list[tuple[int, dict[str, Any]]] = []
    for item in CATEGORY_TAXONOMY:
        score = 0
        for kw in item["keywords"]:
            if kw in text_l:
                score = max(score, len(kw))
        if score:
            hits.append((score, item))
    hits.sort(key=lambda x: x[0], reverse=True)
    # Deduplicate by product_type
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for _, item in hits:
        if item["product_type"] in seen:
            continue
        seen.add(item["product_type"])
        out.append(item)
    return out


def _build_title(
    brand: str,
    product_type: str,
    key_feature: str,
    size_qty: str,
    color: str,
    condition: str,
) -> str:
    parts: list[str] = []
    if brand and brand.lower() not in {"unbranded", "unknown", "n/a"}:
        parts.append(brand.strip())
    if product_type:
        parts.append(product_type.strip())
    if key_feature:
        parts.append(key_feature.strip())
    if size_qty:
        parts.append(size_qty.strip())
    if color:
        parts.append(color.strip())
    if condition:
        # Keep condition short in title
        cond = condition.strip()
        if "open box" in cond.lower():
            parts.append("Open Box")
        elif cond.lower() not in {"new", "used"}:
            parts.append(cond.split("/")[0].strip().title())
        else:
            parts.append(cond.title())
    title = " ".join(p for p in parts if p)
    # Light cleanup
    title = re.sub(r"\s+[–-]\s+", " – ", title)
    title = re.sub(r"\s{2,}", " ", title).strip(" –-")
    return title[:140]


def _marketplace_description(
    condition: str,
    included: str,
    details: str,
    retail_note: str = "",
) -> str:
    lines = [
        f"Condition:\n{condition or 'Open box / condition not fully verified'}",
        "",
        f"Included:\n{included or 'See photos for included items.'}",
        "",
        f"Details:\n{details or 'See photos for visible details.'}",
    ]
    if retail_note:
        lines.extend(["", f"Retail comparison:\n{retail_note}"])
    lines.extend(["", PICKUP_BLOCK])
    return "\n".join(lines)


def _shopify_description(
    product_type: str,
    brand: str,
    color: str,
    material: str,
    dimensions: str,
    included: str,
    condition: str,
    features: str,
) -> str:
    bits: list[str] = []
    if brand:
        bits.append(f"<p><strong>Brand:</strong> {brand}</p>")
    if product_type:
        bits.append(f"<p><strong>Product:</strong> {product_type}</p>")
    if color:
        bits.append(f"<p><strong>Color:</strong> {color}</p>")
    if material:
        bits.append(f"<p><strong>Material:</strong> {material}</p>")
    if dimensions:
        bits.append(f"<p><strong>Dimensions:</strong> {dimensions}</p>")
    if included:
        bits.append(f"<p><strong>Included:</strong> {included}</p>")
    if features:
        bits.append(f"<p><strong>Features:</strong> {features}</p>")
    bits.append(
        f"<p><strong>Condition:</strong> {condition or 'Open box / condition not fully verified'}</p>"
    )
    bits.append("<p>Please review photos for exact appearance and included items.</p>")
    return "\n".join(bits)


def _draft_from_taxonomy(
    match: dict[str, Any],
    *,
    confidence: float,
    source: str,
    hint: str,
    notes: str,
    possible: Optional[list[str]] = None,
    hint_conflict: bool = False,
) -> dict[str, Any]:
    product_type = match["product_type"]
    category = match["category"]
    condition = "Open box / condition not fully verified"
    feature = match["features"][0] if match.get("features") else ""
    title = _build_title("", product_type, feature, "", "", condition)
    details = (
        f"Appears to be a {product_type.lower()} based on "
        f"{'operator hint' if source.startswith('hint') else 'photo cues'}."
    )
    if hint:
        details += f" Operator hint: {hint.strip()}."
    short = f"{product_type} — review photos and edit details before listing."
    uncertain = ["brand", "model", "dimensions", "color", "material", "included_items"]
    status = IdentificationStatus.IDENTIFIED
    if confidence < 0.5:
        status = IdentificationStatus.MANUAL_ENTRY_REQUIRED
    elif confidence < 0.8:
        status = IdentificationStatus.NEEDS_REVIEW

    return empty_identify_result(
        title=title if confidence >= 0.5 else "",
        short_summary=short if confidence >= 0.5 else "",
        marketplace_description=_marketplace_description(condition, "", details)
        if confidence >= 0.5
        else "",
        shopify_description=_shopify_description(
            product_type, "", "", "", "", "", condition, feature
        )
        if confidence >= 0.5
        else "",
        brand="",
        model="",
        category=category,
        product_type=product_type,
        condition=condition,
        color="",
        material="",
        dimensions="",
        quantity="",
        included_items="",
        suggested_price=None,
        suggested_price_low=None,
        suggested_price_high=None,
        price_suggestion_source="",
        confidence=confidence,
        possible_categories=possible
        if possible
        else [f"{category} / {product_type}"],
        uncertain_fields=uncertain,
        hint_conflict=hint_conflict,
        multi_product_warning=False,
        identify_notes=notes,
        source=source,
        identification_status=status.value,
    )


def _low_confidence_result(
    *,
    notes: str,
    possible: Optional[list[str]] = None,
    hint_conflict: bool = False,
    category: str = "",
    product_type: str = "",
) -> dict[str, Any]:
    return empty_identify_result(
        title="",
        short_summary="",
        marketplace_description="",
        shopify_description="",
        category=category,
        product_type=product_type,
        condition="Open box / condition not fully verified",
        confidence=0.35 if possible else 0.2,
        possible_categories=possible or [],
        uncertain_fields=[
            "title",
            "brand",
            "model",
            "category",
            "product_type",
            "dimensions",
            "price",
        ],
        hint_conflict=hint_conflict,
        identify_notes=notes
        or (
            "Could not confidently identify this item. "
            "Add a clearer photo, barcode, label photo, or product hint."
        ),
        source="low_confidence",
        identification_status=IdentificationStatus.MANUAL_ENTRY_REQUIRED.value,
    )


def identify_from_photos(
    photo_paths: list[str],
    hint: str = "",
    *,
    demo_mode: bool = False,
) -> dict[str, Any]:
    """Analyze photos and return a product draft. Never fabricates sample SKUs."""
    if not photo_paths:
        return _low_confidence_result(notes="No photo uploaded.")

    if OPENAI_API_KEY:
        try:
            return _identify_openai(photo_paths, hint=hint)
        except Exception as exc:  # noqa: BLE001
            result = _heuristic_identify(photo_paths, hint=hint)
            result["identify_notes"] = (
                f"Image analysis failed ({exc}). {result.get('identify_notes', '')}"
            ).strip()
            return result

    # Offline / no API key: attribute draft from hint + filename only — no catalog SKU.
    return _heuristic_identify(photo_paths, hint=hint, demo_mode=demo_mode)


def _heuristic_identify(
    photo_paths: list[str],
    hint: str = "",
    *,
    demo_mode: bool = False,
) -> dict[str, Any]:
    del demo_mode  # reserved; demo catalog must not contaminate real uploads
    hint_hits = _match_taxonomy(hint)
    filename_text = " ".join(Path(p).name for p in photo_paths)
    file_hits = _match_taxonomy(filename_text)

    # Hint is a strong correction signal.
    if hint_hits:
        primary = hint_hits[0]
        possible = [f"{h['category']} / {h['product_type']}" for h in hint_hits[:3]]
        conflict = False
        notes = (
            f"Drafted from operator hint “{hint.strip()}”. Review and edit details."
        )
        confidence = 0.88
        if file_hits:
            other = [h for h in file_hits if h["product_type"] != primary["product_type"]]
            if other:
                conflict = True
                confidence = 0.72
                notes = (
                    f"Hint “{hint.strip()}” used as primary category. "
                    "Image cues may conflict — please confirm the category."
                )
                for h in other[:2]:
                    label = f"{h['category']} / {h['product_type']}"
                    if label not in possible:
                        possible.append(label)
        return _draft_from_taxonomy(
            primary,
            confidence=confidence,
            source="hint_taxonomy",
            hint=hint,
            notes=notes,
            possible=possible,
            hint_conflict=conflict,
        )

    if file_hits:
        primary = file_hits[0]
        possible = [f"{h['category']} / {h['product_type']}" for h in file_hits[:3]]
        # Filename-only is weaker than an explicit hint.
        confidence = 0.62 if len(file_hits) == 1 else 0.55
        return _draft_from_taxonomy(
            primary,
            confidence=confidence,
            source="filename_taxonomy",
            hint=hint,
            notes=(
                "Drafted from filename cues only. Add a product hint or clearer photos "
                "to improve confidence."
            ),
            possible=possible,
        )

    # No strong match — do NOT invent a sample product.
    return _low_confidence_result(
        notes=(
            "Could not confidently identify this item. "
            "Add a clearer photo, barcode, label photo, or product hint."
        )
    )


def _identify_openai(photo_paths: list[str], hint: str = "") -> dict[str, Any]:
    prompt = {
        "role": "system",
        "content": (
            "You identify retail products from photos for a resale listing draft. "
            "Return JSON only. Never invent brand, model, dimensions, included parts, "
            "condition, retail price, or compatibility unless clearly visible in the "
            "images or provided in the operator hint. "
            "If unsure, lower confidence and leave fields blank. "
            "Default condition to 'Open box / condition not fully verified' when unclear. "
            "Do not say tested working. "
            "Operator hint is a strong correction signal — prefer it when provided. "
            "If hint conflicts with images, set hint_conflict=true and list alternatives."
        ),
    }
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Analyze these product photos and return JSON with keys: "
                "category, product_type, brand, model, color, material, dimensions, "
                "quantity, condition, included_items, visible_features (array), "
                "readable_text (array), packaging_state, "
                "title, short_summary, marketplace_details, shopify_features, "
                "suggested_price_low, suggested_price_high, price_suggestion_source, "
                "confidence (0-1), possible_categories (array of strings), "
                "uncertain_fields (array), hint_conflict (bool), "
                "multi_product_warning (bool), identify_notes. "
                f"Operator hint: {hint or 'none'}."
            ),
        }
    ]
    for path in photo_paths[:6]:
        raw = _read_image_bytes(path)
        b64 = base64.b64encode(raw).decode("ascii")
        mime = "image/jpeg"
        lower = path.lower()
        if lower.endswith(".png"):
            mime = "image/png"
        elif lower.endswith(".webp"):
            mime = "image/webp"
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [prompt, {"role": "user", "content": user_content}],
        "temperature": 0.1,
    }
    with httpx.Client(timeout=90.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    text = data["choices"][0]["message"]["content"]
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON in model response")
    parsed = json.loads(match.group(0))
    return _normalize_vision_result(parsed, hint=hint)


def _normalize_vision_result(parsed: dict[str, Any], hint: str = "") -> dict[str, Any]:
    confidence = float(parsed.get("confidence") or 0.0)
    category = str(parsed.get("category") or "").strip()
    product_type = str(parsed.get("product_type") or "").strip()
    brand = str(parsed.get("brand") or "").strip()
    if brand.lower() in {"unknown", "n/a", "none", "null"}:
        brand = ""
    model = str(parsed.get("model") or "").strip()
    color = str(parsed.get("color") or "").strip()
    material = str(parsed.get("material") or "").strip()
    dimensions = str(parsed.get("dimensions") or "").strip()
    quantity = str(parsed.get("quantity") or "").strip()
    condition = str(parsed.get("condition") or "").strip() or (
        "Open box / condition not fully verified"
    )
    if "tested" in condition.lower() and "working" in condition.lower():
        condition = "Open box / condition not fully verified"
    included = str(parsed.get("included_items") or "").strip()
    features = parsed.get("visible_features") or []
    if isinstance(features, str):
        feature_text = features
    else:
        feature_text = ", ".join(str(f) for f in features if f)

    # Hint conflict / preference
    hint_conflict = bool(parsed.get("hint_conflict"))
    hint_hits = _match_taxonomy(hint)
    if hint_hits and product_type:
        if hint_hits[0]["product_type"].lower() not in product_type.lower() and hint:
            hint_conflict = True
            # Prefer hint category when conflict
            category = hint_hits[0]["category"]
            product_type = hint_hits[0]["product_type"]

    possible = parsed.get("possible_categories") or []
    if isinstance(possible, str):
        possible = [possible]
    possible = [str(p) for p in possible][:3]
    if not possible and category and product_type:
        possible = [f"{category} / {product_type}"]

    uncertain = parsed.get("uncertain_fields") or []
    if isinstance(uncertain, str):
        uncertain = [uncertain]
    uncertain = [str(u) for u in uncertain]

    if confidence < 0.5:
        return _low_confidence_result(
            notes=str(parsed.get("identify_notes") or "")
            or (
                "Could not confidently identify this item. "
                "Add a clearer photo, barcode, label photo, or product hint."
            ),
            possible=possible,
            hint_conflict=hint_conflict,
            category=category,
            product_type=product_type,
        )

    title = str(parsed.get("title") or "").strip()
    if not title:
        title = _build_title(
            brand,
            product_type,
            feature_text.split(",")[0].strip() if feature_text else "",
            quantity or dimensions,
            color,
            condition,
        )
    short = str(parsed.get("short_summary") or "").strip()
    if not short:
        short = f"{product_type or 'Product'} — review photos before listing."

    details = str(parsed.get("marketplace_details") or feature_text or product_type)
    market = _marketplace_description(condition, included, details)
    shopify = _shopify_description(
        product_type,
        brand,
        color,
        material,
        dimensions,
        included,
        condition,
        str(parsed.get("shopify_features") or feature_text),
    )

    low = parsed.get("suggested_price_low")
    high = parsed.get("suggested_price_high")
    try:
        low_f = float(low) if low is not None else None
    except (TypeError, ValueError):
        low_f = None
    try:
        high_f = float(high) if high is not None else None
    except (TypeError, ValueError):
        high_f = None
    mid = None
    if low_f is not None and high_f is not None:
        mid = round((low_f + high_f) / 2, 2)
    elif low_f is not None:
        mid = low_f
    elif high_f is not None:
        mid = high_f

    status = (
        IdentificationStatus.IDENTIFIED
        if confidence >= 0.8
        else IdentificationStatus.NEEDS_REVIEW
    )

    return empty_identify_result(
        title=title,
        short_summary=short,
        marketplace_description=market,
        shopify_description=shopify,
        brand=brand,
        model=model,
        category=category,
        product_type=product_type,
        condition=condition,
        color=color,
        material=material,
        dimensions=dimensions,
        quantity=quantity,
        included_items=included,
        suggested_price=mid,
        suggested_price_low=low_f,
        suggested_price_high=high_f,
        price_suggestion_source=str(parsed.get("price_suggestion_source") or ""),
        confidence=confidence,
        possible_categories=possible,
        uncertain_fields=uncertain
        or (["brand", "model", "dimensions"] if confidence < 0.8 else []),
        hint_conflict=hint_conflict,
        multi_product_warning=bool(parsed.get("multi_product_warning")),
        identify_notes=str(parsed.get("identify_notes") or "Identified via vision analysis."),
        source="openai_vision",
        identification_status=status.value,
    )


# Back-compat name used by older imports/tests — must not fabricate samples.
SAMPLE_CATALOG = DEMO_CATALOG
