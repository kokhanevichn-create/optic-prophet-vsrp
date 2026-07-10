"""Product identification from photos.

Uses OpenAI vision when OPENAI_API_KEY is set; otherwise a deterministic
heuristic so the five-product test loop works offline.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx

from .config import OPENAI_API_KEY, ROOT, UPLOAD_DIR


# Five cheap reference products for offline / mock identify testing.
SAMPLE_CATALOG: list[dict[str, Any]] = [
    {
        "key": "usb_cable",
        "title": "USB-C to USB-A Cable 3ft Black",
        "short_description": "Durable USB-C charging and data cable.",
        "description": (
            "3-foot black USB-C to USB-A cable for charging and data transfer. "
            "Compatible with phones, tablets, and accessories that use USB-C."
        ),
        "suggested_cost": 1.25,
        "suggested_price": 6.99,
        "keywords": ["usb", "cable", "type-c", "usb-c", "charger"],
    },
    {
        "key": "phone_case",
        "title": "Clear Soft TPU Phone Case",
        "short_description": "Slim clear protective case.",
        "description": (
            "Transparent soft TPU phone case with raised edges for screen and camera "
            "protection. Lightweight everyday cover."
        ),
        "suggested_cost": 1.80,
        "suggested_price": 9.99,
        "keywords": ["case", "phone", "tpu", "cover", "clear"],
    },
    {
        "key": "led_bulb",
        "title": "LED Light Bulb A19 Soft White",
        "short_description": "Energy-efficient soft white LED bulb.",
        "description": (
            "A19 LED bulb, soft white (2700K), equivalent to 60W. "
            "Fits standard E26 sockets. Long life, low energy use."
        ),
        "suggested_cost": 1.10,
        "suggested_price": 4.99,
        "keywords": ["bulb", "led", "light", "lamp", "a19"],
    },
    {
        "key": "notebook",
        "title": "Spiral Notebook College Ruled 70 Sheets",
        "short_description": "Everyday college-ruled spiral notebook.",
        "description": (
            "70-sheet college-ruled spiral notebook with durable cover. "
            "Ideal for notes, lists, and school or office use."
        ),
        "suggested_cost": 0.85,
        "suggested_price": 3.49,
        "keywords": ["notebook", "spiral", "paper", "ruled", "journal"],
    },
    {
        "key": "water_bottle",
        "title": "Stainless Steel Water Bottle 17oz",
        "short_description": "Insulated stainless steel bottle.",
        "description": (
            "17oz double-wall stainless steel water bottle. Keeps drinks cold or hot. "
            "Leak-resistant lid for gym, desk, or travel."
        ),
        "suggested_cost": 3.50,
        "suggested_price": 14.99,
        "keywords": ["bottle", "water", "steel", "flask", "insulated"],
    },
]


def _read_image_bytes(photo_path: str) -> bytes:
    path = Path(photo_path)
    if not path.is_absolute():
        # Stored paths are like /static/uploads/...
        rel = photo_path.lstrip("/")
        path = ROOT / rel
        if not path.exists():
            path = UPLOAD_DIR / Path(photo_path).name
    return path.read_bytes()


def _heuristic_from_filename(filename: str) -> dict[str, Any] | None:
    name = filename.lower()
    for item in SAMPLE_CATALOG:
        for kw in item["keywords"]:
            if kw in name:
                return {**item, "confidence": 0.82, "source": "filename_heuristic"}
    return None


def identify_from_photos(photo_paths: list[str], hint: str = "") -> dict[str, Any]:
    """Return title/descriptions/suggested pricing for the product photos."""
    if OPENAI_API_KEY and photo_paths:
        try:
            return _identify_openai(photo_paths, hint=hint)
        except Exception as exc:  # noqa: BLE001 — fall back for resilience
            fallback = _heuristic_identify(photo_paths, hint=hint)
            fallback["identify_notes"] = f"OpenAI failed ({exc}); used heuristic."
            return fallback
    return _heuristic_identify(photo_paths, hint=hint)


def _heuristic_identify(photo_paths: list[str], hint: str = "") -> dict[str, Any]:
    hint_l = (hint or "").lower()
    for item in SAMPLE_CATALOG:
        if any(kw in hint_l for kw in item["keywords"]):
            return {
                "title": item["title"],
                "short_description": item["short_description"],
                "description": item["description"],
                "suggested_cost": item["suggested_cost"],
                "suggested_price": item["suggested_price"],
                "confidence": 0.9,
                "source": "hint_heuristic",
                "catalog_key": item["key"],
                "identify_notes": "Matched from operator hint against sample catalog.",
            }

    for path in photo_paths:
        hit = _heuristic_from_filename(Path(path).name)
        if hit:
            return {
                "title": hit["title"],
                "short_description": hit["short_description"],
                "description": hit["description"],
                "suggested_cost": hit["suggested_cost"],
                "suggested_price": hit["suggested_price"],
                "confidence": hit["confidence"],
                "source": hit["source"],
                "catalog_key": hit["key"],
                "identify_notes": "Matched filename keywords to sample catalog.",
            }

    # Default: first sample so the pipeline always produces something editable.
    item = SAMPLE_CATALOG[0]
    return {
        "title": item["title"],
        "short_description": item["short_description"],
        "description": item["description"],
        "suggested_cost": item["suggested_cost"],
        "suggested_price": item["suggested_price"],
        "confidence": 0.45,
        "source": "default_sample",
        "catalog_key": item["key"],
        "identify_notes": (
            "No strong match. Defaulted to sample catalog item — edit on Review screen."
        ),
    }


def _identify_openai(photo_paths: list[str], hint: str = "") -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Identify this retail product for a Shopify draft listing. "
                "Return ONLY compact JSON with keys: title, short_description, "
                "description, suggested_cost, suggested_price, confidence (0-1). "
                f"Operator hint: {hint or 'none'}."
            ),
        }
    ]
    for path in photo_paths[:4]:
        raw = _read_image_bytes(path)
        b64 = base64.b64encode(raw).decode("ascii")
        mime = "image/jpeg"
        if path.lower().endswith(".png"):
            mime = "image/png"
        elif path.lower().endswith(".webp"):
            mime = "image/webp"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a product listing assistant. Reply with JSON only.",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
    }
    with httpx.Client(timeout=60.0) as client:
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
    return {
        "title": str(parsed.get("title") or "Untitled product"),
        "short_description": str(parsed.get("short_description") or ""),
        "description": str(parsed.get("description") or ""),
        "suggested_cost": float(parsed["suggested_cost"])
        if parsed.get("suggested_cost") is not None
        else None,
        "suggested_price": float(parsed["suggested_price"])
        if parsed.get("suggested_price") is not None
        else None,
        "confidence": float(parsed.get("confidence") or 0.7),
        "source": "openai_vision",
        "identify_notes": "Identified via OpenAI vision.",
    }
