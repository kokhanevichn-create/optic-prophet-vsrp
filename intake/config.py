"""Runtime configuration for the product intake tool."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

# Vercel’s function filesystem is read-only except /tmp.
# Keep package static assets in-repo; put SQLite + uploads in /tmp there.
ON_VERCEL = os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))
if ON_VERCEL:
    DATA_DIR = Path("/tmp/intake/data")
    UPLOAD_DIR = Path("/tmp/intake/uploads")
else:
    DATA_DIR = ROOT / "data"
    UPLOAD_DIR = ROOT / "static" / "uploads"

DB_PATH = DATA_DIR / "products.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Shopify Admin API (custom app / private app token)
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", "").strip()  # e.g. mystore.myshopify.com
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "").strip()
SHOPIFY_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2024-10").strip()

# When credentials are missing, draft creation is simulated so the flow is testable.
SHOPIFY_MOCK = os.getenv("SHOPIFY_MOCK", "").lower() in {"1", "true", "yes"} or not (
    SHOPIFY_STORE_DOMAIN and SHOPIFY_ACCESS_TOKEN
)

# Optional OpenAI key for vision identify; otherwise heuristic identify is used.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
