# Product Intake — first build (Shopify draft only)

Small four-screen tool that becomes the foundation of Inventory OS later.

**Scope of this build:** Capture → Identify → Review → Shopify draft.  
**Not in this build:** Marketplace handoff (add only after five products validate).

## Screens

1. **Capture** — photos (+ optional hint, shelf, cost, price)
2. **Identify** — title / descriptions / suggested pricing
3. **Review** — edit before push
4. **Shopify Draft** — create draft product + upload images

## Foundation fields saved

Every product record stores:

| Field | Purpose |
| --- | --- |
| `sku` | Internal SKU |
| `photos` | Local image paths |
| `title` | Listing title |
| `description` / `short_description` | Copy |
| `cost` | Unit cost |
| `price` | Sell price |
| `shelf` | Bin / shelf location |
| `shopify_product_id` | Shopify draft ID |
| `marketplace_url` | Reserved (null until handoff) |
| `inventory_status` | `captured` → `identified` → `reviewed` → `shopify_draft` |

## Quick start

```bash
pip install -r requirements.txt
cp intake/.env.example intake/.env
# Optional: set SHOPIFY_STORE_DOMAIN + SHOPIFY_ACCESS_TOKEN for live drafts
python -m intake
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

Without Shopify credentials the app runs in **mock mode** and still assigns a fake Shopify product ID so you can exercise the full path.

## Deployment (Vercel)

**Architecture:** Python FastAPI + Jinja templates + SQLite. Not a Vite app.  
There is no `package.json`, `vite.config`, frontend `src/`, or `index.html` in this repo.

| Setting | Correct value |
| --- | --- |
| Root Directory | `.` (repository root) |
| Install Command | `pip install -r requirements.txt` |
| Build Command | none (`null` — do **not** use `vite build`) |
| Output Directory | none (`null` — not `dist`) |
| Entrypoint | `app.py` → FastAPI instance `app` |

`vercel.json` forces those overrides so a leftover Vite dashboard preset cannot call `vite`.

Verify locally before deploy:

```bash
bash scripts/verify_deploy.sh
```

**Dashboard (once):** Project → Settings → Build & Development → Framework **Other**, clear Build Command / Output Directory overrides that still say `vite` / `dist`. Production must deploy this branch (or merge to `main`); older `main` commits have no FastAPI entrypoint and will keep failing if the project still runs `vite build`.

## Validation gate (do this before Marketplace)

Test **five cheap products**. For each, confirm:

1. Identifies correctly  
2. Saves correctly  
3. Uploads images correctly  
4. Appears in Shopify as a **draft**

Only then add Marketplace handoff.

Sample catalog used for offline identify: USB cable, phone case, LED bulb, notebook, water bottle. Put the product type in the hint (or filename) for a strong match. With `OPENAI_API_KEY`, identify uses vision instead.

## Tests

```bash
python -m pytest intake/tests -q
```

## API (for scripting the five-product loop)

- `GET /health`
- `GET /api/products`
- `GET /api/products/{id}`
- `POST /api/products` (multipart: `photos`, `hint`, `shelf`, `cost`, `price`)
- `POST /api/products/{id}/shopify-draft`

## Layout

```
intake/
  app.py              # FastAPI + four screens
  models.py           # Product foundation schema
  store.py            # SQLite persistence
  identify.py         # Vision / heuristic identify
  shopify_client.py   # Draft create + image upload
  templates/          # Capture, Identify, Review, Shopify
  static/
  tests/test_five_products.py
```

The existing Optic Prophet Streamlit app in this repo is unchanged.
