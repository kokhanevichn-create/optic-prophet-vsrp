"""ASGI entrypoint for Vercel / local uvicorn.

This repository is a FastAPI (Python) app — not a Vite frontend.
Vercel looks for a FastAPI instance named `app` in app.py at the repo root.
"""

from intake.app import app

__all__ = ["app"]
