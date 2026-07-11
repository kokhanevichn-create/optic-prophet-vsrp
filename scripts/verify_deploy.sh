#!/usr/bin/env bash
# Local production-path verification for the FastAPI intake app.
# This repo has no Vite frontend — do not run npm/vite here.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== architecture check =="
if [[ -f package.json || -f vite.config.ts || -f vite.config.js ]]; then
  echo "ERROR: unexpected Vite/Node files present" >&2
  exit 1
fi
echo "no package.json / vite.config — FastAPI only"

echo "== install runtime deps =="
python3 -m pip install -q -r requirements.txt

echo "== import ASGI entrypoint =="
python3 -c "from app import app; assert app.title == 'Product Intake'; print('app:', app.title)"

echo "== pytest (five-product gate) =="
python3 -m pip install -q -r requirements-dev.txt
python3 -m pytest intake/tests -q --tb=line

echo "== boot uvicorn + /health =="
python3 -m uvicorn app:app --host 127.0.0.1 --port 8765 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; }
trap cleanup EXIT
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf http://127.0.0.1:8765/health >/tmp/intake-health.json; then
    break
  fi
  sleep 0.3
done
python3 - <<'PY'
import json
from pathlib import Path
body = json.loads(Path("/tmp/intake-health.json").read_text())
assert body["ok"] is True
assert body["scope"] == "shopify_draft_only"
assert body["marketplace_handoff"] == "deferred"
print("health:", body)
PY
code=$(curl -s -o /tmp/intake-home.html -w "%{http_code}" http://127.0.0.1:8765/)
test "$code" = "200"
grep -q "Capture" /tmp/intake-home.html
grep -q "Intake" /tmp/intake-home.html
echo "GET / -> $code (Capture screen OK)"

echo "== vercel read-only path mode =="
VERCEL=1 python3 - <<'PY'
import os
os.environ["VERCEL"] = "1"
# Fresh import under Vercel mode
import importlib
import sys
for mod in list(sys.modules):
    if mod == "intake" or mod.startswith("intake.") or mod == "app":
        del sys.modules[mod]
from intake import config
assert str(config.DATA_DIR).startswith("/tmp/"), config.DATA_DIR
assert str(config.UPLOAD_DIR).startswith("/tmp/"), config.UPLOAD_DIR
from app import app
assert app.title == "Product Intake"
print("vercel mode ok:", config.DATA_DIR, config.UPLOAD_DIR)
PY

echo
echo "Production path verified."
echo "Vercel: framework=fastapi, install='pip install -r requirements.txt', build=null, output=null"
