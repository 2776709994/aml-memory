#!/usr/bin/env bash
# Recover after an embedding-provider outage (account arrears, expired key, quota).
#
# An outage never breaks writes any more (v1.7+): turns are stored without vectors and
# stay reachable through BM25. What is missing is the dense channel for the turns that
# arrived during the outage. This script probes the provider, backfills those vectors,
# and re-runs the contract self-check. Idempotent; safe to run while the service is up.
#
#   ./deploy/post-outage-recovery.sh              # probe, backfill, verify
#   ./deploy/post-outage-recovery.sh --dry-run    # probe and count, change nothing
set -euo pipefail

AML_DIR="${AML_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$AML_DIR"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
ok()   { printf '%s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s!%s %s\n' "$Y" "$N" "$*" >&2; }
die()  { printf '%s✗%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

env_get() { [ -f "$AML_DIR/.env" ] && sed -n "s/^$1=//p" "$AML_DIR/.env" | head -1 || true; }

PY="${AML_PYTHON:-}"
[ -z "$PY" ] && PY="$(env_get AML_PYTHON)"
[ -z "$PY" ] && [ -x "$AML_DIR/.venv/bin/python" ] && PY="$AML_DIR/.venv/bin/python"
[ -z "$PY" ] && PY="$(command -v python3 || true)"
[ -z "$PY" ] && die "no python interpreter found (set AML_PYTHON)"
[ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || die "$PY is not executable"

TOKEN="$(env_get AML_API_KEY)"
PORT="$(env_get AML_PORT)"; PORT="${PORT:-8000}"
BASE="http://127.0.0.1:$PORT"

printf '%s== 1/4 embedding provider ==%s\n' "$B" "$N"
"$PY" - "$AML_DIR" <<'PY'
import json, sys, urllib.error, urllib.request
from pathlib import Path

root = Path(sys.argv[1])
env = {}
for line in (root / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
provider = env.get("EMBED_PROVIDER", "openai")
model = env.get("EMBED_MODEL", "text-embedding-v4")
dim = env.get("EMBED_DIM", "1024")
base = env.get("EMBED_BASE_URL", "").rstrip("/")
key = env.get("EMBED_API_KEY", "")
if provider in ("hash", "local"):
    print(f"provider={provider} model={model} (no network probe needed)")
    sys.exit(0)
body = json.dumps({"model": model, "input": ["ping"], "dimensions": int(dim)}).encode()
req = urllib.request.Request(
    base + "/embeddings", data=body,
    headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read())
    print(f"provider={provider} model={model} HTTP 200 dim={len(payload['data'][0]['embedding'])}")
except urllib.error.HTTPError as exc:
    detail = exc.read()[:300].decode("utf-8", "replace").replace("\n", " ")
    print(f"provider={provider} model={model} HTTP {exc.code} {detail}")
    sys.exit(3)
except Exception as exc:  # noqa: BLE001
    print(f"provider={provider} model={model} unreachable: {exc}")
    sys.exit(4)
PY
probe=$?
if [ "$probe" -ne 0 ]; then
  warn "the embedding provider is still refused (see the line above)."
  warn "Nothing to repair yet — top up the account / renew the key, then run this again."
  warn "The service keeps accepting Adds meanwhile; those turns are BM25-only until then."
  exit "$probe"
fi
ok "provider answers again"

printf '%s== 2/4 turns stored without vectors ==%s\n' "$B" "$N"
"$PY" scripts/backfill_vectors.py --dry-run | tail -3

printf '%s== 3/4 backfill ==%s\n' "$B" "$N"
if [ "$DRY" = "1" ]; then
  warn "--dry-run: skipping the backfill"
else
  "$PY" scripts/backfill_vectors.py --max-batches 10000 | tail -5
  ok "backfill finished"
fi

printf '%s== 4/4 contract self-check ==%s\n' "$B" "$N"
if [ -n "$TOKEN" ] && "$PY" -c "import httpx" >/dev/null 2>&1; then
  "$PY" scripts/verify_endpoint.py --base-url "https://$(env_get AML_DOMAIN)" --token "$TOKEN" \
    | tail -20 || warn "the public self-check reported a failure (see above)"
else
  warn "skipped: needs httpx and AML_API_KEY in .env"
fi

curl -fsS --max-time 10 "$BASE/health" >/dev/null && ok "local /health is up" || warn "local /health did not answer"
if [ -n "$TOKEN" ]; then
  echo "stats: $(curl -fsS --max-time 15 -H "Authorization: Bearer $TOKEN" "$BASE/admin/stats")"
fi
printf '\n%spost-outage recovery complete%s\n' "$G" "$N"
