#!/usr/bin/env bash
# One-shot, idempotent deployment of the AML memory service on a fresh Linux box.
#
#   sudo AML_DOMAIN=mem.example.com EMBED_API_KEY=sk-xxxx bash deploy/bootstrap.sh
#
# Optional overrides:
#   AML_TAG=v1.2            git tag to deploy (default: v1.2)
#   AML_REPO=<url>          source repository
#   AML_DIR=/opt/aml-memory install location
#   AML_TOKEN=<hex>         fixed API token (default: generated once, then reused)
#   SKIP_DNS_CHECK=1        do not require the A record to resolve yet
#   SKIP_VERIFY=1           skip the in-container contract self-check
#
# Re-running is safe: the .env (and therefore the token) is never regenerated,
# the checkout is updated in place, and the stack is recreated only if something
# changed.  Nothing outside $AML_DIR and Docker's own state is touched.

set -euo pipefail

AML_DOMAIN="${AML_DOMAIN:-}"
EMBED_API_KEY="${EMBED_API_KEY:-}"
AML_TAG="${AML_TAG:-v1.2}"
AML_REPO="${AML_REPO:-https://github.com/2776709994/aml-memory.git}"
AML_DIR="${AML_DIR:-/opt/aml-memory}"
AML_SRC="${AML_SRC:-}"
AML_EMBED_BASE_URL="${EMBED_BASE_URL:-https://ws-i35enzeci2mcdfzq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
SKIP_DNS_CHECK="${SKIP_DNS_CHECK:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[1;32mok\033[0m  %s\n' "$*"; }
warn() { printf '    \033[1;33mwarn\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mFAILED:\033[0m %s\n' "$*" >&2; exit 1; }

on_error() {
  local code=$?
  printf '\n\033[1;31m--- bootstrap aborted (exit %s) ---\033[0m\n' "$code" >&2
  if [ -d "$AML_DIR/deploy" ]; then
    ( cd "$AML_DIR/deploy" && docker compose --env-file ../.env ps 2>&1 | tail -n 8 ) >&2 || true
    ( cd "$AML_DIR/deploy" && docker compose --env-file ../.env logs --tail 40 aml-memory 2>&1 ) >&2 || true
  fi
  printf '\nRead the panel above, fix the cause, then run the same command again.\n' >&2
  exit "$code"
}
trap on_error ERR

# ---------------------------------------------------------------------------
say "Preflight"
[ "$(id -u)" -eq 0 ] || die "run as root (sudo bash deploy/bootstrap.sh)"
[ -n "$AML_DOMAIN" ] || die "AML_DOMAIN is empty — pass the public hostname, e.g. AML_DOMAIN=mem.example.com"
[ -n "$EMBED_API_KEY" ] || die "EMBED_API_KEY is empty — pass the DashScope API key that can call text-embedding-v4"
case "$AML_DOMAIN" in
  *[!a-zA-Z0-9.-]*) die "AML_DOMAIN '$AML_DOMAIN' is not a bare hostname (no scheme, no path)" ;;
esac
ok "domain      $AML_DOMAIN"

# Both values end up inside `sed` replacements and the .env file, so refuse anything
# exotic up front instead of writing a corrupted line.
case "$EMBED_API_KEY" in
  *[!A-Za-z0-9._-]*) die "EMBED_API_KEY contains characters outside [A-Za-z0-9._-] — check for stray quotes or spaces" ;;
esac
if [ -n "${AML_TOKEN:-}" ]; then
  case "$AML_TOKEN" in
    *[!A-Za-z0-9._-]*) die "AML_TOKEN contains characters outside [A-Za-z0-9._-]" ;;
  esac
fi

for tool in curl git openssl; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool (apt-get install -y $tool)"
done
ok "tools       curl git openssl present"

mem_mb=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
[ "$mem_mb" -ge 1800 ] || warn "only ${mem_mb} MB RAM; 2 GB is the documented minimum (search may swap)"
ok "memory      ${mem_mb} MB"

avail_gb=$(df -Pk /var/lib 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')
[ "${avail_gb:-0}" -ge 8 ] || warn "only ${avail_gb} GB free on /var/lib; embeddings and DB growth want 8 GB+"
ok "disk        ${avail_gb:-?} GB free on /var/lib"

if [ "$SKIP_DNS_CHECK" != "1" ]; then
  resolved=$(getent ahostsv4 "$AML_DOMAIN" 2>/dev/null | awk 'NR==1 {print $1}')
  public_ip=$(curl -fsS --max-time 10 https://api.ipify.org 2>/dev/null || true)
  if [ -z "$resolved" ]; then
    die "DNS: $AML_DOMAIN does not resolve.  Create an A record -> $( [ -n "$public_ip" ] && echo "$public_ip" || echo '<this server>' ), wait for propagation, then re-run (or SKIP_DNS_CHECK=1 to defer TLS)"
  fi
  ok "dns         $AML_DOMAIN -> $resolved"
  if [ -n "$public_ip" ] && [ "$resolved" != "$public_ip" ]; then
    warn "$AML_DOMAIN resolves to $resolved but this box's public IP is $public_ip"
    warn "certificate issuance will fail until the record points here (or a proxy fronts this box)"
  fi
fi

for port in 80 443; do
  if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    die "port $port is already in use — stop the other web server (nginx/apache) first, Caddy needs 80 and 443"
  fi
done
ok "ports       80 and 443 free"

# ---------------------------------------------------------------------------
say "Docker"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  ok "engine      $(docker --version)"
else
  warn "installing Docker Engine (Aliyun mirror) — this takes a couple of minutes"
  curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
  systemctl enable --now docker
  docker info >/dev/null 2>&1 || die "Docker installed but the daemon is not running (systemctl status docker)"
  ok "engine      $(docker --version)"
fi
docker compose version >/dev/null 2>&1 \
  || die "the docker compose plugin is missing — apt-get install -y docker-compose-plugin"
ok "compose     $(docker compose version --short)"

# ---------------------------------------------------------------------------
say "Source"
if [ -n "$AML_SRC" ]; then
  # Offline path: the repository arrived as a tarball or a pre-made clone because
  # the target box cannot reach GitHub.  Never overwrite the live .env or data.
  [ -d "$AML_SRC" ] || die "AML_SRC=$AML_SRC is not a directory"
  mkdir -p "$AML_DIR"
  ( cd "$AML_SRC" && tar cf - --exclude=.env --exclude=data --exclude=runs . ) \
    | ( cd "$AML_DIR" && tar xf - )
  ok "copied      $AML_SRC -> $AML_DIR"
elif [ -d "$AML_DIR/.git" ]; then
  git -C "$AML_DIR" fetch --tags --force --quiet
  git -C "$AML_DIR" checkout --quiet "$AML_TAG"
  git -C "$AML_DIR" clean -fdq -e .env -e data -e runs
  ok "updated     $AML_DIR"
elif [ -e "$AML_DIR" ]; then
  die "$AML_DIR exists but is not a git checkout — move it away or pass AML_DIR=<other path>"
else
  mkdir -p "$(dirname "$AML_DIR")"
  git clone --quiet "$AML_REPO" "$AML_DIR"
  git -C "$AML_DIR" checkout --quiet "$AML_TAG"
  ok "cloned      $AML_REPO"
fi
commit=$(git -C "$AML_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
ok "tag         $AML_TAG ($commit)"

# ---------------------------------------------------------------------------
say "Configuration"
ENV_FILE="$AML_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  ok ".env exists — keeping it (token is unchanged)"
else
  cp "$AML_DIR/.env.example" "$ENV_FILE"
  TOKEN="${AML_TOKEN:-$(openssl rand -hex 24)}"
  # Only the deployment-critical values are written; every tuning knob keeps the
  # default that was measured for the leaderboard, so a re-run cannot silently
  # change quality.
  sed -i "s|^AML_DOMAIN=.*|AML_DOMAIN=$AML_DOMAIN|" "$ENV_FILE"
  sed -i "s|^AML_API_KEY=.*|AML_API_KEY=$TOKEN|" "$ENV_FILE"
  sed -i "s|^EMBED_API_KEY=.*|EMBED_API_KEY=$EMBED_API_KEY|" "$ENV_FILE"
  sed -i "s|^EMBED_BASE_URL=.*|EMBED_BASE_URL=$AML_EMBED_BASE_URL|" "$ENV_FILE"
  sed -i "s|^RERANK_API_KEY=.*|RERANK_API_KEY=$EMBED_API_KEY|" "$ENV_FILE"
  ok ".env written from .env.example"
fi
chmod 600 "$ENV_FILE"

# Guard against a half-edited file from an earlier manual attempt.
grep -q "^AML_DOMAIN=$AML_DOMAIN$" "$ENV_FILE" \
  || die "$ENV_FILE has AML_DOMAIN=$(grep '^AML_DOMAIN=' "$ENV_FILE" | cut -d= -f2), expected $AML_DOMAIN — edit it or delete the file and re-run"
grep -q '^AML_API_KEY=.\+' "$ENV_FILE" \
  || die "$ENV_FILE has an empty AML_API_KEY — the endpoint would be world-writable.  Delete the file and re-run to regenerate"

TOKEN=$(grep '^AML_API_KEY=' "$ENV_FILE" | cut -d= -f2)
ok "auth        bearer token is set (printed in the final report)"

# ---------------------------------------------------------------------------
say "Build and start"
cd "$AML_DIR/deploy"
docker compose --env-file ../.env up -d --build
ok "stack up"

say "Waiting for the service to become healthy"
status=""
for _ in $(seq 1 60); do
  status=$(docker inspect -f '{{.State.Health.Status}}' aml-memory 2>/dev/null || echo missing)
  [ "$status" = "healthy" ] && break
  sleep 5
done
[ "$status" = "healthy" ] || die "container did not reach 'healthy' within 5 minutes (last status: $status)"
ok "health      healthy"

# ---------------------------------------------------------------------------
say "Contract self-check (runs inside the container, 13 checks)"
if [ "$SKIP_VERIFY" = "1" ]; then
  warn "skipped on request"
else
  docker compose --env-file ../.env exec -T aml-memory \
    python scripts/verify_endpoint.py --base-url http://127.0.0.1:8000 --token "$TOKEN" \
    || die "self-check failed — the service answers, but not to the AML contract.  See the failing check above"
fi

say "Public HTTPS"
code=""
for _ in $(seq 1 24); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$AML_DOMAIN/health" || true)
  [ "$code" = "200" ] && break
  sleep 5   # ACME issuance on a fresh domain often needs a few seconds
done
if [ "$code" = "200" ]; then
  ok "https://$AML_DOMAIN/health -> 200"
else
  warn "https://$AML_DOMAIN/health -> ${code:-no response}"
  warn "the service itself is healthy; TLS is the only open question:"
  warn "  docker compose --env-file ../.env logs --tail 30 caddy"
  warn "usual causes: DNS not pointing here yet, port 80 blocked by the cloud firewall, ACME rate limit"
fi

public_search=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
  -X POST "https://$AML_DOMAIN/search" -H 'content-type: application/json' \
  -d '{"query":"probe","user_id":"probe"}' || true)
[ "$public_search" = "401" ] && ok "unauthenticated search rejected (401)"

# ---------------------------------------------------------------------------
say "Deployment report"
cat <<REPORT

  Service      aml-memory $AML_TAG ($commit)
  Host         $AML_DOMAIN
  Directory    $AML_DIR

  Register these on the leaderboard:
    Add URL      https://$AML_DOMAIN/add
    Search URL   https://$AML_DOMAIN/search
    Health URL   https://$AML_DOMAIN/health
    Auth         Bearer token: $TOKEN
    Track        Textual Memory
    Division     Open-source Methods
    Embedding    text-embedding-v4 (1024-d)
    LLM          N/A (no generative model is used)
    Reranker     gte-rerank-v2

  Operate it with:
    cd $AML_DIR/deploy
    docker compose --env-file ../.env ps
    docker compose --env-file ../.env logs -f --tail 100 aml-memory
    docker compose --env-file ../.env exec -T aml-memory python scripts/verify_endpoint.py \\
        --base-url http://127.0.0.1:8000 --token "$TOKEN"

  Human must now do the leaderboard registration and the Smoke run.
  Do not run a Full evaluation without an explicit instruction.

REPORT
printf '\n\033[1;32mDeployment complete.\033[0m\n'
