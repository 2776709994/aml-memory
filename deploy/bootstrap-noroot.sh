#!/usr/bin/env bash
# Rootless, Docker-less deployment of aml-memory behind a Cloudflare Tunnel.
#
#   curl -fsSL <url> -o /tmp/aml-bootstrap-noroot.sh
#   AML_DOMAIN=memory.example.com \
#   CF_TUNNEL_TOKEN=eyJ... \
#   EMBED_API_KEY=sk-... \
#   bash /tmp/aml-bootstrap-noroot.sh
#
# What it needs: no sudo, no Docker, no privileged ports. The API listens on
# 127.0.0.1:$AML_PORT and cloudflared (outbound-only) publishes it over HTTPS,
# which also works from behind CGNAT.
#
# Optional env: AML_TAG (default v1.6), AML_REPO, AML_DIR ($HOME/aml-memory),
# AML_PORT (8000), EMBED_BASE_URL, RERANK_API_KEY, RERANK_MODEL, CF_QUICK=1
# (ephemeral trycloudflare hostname instead of a named tunnel), PIP_INDEX_URL,
# AML_SRC (deploy from a local copy), SKIP_VERIFY=1.
set -euo pipefail

# The tag this script belongs to. Bumped with every release, and enforced by
# scripts/audit_consistency.py: a stale default silently *downgrades* anyone who
# re-runs the script without arguments, which is how v1.6 was nearly redeployed
# over v1.7 on 2026-10-24.
DEFAULT_TAG="v1.12"
AML_TAG="${AML_TAG:-$DEFAULT_TAG}"
AML_REPO="${AML_REPO:-https://github.com/2776709994/aml-memory}"
AML_DIR="${AML_DIR:-$HOME/aml-memory}"
AML_PORT="${AML_PORT:-8000}"
AML_DOMAIN="${AML_DOMAIN:-}"
CF_TUNNEL_TOKEN="${CF_TUNNEL_TOKEN:-}"
CF_QUICK="${CF_QUICK:-0}"
# Values the caller passed on the command line WIN over the ones already on disk,
# so rotating a key is `EMBED_API_KEY=sk-new bash bootstrap-noroot.sh`; values
# merely picked up from the existing .env never overwrite it.
EXPLICIT_AML_DOMAIN="${AML_DOMAIN}"
EXPLICIT_CF_TOKEN="${CF_TUNNEL_TOKEN}"
EXPLICIT_EMBED_KEY="${EMBED_API_KEY:-}"
EXPLICIT_RERANK_KEY="${RERANK_API_KEY:-}"
# Re-running to pick up a new tag should not mean re-typing secrets: fall back to
# the values already in $AML_DIR/.env (a fresh install simply has none).
env_existing() { [ -f "$AML_DIR/.env" ] && sed -n "s/^$1=//p" "$AML_DIR/.env" | head -1 || true; }
[ -n "$AML_DOMAIN" ] || AML_DOMAIN="$(env_existing AML_DOMAIN)"
[ -n "$CF_TUNNEL_TOKEN" ] || CF_TUNNEL_TOKEN="$(env_existing CF_TUNNEL_TOKEN)"
EMBED_PROVIDER="${EMBED_PROVIDER:-$(env_existing EMBED_PROVIDER)}"
EMBED_PROVIDER="${EMBED_PROVIDER:-openai}"
EMBED_API_KEY="${EMBED_API_KEY:-$(env_existing EMBED_API_KEY)}"
EMBED_BASE_URL="${EMBED_BASE_URL:-$(env_existing EMBED_BASE_URL)}"
EMBED_BASE_URL="${EMBED_BASE_URL:-https://ws-i35enzeci2mcdfzq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
if [ -z "${EMBED_MODEL:-}" ]; then
  case "$EMBED_PROVIDER" in
    openai) EMBED_MODEL="text-embedding-v4" ;;
    hash)   EMBED_MODEL="dev-hash" ;;
    *)      EMBED_MODEL="" ;;
  esac
fi
EMBED_DIM="${EMBED_DIM:-1024}"
# The hosted reranker costs +0.17 R@1 in the measurements and the same DashScope
# key serves both endpoints, so reusing the embedding key is the right default;
# RERANK_PROVIDER=none is the explicit opt-out when no key is available.
RERANK_API_KEY="${RERANK_API_KEY:-$EMBED_API_KEY}"
RERANK_MODEL="${RERANK_MODEL:-gte-rerank-v2}"
RERANK_BASE_URL="${RERANK_BASE_URL:-https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank}"
# Which of the values above came from the caller rather than from the file on disk.
OVERRIDE_KEYS=""
if [ -n "$EXPLICIT_AML_DOMAIN" ]; then OVERRIDE_KEYS="$OVERRIDE_KEYS,AML_DOMAIN"; fi
if [ -n "$EXPLICIT_CF_TOKEN" ]; then OVERRIDE_KEYS="$OVERRIDE_KEYS,CF_TUNNEL_TOKEN"; fi
if [ -n "$EXPLICIT_EMBED_KEY" ]; then
  OVERRIDE_KEYS="$OVERRIDE_KEYS,EMBED_API_KEY"
  # a rotated embedding key must not leave the reranker on the old one
  if [ -z "$EXPLICIT_RERANK_KEY" ] && [ -n "$RERANK_API_KEY" ]; then
    OVERRIDE_KEYS="$OVERRIDE_KEYS,RERANK_API_KEY"
  fi
fi
if [ -n "$EXPLICIT_RERANK_KEY" ]; then OVERRIDE_KEYS="$OVERRIDE_KEYS,RERANK_API_KEY"; fi
OVERRIDE_KEYS="${OVERRIDE_KEYS#,}"
AML_SRC="${AML_SRC:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
# A provider outage must not block a *code* upgrade: on an existing install the probe
# downgrades to a warning, and SKIP_KEY_CHECK=1 forces that even on a fresh one.
SKIP_KEY_CHECK="${SKIP_KEY_CHECK:-0}"
LOG_DIR="$AML_DIR/logs"

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else B=''; G=''; Y=''; R=''; N=''; fi
say()  { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '  %sOK%s   %s\n' "$G" "$N" "$*"; }
warn() { printf '  %sWARN%s %s\n' "$Y" "$N" "$*"; }
info() { printf '       %s\n' "$*"; }
die()  { printf '\n%sERROR%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
on_error() {
  printf '\n%sFAILED%s\n' "$R" "$N" >&2
  [ -f "$LOG_DIR/api.log" ] && { printf '\n--- last 30 lines of %s ---\n' "$LOG_DIR/api.log" >&2; tail -n 30 "$LOG_DIR/api.log" >&2 || true; }
  [ -f "$LOG_DIR/tunnel.log" ] && { printf '\n--- last 20 lines of %s ---\n' "$LOG_DIR/tunnel.log" >&2; tail -n 20 "$LOG_DIR/tunnel.log" >&2 || true; }
  printf '\nRetry after fixing the cause: rerun this same command (it is idempotent).\n' >&2
}
trap on_error ERR

# ---------------------------------------------------------------------------
say "Preflight"
case "$AML_DIR" in /*) ;; *) die "AML_DIR must be an absolute path (got: $AML_DIR)";; esac

PY3="$(command -v python3 || true)"
[ -n "$PY3" ] || die "python3 not found. Install it (no root: ask the admin, or use a distro that ships it)."
"$PY3" - <<'PY' || die "python3 >= 3.9 is required"
import sys
sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
ok "python3 $("$PY3" -V 2>&1 | awk '{print $2}')"

for c in curl tar; do command -v "$c" >/dev/null || die "$c not found"; done
ok "curl, tar present"

case "$EMBED_PROVIDER" in
  openai|dashscope)
    [ -n "$EMBED_API_KEY" ] || die "EMBED_API_KEY is required (a key that can call $EMBED_MODEL)"
    case "$EMBED_API_KEY" in *[!A-Za-z0-9._-]*) die "EMBED_API_KEY contains unexpected characters";; esac
    ok "embedding: $EMBED_PROVIDER/$EMBED_MODEL via $EMBED_BASE_URL"
    # Prove the key works before spending ten minutes on installs: an expired key
    # otherwise surfaces much later as a bare HTTP 500 on the first /add.
    emb_probe=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
      -X POST "${EMBED_BASE_URL%/}/embeddings" \
      -H "Authorization: Bearer $EMBED_API_KEY" -H 'content-type: application/json' \
      -d "{\"model\":\"$EMBED_MODEL\",\"input\":[\"ping\"],\"dimensions\":$EMBED_DIM}" 2>/dev/null || echo 000)
    case "$emb_probe" in
      200) ok "embedding endpoint answered 200" ;;
      401|403)
        if [ "$SKIP_KEY_CHECK" = "1" ] || [ -f "$AML_DIR/.env" ]; then
          warn "the embedding key was rejected (HTTP $emb_probe): continuing, because this is an upgrade."
          warn "Adds will store text without vectors (BM25-only) until model access is restored — set the key, then run scripts/backfill_vectors.py"
        else
          die "the embedding key was rejected (HTTP $emb_probe). Issue a fresh long-lived key in the Model Studio console; a key that expires mid-run would zero the whole evaluation."
        fi
        ;;
      000) warn "could not reach $EMBED_BASE_URL from this host — continuing, but a full run needs it" ;;
      *)   warn "embedding probe returned HTTP $emb_probe — continuing, but check the key/model/dimensions" ;;
    esac
    ;;
  hash|local)
    warn "EMBED_PROVIDER=$EMBED_PROVIDER: deterministic placeholder vectors — this is a wiring test, NOT a competitive configuration"
    ;;
  *)
    die "unknown EMBED_PROVIDER=$EMBED_PROVIDER (use openai, dashscope, hash or local)"
    ;;
esac

if [ -n "$RERANK_API_KEY" ]; then
  rr_probe=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
    -X POST "$RERANK_BASE_URL" \
    -H "Authorization: Bearer $RERANK_API_KEY" -H 'content-type: application/json' \
    -d "{\"model\":\"$RERANK_MODEL\",\"input\":{\"query\":\"ping\",\"documents\":[\"ping\",\"pong\"]}}" 2>/dev/null || echo 000)
  case "$rr_probe" in
    200) ok "reranker $RERANK_MODEL answered 200 (worth ~+0.17 R@1)" ;;
    *)   warn "reranker probe returned HTTP $rr_probe — retrieval keeps working but falls back to fusion order; set RERANK_PROVIDER=none to silence this" ;;
  esac
else
  warn "no rerank key: the service will fuse dense+BM25 and skip reranking (measured cost ~0.17 R@1)"
fi

if [ "$CF_QUICK" = "1" ]; then
  warn "CF_QUICK=1: the public hostname changes whenever the tunnel restarts — fine for a smoke test, unsafe for a multi-day Full run"
else
  [ -n "$CF_TUNNEL_TOKEN" ] || die "CF_TUNNEL_TOKEN is required (Cloudflare Zero Trust -> Networks -> Tunnels -> your tunnel -> Install connector -> token)"
  [ -n "$AML_DOMAIN" ] || die "AML_DOMAIN is required with a named tunnel (the public hostname configured in the tunnel, e.g. memory.example.com)"
  case "$AML_DOMAIN" in *[!A-Za-z0-9.-]*) die "AML_DOMAIN must be a bare hostname such as memory.example.com";; esac
  ok "tunnel token supplied for $AML_DOMAIN"
fi

mem_mb=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
[ "$mem_mb" -ge 1700 ] || warn "only ${mem_mb} MB RAM — 2 GB is the practical minimum"
disk_avail=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
[ "$disk_avail" -ge 2000000 ] || warn "only $((disk_avail/1024)) MB free under $HOME — 3 GB is safer"
ok "resources: ${mem_mb} MB RAM, $((disk_avail/1024/1024)) GB free under \$HOME"

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$AML_PORT$"; then
  warn "port $AML_PORT already in use on this host — set AML_PORT to something else if it is not aml-memory"
fi

# ---------------------------------------------------------------------------
say "Fetch source ($AML_TAG)"
mkdir -p "$AML_DIR"

fetch_tarball() {  # $1 = destination dir
  local tmp inner u
  tmp="$(mktemp -d)"
  for u in "https://ghfast.top/https://github.com/2776709994/aml-memory/archive/refs/tags/$AML_TAG.tar.gz" \
           "https://github.com/2776709994/aml-memory/archive/refs/tags/$AML_TAG.tar.gz"; do
    if curl -fsSL --max-time 120 --connect-timeout 15 "$u" -o "$tmp/src.tar.gz"; then break; fi
  done
  if [ ! -s "$tmp/src.tar.gz" ]; then rm -rf "$tmp"; return 1; fi
  tar -xzf "$tmp/src.tar.gz" -C "$tmp" || { rm -rf "$tmp"; return 1; }
  rm -f "$tmp/src.tar.gz"
  inner="$(find "$tmp" -maxdepth 1 -mindepth 1 -type d | head -n1)"
  tar -C "$inner" -cf - . | tar -C "$1" -xf - || { rm -rf "$tmp"; return 1; }
  rm -rf "$tmp"
  return 0
}

if [ -n "$AML_SRC" ]; then
  [ -d "$AML_SRC" ] || die "AML_SRC=$AML_SRC is not a directory"
  tar -C "$AML_SRC" --exclude=.env --exclude=data --exclude=runs --exclude=logs --exclude=run \
      --exclude=.venv --exclude=bin -cf - . | tar -C "$AML_DIR" -xf -
  ok "copied from $AML_SRC"
elif [ -d "$AML_DIR/.git" ]; then
  # A git fetch against GitHub can hang indefinitely from a Chinese VPS (no output,
  # no error), which used to stall the whole installer. Bound it and fall back.
  if timeout 150 git -C "$AML_DIR" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
       fetch --depth 1 origin "refs/tags/$AML_TAG:refs/tags/$AML_TAG" >/dev/null 2>&1 \
     && git -C "$AML_DIR" checkout -q "$AML_TAG" 2>/dev/null; then
    ok "updated existing checkout to $AML_TAG"
  else
    warn "git fetch did not finish; downloading the release tarball instead"
    fetch_tarball "$AML_DIR" || die "could not update $AML_DIR (git and tarball both failed)"
    ok "overwrote $AML_DIR with the $AML_TAG tarball"
  fi
elif command -v git >/dev/null 2>&1; then
  if timeout 300 git clone -q --depth 1 --branch "$AML_TAG" "$AML_REPO" "$AML_DIR"; then
    ok "cloned $AML_REPO at $AML_TAG"
  else
    warn "git clone failed; downloading the release tarball instead"
    fetch_tarball "$AML_DIR" || die "could not fetch the source (git and tarball both failed)"
    ok "downloaded tarball for $AML_TAG"
  fi
else
  fetch_tarball "$AML_DIR" || die "could not download the source tarball"
  ok "downloaded tarball for $AML_TAG"
fi
[ -f "$AML_DIR/aml_memory/main.py" ] || die "$AML_DIR does not look like the aml-memory repo"
[ -x "$AML_DIR/deploy/service-noroot.sh" ] || chmod +x "$AML_DIR/deploy/service-noroot.sh"

# The release tag and the version the service reports have to agree, or every later
# diagnosis is wrong: /health said 1.7.0 while the code was v1.8, and the deploy agent
# spent a round proving the four download sources agreed. Cheap to check, so check it.
SRC_VERSION="$(sed -n 's/^VERSION = "\([^"]*\)".*/\1/p' "$AML_DIR/aml_memory/main.py" | head -1)"
TAG_VERSION="${AML_TAG#v}"
if [ -n "$SRC_VERSION" ] && [ "${SRC_VERSION#"$TAG_VERSION"}" = "$SRC_VERSION" ]; then
  warn "release mismatch: $AML_TAG ships VERSION=$SRC_VERSION — /health will not report $TAG_VERSION."
  warn "the code is whatever $AML_TAG contains; trust 'git describe --tags', not the version string"
fi

# ---------------------------------------------------------------------------
say "Python environment"
# Preference order: the interpreter the pins were validated on first. A newer
# python whose wheels do not exist yet would try to compile numpy from source.
PIP_SPEC="${PIP_SPEC:--r $AML_DIR/requirements.txt}"

install_pip_via_getpip() {  # $1 = interpreter inside the new venv
  local py="$1" tmp u
  tmp="$(mktemp -d)"
  for u in "https://ghfast.top/https://raw.githubusercontent.com/pypa/get-pip/main/public/get-pip.py" \
           "https://bootstrap.pypa.io/get-pip.py"; do
    if curl -fsSL --max-time 90 "$u" -o "$tmp/get-pip.py" 2>/dev/null && [ -s "$tmp/get-pip.py" ]; then break; fi
  done
  if [ ! -s "$tmp/get-pip.py" ]; then rm -rf "$tmp"; return 1; fi
  "$py" "$tmp/get-pip.py" -q --disable-pip-version-check >/dev/null 2>&1 || true
  rm -rf "$tmp"
  "$py" -m pip --version >/dev/null 2>&1
}

build_env() {
  local py
  for py in python3.12 python3.11 python3.10 python3.13 python3; do
    command -v "$py" >/dev/null 2>&1 || continue
    rm -rf "$AML_DIR/.venv.tmp"
    if "$py" -m venv "$AML_DIR/.venv.tmp" 2>/dev/null \
       && "$AML_DIR/.venv.tmp/bin/python" -m pip --version >/dev/null 2>&1; then
      mv "$AML_DIR/.venv.tmp" "$AML_DIR/.venv"; echo "venv ($py)"; return 0
    fi
    rm -rf "$AML_DIR/.venv.tmp"
    if "$py" -m venv --without-pip "$AML_DIR/.venv.tmp" 2>/dev/null \
       && install_pip_via_getpip "$AML_DIR/.venv.tmp/bin/python"; then
      mv "$AML_DIR/.venv.tmp" "$AML_DIR/.venv"; echo "venv without ensurepip ($py)"; return 0
    fi
    rm -rf "$AML_DIR/.venv.tmp"
  done
  return 1
}

RUNTIME_PY=""
if [ -x "$AML_DIR/.venv/bin/python" ] && "$AML_DIR/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
  RUNTIME_PY="$AML_DIR/.venv/bin/python"
  ok "reusing $AML_DIR/.venv ($("$RUNTIME_PY" -V 2>&1))"
elif env_desc="$(build_env)"; then
  RUNTIME_PY="$AML_DIR/.venv/bin/python"
  ok "created $AML_DIR/.venv — $env_desc, $("$RUNTIME_PY" -V 2>&1)"
elif "$PY3" -m pip --version >/dev/null 2>&1 || command -v pip3 >/dev/null 2>&1; then
  warn "no venv module and no ensurepip on this host — falling back to a user-level install"
  warn "the service will run on the system python; ask your admin for python3-venv to avoid this"
  RUNTIME_PY="$PY3"
  USER_MODE=1
else
  die "cannot create a Python environment: no venv module, no ensurepip, no pip.
       Ask the admin to install the venv package for $(basename "$PY3") (e.g. 'apt install python3-venv'),
       or point AML_SRC at a tree with a ready .venv."
fi

MIRRORS="https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.aliyun.com/pypi/simple https://mirrors.ustc.edu.cn/pypi/simple https://pypi.org/simple"
if [ -z "$PIP_INDEX_URL" ]; then
  for cand in $MIRRORS; do
    if curl -fsS --max-time 6 -o /dev/null "$cand/" 2>/dev/null; then
      PIP_INDEX_URL="$cand"; break
    fi
  done
  PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
fi
info "pip index: $PIP_INDEX_URL"

pip_install() {
  if [ "${USER_MODE:-0}" = "1" ]; then
    # shellcheck disable=SC2086
    "$RUNTIME_PY" -m pip install --user --break-system-packages --disable-pip-version-check \
      --timeout 25 --retries 1 $PIP_SPEC -i "$1"
  else
    # shellcheck disable=SC2086
    "$RUNTIME_PY" -m pip install --disable-pip-version-check \
      --timeout 25 --retries 1 $PIP_SPEC -i "$1"
  fi
}

# A mirror that answers for the index root can still be slow or half-broken, and
# pypi.org from mainland China can hang for many minutes, so never stop at one try.
installed=0
for idx in "$PIP_INDEX_URL" $MIRRORS; do
  info "installing dependencies from $idx"
  if pip_install "$idx"; then installed=1; PIP_INDEX_URL="$idx"; break; fi
  warn "that index did not work, trying the next one"
done
[ "$installed" = "1" ] || die "pip install failed on every mirror — check outbound network or pass PIP_INDEX_URL"
info "effective index: $PIP_INDEX_URL"
"$RUNTIME_PY" -c "import fastapi, uvicorn, numpy, httpx, pydantic, dotenv, orjson" \
  || die "dependencies are installed but not importable"
ok "dependencies installed"


# ---------------------------------------------------------------------------
say "Configuration (.env)"
ENV_FILE="$AML_DIR/.env"
[ -f "$AML_DIR/.env.example" ] || die ".env.example missing from the source tree"
# Merge, do not overwrite: an existing .env keeps its token and secrets, but every
# key that is missing or empty is filled in.  (A first run with a quick tunnel used
# to leave AML_DOMAIN and CF_TUNNEL_TOKEN empty, and a re-run skipped the file
# entirely — the service then came up with no hostname and no tunnel token.)
AMLDOMAIN="$AML_DOMAIN" EMBEDKEY="$EMBED_API_KEY" EMBEDURL="$EMBED_BASE_URL" \
EMBPROV="$EMBED_PROVIDER" EMBMODEL="$EMBED_MODEL" EMBDIM="$EMBED_DIM" \
RRKEY="$RERANK_API_KEY" RRMODEL="$RERANK_MODEL" RRURL="$RERANK_BASE_URL" \
CFTOKEN="$CF_TUNNEL_TOKEN" ENVFILE="$ENV_FILE" SRCENV="$AML_DIR/.env.example" \
OVERRIDE="$OVERRIDE_KEYS" \
"$RUNTIME_PY" - <<'PY'
import os, pathlib, secrets

envfile = pathlib.Path(os.environ["ENVFILE"])
fresh = not envfile.exists()
base_text = envfile.read_text(encoding="utf-8") if not fresh else \
    pathlib.Path(os.environ["SRCENV"]).read_text(encoding="utf-8")
lines = base_text.splitlines()

def current(key):
    if fresh:  # .env.example values are placeholders, not decisions
        return None
    for line in lines:
        if line.startswith(key + "=") and not line.lstrip().startswith("#"):
            return line.split("=", 1)[1].strip()
    return None

wanted = {
    "AML_DOMAIN": os.environ.get("AMLDOMAIN", ""),
    "EMBED_PROVIDER": os.environ["EMBPROV"],
    "EMBED_MODEL": os.environ["EMBMODEL"],
    "EMBED_BASE_URL": os.environ["EMBEDURL"],
    "EMBED_API_KEY": os.environ.get("EMBEDKEY", ""),
    "EMBED_DIM": os.environ["EMBDIM"],
    "AML_INDEX_CACHE_VECTORS": "60000",
}
rr = os.environ.get("RRKEY", "")
if rr:
    wanted.update({
        "RERANK_PROVIDER": "dashscope",
        "RERANK_MODEL": os.environ["RRMODEL"],
        "RERANK_BASE_URL": os.environ["RRURL"],
        "RERANK_API_KEY": rr,
        "RERANK_DEPTH": "60",
    })
cf = os.environ.get("CFTOKEN", "")
if cf:
    wanted["CF_TUNNEL_TOKEN"] = cf

# Fill what is absent or blank, and overwrite what the caller passed explicitly
# (that is how a key gets rotated); never rewrite a value the operator set earlier.
override = {k.strip() for k in os.environ.get("OVERRIDE", "").split(",") if k.strip()}
fill = {k: v for k, v in wanted.items() if v and (k in override or not current(k))}
if not current("AML_API_KEY"):
    fill["AML_API_KEY"] = "aml-" + secrets.token_hex(24)

out, seen = [], set()
for line in lines:
    key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
    if key in fill:
        out.append(f"{key}={fill[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, val in fill.items():
    if key not in seen:
        out.append(f"{key}={val}")
envfile.write_text("\n".join(out) + "\n", encoding="utf-8")
if envfile.exists() and fill:
    print("       filled in: " + ", ".join(sorted(fill)))
elif not fill:
    print("       nothing to change")
PY
chmod 600 "$ENV_FILE"
ok "configuration ready at $ENV_FILE"

if [ "${USER_MODE:-0}" = "1" ]; then
  # The service must use the same interpreter the dependencies went into.
  grep -q '^AML_PYTHON=' "$ENV_FILE" || printf 'AML_PYTHON=%s\n' "$RUNTIME_PY" >>"$ENV_FILE"
  ok "recorded AML_PYTHON=$RUNTIME_PY in .env (user-level install)"
fi

grep -q '^AML_API_KEY=.\+' "$ENV_FILE" || die "$ENV_FILE has an empty AML_API_KEY"
env_provider=$(grep -m1 '^EMBED_PROVIDER=' "$ENV_FILE" | cut -d= -f2-)
case "$env_provider" in
  openai|dashscope)
    grep -q '^EMBED_API_KEY=.\+' "$ENV_FILE" || die "$ENV_FILE has an empty EMBED_API_KEY"
    grep -q '^EMBED_MODEL=.\+' "$ENV_FILE" || die "$ENV_FILE has an empty EMBED_MODEL"
    ;;
esac
TOKEN=$(grep -m1 '^AML_API_KEY=' "$ENV_FILE" | cut -d= -f2-)
ok "auth token is set (printed in the final report)"

# ---------------------------------------------------------------------------
say "cloudflared"
CF_BIN="${CF_BIN:-$AML_DIR/bin/cloudflared}"
mkdir -p "$AML_DIR/bin"
if [ -x "$CF_BIN" ] && "$CF_BIN" --version >/dev/null 2>&1; then
  ok "reusing $("$CF_BIN" --version 2>&1 | head -n1)"
else
  case "$(uname -m)" in
    x86_64|amd64) cfa=cloudflared-linux-amd64 ;;
    aarch64|arm64) cfa=cloudflared-linux-arm64 ;;
    *) die "unsupported architecture $(uname -m) — download cloudflared manually and set CF_BIN" ;;
  esac
  got=0
  for u in "https://ghfast.top/https://github.com/cloudflare/cloudflared/releases/latest/download/$cfa" \
           "https://github.com/cloudflare/cloudflared/releases/latest/download/$cfa"; do
    if curl -fsSL --max-time 180 "$u" -o "$CF_BIN.tmp"; then got=1; break; fi
  done
  [ "$got" = "1" ] || die "could not download cloudflared — grab $cfa manually from https://github.com/cloudflare/cloudflared/releases and set CF_BIN=<path>"
  mv "$CF_BIN.tmp" "$CF_BIN"; chmod +x "$CF_BIN"
  "$CF_BIN" --version >/dev/null 2>&1 || die "the downloaded cloudflared does not run"
  ok "installed $("$CF_BIN" --version 2>&1 | head -n1)"
fi

# ---------------------------------------------------------------------------
say "Start the service"
export AML_DIR AML_PORT CF_BIN CF_QUICK
"$AML_DIR/deploy/service-noroot.sh" stop >/dev/null 2>&1 || true
START_TUNNEL=1 "$AML_DIR/deploy/service-noroot.sh" start

say "Waiting for the local API to answer (first start embeds nothing, should be seconds)"
local_ok=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$AML_PORT/health" >/dev/null 2>&1; then local_ok=1; break; fi
  sleep 2
done
[ "$local_ok" = "1" ] || die "no healthy answer on http://127.0.0.1:$AML_PORT/health within 2 minutes"
ok "local health 200"

if [ "$CF_QUICK" = "1" ]; then
  info "waiting for the trycloudflare hostname…"
  for _ in $(seq 1 60); do
    [ -s "$AML_DIR/run/public-url" ] && break
    sleep 2
  done
  PUBLIC_URL="$(cat "$AML_DIR/run/public-url" 2>/dev/null || true)"
  [ -n "$PUBLIC_URL" ] || die "cloudflared did not publish a hostname — see $LOG_DIR/tunnel.log"
else
  PUBLIC_URL="https://$AML_DOMAIN"
fi

say "Waiting for $PUBLIC_URL/health (DNS + tunnel + certificate on Cloudflare's side)"
public_ok=0
for _ in $(seq 1 40); do
  if curl -fsS --max-time 8 "$PUBLIC_URL/health" >/dev/null 2>&1; then public_ok=1; break; fi
  sleep 5
done
[ "$public_ok" = "1" ] || die "public health check failed for $PUBLIC_URL — check 'logs/tunnel.log' and that the tunnel's public hostname points at http://127.0.0.1:$AML_PORT"
ok "public health 200 over HTTPS"

# ---------------------------------------------------------------------------
if [ "$SKIP_VERIFY" != "1" ]; then
  say "Contract self-check (15 checks against $PUBLIC_URL)"
  "$RUNTIME_PY" "$AML_DIR/scripts/verify_endpoint.py" --base-url "$PUBLIC_URL" --token "$TOKEN" \
    || die "the endpoint failed the contract self-check — fix before registering it"
else
  warn "SKIP_VERIFY=1: contract self-check skipped"
fi

# ---------------------------------------------------------------------------
say "Deployment complete."
cat <<EOF

  Add URL      $PUBLIC_URL/add
  Search URL   $PUBLIC_URL/search
  Health URL   $PUBLIC_URL/health
  Auth header  Authorization: Bearer $TOKEN
  (same token also accepted as X-Api-Token or X-Api-Key)

  Division     Open-source Methods  (track: Textual Memory)
  Embedding    $(grep -m1 '^EMBED_MODEL=' "$ENV_FILE" | cut -d= -f2-) ($(grep -m1 '^EMBED_DIM=' "$ENV_FILE" | cut -d= -f2-)-d)
  Generative   N/A  (no LLM is called inside Add/Search)
  Reranker     $(grep -m1 '^RERANK_PROVIDER=' "$ENV_FILE" | cut -d= -f2- | grep -qv '^none$' && grep -m1 '^RERANK_MODEL=' "$ENV_FILE" | cut -d= -f2- || echo none)

  Data dir     $AML_DIR (SQLite at $AML_DIR/data/aml.sqlite3)
  Processes    $(AML_DIR=$AML_DIR "$AML_DIR/deploy/service-noroot.sh" status | tr '\n' ' ')
  Logs         $LOG_DIR/api.log, $LOG_DIR/tunnel.log
  Restart      $AML_DIR/deploy/service-noroot.sh restart
  Verify again $AML_DIR/.venv/bin/python $AML_DIR/scripts/verify_endpoint.py --base-url $PUBLIC_URL --token \$TOKEN

  Keep this host awake and online while the platform evaluates: a brief outage is
  covered by the platform's retries, a long one wastes a Full run (only 2 per key).
EOF

if [ "$EMBED_PROVIDER" = "hash" ] || [ "$EMBED_PROVIDER" = "local" ]; then
  printf '\n%sWARNING%s this deployment uses %s embeddings: the wiring works, but do NOT\n' "$Y" "$N" "$EMBED_PROVIDER" >&2
  printf '  register it. Rerun with EMBED_PROVIDER=openai EMBED_API_KEY=sk-... for a real submission.\n' >&2
fi
