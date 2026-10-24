#!/usr/bin/env bash
# Watchdog for hosts without cron/systemd: call it from any supervisor (or by hand)
# as often as you like.  It only touches the service when something is actually
# down, so a healthy endpoint is left alone.
#
#   */5 * * * *  ~/aml-memory/deploy/watchdog.sh >> ~/aml-memory/logs/watchdog.log 2>&1
#
# Exit codes: 0 = healthy (or restarted successfully), 1 = still broken.
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
AML_DIR="${AML_DIR:-$(dirname "$SELF_DIR")}"
ENV_FILE="${AML_ENV_FILE:-$AML_DIR/.env}"
SERVICE="$AML_DIR/deploy/service-noroot.sh"
LOG_DIR="$AML_DIR/logs"
mkdir -p "$LOG_DIR"

env_get() { [ -f "$ENV_FILE" ] && sed -n "s/^$1=//p" "$ENV_FILE" | head -1 || true; }
PORT="${AML_PORT:-$(env_get AML_PORT)}"; PORT="${PORT:-8000}"
STAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
rc=0

healthy() { curl -fsS --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

if ! healthy; then
  echo "$STAMP api down (no answer on 127.0.0.1:$PORT) -> restarting"
  bash "$SERVICE" restart >>"$LOG_DIR/watchdog.log" 2>&1 || true
  for _ in $(seq 1 30); do healthy && break; sleep 1; done   # uvicorn needs a few seconds
  if healthy; then
    echo "$STAMP api recovered"
  else
    echo "$STAMP api STILL DOWN after 30 s - inspect $LOG_DIR/api.log"; rc=1
  fi
fi

# The tunnel is a separate process; without it the public URL answers 502.
if [ -n "$(env_get CF_TUNNEL_TOKEN)" ]; then
  if ! ps -eo cmd 2>/dev/null | grep -q "[c]loudflared tunnel"; then
    echo "$STAMP tunnel down -> restarting"
    bash "$SERVICE" start >>"$LOG_DIR/watchdog.log" 2>&1 || true
  fi
fi

# A full disk stops every write, so report it before it happens.  The database
# grows with the number of turns ingested; if this line appears, look at
# /admin/stats (db_bytes) and lower EMBED_CACHE_ROWS or move the data dir.
FREE_MB="$(df -Pm "$AML_DIR" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "${FREE_MB:-}" ] && [ "$FREE_MB" -lt 2048 ] 2>/dev/null; then
  echo "$STAMP WARNING low disk: ${FREE_MB} MB free on the data volume"
fi

# An embedding outage is silent from the outside: Adds still return 200, they just
# store turns without vectors, and every search quietly loses its dense channel.
# The counter is the only signal, so surface it here.
TOKEN="$(env_get AML_API_KEY)"
if [ -n "$TOKEN" ]; then
  STATS="$(curl -fsS --max-time 15 -H "Authorization: Bearer $TOKEN" \
           "http://127.0.0.1:$PORT/admin/stats" 2>/dev/null || true)"
  case "$STATS" in
    *'"embed_failures"'*)
      case "$STATS" in
        *'"add": 0'*) : ;;
        *) echo "$STAMP WARNING embedding failures accumulating: ${STATS:0:240}" ;;
      esac
      ;;
  esac
  case "$STATS" in
    *'"fallbacks": 0'*) : ;;
    *'"fallbacks"'*) echo "$STAMP WARNING reranker is falling back (searches are in fused order): ${STATS:0:240}" ;;
  esac
fi

exit "$rc"
