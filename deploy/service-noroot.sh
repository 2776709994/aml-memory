#!/usr/bin/env bash
# Process control for a rootless, Docker-less deployment (uvicorn + cloudflared).
#
#   service-noroot.sh start|stop|restart|status|logs|tunnel|tunnel-stop|url
#
# No sudo, no systemd, no privileged ports: the API listens on 127.0.0.1:$AML_PORT
# and Cloudflare Tunnel publishes it over HTTPS. Every process is supervised by a
# restart loop, so a crash costs seconds instead of the whole evaluation run.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AML_DIR="${AML_DIR:-$(cd "$SELF_DIR/.." && pwd)}"
SELF="$SELF_DIR/$(basename "${BASH_SOURCE[0]}")"
RUN_DIR="$AML_DIR/run"
LOG_DIR="$AML_DIR/logs"
PY="$AML_DIR/.venv/bin/python"
PORT="${AML_PORT:-8000}"
CF_BIN="${CF_BIN:-$AML_DIR/bin/cloudflared}"
API_PID="$RUN_DIR/api.pid"
TUN_PID="$RUN_DIR/tunnel.pid"
URL_FILE="$RUN_DIR/public-url"

mkdir -p "$RUN_DIR" "$LOG_DIR"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

env_get() {  # env_get KEY  -> value from .env (empty if absent)
  [ -f "$AML_DIR/.env" ] || return 0
  grep -m1 "^$1=" "$AML_DIR/.env" 2>/dev/null | cut -d= -f2-
}

# Prefer the interpreter recorded by the bootstrap (a user-level install has no
# .venv), then the venv, then the ambient python3.
PY="${AML_PYTHON:-}"
[ -n "$PY" ] || PY="$(env_get AML_PYTHON)"
[ -n "$PY" ] || PY="$AML_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || echo python3)"

alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; }

# --- supervisor loops (re-executed by start) --------------------------------
__api_loop() {
  cd "$AML_DIR" || exit 1
  while true; do
    "$PY" -m uvicorn aml_memory.main:app --host 127.0.0.1 --port "$PORT" \
      --workers 1 --timeout-keep-alive 75 --no-server-header >>"$LOG_DIR/api.log" 2>&1
    printf '[supervisor] api exited code=%s at %s; restarting in 3s\n' \
      "$?" "$(date -Is)" >>"$LOG_DIR/api.log"
    sleep 3
  done
}

__tunnel_quick_loop() {
  local url
  : >"$LOG_DIR/tunnel.log"
  while true; do
    "$CF_BIN" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" \
      >>"$LOG_DIR/tunnel.log" 2>&1 &
    local tpid=$!
    # publish the assigned hostname as soon as cloudflared prints it
    for _ in $(seq 1 60); do
      url=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" | head -n1)
      [ -n "$url" ] && printf '%s\n' "$url" >"$URL_FILE"
      sleep 1
    done
    wait "$tpid" 2>/dev/null
    printf '[supervisor] quick tunnel dropped at %s; new hostname in a few seconds\n' \
      "$(date -Is)" >>"$LOG_DIR/tunnel.log"
    sleep 5
  done
}

__tunnel_named_loop() {
  local token; token="$(env_get CF_TUNNEL_TOKEN)"
  [ -n "$token" ] || { printf 'ERROR: CF_TUNNEL_TOKEN missing from .env\n' >&2; exit 1; }
  while true; do
    "$CF_BIN" tunnel --no-autoupdate run --token "$token" >>"$LOG_DIR/tunnel.log" 2>&1
    printf '[supervisor] named tunnel exited at %s; restarting in 5s\n' \
      "$(date -Is)" >>"$LOG_DIR/tunnel.log"
    sleep 5
  done
}

# --- commands ---------------------------------------------------------------
start_api() {
  if alive "$API_PID"; then say "api      already running (pid $(cat "$API_PID"))"; return 0; fi
  [ -x "$PY" ] || die "no virtualenv at $AML_DIR/.venv — run deploy/bootstrap-noroot.sh first"
  : >"$API_PID"
  nohup setsid "$SELF" __api-loop >>"$LOG_DIR/supervisor.log" 2>&1 &
  printf '%s\n' "$!" >"$API_PID"
  say "api      started (pid $(cat "$API_PID")), listening on 127.0.0.1:$PORT"
}

stop_api() {
  alive "$API_PID" || { say "api      not running"; rm -f "$API_PID"; return 0; }
  local pid; pid="$(cat "$API_PID")"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 20); do alive "$API_PID" || break; sleep 0.5; done
  alive "$API_PID" && kill -KILL -- "-$pid" 2>/dev/null
  rm -f "$API_PID"
  say "api      stopped"
}

start_tunnel() {
  if [ "${CF_QUICK:-0}" = "1" ]; then
    if alive "$TUN_PID"; then say "tunnel   already running (pid $(cat "$TUN_PID"))"; return 0; fi
    [ -x "$CF_BIN" ] || die "cloudflared not found at $CF_BIN"
    rm -f "$URL_FILE"
    nohup setsid "$SELF" __tunnel-quick-loop >>"$LOG_DIR/supervisor.log" 2>&1 &
    printf '%s\n' "$!" >"$TUN_PID"
    say "tunnel   quick tunnel starting (public hostname lands in $URL_FILE)"
  else
    if alive "$TUN_PID"; then say "tunnel   already running (pid $(cat "$TUN_PID"))"; return 0; fi
    [ -x "$CF_BIN" ] || die "cloudflared not found at $CF_BIN"
    nohup setsid "$SELF" __tunnel-named-loop >>"$LOG_DIR/supervisor.log" 2>&1 &
    printf '%s\n' "$!" >"$TUN_PID"
    say "tunnel   named tunnel running (hostname configured in the Cloudflare dashboard)"
  fi
}

stop_tunnel() {
  alive "$TUN_PID" || { say "tunnel   not running"; rm -f "$TUN_PID"; return 0; }
  local pid; pid="$(cat "$TUN_PID")"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  sleep 1
  pkill -f "cloudflared.*$PORT" 2>/dev/null || true
  rm -f "$TUN_PID"
  say "tunnel   stopped"
}

case "${1:-status}" in
  __api-loop) __api_loop ;;
  __tunnel-quick-loop) __tunnel_quick_loop ;;
  __tunnel-named-loop) __tunnel_named_loop ;;
  start)
    start_api
    [ "${START_TUNNEL:-1}" = "1" ] && start_tunnel
    ;;
  stop) stop_tunnel; stop_api ;;
  restart) "$SELF" stop; sleep 1; START_TUNNEL="${START_TUNNEL:-1}" "$SELF" start ;;
  tunnel) start_tunnel ;;
  tunnel-stop) stop_tunnel ;;
  status)
    if alive "$API_PID"; then
      if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        say "api      healthy   pid $(cat "$API_PID")  http://127.0.0.1:$PORT"
      else
        say "api      starting  pid $(cat "$API_PID")  (log: $LOG_DIR/api.log)"
      fi
    else
      say "api      stopped"
    fi
    if alive "$TUN_PID"; then say "tunnel   running   pid $(cat "$TUN_PID")"; else say "tunnel   stopped"; fi
    [ -f "$URL_FILE" ] && say "public   $(cat "$URL_FILE")"
    [ -n "$(env_get AML_DOMAIN)" ] && say "hostname $(env_get AML_DOMAIN)"
    ;;
  url)
    if [ -f "$URL_FILE" ]; then cat "$URL_FILE"
    elif [ -n "$(env_get AML_DOMAIN)" ]; then printf 'https://%s\n' "$(env_get AML_DOMAIN)"; fi
    ;;
  logs)
    tail -n "${2:-60}" -f "$LOG_DIR/api.log"
    ;;
  *)
    die "usage: $0 {start|stop|restart|status|logs|tunnel|tunnel-stop|url}"
    ;;
esac
