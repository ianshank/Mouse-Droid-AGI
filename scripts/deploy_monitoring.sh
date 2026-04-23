#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI — Monitoring Stack Deployment Helper
# =============================================================================
# Brings up the compose-based Prometheus / Grafana / Loki stack on the Jetson,
# optionally configures secure Prometheus bearer-token scraping, and verifies
# end-to-end health.
#
# Usage:
#   sudo bash scripts/deploy_monitoring.sh [OPTIONS]
#
# Options:
#   --secure-scrape     Protect /metrics with bearer auth and configure Prometheus token scrape
#   --verify-only       Skip compose changes and only run verification checks
#   --down              Stop the monitoring stack and remove secure scrape token file
#   --skip-app-restart  Do not restart the MouseDroid app container in secure mode
#   --help              Show this help message
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INSTALL_DIR="${MOUSEDROID_INSTALL_DIR:-/opt/mousedroid}"
CONFIG_DIR="${MOUSEDROID_CONFIG_DIR:-/etc/mousedroid}"
DOCKER_ENV_FILE="${MOUSEDROID_DOCKER_ENV_FILE:-${CONFIG_DIR}/docker.env}"

if [ -f "$DOCKER_ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$DOCKER_ENV_FILE"
    set +a
fi

APP_COMPOSE_FILE="${MOUSEDROID_APP_COMPOSE_FILE:-${INSTALL_DIR}/docker-compose.jetson.yml}"
MONITORING_COMPOSE_FILE="${MOUSEDROID_MONITORING_COMPOSE_FILE:-${INSTALL_DIR}/docker-compose.monitoring.yml}"
PROM_RUNTIME_DIR="${MOUSEDROID_PROM_RUNTIME_DIR:-${INSTALL_DIR}/config/prometheus/runtime}"
PROM_TOKEN_FILE="${PROM_RUNTIME_DIR}/mousedroid_token"
SECURE_OVERLAY_SOURCE="${PROJECT_DIR}/config/jetson_secure_metrics.yaml"
SECURE_OVERLAY_DEST="${CONFIG_DIR}/jetson_secure_metrics.yaml"
DEFAULT_CONFIG_FILES="/etc/mousedroid/default.yaml /etc/mousedroid/jetson_production.yaml"
SECURE_OVERLAY_FILE="/etc/mousedroid/jetson_secure_metrics.yaml"
DEFAULT_PROMETHEUS_CONFIG="./config/prometheus/scrape_host_gateway.yml"
SECURE_PROMETHEUS_CONFIG="./config/prometheus/scrape_host_gateway_secure.yml"

TELEMETRY_PORT="${MOUSEDROID_TELEMETRY_PORT:-8080}"
HEALTH_PATH="${MOUSEDROID_HEALTH_PATH:-/api/v1/health}"
METRICS_PATH="${MOUSEDROID_METRICS_PATH:-/metrics}"
GRAFANA_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-mousedroid}"

SECURE_SCRAPE=false
VERIFY_ONLY=false
STOP_STACK=false
SKIP_APP_RESTART=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --secure-scrape) SECURE_SCRAPE=true; shift ;;
        --verify-only) VERIFY_ONLY=true; shift ;;
        --down) STOP_STACK=true; shift ;;
        --skip-app-restart) SKIP_APP_RESTART=true; shift ;;
        --help|-h)
            sed -n '2,/^# ====/{ /^# ====/d; s/^# \?//p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        error "Required command not found: $cmd"
        exit 1
    fi
}

ensure_env_file_parent() {
    mkdir -p "$(dirname "$DOCKER_ENV_FILE")"
    touch "$DOCKER_ENV_FILE"
}

set_env_var() {
    local key="$1"
    local value="$2"

    ensure_env_file_parent
    python3 - "$DOCKER_ENV_FILE" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
replacement = f'{key}="{value}"' if (' ' in value or value.startswith('./')) else f'{key}={value}'

lines = path.read_text().splitlines() if path.exists() else []
updated = False
new_lines = []
for line in lines:
    if line.startswith(f"{key}="):
        new_lines.append(replacement)
        updated = True
    else:
        new_lines.append(line)
if not updated:
    new_lines.append(replacement)
path.write_text("\n".join(new_lines) + "\n")
PY
}

remove_secure_overlay_from_config_files() {
    local current="${MOUSEDROID_CONFIG_FILES:-$DEFAULT_CONFIG_FILES}"
    current="${current//${SECURE_OVERLAY_FILE}/}"
    current="$(printf '%s' "$current" | xargs)"
    if [ -z "$current" ]; then
        current="$DEFAULT_CONFIG_FILES"
    fi
    MOUSEDROID_CONFIG_FILES="$current"
    export MOUSEDROID_CONFIG_FILES
    set_env_var "MOUSEDROID_CONFIG_FILES" "$MOUSEDROID_CONFIG_FILES"
}

restart_mousedroid_app() {
    if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet mousedroid-docker; then
        info "Restarting mousedroid-docker systemd service"
        systemctl restart mousedroid-docker
        return
    fi

    info "Restarting MouseDroid via docker compose"
    docker compose -f "$APP_COMPOSE_FILE" up -d
}

wait_for_http() {
    local name="$1"
    local url="$2"
    shift 2

    local attempt
    for attempt in $(seq 1 20); do
        if curl -fsS "$@" "$url" >/dev/null 2>&1; then
            info "  ${name}: OK (${url})"
            return 0
        fi
        sleep 2
    done

    error "  ${name}: not ready (${url})"
    return 1
}

ensure_secure_overlay() {
    if [ ! -f "$SECURE_OVERLAY_SOURCE" ]; then
        error "Secure telemetry overlay not found: $SECURE_OVERLAY_SOURCE"
        exit 1
    fi

    mkdir -p "$CONFIG_DIR"
    cp "$SECURE_OVERLAY_SOURCE" "$SECURE_OVERLAY_DEST"
}

configure_secure_scrape() {
    if [ -z "${MOUSEDROID_TELEMETRY_TOKEN:-}" ] || [ "${MOUSEDROID_TELEMETRY_TOKEN}" = "changeme" ]; then
        error "Set a real MOUSEDROID_TELEMETRY_TOKEN in ${DOCKER_ENV_FILE} before enabling secure scrape"
        exit 1
    fi

    mkdir -p "$PROM_RUNTIME_DIR"
    printf '%s\n' "$MOUSEDROID_TELEMETRY_TOKEN" > "$PROM_TOKEN_FILE"
    chmod 600 "$PROM_TOKEN_FILE"
    export MOUSEDROID_PROMETHEUS_CONFIG="$SECURE_PROMETHEUS_CONFIG"
    set_env_var "MOUSEDROID_PROMETHEUS_CONFIG" "$MOUSEDROID_PROMETHEUS_CONFIG"

    if [ "$SKIP_APP_RESTART" = true ]; then
        warn "Skipping app restart; secure Prometheus scrape requires MouseDroid to already be running with jetson_secure_metrics.yaml"
        return
    fi

    ensure_secure_overlay

    local app_config_files="${MOUSEDROID_CONFIG_FILES:-$DEFAULT_CONFIG_FILES}"
    if [[ " ${app_config_files} " != *" ${SECURE_OVERLAY_FILE} "* ]]; then
        app_config_files="${app_config_files} ${SECURE_OVERLAY_FILE}"
    fi

    export MOUSEDROID_CONFIG_FILES="$app_config_files"
    set_env_var "MOUSEDROID_CONFIG_FILES" "$MOUSEDROID_CONFIG_FILES"
    info "Restarting MouseDroid with secure telemetry metrics overlay"
    restart_mousedroid_app
}

verify_monitoring_stack() {
    info "Running monitoring verification"

    wait_for_http "Telemetry health" "http://127.0.0.1:${TELEMETRY_PORT}${HEALTH_PATH}"
    wait_for_http "Prometheus ready" "http://127.0.0.1:9090/-/ready"
    wait_for_http "Loki ready" "http://127.0.0.1:3100/ready"
    wait_for_http \
        "Grafana health" \
        "http://127.0.0.1:3000/api/health" \
        -u "${GRAFANA_ADMIN_USER}:${GRAFANA_ADMIN_PASSWORD}"

    if [ "$SECURE_SCRAPE" = true ]; then
        local unauth_status
        unauth_status="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${TELEMETRY_PORT}${METRICS_PATH}")"
        if [ "$unauth_status" != "401" ]; then
            error "Expected unauthenticated metrics scrape to return 401, got ${unauth_status}"
            return 1
        fi
        info "  Secure metrics auth: unauthenticated scrape correctly rejected"

        wait_for_http \
            "Authenticated metrics" \
            "http://127.0.0.1:${TELEMETRY_PORT}${METRICS_PATH}" \
            -H "Authorization: Bearer ${MOUSEDROID_TELEMETRY_TOKEN}"
    fi

    local prom_query
    for _attempt in $(seq 1 20); do
        prom_query="$(curl -fsS --get "http://127.0.0.1:9090/api/v1/query" --data-urlencode 'query=up{job="mousedroid"}')" || true
        if printf '%s' "$prom_query" | python3 -c 'import json, sys
payload = json.load(sys.stdin)
results = payload.get("data", {}).get("result", [])
ok = any(sample.get("value", [None, "0"])[1] == "1" for sample in results)
raise SystemExit(0 if ok else 1)
'; then
            info "  Prometheus target state: mousedroid job is up"
            return 0
        fi
        sleep 2
    done

    error "Prometheus did not report the mousedroid target as up"
    return 1
}

require_command docker
require_command curl
require_command python3

if [ "$STOP_STACK" = true ]; then
    info "Stopping monitoring stack"
    cd "$INSTALL_DIR"
    docker compose -f "$MONITORING_COMPOSE_FILE" down --timeout 30
    rm -f "$PROM_TOKEN_FILE"
    export MOUSEDROID_PROMETHEUS_CONFIG="$DEFAULT_PROMETHEUS_CONFIG"
    set_env_var "MOUSEDROID_PROMETHEUS_CONFIG" "$MOUSEDROID_PROMETHEUS_CONFIG"
    remove_secure_overlay_from_config_files
    exit 0
fi

cd "$INSTALL_DIR"

if [ "$SECURE_SCRAPE" = true ]; then
    configure_secure_scrape
else
    export MOUSEDROID_PROMETHEUS_CONFIG="$DEFAULT_PROMETHEUS_CONFIG"
fi

if [ "$VERIFY_ONLY" = false ]; then
    info "Starting monitoring stack"
    docker compose -f "$MONITORING_COMPOSE_FILE" up -d
fi

verify_monitoring_stack