#!/bin/bash
# Full Jetson smoke run wrapper (container-first).
#
# All Python execution is delegated to `docker exec mousedroid python3` via a
# transient wrapper script exported as MOUSEDROID_SMOKE_PYTHON, so
# scripts/jetson_smoke_test.sh runs unmodified but executes inside the
# container which already has `mousedroid`, pytest, torch, tensorrt and the
# device passthroughs. Stage blocking defaults follow the Phase 1 roadmap
# (LiDAR -> Camera -> OLED -> Motors); per-stage overrides are exposed via
# MOUSEDROID_SMOKE_BLOCKING_<STAGE> env vars (yes|no).
#
# Usage (run on the Jetson host, repo at /opt/mousedroid):
#   bash scripts/jetson_full_smoke_run.sh
#
# Optional overrides:
#   MOUSEDROID_SMOKE_REPORT_ROOT  -- defaults to <repo>/reports/jetson_smoke
#   MOUSEDROID_SMOKE_BUS          -- I2C bus for OLED stage (default 7)
#   MOUSEDROID_SMOKE_CONTAINER    -- container name (default mousedroid)
#   MOUSEDROID_JETSON_CONFIGS     -- forwarded to jetson_smoke_test.sh + container

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

CONTAINER="${MOUSEDROID_SMOKE_CONTAINER:-mousedroid}"
OLED_BUS="${MOUSEDROID_SMOKE_BUS:-7}"
REPORT_ROOT="${MOUSEDROID_SMOKE_REPORT_ROOT:-${REPO_DIR}/reports/jetson_smoke}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${REPORT_ROOT}/${STAMP}"
mkdir -p "${RUN_DIR}"

SUMMARY="${RUN_DIR}/SUMMARY.md"
declare -a RESULTS=()
OVERALL_FAIL=0

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

record() {
    local name="$1" status="$2" note="${3:-}"
    RESULTS+=("${status}|${name}|${note}")
    log "${status}: ${name} ${note}"
}

resolve_blocking() {
    # Resolve effective blocking flag for a stage. Operators can promote or
    # demote any stage without code edits via:
    #   MOUSEDROID_SMOKE_BLOCKING_<STAGE_UPPER>=yes|no
    # Falls back to the default literal passed in by the caller. Values are
    # validated; unknown values fall back to the default.
    local label="$1" default_blocking="$2"
    local upper
    upper="$(printf '%s' "${label}" | tr '[:lower:]-' '[:upper:]_')"
    local override_var="MOUSEDROID_SMOKE_BLOCKING_${upper}"
    local override="${!override_var:-}"
    case "${override}" in
        yes|no) printf '%s' "${override}" ;;
        "")     printf '%s' "${default_blocking}" ;;
        *)
            log "WARN: ${override_var}='${override}' is invalid; using default '${default_blocking}'"
            printf '%s' "${default_blocking}"
            ;;
    esac
}

run_stage() {
    # $1 = stage label, $2 = blocking? (yes|no), $3 = timeout seconds (0 = none),
    # $4.. = command
    local label="$1" default_blocking="$2" tmo="$3"; shift 3
    local blocking
    blocking="$(resolve_blocking "${label}" "${default_blocking}")"
    local logfile="${RUN_DIR}/${label}.log"
    log "=== STAGE ${label} (blocking=${blocking} timeout=${tmo}s) ==="
    log "    cmd: $*"
    log "    log: ${logfile}"
    set +e
    if [[ "${tmo}" != "0" ]]; then
        MOUSEDROID_SMOKE_STAGE_TIMEOUT="${tmo}" \
            timeout --signal=INT --kill-after=10 "${tmo}" "$@" >"${logfile}" 2>&1
    else
        MOUSEDROID_SMOKE_STAGE_TIMEOUT="0" "$@" >"${logfile}" 2>&1
    fi
    local rc=$?
    set +e
    if [[ ${rc} -eq 0 ]]; then
        record "${label}" "PASS"
        return 0
    fi
    if [[ "${blocking}" == "no" ]]; then
        local why="rc=${rc} (non-blocking)"
        [[ ${rc} -eq 124 || ${rc} -eq 137 ]] && why="rc=${rc} (timeout after ${tmo}s, non-blocking)"
        record "${label}" "EXPECTED-FAIL" "${why}"
        return 0
    fi
    record "${label}" "FAIL" "rc=${rc}"
    OVERALL_FAIL=1
    return 1
}

container_running() {
    docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

# --- Container python wrapper --------------------------------------------
# jetson_smoke_test.sh resolves "${PYTHON}" from MOUSEDROID_SMOKE_PYTHON
# (must be executable). We point it at this shim that proxies into the
# running container with relevant MOUSEDROID_* env vars forwarded.
PY_WRAPPER="${RUN_DIR}/python3-in-container"
cat > "${PY_WRAPPER}" <<EOF
#!/bin/bash
set -uo pipefail

stage_timeout="\${MOUSEDROID_SMOKE_STAGE_TIMEOUT:-0}"
inner_timeout="\${stage_timeout}"
if [[ "\${stage_timeout}" =~ ^[0-9]+$ && "\${stage_timeout}" -gt 5 ]]; then
    inner_timeout="\$((stage_timeout - 5))"
fi

docker_args=(
    docker exec
    -w ${REPO_DIR}
    -e MOUSEDROID_MOCK_HARDWARE=false
    -e MOUSEDROID_JETSON_CONFIGS
    -e MOUSEDROID_FACE_DISPLAY_SMOKE
    -e MOUSEDROID_FACE_DISPLAY_BUS
    -e MOUSEDROID_LLM__N_GPU_LAYERS
    -e MOUSEDROID_LLM__MAX_TOKENS
    -e PYTHONPATH
    ${CONTAINER}
)

if [[ "\${inner_timeout}" != "0" ]]; then
    docker_args+=(timeout --signal=INT --kill-after=5 "\${inner_timeout}")
fi

docker_args+=(python3 "\$@")
exec "\${docker_args[@]}"
EOF
chmod +x "${PY_WRAPPER}"
export MOUSEDROID_SMOKE_PYTHON="${PY_WRAPPER}"

if ! container_running; then
    log "FATAL: container ${CONTAINER} is not running. Start it before running smoke."
    exit 2
fi

# --- Stage 0: container_health (informational, non-blocking) -------------
{
    echo "Container: ${CONTAINER}"
    docker inspect --format 'Image: {{.Image}}' "${CONTAINER}"
    docker inspect --format 'Status: {{.State.Status}}' "${CONTAINER}"
    docker inspect --format 'Health.Status: {{.State.Health.Status}}' "${CONTAINER}" 2>/dev/null || true
    docker inspect --format 'Health.FailingStreak: {{.State.Health.FailingStreak}}' "${CONTAINER}" 2>/dev/null || true
    echo "--- last 3 healthcheck log entries ---"
    docker inspect --format '{{range .State.Health.Log}}exit={{.ExitCode}} | out={{.Output}}{{println}}{{end}}' "${CONTAINER}" 2>/dev/null | tail -3 || true
    echo "--- container python3 sanity ---"
    docker exec "${CONTAINER}" python3 -c "import sys, mousedroid, pytest; print('python', sys.version.split()[0]); print('mousedroid', mousedroid.__file__); print('pytest', pytest.__version__)"
} > "${RUN_DIR}/container_health.log" 2>&1
record "container_health" "INFO" "see container_health.log"

# --- Stages 1-8: delegated to jetson_smoke_test.sh via PY_WRAPPER --------
# Bench-debug sensors (camera/audio/lidar/speaker/voice) are non-blocking until the
# physical wiring/overlays land; system/gpio/serial and the bounded motor smoke are hard gates.
for stage in system gpio serial; do
    run_stage "${stage}" "yes" 60 bash scripts/jetson_smoke_test.sh "${stage}" || break
done

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    # Motor remains soft until M4 (rover power + encoder wiring + speaker
    # write timeout). Promote with MOUSEDROID_SMOKE_BLOCKING_MOTOR=yes once
    # encoder loopback consistently passes on the bench.
    run_stage "motor" "no" 120 \
        env \
            MOUSEDROID_SMOKE_ALLOW_MOTION=1 \
            bash scripts/jetson_smoke_test.sh motor || true
fi

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    # M1: lidar is now a hard gate by default (re-blocked after LD19 hardening).
    # Operators can demote at runtime with MOUSEDROID_SMOKE_BLOCKING_LIDAR=no.
    run_stage "lidar" "yes" 60 bash scripts/jetson_smoke_test.sh lidar || true
fi

if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    for stage in camera audio speaker voice; do
        run_stage "${stage}" "no" 45 bash scripts/jetson_smoke_test.sh "${stage}"
    done
fi

# --- Stage 9: OLED (non-blocking, container stays up) ---------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "oled" "no" 60 \
        env \
            MOUSEDROID_FACE_DISPLAY_SMOKE=1 \
            MOUSEDROID_FACE_DISPLAY_BUS="${OLED_BUS}" \
            "${PY_WRAPPER}" -m pytest -m hardware -v \
            tests/hardware/test_ssd1306_smoke.py
fi

# --- Stage 10: app health check -------------------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "app_health" "yes" 60 bash scripts/jetson_smoke_test.sh app
fi

# --- Stage 11: hardware pytest suite (non-blocking until camera/USB speaker/HC-SR04 are fixed)
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "hardware_pytest" "no" 300 \
        bash scripts/jetson_smoke_test.sh pytest
fi

# --- Stage 12: orchestrator E2E 5s (non-blocking until camera/dev/video0 fixed)
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "e2e" "no" 60 bash scripts/jetson_smoke_test.sh e2e
fi

# --- Stage 13a: MCP + motor smoke ----------------------------------------
# Runs the rover motor smoke (velocity round-trip, e-stop latency, MCP
# resource polling under load) with the optional MCP server enabled.
#
# SAFETY: motion is disabled by default via the smoke_test_allow_motion=False
# default in ESP32Config — the velocity round-trip sends a zero command so
# the rover does not roll while running unattended (e.g. on a table).
# Override with MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true ONLY when
# the rover is on rollers / tethered / under direct supervision.
#
# Non-blocking so an MCP/motor failure does not mask later stages but
# still surfaces in SUMMARY.md.
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "mcp_motor_smoke" "no" 300 \
        env \
            MOUSEDROID_MCP__ENABLED=true \
            MOUSEDROID_MCP__BIND_TRANSPORT=false \
            MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION="${MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION:-false}" \
            "${PY_WRAPPER}" -m pytest tests/hardware/test_motor_smoke.py \
                -v -m hardware --tb=short --no-cov
fi

# --- Stage 14: LLM live probe ---------------------------------------------
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    LLM_PROBE='import asyncio
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_llm_gateway
from mousedroid.validation.runtime import resolve_runtime_config_paths

cfg = load_settings(*resolve_runtime_config_paths())
gw = build_llm_gateway(cfg)

async def main() -> None:
    await gw.start()
    try:
        degraded = getattr(gw, "is_degraded", False)
        if degraded:
            raise RuntimeError("LLM gateway entered degraded mode during startup")
        if not gw.is_ready:
            raise RuntimeError("LLM gateway is not ready after startup")
        print("LLM_PROBE_READY")
    finally:
        await gw.stop()

asyncio.run(main())'
    run_stage "llm_probe" "yes" 120 \
        env \
            MOUSEDROID_LLM__N_GPU_LAYERS=-1 \
            "${PY_WRAPPER}" -c "${LLM_PROBE}"
fi

# --- Stage 15: New features probes (Phase B) ------------------------------
# Exercises operator-observable surfaces introduced by PRs #75-#82 that the
# 14-stage smoke does NOT cover (lidar raw WS, sensor liveness, port
# discovery, mDNS readiness, hello negotiation, voice event fairness,
# orchestrator FailureRecorder, ClockProtocol, dashboard data flow, logs WS).
# Blocking by default — a green Phase A with a red Phase B should NOT
# advance the deployment SHA bump.
#
# When Phase A already failed we still record an explicit SKIPPED entry
# so the SUMMARY has a Phase B row instead of looking like "the stage
# never existed" — operators triaging a partial failure need to see the
# stage is intentionally not run (Copilot review on PR #83).
mkdir -p "${RUN_DIR}/new_features"
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    run_stage "new_features" "yes" 240 \
        env \
            MOUSEDROID_SMOKE_CONTAINER="${CONTAINER}" \
            MOUSEDROID_PROBE_REPORT_DIR="${RUN_DIR}/new_features" \
            bash scripts/jetson_new_features_probe.sh
else
    {
        echo "SKIPPED: Phase B was not run because Phase A failed earlier."
        echo "Re-run scripts/jetson_full_smoke_run.sh after fixing the blocking"
        echo "stage(s) above. Phase B requires a healthy app_health + llm_probe."
    } > "${RUN_DIR}/new_features.log"
    record "new_features" "SKIPPED" "Phase A failed; see preceding stages"
fi

# --- Summary -------------------------------------------------------------

# Enrich notes for the voice stage so the precise reason for any failure is
# visible directly in SUMMARY.md without having to open voice.log.
voice_log="${RUN_DIR}/voice.log"
voice_remediation=""
if [[ -f "${voice_log}" ]]; then
    for i in "${!RESULTS[@]}"; do
        IFS='|' read -r status name note <<<"${RESULTS[$i]}"
        if [[ "${name}" != "voice" || "${status}" == "PASS" ]]; then
            continue
        fi
        hints=()
        if grep -q "voice.tts_model_path is not configured" "${voice_log}"; then
            hints+=("voice.tts_model_path missing in runtime config")
        fi
        if grep -q "piper_tts_not_installed" "${voice_log}"; then
            hints+=("piper-tts python package not installed in container")
        fi
        if grep -q "piper_tts_no_model_path" "${voice_log}"; then
            hints+=("piper TTS started without a model path")
        fi
        if grep -q "piper_tts_load_failed" "${voice_log}"; then
            hints+=("piper voice model failed to load (see voice.log)")
        fi
        if grep -qE "Piper voice model failed to load from" "${voice_log}"; then
            hints+=("Piper model file unreadable at configured tts_model_path")
        fi
        if grep -q "Rocky voice TTS returned silent audio" "${voice_log}"; then
            hints+=("synthesised audio was silent (model loaded but produced no signal)")
        fi
        if grep -q "configured speaker unavailable for Rocky voice" "${voice_log}"; then
            hints+=("speaker device unavailable for Rocky voice")
        fi
        if [[ ${#hints[@]} -eq 0 ]]; then
            hints+=("see voice.log for details")
        fi
        joined=$(printf '; %s' "${hints[@]}")
        joined="${joined:2}"
        RESULTS[$i]="${status}|${name}|${note} -- ${joined}"
        voice_remediation="${joined}"
    done
fi

{
    echo "# Jetson Full Smoke Run ${STAMP}"
    echo
    echo "- Host: $(hostname)"
    echo "- Repo HEAD: $(git -C "${REPO_DIR}" rev-parse --short HEAD)"
    echo "- Branch: $(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
    if container_running; then
        image_id="$(docker inspect -f '{{.Image}}' "${CONTAINER}" 2>/dev/null || true)"
        echo "- Container: ${CONTAINER} image=${image_id}"
    else
        echo "- Container: ${CONTAINER} (not running at summary time)"
    fi
    echo "- Run dir: ${RUN_DIR}"
    echo
    echo "| Stage | Status | Note |"
    echo "|-------|--------|------|"
    for entry in "${RESULTS[@]}"; do
        IFS='|' read -r status name note <<<"${entry}"
        echo "| ${name} | ${status} | ${note} |"
    done
    if [[ -n "${voice_remediation}" ]]; then
        echo
        echo "## Rocky voice prerequisites"
        echo
        echo "Voice stage diagnostics: ${voice_remediation}"
        echo
        echo "To resolve, ensure ALL of the following are true on the Jetson:"
        echo
        echo "1. \`piper-tts\` is installed inside the \`${CONTAINER}\` container"
        echo "   (rebuild image: \`docker compose -f docker-compose.jetson.yml build --no-cache mousedroid\`)."
        echo "2. The Piper voice model exists at the path referenced by"
        echo "   \`voice.tts_model_path\` in the active runtime config"
        echo "   (default: \`/opt/voice_models/en_US-lessac-medium.onnx\` plus its \`.onnx.json\` sibling)."
        echo "3. The configured USB speaker is enumerated and not held by another process."
        echo
        echo "Full per-stage logs are in \`${RUN_DIR}\` (\`voice.log\` for this stage)."
    fi
    # Ten Pillars section — appended when validate_pillar.sh has run against
    # this same RUN_DIR (MOUSEDROID_PILLAR_REPORT_DIR="${RUN_DIR}").
    ten_pillars_log="${RUN_DIR}/ten_pillars.log"
    if [[ -f "${ten_pillars_log}" ]]; then
        echo
        echo "## Ten Pillars Validation"
        echo
        cat "${ten_pillars_log}"
    fi
} > "${SUMMARY}"

log "Summary written to ${SUMMARY}"
log "Overall: $([[ "${OVERALL_FAIL}" -eq 0 ]] && echo PASS || echo FAIL)"
exit "${OVERALL_FAIL}"
