#!/usr/bin/env bash
# validate_pillar.sh — Ten Pillars on-Nano validation dispatcher.
#
# Runs one pillar's headless pytest set plus a container-backed runtime factory
# probe using the same python3-in-container shim as jetson_full_smoke_run.sh.
#
# Usage (run in /opt/mousedroid on the Jetson host):
#   export MOUSEDROID_SMOKE_PYTHON=<path-to-python3-in-container>
#   bash scripts/validate_pillar.sh safety
#   bash scripts/validate_pillar.sh world_model
#   # … or all pillars at once:
#   bash scripts/validate_pillar.sh all
#
# Optional overrides:
#   MOUSEDROID_PILLAR_REPORT_DIR  -- directory for pillar log files
#                                    (defaults to <repo>/reports/jetson_smoke/<stamp>)
#   MOUSEDROID_SMOKE_PYTHON       -- python3-in-container shim from a prior smoke run;
#                                    if absent, falls back to docker exec directly
#   MOUSEDROID_PILLAR_TIMEOUT     -- per-pillar timeout in seconds (default 180)
#   MOUSEDROID_PILLAR_BLOCKING_<UPPER> -- override blocking for one pillar (yes|no)
#
# Exit codes:
#   0  all selected pillars passed
#   1  one or more blocking pillar failures
#   2  usage / precondition error

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

CONTAINER="${MOUSEDROID_SMOKE_CONTAINER:-mousedroid}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${MOUSEDROID_PILLAR_REPORT_DIR:-${REPO_DIR}/reports/jetson_smoke/${STAMP}}"
mkdir -p "${REPORT_DIR}"

PILLAR_TIMEOUT="${MOUSEDROID_PILLAR_TIMEOUT:-180}"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

# ---------------------------------------------------------------------------
# Python shim: prefer MOUSEDROID_SMOKE_PYTHON from a prior smoke run; fall
# back to a transient docker exec wrapper so this script is self-contained.
# ---------------------------------------------------------------------------
if [[ -n "${MOUSEDROID_SMOKE_PYTHON:-}" && -x "${MOUSEDROID_SMOKE_PYTHON}" ]]; then
    PYTHON="${MOUSEDROID_SMOKE_PYTHON}"
    log "Using MOUSEDROID_SMOKE_PYTHON: ${PYTHON}"
else
    PYTHON="${REPORT_DIR}/python3-in-container"
    cat > "${PYTHON}" <<SHIMEOF
#!/bin/bash
set -uo pipefail
exec docker exec -w ${REPO_DIR} \
    -e MOUSEDROID_MOCK_HARDWARE=false \
    -e MOUSEDROID_JETSON_CONFIGS \
    -e PYTHONPATH \
    ${CONTAINER} python3 "\$@"
SHIMEOF
    chmod +x "${PYTHON}"
    log "Created fallback python3-in-container shim: ${PYTHON}"
fi

# Mapping of config names used in MOUSEDROID_JETSON_CONFIGS
JETSON_CONFIGS="${MOUSEDROID_JETSON_CONFIGS:-/etc/mousedroid/default.yaml,/etc/mousedroid/jetson_production.yaml}"
export MOUSEDROID_JETSON_CONFIGS="${JETSON_CONFIGS}"

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
declare -a PILLAR_RESULTS=()
OVERALL_FAIL=0

record_pillar() {
    local name="$1" status="$2" note="${3:-}"
    PILLAR_RESULTS+=("${status}|${name}|${note}")
    log "${status}: pillar/${name} ${note}"
}

resolve_blocking() {
    local label="$1" default_blocking="$2"
    local upper
    upper="$(printf '%s' "${label}" | tr '[:lower:]-' '[:upper:]_')"
    local var="MOUSEDROID_PILLAR_BLOCKING_${upper}"
    local val="${!var:-}"
    case "${val}" in
        yes|no) printf '%s' "${val}" ;;
        "")     printf '%s' "${default_blocking}" ;;
        *)
            log "WARN: ${var}='${val}' invalid; using default '${default_blocking}'"
            printf '%s' "${default_blocking}"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# run_pillar_check:
#   $1 = pillar label (e.g. safety)
#   $2 = blocking (yes|no)
#   $3 = timeout seconds
#   $4 = check type: headless | runtime
#   $5.. = command
# ---------------------------------------------------------------------------
run_pillar_check() {
    local label="$1" blocking_default="$2" tmo="$3" kind="$4"; shift 4
    local blocking
    blocking="$(resolve_blocking "${label}" "${blocking_default}")"
    local logfile="${REPORT_DIR}/pillar_${label}_${kind}.log"

    log "--- pillar:${label} kind:${kind} blocking:${blocking} timeout:${tmo}s ---"
    log "    cmd: $*"
    # Save the caller's errexit state so we don't permanently force `set -e`
    # if it wasn't already enabled.  The stage command is allowed to fail; we
    # capture the rc and let the caller decide blocking semantics.
    local prev_errexit="off"
    case "$-" in *e*) prev_errexit="on";; esac
    set +e
    if [[ "${tmo}" != "0" ]]; then
        MOUSEDROID_SMOKE_STAGE_TIMEOUT="${tmo}" \
            timeout --signal=INT --kill-after=15 "${tmo}" \
            "$@" >"${logfile}" 2>&1
    else
        MOUSEDROID_SMOKE_STAGE_TIMEOUT="0" "$@" >"${logfile}" 2>&1
    fi
    local rc=$?
    if [[ "${prev_errexit}" == "on" ]]; then
        set -e
    fi

    if [[ ${rc} -eq 0 ]]; then
        record_pillar "${label}" "PASS" "kind=${kind}"
        return 0
    fi

    if [[ "${blocking}" == "no" ]]; then
        local why="rc=${rc} (non-blocking, kind=${kind})"
        [[ ${rc} -eq 124 || ${rc} -eq 137 ]] && why="rc=${rc} (timeout, kind=${kind})"
        record_pillar "${label}" "EXPECTED-FAIL" "${why}"
        return 0
    fi

    record_pillar "${label}" "FAIL" "rc=${rc} kind=${kind}"
    OVERALL_FAIL=1
    return 1
}

# ---------------------------------------------------------------------------
# Container sanity check
# ---------------------------------------------------------------------------
assert_container_running() {
    if ! docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
        log "FATAL: container '${CONTAINER}' is not running."
        log "Start with: docker compose -f docker-compose.jetson.yml up -d mousedroid"
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# Pillar definitions
# Encoding: name | default_blocking | headless_pytest_targets | runtime_probe_python
# ---------------------------------------------------------------------------

# Runtime probe snippets — kept here so each is a single, short bash variable
# and can be tested / quoted without nesting issues.

_probe_safety='
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_safety_monitor
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
m = build_safety_monitor(cfg)
print("safety_monitor_ok", type(m).__name__)
'

_probe_world_model='
import torch
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_world_model
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
wm = build_world_model(cfg)
params = sum(p.numel() for p in wm.parameters()) if hasattr(wm, "parameters") else -1
print("world_model_ok", type(wm).__name__, "params=" + str(params))
'

_probe_memory='
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_memory_tier
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
mm = build_memory_tier(cfg)
print("memory_ok", type(mm).__name__ if mm is not None else "disabled")
'

_probe_cognitive='
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_cognitive_core
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
cc = build_cognitive_core(cfg)
print("cognitive_ok", type(cc).__name__)
'

_probe_reward='
from mousedroid.config.loader import load_settings
from mousedroid.reward.model import MultiObjectiveRewardModel
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
rm = MultiObjectiveRewardModel(cfg.model, cfg.reward)
print("reward_ok", type(rm).__name__, "params="+str(sum(p.numel() for p in rm.parameters())))
'

_probe_curiosity='
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_curiosity_module
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
icm = build_curiosity_module(cfg)
print("curiosity_ok", type(icm).__name__ if icm is not None else "disabled")
'

_probe_continual='
import torch.nn as nn
from mousedroid.config.loader import load_settings
from mousedroid.learning.ewc import EWCAgent
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
dummy = nn.Linear(cfg.learning.ewc_fallback_input_dim, 1)
ewc = EWCAgent(cfg.learning, dummy)
print("continual_ok", type(ewc).__name__, "lambda="+str(cfg.learning.ewc_lambda))
'

_probe_meta='
import torch.nn as nn
from mousedroid.config.loader import load_settings
from mousedroid.meta.maml import MAMLAdapter
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
dummy = nn.Linear(cfg.model.obs_dim, cfg.model.obs_dim)
m = MAMLAdapter(dummy, inner_lr=1e-3, outer_lr=1e-4, inner_steps=5)
print("meta_ok", type(m).__name__)
'

_probe_scaling='
from mousedroid.config.loader import load_settings
from mousedroid.scaling.adaptive import AdaptiveCompute
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
s = AdaptiveCompute(input_dim=cfg.model.obs_dim, max_steps=8)
print("scaling_ok", type(s).__name__, "obs_dim="+str(cfg.model.obs_dim))
'

_probe_growth='
import tensorrt
import torch.nn as nn
from mousedroid.config.loader import load_settings
from mousedroid.growth.distillation import KnowledgeDistiller
from mousedroid.validation.runtime import resolve_runtime_config_paths
cfg = load_settings(*resolve_runtime_config_paths())
teacher = nn.Linear(cfg.model.obs_dim, cfg.model.obs_dim)
student = nn.Linear(cfg.model.obs_dim, cfg.model.obs_dim)
g = KnowledgeDistiller(teacher=teacher, student=student, temperature=2.0, alpha=0.5, lr=1e-3)
print("growth_ok", "trt=" + tensorrt.__version__, type(g).__name__)
'

# ---------------------------------------------------------------------------
# run_pillar: dispatch headless + runtime checks for one named pillar
#
# Default blocking for each pillar:
#   safety        → yes (foundational)
#   world_model   → yes (foundational)
#   memory        → yes (foundational)
#   cognitive     → yes
#   reward        → yes
#   curiosity     → no  (training-path, not on critical run path)
#   continual     → no
#   meta          → no
#   scaling       → no
#   growth        → no
# Operators can override with MOUSEDROID_PILLAR_BLOCKING_<UPPER>=yes|no.
# ---------------------------------------------------------------------------
run_pillar() {
    local pillar="$1"
    log "====== PILLAR: ${pillar} ======"

    case "${pillar}" in

    safety)
        run_pillar_check safety yes "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_safety_monitor.py \
                tests/unit/test_safety_context.py \
                tests/unit/test_three_laws.py \
                -q --tb=short || return 1
        run_pillar_check safety yes 60 runtime \
            "${PYTHON}" -c "${_probe_safety}"
        ;;

    world_model)
        run_pillar_check world_model yes "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_rssm.py \
                tests/unit/test_dual_stream_rssm.py \
                tests/unit/test_mcts.py \
                -q --tb=short || return 1
        run_pillar_check world_model yes 60 runtime \
            "${PYTHON}" -c "${_probe_world_model}"
        ;;

    memory)
        run_pillar_check memory yes "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_memory_working.py \
                tests/unit/test_memory_episodic.py \
                tests/unit/test_memory_semantic.py \
                tests/unit/test_memory_tier.py \
                tests/unit/test_consolidation.py \
                -q --tb=short || return 1
        run_pillar_check memory yes 60 runtime \
            "${PYTHON}" -c "${_probe_memory}"
        ;;

    cognitive)
        run_pillar_check cognitive yes "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_cognitive_core.py \
                tests/unit/test_metacognitive.py \
                tests/unit/test_orchestrator.py \
                tests/unit/test_orchestrator_telemetry.py \
                -q --tb=short || return 1
        run_pillar_check cognitive yes 60 runtime \
            "${PYTHON}" -c "${_probe_cognitive}"
        ;;

    reward)
        run_pillar_check reward yes "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_reward_model.py \
                tests/unit/test_reward_aggregator.py \
                tests/unit/test_three_laws_reward.py \
                -q --tb=short || return 1
        run_pillar_check reward yes 60 runtime \
            "${PYTHON}" -c "${_probe_reward}"
        ;;

    curiosity)
        run_pillar_check curiosity no "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_icm.py \
                tests/unit/test_curiosity_wiring.py \
                tests/unit/test_novelty_decay.py \
                -q --tb=short
        run_pillar_check curiosity no 60 runtime \
            "${PYTHON}" -c "${_probe_curiosity}"
        ;;

    continual)
        run_pillar_check continual no "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_ewc.py \
                tests/unit/test_progressive.py \
                -q --tb=short
        run_pillar_check continual no 60 runtime \
            "${PYTHON}" -c "${_probe_continual}"
        ;;

    meta)
        run_pillar_check meta no "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_maml.py \
                tests/unit/test_in_context.py \
                -q --tb=short
        run_pillar_check meta no 60 runtime \
            "${PYTHON}" -c "${_probe_meta}"
        ;;

    scaling)
        run_pillar_check scaling no "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_moe.py \
                tests/unit/test_adaptive_compute.py \
                tests/unit/test_batch_tuner.py \
                -q --tb=short
        run_pillar_check scaling no 60 runtime \
            "${PYTHON}" -c "${_probe_scaling}"
        ;;

    growth)
        run_pillar_check growth no "${PILLAR_TIMEOUT}" headless \
            "${PYTHON}" -m pytest \
                tests/unit/test_distillation.py \
                -q --tb=short
        run_pillar_check growth no 60 runtime \
            "${PYTHON}" -c "${_probe_growth}"
        ;;

    *)
        log "ERROR: unknown pillar '${pillar}'"
        log "Valid pillars: safety world_model memory cognitive reward curiosity continual meta scaling growth"
        exit 2
        ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    log "Usage: $0 <pillar|all> [pillar2 …]"
    log "Pillars: safety world_model memory cognitive reward curiosity continual meta scaling growth"
    exit 2
fi

assert_container_running

PILLARS_TO_RUN=()
if [[ "$1" == "all" ]]; then
    PILLARS_TO_RUN=(safety world_model memory cognitive reward curiosity continual meta scaling growth)
else
    for arg in "$@"; do
        PILLARS_TO_RUN+=("${arg}")
    done
fi

for p in "${PILLARS_TO_RUN[@]}"; do
    run_pillar "${p}" || true   # run_pillar propagates blocking failures via OVERALL_FAIL
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
SUMMARY_FILE="${REPORT_DIR}/ten_pillars.log"
{
    printf "# Ten Pillars Validation — %s\n\n" "$(ts)"
    printf "| Pillar | Status | Note |\n"
    printf "|--------|--------|------|\n"
    for entry in "${PILLAR_RESULTS[@]}"; do
        IFS='|' read -r status name note <<<"${entry}"
        printf "| %s | %s | %s |\n" "${name}" "${status}" "${note}"
    done
    printf "\n"
    if [[ ${OVERALL_FAIL} -eq 0 ]]; then
        printf "Overall: PASS\n"
    else
        printf "Overall: FAIL\n"
    fi
} | tee "${SUMMARY_FILE}"

log "Report dir: ${REPORT_DIR}"
log "Summary:    ${SUMMARY_FILE}"
exit "${OVERALL_FAIL}"
