#!/usr/bin/env bash
# F-037 — HTTP gateway self-builds RegexInjectionFilter; CLIs pass the factory filter.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Explicit `if ! ...` rather than `assert` — Jetson PYTHONOPTIMIZE=1 strips
# Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      tests/regression/test_f037_aqa.py \
      tests/regression/test_f037_backwards_compat.py \
      tests/unit/llm_gateway/test_openai_compatible_injection_filter.py \
      tests/unit/scripts/test_translate_mission_cli.py \
      tests/unit/scripts/test_ask_rover_cli.py \
      tests/security/test_openai_compatible_cli_sanitize.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-037 FAIL: HTTP self-build filter / CLI factory threading drifted" >&2
  exit 1
fi

echo "F-037 OK: OpenAICompatibleLLMGateway self-builds filter; CLIs pass build_injection_filter"
