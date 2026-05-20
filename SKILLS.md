# SKILLS.md — Skill index for MouseDroidAGI

> Maps high-level capability to the *files an agent should read* + *commands
> an agent should run* to exercise that capability. Companion to
> `AGENTS.md` (rules) and `CLAUDE.md` (project facts).

Skills are organised by **trigger**: when you see the trigger, invoke the
skill. If no project-local skill matches, fall back to the
`plugin:superpowers:*` family or the Anthropic skill index.

---

## Operational skills (operator-facing)

### dashboard-proxy

**Trigger:** "open the dashboard", "browse the rover from my workstation",
"the Jetson telemetry server is auth-gated".

**Read:**
- `tools/dashboard_proxy.py` — the proxy itself
- `launch_dashboard.ps1` — Windows / PowerShell launcher
- `config/dev_dashboard.yaml.example` — overlay template
- `docs/architecture/c4-dashboard-proxy.md` — C4 diagrams

**Run:**
```bash
# CLI form
python tools/dashboard_proxy.py 8081 http://192.168.55.1:8080 "$JETSON_TOKEN"
# Env form
PROXY_PORT=8081 JETSON_HTTP=http://192.168.55.1:8080 JETSON_TOKEN=... \
    python tools/dashboard_proxy.py
```

Then open `http://127.0.0.1:8081/lidar` in a browser; the proxy forwards
the auth header + handles MJPEG + WebSockets transparently.

### live-camera-verification

**Trigger:** "the camera feed is solid green", "no JPEG appearing on
`/camera/frame.jpg`", "IMX708 / Bayer / RG10".

**Read:**
- `src/mousedroid/hardware/camera/jetson_csi.py` — focus on
  `capture_raw_jpeg` + `_frame_to_rgb_for_snapshot`
- `src/mousedroid/config/schema.py` — `CameraConfig.v4l2_grayscale_extract`
  field docstring (background on the IMX708 Bayer-misinterpretation)
- `tests/unit/test_jetson_csi.py` — the test surface that pins the per-
  backend colour-conversion paths.

**Run:**
```bash
# Live JPEG fetch through the dashboard proxy (workstation)
curl -o /tmp/snap.jpg http://127.0.0.1:8081/camera/frame.jpg
# Offline snapshot via the verify script (Jetson or mock workstation)
MOUSEDROID_MOCK_HARDWARE=true python scripts/verify_sensors.py \
    --config config/default.yaml --sensor camera --frames 3 \
    --save-frame /tmp/snap.jpg
```

### esp32-disconnected-mode

**Trigger:** "the rover won't boot — orchestrator crashes on connect",
"no ESP32 plugged in", "running camera + LiDAR without motors".

**Read:**
- `src/mousedroid/factory.py:90-127` — the `build_esp32_driver` branch.
- `src/mousedroid/config/schema.py` — `ESP32Config.enabled` docstring.
- `tests/integration/test_pr104_esp32_disabled_integration.py` —
  reference tests.

**Run:**
```bash
# YAML override
echo "esp32:\n  enabled: false" >> /etc/mousedroid/jetson_production.yaml
# Or env override (docker.env)
MOUSEDROID_ESP32__ENABLED=false python -m mousedroid.main \
    --config /etc/mousedroid/jetson_production.yaml
```

### preflight-validation

**Trigger:** "smoke test the rover", "is hardware healthy before mission?".

**Read:**
- `src/mousedroid/validation/preflight.py` — hardware probe dispatcher.
- `src/mousedroid/validation/pillars.py` — 10-pillar validator.
- `src/mousedroid/cli/preflight.py` + `cli/validate_pillars.py` — argparse
  wrappers (exit-code contract: 0=OK/DEGRADED, 1=FAIL).

**Run:**
```bash
python -m mousedroid.cli.preflight --config config/default.yaml
python -m mousedroid.cli.validate_pillars --config config/default.yaml
```

---

## Engineering skills (developer-facing)

### add-schema-field

**Trigger:** "add a new config knob", "expose this threshold to YAML".

**Process:**

1. Add the `Field(default=..., ge=..., le=..., description=...)` to the
   appropriate Pydantic model in `src/mousedroid/config/schema.py`.
2. The `description` MUST be ≥ 20 chars + explain *why* the operator
   would change it.
3. Default MUST preserve legacy behaviour (an existing YAML without the
   field must load identically).
4. Add a regression test in `tests/regression/` pinning the default.
5. Add an AQA test in `tests/regression/test_pr*_aqa.py` confirming the
   description + default + range are reachable via `FieldInfo`.
6. If the field gates a code branch, add an integration test in
   `tests/integration/` exercising the wiring through `factory.py`.

Reference: PR #104 added 3 fields this way — see commit `1b7a12e` for the
shape.

### add-hardware-driver

**Trigger:** "implement the IMX500 / VL53L0X / new sensor driver".

**Process:**

1. Define `@runtime_checkable Protocol` in
   `src/mousedroid/hardware/protocols.py` (or the appropriate
   subsystem-local protocol module). Keep the interface narrow.
2. Implement the real driver in `src/mousedroid/hardware/<subsystem>/`.
3. Implement a `MockX` mirror — same protocol, no real I/O. Mock must
   produce sensible synthetic data (`np.random.default_rng(seed).normal()`
   shaped to match the real driver's output).
4. Add a `build_<driver>(cfg, ...)` builder in `factory.py`. Honour
   `mock_hardware` AND any per-subsystem `enabled: bool`.
5. Tests: unit (mock-driven), integration (factory wiring), hardware
   (rover-only, `@pytest.mark.hardware`).
6. Update `docs/architecture/` if you added a new external boundary.

### run-pre-pr-validation

**Trigger:** "I'm about to push a PR", "is this ready to merge?"

**Process:**

```bash
# Local — must all be green
"/c/Program Files/Python311/python.exe" -m ruff check src/ tests/ tools/ scripts/
"/c/Program Files/Python311/python.exe" -m ruff format --check src/ tests/ tools/ scripts/
"/c/Program Files/Python311/python.exe" -m mypy --strict --no-incremental src/mousedroid/<touched>.py
"/c/Program Files/Python311/python.exe" -m pytest tests/unit tests/integration tests/property \
    -m "not hardware" --import-mode=importlib --timeout=180 -q
ALLOW_PYTEST_COLLECTION_SKIP=1 \
    "/c/Program Files/Python311/python.exe" scripts/check_branch_coverage.py \
        --min 85 --tests tests/unit/<your-test>.py
```

Then delegate to subagents in parallel:
- `security-auditor` — focused on the diff's files.
- `code-quality` — ruff + mypy + branch-coverage gate.
- `feature-dev:code-reviewer` — independent reviewer for high-confidence
  issues.

When all three return green, push + open the PR with the body template
from `.github/pull_request_template.md` (or PR #104 as a worked example).

---

## Subagent skills (delegation-facing)

| Skill | Subagent | Use when |
|-------|----------|----------|
| Security audit | `security-auditor` | Before any PR that touches auth, networking, or shells out |
| Code quality | `code-quality` | Before any PR — runs ruff + mypy + branch-coverage gates |
| Code review | `feature-dev:code-reviewer` | Independent second-opinion on a diff |
| ML review | `ai-ml-toolkit:ml-engineer` | Training-loop changes, model deployment, A/B tests |
| Architecture | `feature-dev:code-architect` | Designing a new feature with multi-file scope |
| Test engineering | `testing-suite:test-engineer` | When the test pyramid needs a structural change |
| Documentation | `documentation-generator:technical-writer` | README rewrites, ADRs, user guides |

When dispatching, ALWAYS:

- Provide the worktree path explicitly.
- Provide the Python interpreter path (`"/c/Program Files/Python311/python.exe"`)
  on Windows.
- Provide the exact files / paths to focus on.
- Ask for a length-capped report ("under 250 words inline").
- Use `run_in_background=true` for any check that doesn't block your next
  step — then continue work in parallel and respond to the completion
  notification.

---

## Discovery

To find more skills, in this order:
1. Grep this file: `grep -i "<keyword>" SKILLS.md`.
2. Check `~/.claude/plugins/cache/claude-plugins-official/superpowers/*/skills/`
   for general-purpose Anthropic skills.
3. Check `docs/operations/` for operator runbooks (which often map to
   skills here).
