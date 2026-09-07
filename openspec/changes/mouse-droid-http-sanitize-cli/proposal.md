# Proposal — HTTP gateway constructor sanitise + CLI filter threading

- change_id: mouse-droid-http-sanitize-cli
- project: mouse-droid
- status: active
- feature_id: F-037
- epic: Security
- owner: ianshank
- created: 2026-09-07
- rev: A

## Why

`OpenAICompatibleLLMGateway` skipped `sanitize()` when `injection_filter is None`.
The orchestrator always threads `build_injection_filter`, but
`scripts/translate_mission.py` and `scripts/ask_rover.py` call
`build_llm_gateway(settings)` with no filter. CHARTER §3 names
`RegexInjectionFilter.sanitize()` as the control that makes cloud egress
acceptable. Those scripts are the probe path the runbooks teach.

## What Changes

- HTTP gateway self-builds `RegexInjectionFilter` from `cfg.injection_patterns`
  / `cfg.max_command_len` when the caller passes `None`.
- Both operator CLIs pass `build_injection_filter(settings)`.
- Factory docstring no longer claims the HTTP backend skips sanitisation.

## Impact

Orchestrator path is unchanged (it already supplied a filter). Direct
construction and CLI probes gain the same envelope as Anthropic / llama_cpp.
The HTTP "never raises on backend failure" contract still swallows sanitiser
exceptions into a neutral `GoalVector` / `""`.

## Charter

No CHARTER §3 carve-out: (1) no new actuation; (2) sanitisation is pre-egress,
not inside 30 Hz; (3) no schema field added. Frozen `arm/**` stays untouched.

## Spec Deltas

`openspec/changes/mouse-droid-http-sanitize-cli/specs/http-sanitize-cli/spec.md`

## Tasks

See `openspec/changes/mouse-droid-http-sanitize-cli/tasks.md`.

## Validation

`bash scripts/validations/F-037.sh`
