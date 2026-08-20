---
description: Commit evidence artifacts to reports/ and link them from features.yaml
status: active
---

# Evidence Commit

Place commit-safe evidence artifacts under the tracked roots and link them to
features.yaml. For gitignored-by-policy families, record the declared
local-only evidence chain instead.

## Tracked Evidence

Artifacts that ship with the repo go under `reports/<surface>/<date>/`:

```bash
mkdir -p reports/<surface>/$(date +%Y-%m-%d)
cp <artifact> reports/<surface>/$(date +%Y-%m-%d)/
git add reports/<surface>/$(date +%Y-%m-%d)/
```

Then update `features.yaml` notes for the relevant feature with the path.

## Local-Only Evidence Chains

Some report families are gitignored by policy (configured in
`.claude/workforce.yaml` under `evidence.local_only_declared`):

- `reports/trunk_sync`
- `reports/jetson_full_validation`
- `reports/jetson_smoke`
- `reports/dead_code`
- `reports/local_runs`
- `reports/endurance`

For these, the evidence chain is: the gitignored artifact EXISTS locally +
a CHANGELOG.md or plan-doc reference documents when it was produced and what
it showed.

## Guardrails

- Never close a hardware/performance claim with NEITHER a tracked artifact
  NOR a declared local-only chain — the hw-evidence-auditor will flag it
- Staleness: artifacts older than `evidence.stale_after_days` (from
  `.claude/workforce.yaml`) are flagged
- `features.yaml` `implemented_in` must be a hex commit SHA, not a branch name
