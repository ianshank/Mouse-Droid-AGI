# Secret scanning — operator runbook (F-015, WS-0.4)

The repo carries a gitleaks-based secret-scan gate so the *next* leaked
credential fails at PR time instead of being discovered in a transcript.
Configuration lives in `.gitleaks.toml`; the CI job is `gitleaks` in
`.github/workflows/ci.yml`; a local advisory stage runs in `scripts/ci.sh`
when the binary is on PATH.

## What runs where

| Surface | Mode | Scope |
|---------|------|-------|
| CI job `gitleaks` | **blocking** (promoted 2026-08-07) | **full history** (`fetch-depth: 0`) |
| `scripts/ci.sh` stage | **blocking** when the binary is present; skipped (loudly) when absent | working tree |
| Local pre-commit hook (below) | blocking for the committer | staged files |

## Allowlist policy — the one rule

**Allowlist by exact fake-key regex only. NEVER by path.** A path allowlist
(waiving `tests/` or `docs/`) blinds the scanner to a real secret landing in
exactly those files — the original `ANTHROPIC_API_KEY` incident started in
documentation. Placeholder conventions already covered: `sk-ant-...`,
`sk-ant-xyz`, `sk-ant-test`, the `changeme` telemetry-token stand-in, and
`test-api-key-abc123` (the bearer-token test fixture).
When adding a new placeholder to docs or tests, keep it obviously
non-key-shaped (short, no base64 tail) and add its literal regex to
`.gitleaks.toml` if the scan fires.

## Promotion protocol

**Promoted advisory → blocking 2026-08-07.** The job ran advisory from
2026-07-03; `scripts/check_advisory_promotions.py` flagged it overdue at 34
days against its 30-day window, with the job green across recent runs
(including runs where unrelated jobs failed). `continue-on-error: true` was
dropped and the `.github/advisory_stages.yaml` entry retired (the tracker
covers only jobs carrying the flag). Re-demoting to advisory requires
re-adding both, with a dated reason —
`tests/regression/test_secret_scan_gate.py` pins the blocking state.

## Local pre-commit hook (recommended)

The repo intentionally ships no `.pre-commit-config.yaml` (script+runbook
convention). To block staged secrets locally:

```bash
cat > .git/hooks/pre-commit <<'EOF'
#!/usr/bin/env bash
# Block staged secrets before they enter history (advisory if gitleaks absent).
if command -v gitleaks >/dev/null 2>&1; then
    exec gitleaks protect --staged --config .gitleaks.toml --redact --no-banner
fi
echo "gitleaks not on PATH - staged secret scan skipped (CI will scan)"
EOF
chmod +x .git/hooks/pre-commit
```

Install the binary from <https://github.com/gitleaks/gitleaks/releases>
(pin the same major/minor as the CI image tag in `ci.yml`).

## If the scan fires

1. **Do not push.** Rotate the credential first if it is real (see
   `NEXT_STEPS.md` item 1 for the ANTHROPIC_API_KEY rotation procedure).
2. Scrub the secret from the working tree; prefer a pointer to the secret
   store (`/etc/mousedroid/docker.env`, GitHub Actions secrets) over any
   inline value.
3. If it is a placeholder false-positive, add its literal regex to
   `.gitleaks.toml` — never its path.
4. History rewrite is a separate, owner-level decision — a secret that
   reached the remote must be treated as compromised regardless (GitHub
   retains dangling commits), so rotation, not rewrite, is the remediation.
