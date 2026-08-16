# Security Policy

## Supported Versions

MouseDroid is an actively developed edge-AI / robotics portfolio project.
Security fixes are applied to the latest `main` and the most recent tagged
release only.

| Version        | Supported |
| -------------- | --------- |
| `main` (HEAD)  | ✅        |
| latest release | ✅        |
| older tags     | ❌        |

## Reporting a Vulnerability

Please report vulnerabilities **privately** — do not open a public issue.

1. **Preferred:** GitHub → this repository → *Security* → *Report a vulnerability*
   (private vulnerability reporting).
2. **Fallback:** email `ianshank@gmail.com` with steps to reproduce and impact.

You can expect an acknowledgement within a few days. Please allow reasonable
time for a fix before any public disclosure.

## Scope & Existing Tooling

Secret hygiene and dependency posture are checked in CI:

- **Secret scanning** — `.gitleaks.toml` + [`docs/runbooks/secret-scanning.md`](docs/runbooks/secret-scanning.md)
  (the `gitleaks` job — **blocking** since 2026-08-07, promoted from advisory after its green-run window)
- **Dependency audit** — the `security` job (`pip-audit --skip-editable`) in
  `.github/workflows/ci.yml` — currently **advisory** via `continue-on-error`
  pending triage of open findings; a real vulnerability turns the job red
  without blocking the merge
- **Advisory-stage tracking** — `.github/advisory_stages.yaml` (each advisory
  job carries a promotion clock enforced by `scripts/check_advisory_promotions.py`)

Secrets are never committed; per-host runtime secrets live only in
`/etc/mousedroid/docker.env` (documented, without live values, in
`config/docker.env.example`).

## Safety vs. Security

The rover's *safety* boundary — the Three Laws / emergency-stop path, kept LLM-
and training-free inside the deterministic 30 Hz control loop — is a distinct
concern documented in [`docs/CHARTER.md`](docs/CHARTER.md). Report
safety-relevant defects through the same private channel above.
