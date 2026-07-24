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

Secret hygiene and dependency posture are enforced in CI:

- **Secret scanning** — `.gitleaks.toml` + [`docs/runbooks/secret-scanning.md`](docs/runbooks/secret-scanning.md)
- **Dependency audit** — the `pip-audit` job in `.github/workflows/ci.yml`
- **Advisory-stage tracking** — `.github/advisory_stages.yaml`

Secrets are never committed; per-host runtime secrets live only in
`/etc/mousedroid/docker.env` (documented, without live values, in
`config/docker.env.example`).

## Safety vs. Security

The rover's *safety* boundary — the Three Laws / emergency-stop path, kept LLM-
and training-free inside the deterministic 30 Hz control loop — is a distinct
concern documented in [`docs/CHARTER.md`](docs/CHARTER.md). Report
safety-relevant defects through the same private channel above.
