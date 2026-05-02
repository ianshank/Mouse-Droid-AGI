# promtool Install Playbook

Use this playbook when `bash scripts/ci.sh` reports
`promtool not on PATH - skipping Prometheus rule validation` and you want to
make the Prometheus rule stage enforced rather than skipped.

## What This Covers

- Installing `promtool` (Prometheus's command-line rule + metrics validator)
  on Linux, macOS, and Windows validation hosts
- Verifying the install against the alert rules shipped in
  `config/prometheus/alerts.yml`
- Wiring the install into the local CI run so the `=== Prometheus Rules
  Validation (promtool) ===` stage of `scripts/ci.sh` becomes a hard gate

## Why This Matters

`scripts/ci.sh` already gracefully handles a missing `promtool` (the stage
prints a skip notice and exits successfully). The GitHub Actions `CI`
workflow auto-installs `promtool` from the official release tarball
(`.github/workflows/ci.yml` — see the `Install promtool (graceful skip if
unavailable)` step). However, a missing local install means changes to
`config/prometheus/alerts.yml` are validated only in CI, not before the
commit lands. Installing locally closes that gap.

## First Checks

1. Confirm whether the tool is already on PATH:
   - Linux/macOS: `command -v promtool`
   - Windows (Git Bash): `command -v promtool`
   - Windows (PowerShell): `Get-Command promtool -ErrorAction SilentlyContinue`
2. If absent, choose the matching install path below and follow it.

## Install — Linux (x86_64)

```bash
PROMTOOL_VERSION="2.51.0"
curl -sSLf "https://github.com/prometheus/prometheus/releases/download/v${PROMTOOL_VERSION}/prometheus-${PROMTOOL_VERSION}.linux-amd64.tar.gz" -o /tmp/prometheus.tar.gz
tar -xzf /tmp/prometheus.tar.gz --strip-components=1 -C /tmp "prometheus-${PROMTOOL_VERSION}.linux-amd64/promtool"
sudo install -m 0755 /tmp/promtool /usr/local/bin/promtool
rm /tmp/prometheus.tar.gz /tmp/promtool
```

Verify:

```bash
promtool --version
```

## Install — Linux (aarch64 / Jetson)

```bash
PROMTOOL_VERSION="2.51.0"
curl -sSLf "https://github.com/prometheus/prometheus/releases/download/v${PROMTOOL_VERSION}/prometheus-${PROMTOOL_VERSION}.linux-arm64.tar.gz" -o /tmp/prometheus.tar.gz
tar -xzf /tmp/prometheus.tar.gz --strip-components=1 -C /tmp "prometheus-${PROMTOOL_VERSION}.linux-arm64/promtool"
sudo install -m 0755 /tmp/promtool /usr/local/bin/promtool
rm /tmp/prometheus.tar.gz /tmp/promtool
```

## Install — macOS (Apple Silicon or Intel)

```bash
brew install prometheus
# `brew install prometheus` ships both the server and `promtool`. If you only
# want `promtool`, the official tarball under
# https://github.com/prometheus/prometheus/releases is also fine — substitute
# `darwin-arm64` or `darwin-amd64` for the platform suffix in the Linux
# instructions above.
```

## Install — Windows (10 / 11)

1. Download the official release archive (matches the `2.51.0` version pinned
   in `.github/workflows/ci.yml` so local and CI behaviour stay aligned):
   <https://github.com/prometheus/prometheus/releases/tag/v2.51.0>
   - File: `prometheus-2.51.0.windows-amd64.zip`
2. Extract the archive (e.g. via Explorer → Extract All, or `Expand-Archive`
   in PowerShell). Inside, find `promtool.exe`.
3. Move `promtool.exe` to a stable directory that is on PATH, for example
   `C:\Tools\bin\promtool.exe`.
4. If that directory is not yet on PATH, add it. PowerShell example
   (run as a regular user — `User` scope, no admin required):

   ```powershell
   $tools = "C:\Tools\bin"
   if ($Env:Path -notlike "*$tools*") {
       [Environment]::SetEnvironmentVariable(
           "Path",
           [Environment]::GetEnvironmentVariable("Path", "User") + ";$tools",
           "User"
       )
   }
   ```

5. Open a fresh terminal (so the new PATH is picked up) and verify:

   ```powershell
   promtool --version
   ```

   In Git Bash:

   ```bash
   promtool --version
   ```

## Verify Against the Repo's Alert Rules

From the repo root:

```bash
promtool check rules config/prometheus/alerts.yml
```

Expected output: `SUCCESS: <N> rules found`. Any other output indicates a
syntax or semantics error in `alerts.yml` and must be fixed before the rules
are deployed to a Prometheus server.

To exercise the same flow `scripts/ci.sh` runs:

```bash
bash scripts/ci.sh
```

The Prometheus stage should now print `SUCCESS: <N> rules found` rather than
the skip notice.

## Pin a Specific Version

The CI workflow pins `PROMTOOL_VERSION="2.51.0"` so local installs drift the
moment a newer release is published. To match CI exactly, install the same
version. To upgrade, bump both this playbook and
`.github/workflows/ci.yml`'s `PROMTOOL_VERSION` in the same commit.

## Troubleshooting

- `promtool: command not found` after installing: open a new terminal so the
  shell re-reads PATH, or `source ~/.bashrc` (`source ~/.zshrc` on macOS).
- `promtool check rules` exits with code `3`: this is a **lint warning**, not
  an error. The CI workflow treats `rc == 3` as non-fatal (see the `Validate
  Prometheus metrics format` step in `.github/workflows/ci.yml`) for
  Grafana/alert-rule backwards compatibility. Inspect the warnings; if they
  point at intentional naming choices, leave as-is.
- `promtool check metrics` requires a `metrics_sample.txt` produced by the
  telemetry server. The CI flow generates one programmatically — locally,
  start the server and `curl http://localhost:8000/metrics > metrics_sample.txt`
  before piping into `promtool check metrics`.

## Cross-Reference

- `.github/workflows/ci.yml` — auto-installs `promtool` for ubuntu-latest CI
  runs (graceful skip if download fails)
- `scripts/ci.sh` — local CI entry; the Prometheus stage soft-skips when
  `promtool` is missing
- `config/prometheus/alerts.yml` — the rule file this playbook validates
- `config/prometheus/scrape_*.yml` — scrape configs (not validated by
  `promtool check rules`; use `promtool check config` instead if you ever
  add a top-level `prometheus.yml`)
