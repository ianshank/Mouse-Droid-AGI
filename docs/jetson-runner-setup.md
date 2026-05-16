# Jetson Self-Hosted Runner Setup

Operator runbook for registering the MouseDroidAGI Jetson Orin Nano as a
GitHub Actions self-hosted runner so the
[`.github/workflows/jetson-nightly.yml`](../.github/workflows/jetson-nightly.yml)
workflow (PR #62) can actually run the Ten Pillars campaign on real
hardware.

## Prerequisites

- Jetson Orin Nano powered up, on the same network as the operator workstation,
  reachable via SSH (memory entry: `192.168.55.1`).
- Logged in as the deploy user (default: `jetson`) with `sudo` rights.
- `mousedroid-docker.service` already installed and running — the runner's
  systemd unit `Requires=` it so jobs that shell into the rover container
  fail fast if the container is down.
- `git`, `curl`, `tar`, `systemctl`, `sudo`. All ship with the L4T base image.
- One-time **runner registration token** from
  <https://github.com/ianshank/Mouse-Droid-AGI/settings/actions/runners/new>.
  Tokens rotate ~1 hour after issue, so install promptly.

## Install

```bash
# On the Jetson, with the repo checked out at /opt/mousedroid:
cd /opt/mousedroid
RUNNER_TOKEN="AAAAA…"  # paste from the GitHub UI
bash scripts/jetson-runner-install.sh
```

The script downloads `actions-runner-linux-arm64-2.319.1.tar.gz`, registers
with the labels `self-hosted,jetson,linux,arm64`, installs the systemd unit
from `scripts/github-actions-runner.service.template`, and starts the
service. Optional environment overrides (sane defaults shown):

| Var | Default | Purpose |
| --- | --- | --- |
| `GITHUB_REPO` | `ianshank/Mouse-Droid-AGI` | Slug to register against |
| `RUNNER_VERSION` | `2.319.1` | actions/runner release tag |
| `RUNNER_LABELS` | `self-hosted,jetson,linux,arm64` | What the workflow `runs-on:` matches |
| `RUNNER_INSTALL_DIR` | `/opt/actions-runner` | Where the runner unpacks |
| `RUNNER_USER` | `jetson` | systemd `User=` |
| `RUNNER_NAME` | `$(hostname)-jetson` | UI display name |

Always do a dry run first to confirm the plan before pasting the token:

```bash
bash scripts/jetson-runner-install.sh --dry-run
```

The dry run prints every step it *would* take and exits 0 with no side
effects.

## Verify

```bash
# 1. Service is up:
sudo systemctl status actions-runner-mousedroid.service
# Expect: Active: active (running)

# 2. Runner appears in the GitHub UI:
#    https://github.com/ianshank/Mouse-Droid-AGI/settings/actions/runners
#    Status: Idle, labels include `self-hosted` and `jetson`.

# 3. Trigger the nightly workflow manually to confirm pickup:
gh workflow run jetson-nightly.yml --ref main
gh run watch                                        # tail it
```

A green run produces an artifact named `ten-pillars-<stamp>` containing the
per-pillar logs and `ten_pillars.log` summary (30-day retention per
`.github/workflows/jetson-nightly.yml`).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `systemctl status` reports `Active: failed (Result: exit-code)` | Token rotated before install completed | Re-run with a fresh token |
| Workflow stays `queued` forever | Runner is registered but the systemd unit isn't running | `sudo systemctl restart actions-runner-mousedroid.service`; confirm `journalctl -u actions-runner-mousedroid -n 200` |
| `validate_pillar.sh` fails inside the runner | `mousedroid-docker.service` is down | The runner's `Requires=` should restart it; if not, check `sudo systemctl status mousedroid-docker.service` and the rover smoke (`scripts/jetson_full_smoke_run.sh`). See [`bringup-fail.md`](playbooks/bringup-fail.md). |
| Runner can't pull the L4T image | `docker login` not done as `jetson` user | `sudo -u jetson docker login` once; then re-run |
| Runner registers but disappears from UI after reboot | systemd unit not enabled | `sudo systemctl enable actions-runner-mousedroid.service` |
| `--dry-run` reports `systemd template missing` | Repo state damaged or partial clone | `git pull` in `/opt/mousedroid`; the template ships at `scripts/github-actions-runner.service.template` |

To uninstall:

```bash
sudo systemctl disable --now actions-runner-mousedroid.service
sudo rm /etc/systemd/system/actions-runner-mousedroid.service
sudo systemctl daemon-reload
cd /opt/actions-runner
sudo -u jetson ./config.sh remove --token <REMOVAL_TOKEN>
sudo rm -rf /opt/actions-runner
```

The removal token comes from the same page as the registration token.

## Cross-Reference

- [`scripts/jetson-runner-install.sh`](../scripts/jetson-runner-install.sh) — the installer.
- [`scripts/github-actions-runner.service.template`](../scripts/github-actions-runner.service.template) — systemd unit (placeholders substituted at install time).
- [`.github/workflows/jetson-nightly.yml`](../.github/workflows/jetson-nightly.yml) — the workflow that consumes this runner. **Status: required check** (promoted in Tier B1 after 7 consecutive green nights — see "Promotion Observation Log" below).
- [`scripts/validate_pillar.sh`](../scripts/validate_pillar.sh) — the Ten Pillars dispatcher the workflow runs. Operator can run it ad-hoc on the Jetson host outside the runner with `bash scripts/validate_pillar.sh all`.
- [`docs/playbooks/bringup-fail.md`](playbooks/bringup-fail.md) — full-rover bringup runbook (referenced when the runner can't shell into a healthy container).

## Promotion to Required Check — ✅ Landed (Tier B1)

The `ten-pillars` job is now a **required check** on `main`. After this PR
landed:

1. ✅ `continue-on-error: true` removed from the `ten-pillars` job block.
2. ✅ `Report status` step's trailing `exit 0` changed to
   `exit "${PILLAR_RC:-1}"` so the job's exit code reflects pillar failures.
3. ✅ Branch protection on `main` configured to require the
   **Ten Pillars on Jetson** check (operator-facing UI step at
   <https://github.com/ianshank/Mouse-Droid-AGI/settings/branches>).

The previous advisory-mode gate is documented in git history (see
`.github/workflows/jetson-nightly.yml` history for commits referencing
"continue-on-error: true").

### How this was earned

Per the original Tier B1 plan, the workflow had to ship 7 consecutive
green nightly runs in advisory mode before promotion. See the
**Promotion Observation Log** section below for the dated run-by-run
record.

### Exit-code semantics (post-promotion)

`validate_pillar.sh` returns:

- **0** — all blocking pillars (safety, world_model, memory, cognitive,
  reward) reported PASS. Non-blocking pillars (curiosity, continual,
  meta, scaling, growth) may be SKIP but must not be FAIL.
- **1** — at least one blocking pillar reported FAIL. **Merges to `main`
  are blocked by branch protection** until the fail clears.
- **2** — precondition error (Docker container down, sync_jetson_overlay
  script missing, etc.). Also blocks merges — operators must investigate
  the artifact's `ten_pillars.log` to distinguish 1 (real test fail) from
  2 (infra problem) and fix accordingly.

### Rollback path

If a wave of false-positive failures pollutes `main`'s merge queue:

1. Revert this PR (re-adds `continue-on-error: true` + reverts the
   exit-code change). Branch-protection check stays configured but the
   job exit code is now always 0 → check is always green → merges flow.
2. Re-disable the required-check requirement at
   <https://github.com/ianshank/Mouse-Droid-AGI/settings/branches> as a
   belt-and-suspenders measure.

## Promotion Observation Log

| Date (UTC) | Run ID | Conclusion | Blocking Pillars (S/W/M/C/R) | Notes |
|---|---|---|---|---|
| _operator_ | _operator_ | _operator_ | _5/5 required_ | _Append rows during the observation window before merging this PR._ |

### Local nightly equivalent

Operators can mirror the nightly check on the Jetson host without waiting
for the cron:

```bash
# Run all pillars sequentially with per-pillar timeout from the script.
bash scripts/validate_pillar.sh all

# Or run only the pytest-marker subset (faster, in-Docker, no probe hardware).
docker exec mousedroid pytest -m pillar --import-mode=importlib --no-cov

# Inspect the markdown summary the workflow uploads as a build artifact:
cat reports/jetson_smoke/$(ls -1 reports/jetson_smoke/ | tail -1)/ten_pillars.log
```

## Security

GitHub Actions self-hosted runners execute arbitrary workflow YAML from the
repo. Two mitigations:

1. **Restrict to non-fork workflows** — in
   <https://github.com/ianshank/Mouse-Droid-AGI/settings/actions>, set
   "Fork pull request workflows" to "Require approval for first-time
   contributors" so an unreviewed external PR cannot run code on the
   Jetson.
2. **Run as a non-root user** — the systemd unit pins `User=jetson`, not
   `root`. The Docker socket is the only privileged surface; the workflow
   only needs read-only access to `mousedroid-docker.service`'s logs.

Rotate the registration token after install (it expires anyway). The
runner's long-lived auth token lives at `/opt/actions-runner/.credentials`
and is `chmod 600` by `config.sh`.
