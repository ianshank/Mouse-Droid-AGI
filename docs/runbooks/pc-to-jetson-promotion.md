# Runbook — PC-to-Jetson promotion and rollback

Promote a reviewed commit from the PC to the rover, prove it, and roll it back
without a rebuild or a network. Feature: **F-051** (PC-to-rover delivery).
Change bundle: `openspec/changes/mouse-droid-jetson-onnx-delivery/`.

## Where this sits among the deploy docs (read before adding a fifth path)

Four documents already describe getting code onto this rover. This runbook is
**not** a replacement for any of them; it is the promotion-and-rollback slice,
and it defers everywhere else:

| Document | Owns | Overlap with this runbook |
| --- | --- | --- |
| `docs/deployment.md` | Flash, NVMe, service install, probe-first bring-up | The container/systemd spine. This runbook assumes it is already done. |
| `docs/runbooks/jetson-full-bringup.md` | First bring-up of a dead-board rover, probe-first ordering | Preconditions only. A rover that has never come up is not a promotion. |
| `docs/runbooks/jetson-claude-pilot-deploy.md` | The `--force-recreate` recipe for the LLM tier, image-SHA authority | The recreate step and the "read the SHA out of the record" rule are that document's; not restated here. |
| `docs/planning/JETSON_DEPLOY_RUNBOOK.md` | Historical v0.3-era deploy narrative | Superseded for anything operational; kept for provenance. |

**What is only here:** the dirty-target refusal and WIP preservation, the
`--strict-health` promotion gate, and the offline rollback drill.

## 0. Preconditions

- Bench time. The on-rover phase is scheduled **behind F-008** per the F-024
  hardware-priority rule ("while F-008 is bench-blocked"), and competes for the
  same bench.
- `ssh` to the rover works; `/etc/mousedroid/docker.env` exists, owner-only
  (`600`). **Presence-check secrets, never echo them.**
- **Disk:** `df -h /` + `docker system df -v`, **>= 8 GiB free**; reclaim with
  `scripts/jetson_disk_cleanup.sh`.
- **Memory:** record `swapon --show` + `zramctl`. The container cap is 6 GiB of
  ~7.4 GiB usable RAM, and swap is deliberately unbounded — the reasoning, and
  the three knobs to change together if that is ever revisited, are recorded in
  `docker-compose.jetson.yml` under `RESOURCE BUDGET DECISION`.
- Note where Prometheus/Grafana/Loki run: host-side services compete with the
  container for the same RAM.

## 1. Preserve rover-local work — before anything destructive

`/opt/mousedroid` is a **bind-mounted git checkout** with an editable install,
and it routinely holds real uncommitted bench work. `rsync --delete` destroys
it, so `scripts/deploy_remote.sh` now refuses to run against a dirty target:

```bash
# Refuses, listing every uncommitted path, and never reaches rsync:
bash scripts/deploy_remote.sh "$JETSON_HOST" --code-only

# Preserve, then sync:
bash scripts/deploy_remote.sh "$JETSON_HOST" --code-only --confirm-dirty
```

With `--confirm-dirty` the guard (`scripts/rover_wip_guard.sh`, piped over
`ssh` so it does not have to already be on the rover) does two things, in this
order:

1. **Archives off-device** — a whitespace-insensitive diff (`git diff
   --ignore-all-space HEAD`), the `git status` inventory, and the untracked
   files' *contents* — streamed straight to
   `MOUSEDROID_DEPLOY_ARCHIVE_DIR` (default `~/.mousedroid/rover-wip`) on the
   operator's machine. The transfer is then verified (valid gzip, expected
   members) and the sync is refused if it is not intact. A branch on a rover
   whose microSD dies is not a backup; this is the artifact that survives.
2. **Commits** the same state to `rover/wip-<date>`, leaving the checkout clean
   and the operator able to keep working from the rover's own history. A second
   rescue on the same day gets its own branch rather than landing on the first.

Hard rules, enforced by `tests/unit/scripts/test_deploy_remote_guard.py`:
**`git clean` is never run**, and rsync-delete never runs over unpreserved work.
A target the guard cannot *read* (the known root-ownership drift makes
`git status` fail) is a **refusal**, not a "clean" — repair ownership first.

The archive may contain rover-local files, so treat it as sensitive: it stays on
the operator machine and is never published as a CI artifact.
`/etc/mousedroid/docker.env` is outside the sync target and is never
transferred.

## 2. Pre-tag the rollback anchor — before the rebuild

This is the established anchor, and it is what makes step 5 work offline:

```bash
docker tag mousedroid:jetson "mousedroid:jetson-rollback-$(date -u +%Y%m%d)"
git -C /opt/mousedroid rev-parse HEAD > ~/rollback-commit.txt   # the other half
```

Record **both** halves. The image tag alone rolls back the baked layers; the
bind-mounted checkout is the running code, so the commit is what actually
reverts behaviour.

## 3. Promote

Sync (step 1), then recreate the container per
`docs/runbooks/jetson-claude-pilot-deploy.md`. Nothing in the promotion path
introduces a release directory or a `current`/`previous` symlink: the editable
install resolves through an absolute path, so replacing the mount root with a
symlink farm breaks it, and would also change what `MOUSEDROID_INSTALL_DIR`
means for `scripts/sync_jetson_overlay.sh`, `scripts/preflight_check.sh` and
`scripts/docker_deploy.sh`.

New runtime fields are selected through `MOUSEDROID_*` environment variables in
`/etc/mousedroid/docker.env`, **never** by adding a `world_model:` block to a
tracked `config/*.yaml`: at the schema SHA that `config-compat` pins,
`WorldModelConfig` is a plain `BaseModel` with `extra="ignore"`, so such a key
would pass the gate and then be silently dropped — worse than a hard failure.

## 4. Prove it — `--strict-health`

```bash
bash scripts/docker_deploy.sh --health-only --strict-health
```

Default runs are unchanged (bring-up still gets a warn when telemetry is not yet
listening). Under `--strict-health` the run **fails**, and the prior release
stays active, on any of:

- a dead telemetry endpoint;
- an observed ORT execution provider other than the configured primary;
- a model digest that does not match `model_sha256` in
  `deployments/jetson-image.json` — including the case where that field is
  `null` while `world_model.engine` is `onnx_trt`. "Cannot tell" is not "fine".

Under the default `engine: torch` the provider and digest legs report `n/a` and
pass: there is no ONNX artifact in the 30 Hz path to prove. That is expected —
the task-2.1 decision gate closed against the ONNX fast path (end-to-end ceiling
1.0016x–1.0039x), so this is infrastructure that is correct *if* the engine is
ever enabled, not a live switch.

## 5. Rollback — offline, no rebuild

> **STATUS: this drill has NOT been run.** It requires the physical rover with
> networking disabled, and the on-rover phase is scheduled behind F-008. The
> procedure below is the intended one and is written to be followed exactly;
> until an operator runs it and records the result, treat it as **unverified**.
> Task 8.7 of the bundle stays open. Run it **before** the prior image becomes
> eligible for pruning — a pruned anchor is a rollback that cannot happen.

```bash
# 1. Disconnect networking first: the point is to prove no pull is needed.
# 2. Restore the image anchor and the commit, in that order.
docker tag "mousedroid:jetson-rollback-<date>" mousedroid:jetson
git -C /opt/mousedroid checkout "$(cat ~/rollback-commit.txt)"
docker compose -f docker-compose.jetson.yml up -d --force-recreate
# 3. Re-prove before re-enabling actuation.
bash scripts/docker_deploy.sh --health-only --strict-health
```

Rollback must clear container health, telemetry health, config parse, model
digest and the provider probe **before** actuation is re-enabled, and must write
a timestamped report. Leave the service in a known state either way.

## 6. Re-pin the deploy record — post-merge only

`deployments/jetson-image.json` is the per-platform record, and `config-compat`
`git worktree`s the SHA in it. Re-pinning to a **feature-branch** SHA reproduces
the `9c31968` failure the record itself documents, which killed that gate
repo-wide when the branch was deleted. So:

- during the campaign, leave the record untouched;
- after the PR merges, re-pin to the squash-merge **trunk** commit, and fill
  `model_sha256` / `ort_provider` in the same edit if real values exist by then
  (both are explicitly `null` today, and the record says why);
- run `bash scripts/repin_tags.sh` for the plan and `--push` (an operator
  action) so the SHA has **tag** reachability, not branch reachability. The
  record states the current pin has zero tags and only stale branch carriers.

## Related

- `docs/runbooks/jetson-onnx-benchmark.md` — measuring the stage this promotion
  path would carry.
- `docs/runbooks/jetson-full-validation.md` — cold-then-warm on-device
  validation.
- `docs/runbooks/worktrees.md` — multi-agent worktree isolation on the PC side.
