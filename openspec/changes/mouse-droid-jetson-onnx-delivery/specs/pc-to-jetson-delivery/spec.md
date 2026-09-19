# Spec delta — PC-to-rover delivery and rollback

## MODIFIED Requirements

### Requirement: The deployment spine SHALL remain a bind-mounted git checkout

`/opt/mousedroid` SHALL remain a git checkout bind-mounted at the same path inside the
container (`docker-compose.jetson.yml:79-80`) carrying an editable install
(`Dockerfile.jetson:141`). No release-directory or `current`/`previous` symlink scheme SHALL
be introduced by this change.

An editable install resolves through an absolute path, so replacing the mount root with a
symlink farm breaks it. The same substitution would also change the meaning of
`MOUSEDROID_INSTALL_DIR` for `sync_jetson_overlay.sh`, break
`jetson-nightly.yml:87`, `preflight_check.sh:29-31`, and `docker_deploy.sh:165-169` (which
requires `${INSTALL_DIR}/pyproject.toml` to exist), and discard the rover-local git workflow
that is the repository's actual WIP-preservation mechanism.

#### Scenario: Container recreation after a promotion

- **GIVEN** a promotion that checked out a new commit in `/opt/mousedroid`
- **WHEN** the container is recreated
- **THEN** the editable install resolves and the bind-mounted source is the running code

### Requirement: Rollback SHALL use the established Docker tag anchor and work offline

Before any rebuild, promotion SHALL pre-tag the current image
(`docker tag mousedroid:jetson mousedroid:jetson-rollback-<date>`). Rollback SHALL restore
that tag plus the prior commit checkout, and SHALL require neither network access nor a
rebuild.

Rollback SHALL then require container health, telemetry health, config parse, model digest
and an observed-provider probe before actuation is re-enabled, and SHALL write a timestamped
report. An EXIT trap SHALL leave the service in a known state.

#### Scenario: Rollback with networking disabled

- **GIVEN** a failed promotion and no network
- **WHEN** rollback runs
- **THEN** the prior image tag and commit are restored and the service returns to health

### Requirement: Rover-local work SHALL be preserved before any destructive sync

`scripts/deploy_remote.sh` SHALL refuse to proceed against a dirty target unless the
operator confirms, and SHALL commit rover-local state to a `rover/wip-<date>` branch with an
off-device archive before the `rsync -avz --delete` at `:151` runs.

`git clean` SHALL NEVER be used, and rsync-delete SHALL NEVER run over uncommitted rover
work. Because no shellcheck gate exists in CI, this behaviour SHALL be covered by a Python
test with a real fixture, following `tests/unit/scripts/test_repin_tags.py`.

Target secrets SHALL stay outside any transferred bundle. `/etc/mousedroid` is mounted
read-only (`docker-compose.jetson.yml:82`); `/etc/mousedroid/docker.env` SHALL be preserved
and never transferred.

#### Scenario: Dirty target

- **GIVEN** uncommitted changes under `/opt/mousedroid`
- **WHEN** a sync is attempted
- **THEN** it refuses, and with confirmation creates `rover/wip-<date>` plus an archive first

### Requirement: Promotion health checks SHALL prove provider, digest and telemetry

`scripts/docker_deploy.sh` SHALL gain a `--strict-health` mode that fails on a dead
telemetry endpoint, an observed ORT provider other than the configured primary, or a model
digest mismatch. The default permissive behaviour SHALL be unchanged.

Today the telemetry leg is a `warn` with "(This is normal if telemetry is disabled or still
starting)" (`:116-123`), and the whole function is softened at `:241`
(`health_check || warn "…non-fatal for deployment"`), so even the legs that return non-zero
are downgraded. The container `HEALTHCHECK`
(`scripts/mousedroid_healthcheck.sh`) only checks heartbeat freshness and cannot express
provider state.

Any Python-side guard SHALL raise a named exception, never `assert`
(`Dockerfile.jetson:178` sets `PYTHONOPTIMIZE=1`).

#### Scenario: Provider downgrade blocks promotion

- **GIVEN** a promotion configured for TensorRT
- **WHEN** the strict probe observes CUDA or CPU
- **THEN** promotion fails and the prior release remains active

#### Scenario: Bring-up before telemetry is up

- **GIVEN** the general-purpose helper run without `--strict-health`
- **WHEN** telemetry is not yet listening
- **THEN** the check still warns rather than failing, preserving today's bring-up flows

### Requirement: The deploy record SHALL be extended, not duplicated

Provenance SHALL be recorded in `deployments/<platform>-image.json`, which
`config-compat.yml:8-10` documents as the extensible per-platform matrix, gaining the model
digest and the observed ORT provider. No parallel manifest format SHALL be introduced.

`tests/regression/test_config_no_dup_keys_and_deploy_record.py` pins
`("sha","platform","image_tag")` plus a full 40-hex SHA, and
`scripts/check_config_compat.py` worktrees that SHA. The record SHALL therefore be re-pinned
**post-merge only**, to the squash-merge trunk commit: re-pinning to a feature-branch SHA
reproduces the `9c31968` failure the record itself documents, which killed the
`config-compat` gate repo-wide.

Any SHA recorded SHALL satisfy `mouse-droid-deploy-repin`'s pin-reachability spec —
remote-*tag* reachability, not branch reachability. `deployments/jetson-image.json` states
that zero tags currently exist and the pin's only reachability is ten stale feature
branches, so `scripts/repin_tags.sh` (and `--push`, an operator action) SHALL run before any
recorded SHA is relied upon.

#### Scenario: Pre-merge re-pin is refused

- **GIVEN** a feature-branch SHA
- **WHEN** a re-pin is attempted before merge
- **THEN** it is refused and deferred to the post-merge step

### Requirement: No `world_model:` key SHALL be added to a tracked overlay

New runtime fields SHALL live on Pydantic schema defaults, and the FP16/cache profile SHALL
be selected through `MOUSEDROID_WORLD_MODEL__*` environment variables in
`/etc/mousedroid/docker.env`. No tracked `config/*.yaml` SHALL carry a `world_model:` block,
including a new opt-in overlay: `check_config_compat.py` validates changed config files
against the pinned schema, where `WorldModelConfig` is a plain `BaseModel` with
`extra="ignore"`, so such a file passes the gate and has every key silently dropped.

`config/default.yaml` is the deep-merge base for all 17 overlays (`loader.py:76-96`), so a
block there would apply everywhere. More importantly, `config-compat` validates new YAML
against the schema at the pinned SHA, where `WorldModelConfig` is a plain `BaseModel`
(`032942b5…:src/mousedroid/config/schema.py:1597`) with `extra="ignore"` — a new
`world_model.*` key would **pass the gate and then be silently ignored by the pinned
schema**, which is worse than a hard failure from the gate that exists to catch "YAML
merges, rover crash-loops". A new top-level block hard-fails instead.

#### Scenario: The silent-ignore window is avoided

- **GIVEN** new ONNX runtime fields
- **WHEN** `config-compat` runs on the PR
- **THEN** no tracked overlay carries a `world_model:` key and nothing is silently dropped

### Requirement: A TensorRT cache SHALL persist across container recreation

`docker-compose.jetson.yml` SHALL gain a named volume for the engine/timing cache beside
`mousedroid_experience` and `promtail_positions` (`:165-169`). The directory SHALL come from
`cfg.jetson.tensorrt_cache_dir`, not a literal in the compose file.

Nothing currently persists a TensorRT cache or `HF_HOME`. For the record: there is **no**
weights bind mount — weights persist transitively via `WORKDIR /opt/mousedroid`
(`Dockerfile.jetson:49`) plus the relative `onnx_cache_dir` default (`world_model.py:68`)
resolving inside the `/opt/mousedroid` mount.

#### Scenario: Warm cache survives `--force-recreate`

- **GIVEN** an engine plan built on the rover
- **WHEN** the container is recreated
- **THEN** the next start is a warm build and is recorded as such

## ADDED Requirements

### Requirement: The GitHub trust boundary SHALL be pinned, not changed

All five workflows already declare top-level `permissions: contents: read`
(`ci.yml:49-50`, `config-compat.yml:28-29`, `harness.yml:35-36`,
`jetson-nightly.yml:41-42`, `release.yml:28-29`), no workflow references `secrets.*`, there
is no `pull_request_target` trigger, and the only protected environment is `pypi`
(`release.yml:193`). This change SHALL add a regression pin so that stays true, rather than
describing it as a change.

A self-hosted Jetson runner already exists — `jetson-nightly.yml:53`
(`runs-on: [self-hosted, jetson]`), a validation workflow, not a deploy one. This change
SHALL NOT add a job to it. If a Jetson deploy job is ever added, it SHALL require a
protected `jetson-production` environment with reviewers, branch restrictions and
concurrency of one; until then promotion stays an operator-run PC command.

Reports published as artifacts SHALL redact hostnames, usernames, paths and secrets.

#### Scenario: A PR from an untrusted fork

- **GIVEN** a pull request
- **WHEN** workflows run
- **THEN** contents permission is read-only and no job receives rover credentials

#### Scenario: The existing self-hosted runner is untouched

- **GIVEN** this change
- **WHEN** `jetson-nightly.yml` is inspected
- **THEN** no promotion job has been added to the self-hosted runner

### Requirement: On-rover work SHALL be sequenced behind F-008

F-008 ("USB-C rover smoke passes on the physical Jetson") is `todo`, and the F-024 rule
states hardware readiness preempts in-flight software streams. No path in this change is
blocked by `freeze_gate.py` — `.claude/workforce.yaml` freezes only
`src/mousedroid/arm/**` — but the on-rover phase competes with F-008 for the same bench and
SHALL be scheduled behind it, in the house idiom "while F-008 is bench-blocked".

#### Scenario: Bench contention

- **GIVEN** F-008 not yet `done`
- **WHEN** the on-rover phase is scheduled
- **THEN** it yields to F-008 unless the operator explicitly allocates bench time
