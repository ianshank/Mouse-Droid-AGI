# Model Artifact Integrity Failure Playbook

Use this playbook when `ModelArtifactDigestMismatch` or
`CloudWeightUpdateDigestMismatch` pages: a model artifact failed its SHA-256
manifest check and was refused. Both page because the consequence of loading
the wrong weights is wrong inference, silently.

Every command below runs **on the rover**. None of them changes state except
the steps under Remediation, which say so.

## What This Covers

- `ModelArtifactDigestMismatch{artifact="bdi_weights"}` — the BDI weight set
  failed its check at boot. With `cognitive.fallback_to_mcts` enabled (the
  production default) the rover **kept running, degraded to the MCTS
  planner**, and stays degraded until it is restarted with verified weights.
  The alert keeps firing for that whole time by design.
- `ModelArtifactDigestMismatch{artifact="world_model_onnx"}` — listed for
  completeness, but **expect it never to fire**. A world-model refusal is not
  caught: the process exits before `/metrics` exists. It shows up instead as
  `TelemetryDown` plus a container crash loop — start at First Checks step 2.
- `CloudWeightUpdateDigestMismatch{repo_id=...}` — an over-the-air weight
  update was downloaded, failed its check, and the swap was refused. The
  rover is **still on its previous weights**. This is an event, not a
  degraded state: the alert resolves on its own after 15 minutes.

## First Checks

1. Find the refusal in the journal. Event names are `<artifact>_sha256_mismatch`,
   so one grep covers all three cases:
   ```bash
   docker logs mousedroid --since 1h 2>&1 \
     | grep -E 'bdi_weights_sha256_mismatch|world_model_onnx_sha256_mismatch|cloud_weight_update_sha256_mismatch|refusing to load'
   ```
   The `refusing to load ...` line names the file, the manifest it was checked
   against, and the `repo@revision` it came from.
2. If the container is crash-looping, it is almost certainly the world model:
   ```bash
   docker ps -a --filter name=mousedroid --format '{{.Status}}'
   docker logs mousedroid --tail 50 2>&1 | grep -E 'ArtifactIntegrityError|refusing to load world_model_onnx'
   ```
3. If `bdi_weights` fired, confirm the rover really is on the fallback planner:
   ```bash
   docker logs mousedroid 2>&1 | grep cognitive_core_init_failed_falling_back_to_mcts
   ```
   `error_type=ArtifactIntegrityError` on that line means the digest gate, not
   some other construction failure, put it there.
4. Read the counters directly. The port is the one the rover resolved from its
   own config, read from the env file its entrypoint writes — not a guessed
   `8080`:
   ```bash
   PORT=$(docker exec mousedroid sh -c 'cat "${MOUSEDROID_HEALTHCHECK_ENV_FILE:-/run/mousedroid.env}"' \
          | sed -n "s/^MOUSEDROID_RESOLVED_TELEMETRY_PORT='\([0-9]*\)'$/\1/p")
   curl -s "http://127.0.0.1:${PORT}/metrics" | grep sha256_mismatches
   ```

## Remediation Steps

Do **not** turn off `cognitive.require_sha256_manifest` or
`world_model.onnx_require_sha256_manifest` to get the rover running. The gate
did its job; disabling it loads exactly the weights it just refused.

1. Decide which side is wrong. If the cached file is corrupt or stale, the
   manifest is right. If a new model was published without regenerating its
   `sha256.txt`, the manifest is wrong. The `refusing to load` line from First
   Checks step 1 names both paths.
2. **Cached file wrong** — delete the cached artifact it names (this is a
   state change) and restart, so it is fetched and verified again:
   ```bash
   sudo systemctl restart mousedroid-docker
   ```
3. **Manifest wrong** — regenerate and republish `sha256.txt` alongside the
   artifact in its repository, then restart as above. Never edit the cached
   manifest on the rover to match a file you have not independently verified.
4. `ModelArtifactDigestMismatch` clears only when the process restarts with
   verified weights, because the counter lives for the life of the process.
   `CloudWeightUpdateDigestMismatch` clears on its own; investigate the
   publishing side before re-publishing.

## Cross-Reference

- Alert rules: `config/prometheus/alerts.yml` (group `mousedroid_artifact_integrity`).
  Why the boot-path rule checks the counter's value rather than its rate is
  documented on the rule itself and proven in `config/prometheus/alerts_test.yml`.
- The gate: `src/mousedroid/utils/artifact_integrity.py` (`enforce_artifact_digests`).
- The OTA gate: `src/mousedroid/cloud/weight_update_poller.py`.
- Why a BDI refusal degrades rather than refuses to boot:
  `src/mousedroid/factory/orchestrator.py` (`fallback_to_mcts`, recorded there
  as a known gap).
- Crash loops and `TelemetryDown` in general: `docs/playbooks/bringup-fail.md`.
