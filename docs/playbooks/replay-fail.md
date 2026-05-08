# Phase 2 Replay Loop Failure Playbook

Use this playbook when the Phase 2 sim/real replay loop misbehaves —
schema mismatches in the LMDB store, mixer ratio drift away from
`alpha_target`, BC aux loss explosions during PPO, or `LMDBReplayReader`
chunk yields that don't match the operator's mental model.

## What This Covers

- Schema-version mismatch in LMDB experience records
- Mixer realized α drifting from `cfg.training.replay_mixer.alpha_target`
- BC aux loss explosions in `train_constitutional_rl.py` (gradient norms)
- Empty / missing LMDB store on a fresh Jetson
- LMDB lock contention between rover (writer) + export script (reader)
- Live triage: enabling `debug_log_every_n` on a running rover

## First Checks

1. Confirm Phase 2 is actually opted in (defaults are sim-only):
   ```bash
   docker exec mousedroid python3 -c "
   from mousedroid.config.loader import load_settings
   from mousedroid.validation.runtime import resolve_runtime_config_paths
   cfg = load_settings(*resolve_runtime_config_paths())
   print('alpha_target:', cfg.training.replay_mixer.alpha_target)
   print('debug_log_every_n:', cfg.training.replay_mixer.debug_log_every_n)
   print('experience.path:', cfg.experience.path)
   "
   ```
   Default `alpha_target=0.0` means the mixer pulls only synthetic data —
   any 'replay' fault here is misconfiguration, not a runtime failure.
2. Inspect the `mixer_ratio_check` cadence in the journal:
   ```bash
   docker logs mousedroid --since 10m 2>&1 | grep mixer_ratio_check | tail -5
   ```
   Each line shows `step`, `current_alpha` (ramped value), `realized_alpha`
   (empirical fraction), `real_drawn`, `sim_drawn`. Drift > 1% over ≥ 1000
   draws is a real signal.
3. Inspect the reader's running counters from `/metrics`:
   ```bash
   curl -s http://192.168.55.1:8080/metrics | grep -E 'replay|skipped_schema'
   ```
4. Read the on-disk LMDB stats directly:
   ```bash
   docker exec mousedroid python3 -c "
   import lmdb
   env = lmdb.open('/home/jetson/mousedroid_experience', readonly=True, lock=False)
   with env.begin() as txn:
       print('records:', txn.stat()['entries'])
   env.close()
   "
   ```
5. **Live-triage knob** — flip `debug_log_every_n` to surface
   per-N-operations DEBUG lines without a rebuild:
   ```yaml
   # /etc/mousedroid/jetson_production.yaml on the host:
   training:
     replay_mixer:
       debug_log_every_n: 100   # one mixer_draw + one replay_chunk_decoded per 100 ops
   ```
   Then `sudo systemctl restart mousedroid-docker.service` so
   `sync_jetson_overlay.sh` re-stages the overlay. Reset to `0` after
   triage — DEBUG lines flood at high mixer rate.

## Remediation Steps

1. **Schema-mismatch counter climbing**: the rover's running code is on a
   newer `MouseDroidExperienceRecord.schema_version` than the on-disk
   records were written with. Two options:
   - Migrate the LMDB store with a one-shot Python script that
     deserialises old records and re-writes them under the new version.
   - Wipe the store and restart episode logging:
     `docker exec mousedroid rm -rf /home/jetson/mousedroid_experience`
     (only safe if the data is already exported via
     `scripts/export_experience_to_training.py`).
2. **Mixer realized α drifts**: check `current_alpha` vs `alpha_target` —
   during the first `alpha_ramp_steps` (default 1000) the realized α is
   *expected* to be below target (linear ramp). Drift after ramp-end
   indicates the real source is exhausted (`real_exhausted` counter
   > 0); ramp `alpha_target` down or extend `alpha_ramp_steps`.
3. **BC aux loss explosion**: if `train_constitutional_rl.py` reports a
   gradient norm > 100, the real-action distribution is far from the
   policy's. Reduce `cfg.offline_rl.real_supervised_weight` (default 0.0
   = off) and re-run. Add structured logs at the loss site if not
   already present.
4. **Empty LMDB on a fresh Jetson**: `LMDBReplayReader.stream` logs a
   single `replay_empty_db` warning and yields no chunks. This is the
   intended safe behaviour — the mixer falls back entirely to sim. Once
   the rover has logged ≥ a few episodes, the next training run will
   pick them up automatically.
5. **Reader/exporter lock contention**: both
   `LMDBReplayReader` and `scripts/export_experience_to_training.py`
   open the env with `readonly=True, lock=False`. The rover writer holds
   no lock that blocks them. If you see `lmdb.LockError`, something else
   (probably an old `mousedroid` process from a stale container) is
   holding the lock. `docker compose -f docker-compose.jetson.yml down
   && up` clears it.

## Cross-Reference

- [`src/mousedroid/training/replay/lmdb_reader.py`](../../src/mousedroid/training/replay/lmdb_reader.py) — async chunked reader, env opened once per stream() call.
- [`src/mousedroid/training/replay/mixer.py`](../../src/mousedroid/training/replay/mixer.py) — sim/real mixer + alpha ramp; `mixer_ratio_check` + `mixer_draw` log lines.
- [`src/mousedroid/experience/record.py`](../../src/mousedroid/experience/record.py) — `MouseDroidExperienceRecord.SCHEMA_VERSION`.
- [`config/jetson_production.yaml`](../../config/jetson_production.yaml) — `training.replay_mixer.*` block.
- [`scripts/export_experience_to_training.py`](../../scripts/export_experience_to_training.py) — LMDB → msgpack-gz exporter.
- [`tests/integration/test_phase2_replay_pipeline.py`](../../tests/integration/test_phase2_replay_pipeline.py) — 10-episode synthetic LMDB acceptance test.
- [`tests/regression/test_phase2_rssm_golden.py`](../../tests/regression/test_phase2_rssm_golden.py) — golden RSSM loss-curve regression.
