# Voice Failure Playbook

Use this playbook when the Rocky voice stage fails in `scripts/jetson_full_smoke_run.sh` or
`scripts/verify_sensors.py`.

## What This Covers

- Piper model missing or not loadable
- Speaker device unavailable
- Wrong production voice mapping or thresholds
- Jetson deployment drift outside the normal service-managed path

## First Checks

1. Confirm the service-managed deployment path is in use:
   `systemctl status mousedroid-docker`
2. Check the current production config on the Jetson host:
   `sudo sed -n '/^voice:/,/^logging:/p' /etc/mousedroid/jetson_production.yaml`
3. Inspect the live container logs:
   `docker logs mousedroid --tail 200 | grep -i "piper\|voice\|speaker"`

## Expected Production Keys

- `voice.tts_model_path`
- `voice.personality_to_model_map`
- `voice.event_intensity_thresholds`
- `voice.output_volume`

If the configured personality is present in `personality_to_model_map`, that path overrides
`tts_model_path`.

## Remediation Steps

1. Verify that the selected model file exists inside the container:
   `docker exec -w /opt/mousedroid mousedroid python3 -c "from mousedroid.config.loader import load_settings; cfg = load_settings('config/jetson_production.yaml'); print(cfg.voice.personality, cfg.voice.resolved_tts_model_path())"`
2. If the model path is wrong, fix `config/jetson_production.yaml` in the repo and restart the service.
3. If the service path is bypassed, run the overlay sync manually once:
   `sudo /opt/mousedroid/scripts/sync_jetson_overlay.sh`
4. Rebuild or restart the container if Piper or the model staging changed:
   `docker compose -f docker-compose.jetson.yml build mousedroid`
   `systemctl restart mousedroid-docker`
5. Re-run the focused checks:
   `python scripts/verify_sensors.py --sensor speaker`
   `bash scripts/jetson_smoke_test.sh voice`

## If The Speaker Device Is Missing

1. Check ALSA devices on the Jetson host:
   `aplay -l`
2. Confirm `/dev/snd` is mounted into the container.
3. Re-seat the USB audio adapter and restart `mousedroid-docker`.

## Exit Criteria

- `bash scripts/jetson_smoke_test.sh voice` passes
- `docker logs mousedroid` shows `piper_tts_model_loaded`
- Voice output is heard or the mock speaker path reports successful sample playback