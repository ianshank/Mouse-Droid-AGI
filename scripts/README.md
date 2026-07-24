# Operational Scripts

Deploy, validate, and operate the rover. Highlights:

- **CI / validation** — `ci.sh`, `validate.py` (spec harness), `check_config_compat.py`,
  `check_branch_coverage.py`, `validate_pillar.sh`
- **Jetson ops** — `jetson_full_validation.sh`, `jetson_smoke_test.sh`, `deploy_jetson.sh`, `docker_deploy.sh`
- **Sensors / mission** — `verify_sensors.py`, `translate_mission.py`, `ask_rover.py`, `preflight_check.sh`
- **systemd units** — `mousedroid.service`, `mousedroid-docker.service`, `mousedroid-trend.{service,timer}`

Full local CI gate: `bash scripts/ci.sh`. Operator runbooks live in [`../docs/runbooks/`](../docs/runbooks/).
