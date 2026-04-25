# Jetson Full Smoke Run 20260425T165906Z

- Host: mousedroid
- Repo HEAD: 0cf7ddc
- Branch: hardware/jetson-full-smoke
- Container: mousedroid image=sha256:e50897f10d4cfe64c308c9ea17331c0ecb7d81fbea94a84840af6da879145cab
- Run dir: /opt/mousedroid/reports/jetson_smoke/20260425T165906Z

| Stage | Status | Note |
|-------|--------|------|
| container_health | INFO | see container_health.log |
| system | PASS |  |
| gpio | PASS |  |
| serial | PASS |  |
| camera | EXPECTED-FAIL | rc=1 (non-blocking) |
| audio | PASS |  |
| lidar | PASS |  |
| speaker | EXPECTED-FAIL | rc=124 (timeout after 45s, non-blocking) |
| oled | EXPECTED-FAIL | rc=1 (non-blocking) |
| app_health | PASS |  |
| hardware_pytest | EXPECTED-FAIL | rc=1 (non-blocking) |
| e2e | FAIL | rc=1 |
