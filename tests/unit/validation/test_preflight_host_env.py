"""Unit tests for the ``host_env_keys`` preflight check (F-017, WS-3.1).

The check is WARN-only by contract: a missing override is operator-actionable
drift (rerun ``scripts/host_bootstrap.sh``), never a FAIL. Values must never
leak into the result detail — only key NAMES.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import HostEnvConfig, Settings
from mousedroid.validation.preflight import (
    PreflightStatus,
    _check_host_env_keys,
    _parse_env_keys,
    run_preflight,
)

_CANARY_VALUE = "super-secret-value-never-logged"


def _cfg(tmp_path: Path, *, enabled: bool = True, mock: bool = False) -> Settings:
    # Construct in mock mode (satisfies the distance-sensor validator) and
    # flip the flag post-init — the same pattern cli/preflight.py uses for
    # its --mock-hardware override.
    cfg = Settings(mock_hardware=True)
    cfg.mock_hardware = mock
    cfg.host_env = HostEnvConfig(
        enabled=enabled,
        env_file=tmp_path / "docker.env",
        template_file=tmp_path / "docker.env.example",
    )
    return cfg


class TestParseEnvKeys:
    def test_extracts_keys_and_discards_values(self) -> None:
        keys = _parse_env_keys(f"A=1\nB={_CANARY_VALUE}\n")
        assert keys == {"A", "B"}

    def test_skips_blanks_comments_and_valueless_lines(self) -> None:
        text = "\n# comment\nJUSTTEXT\nKEY=1\n  # indented comment\n"
        assert _parse_env_keys(text) == {"KEY"}

    def test_tolerates_export_prefix(self) -> None:
        assert _parse_env_keys("export KEY=1\n") == {"KEY"}


class TestCheckHostEnvKeys:
    async def test_disabled_config_is_ok_noop(self, tmp_path: Path) -> None:
        result = await _check_host_env_keys(_cfg(tmp_path, enabled=False))
        assert result.status is PreflightStatus.OK
        assert "disabled" in result.detail

    async def test_absent_config_block_is_ok_noop(self) -> None:
        cfg = Settings(mock_hardware=True)
        cfg.mock_hardware = False
        assert cfg.host_env is None
        result = await _check_host_env_keys(cfg)
        assert result.status is PreflightStatus.OK

    async def test_mock_hardware_short_circuits_ok(self, tmp_path: Path) -> None:
        result = await _check_host_env_keys(_cfg(tmp_path, mock=True))
        assert result.status is PreflightStatus.OK
        assert "mock_hardware" in result.detail

    async def test_superset_deployment_is_ok(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cfg.host_env.template_file.write_text("A=1\nB=2\n", encoding="utf-8")
        cfg.host_env.env_file.write_text("A=x\nB=y\nEXTRA=z\n", encoding="utf-8")
        result = await _check_host_env_keys(cfg)
        assert result.status is PreflightStatus.OK

    async def test_missing_keys_warn_with_names_only(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cfg.host_env.template_file.write_text(
            f"MOUSEDROID_LLM__ENABLED=true\nKEPT={_CANARY_VALUE}\n", encoding="utf-8"
        )
        cfg.host_env.env_file.write_text(f"KEPT={_CANARY_VALUE}\n", encoding="utf-8")
        result = await _check_host_env_keys(cfg)
        assert result.status is PreflightStatus.WARN
        assert "MOUSEDROID_LLM__ENABLED" in result.detail
        assert _CANARY_VALUE not in result.detail, "values must never reach the detail"

    async def test_missing_template_warns_not_fails(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cfg.host_env.env_file.write_text("A=1\n", encoding="utf-8")
        result = await _check_host_env_keys(cfg)
        assert result.status is PreflightStatus.WARN
        assert "template" in result.detail

    async def test_missing_env_file_warns_with_remediation(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        cfg.host_env.template_file.write_text("A=1\n", encoding="utf-8")
        result = await _check_host_env_keys(cfg)
        assert result.status is PreflightStatus.WARN
        assert "host_bootstrap" in result.detail

    async def test_never_fails(self, tmp_path: Path) -> None:
        # Exhaustive over the failure-ish states: neither file, only template,
        # only env, drifted — all must stay OK/WARN (FAIL is reserved for
        # driver crashes elsewhere in preflight).
        cfg = _cfg(tmp_path)
        for setup in range(4):
            for f in (cfg.host_env.template_file, cfg.host_env.env_file):
                f.unlink(missing_ok=True)
            if setup in (1, 3):
                cfg.host_env.template_file.write_text("A=1\n", encoding="utf-8")
            if setup in (2, 3):
                cfg.host_env.env_file.write_text("B=1\n", encoding="utf-8")
            result = await _check_host_env_keys(cfg)
            assert result.status is not PreflightStatus.FAIL, f"setup {setup} FAILed"


class TestDispatchIntegration:
    async def test_check_is_dispatchable_by_name(self, tmp_path: Path) -> None:
        report = await run_preflight(_cfg(tmp_path, mock=True), check_names={"host_env_keys"})
        assert [c.name for c in report.checks] == ["host_env_keys"]
        assert report.overall_status is PreflightStatus.OK

    async def test_warn_degrades_but_never_fails_report(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)  # both files absent -> WARN
        report = await run_preflight(cfg, check_names={"host_env_keys"})
        assert report.overall_status is PreflightStatus.DEGRADED


@pytest.mark.parametrize("field", ["enabled", "env_file", "template_file"])
def test_host_env_config_fields_have_defaults(field: str) -> None:
    # Backwards compatibility: a bare block must load with schema defaults.
    cfg = HostEnvConfig()
    assert getattr(cfg, field) is not None
    assert cfg.enabled is False, "the check must be OFF by default"
