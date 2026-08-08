"""Regression pins for the secret-scan gate (F-015, WS-0.4).

Binary-free assertions over the three gate surfaces:

* the ``gitleaks`` CI job in ``.github/workflows/ci.yml`` (blocking since the
  2026-08-07 promotion, full-history, image pinned to an exact patch tag),
* the ``.gitleaks.toml`` allowlist (regex-only — a ``paths`` allowlist would
  blind the scanner to real secrets landing in waived files),
* the advisory ``scripts/ci.sh`` stage (guarded so a missing binary skips
  instead of failing the local gate).

These tests never invoke gitleaks itself — the tool is tested upstream; what
this repo owns is the wiring and the allowlist policy.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10 CI legs
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_GITLEAKS_TOML = _REPO_ROOT / ".gitleaks.toml"
_CI_SH = _REPO_ROOT / "scripts" / "ci.sh"

_PINNED_IMAGE_RE = re.compile(r"^docker://zricethezav/gitleaks:v\d+\.\d+\.\d+$")


def _load_ci_jobs() -> dict:
    """Parse ci.yml and return its jobs mapping.

    Assert via ``jobs`` only — PyYAML maps the workflow ``on:`` key to the
    boolean ``True``, so top-level trigger assertions are a known trap.
    """
    data = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "ci.yml did not parse to a mapping"
    jobs = data.get("jobs")
    assert isinstance(jobs, dict), "ci.yml has no jobs mapping"
    return jobs


class TestGitleaksCiJob:
    """The CI job exists, is blocking, scans full history, pins its image."""

    def test_job_exists_and_is_blocking(self) -> None:
        jobs = _load_ci_jobs()
        assert "gitleaks" in jobs, "gitleaks job missing from ci.yml"
        job = jobs["gitleaks"]
        # Assert the SEMANTIC contract, not the YAML shape: an explicit
        # `continue-on-error: false` is still blocking and must pass.
        assert job.get("continue-on-error") is not True, (
            "gitleaks was promoted advisory -> blocking 2026-08-07 "
            "(docs/runbooks/secret-scanning.md) - re-demoting it to advisory "
            "requires a recorded decision, not a workflow edit"
        )
        assert job.get("needs") == "lint"

    def test_checkout_fetches_full_history(self) -> None:
        job = _load_ci_jobs()["gitleaks"]
        checkout_steps = [
            step
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert checkout_steps, "gitleaks job has no checkout step"
        fetch_depth = checkout_steps[0].get("with", {}).get("fetch-depth")
        assert fetch_depth == 0, "full-history scan requires fetch-depth: 0"

    def test_image_is_pinned_patch_tag(self) -> None:
        job = _load_ci_jobs()["gitleaks"]
        scan_steps = [
            step for step in job["steps"] if str(step.get("uses", "")).startswith("docker://")
        ]
        assert scan_steps, "gitleaks job has no docker:// scan step"
        image = str(scan_steps[0]["uses"])
        assert _PINNED_IMAGE_RE.match(image), (
            f"gitleaks image {image!r} must be pinned to an exact vX.Y.Z patch "
            "tag (never a floating tag like 'latest')"
        )
        args = str(scan_steps[0].get("with", {}).get("args", ""))
        assert "--config=.gitleaks.toml" in args
        assert "--redact" in args, "scan output must never echo secret values"


class TestGitleaksConfig:
    """.gitleaks.toml parses, extends defaults, and allowlists by regex only."""

    def test_config_parses_and_extends_default(self) -> None:
        config = tomllib.loads(_GITLEAKS_TOML.read_text(encoding="utf-8"))
        assert config.get("extend", {}).get("useDefault") is True

    def test_allowlist_is_regex_only_never_paths(self) -> None:
        config = tomllib.loads(_GITLEAKS_TOML.read_text(encoding="utf-8"))
        allowlist = config.get("allowlist", {})
        assert "paths" not in allowlist, (
            "path-based allowlisting is forbidden — it blinds the scanner to "
            "real secrets in waived files (see .gitleaks.toml header)"
        )
        regexes = allowlist.get("regexes", [])
        assert regexes, "allowlist must document the fake-key placeholder regexes"
        for pattern in regexes:
            re.compile(pattern)  # raises on an invalid pattern

    def test_allowlist_covers_documented_placeholders(self) -> None:
        config = tomllib.loads(_GITLEAKS_TOML.read_text(encoding="utf-8"))
        patterns = [re.compile(p) for p in config["allowlist"]["regexes"]]
        for placeholder in (
            "sk-ant-...",
            "sk-ant-xyz",
            "sk-ant-test",
            "test-api-key-abc123",
        ):
            assert any(p.search(placeholder) for p in patterns), (
                f"documented placeholder {placeholder!r} is not covered by the allowlist regexes"
            )


class TestLocalCiStage:
    """scripts/ci.sh carries the guarded advisory stage."""

    def test_ci_sh_has_guarded_gitleaks_stage(self) -> None:
        source = _CI_SH.read_text(encoding="utf-8")
        assert "command -v gitleaks" in source, (
            "ci.sh must guard the gitleaks stage so a missing binary skips "
            "instead of failing the local gate"
        )
        assert "--config .gitleaks.toml" in source
        assert "--redact" in source
