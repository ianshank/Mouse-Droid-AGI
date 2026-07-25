"""Regression: config overlays have no duplicate keys, and the deployed-image
record stays valid.

* Duplicate YAML keys are silently accepted by ``yaml.safe_load`` (last one
  wins), which can discard an operator's intended overrides. This exact bug hit
  ``config/jetson_production.yaml`` — a duplicate ``domain_randomization`` block
  silently dropped the tightened ``ultrasonic_*`` ranges (fixed in PR #113).
  This test fails loudly on any duplicate key *at any depth* in any overlay.

  Detection uses ``yaml.compose`` with ``SafeLoader`` (returns the node tree
  WITHOUT constructing any Python objects — no ``!!python/object`` risk, no
  custom Loader) and walks the nodes for repeated mapping keys.
* The ``config-compat`` CI gate ``git worktree``s the SHA in
  ``deployments/jetson-image.json``, so that record must have the required keys
  and a full 40-hex commit SHA (never a short/abbreviated/branch ref).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_DEPLOY_RECORD = _REPO_ROOT / "deployments" / "jetson-image.json"
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _find_duplicate_keys(node: yaml.Node | None, path: str = "<root>") -> Iterator[str]:
    """Yield ``path``-qualified locations of duplicate keys in a YAML node tree.

    Walks composed nodes only (no object construction). Recurses into nested
    mappings and sequences so a duplicate at any depth is caught.
    """
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = str(getattr(key_node, "value", key_node))
            loc = f"{path}.{key}"
            if key in seen:
                yield loc
            seen.add(key)
            yield from _find_duplicate_keys(value_node, loc)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            yield from _find_duplicate_keys(item, f"{path}[{index}]")


def _config_yaml_files() -> list[Path]:
    files = sorted(_CONFIG_DIR.rglob("*.yaml")) + sorted(_CONFIG_DIR.rglob("*.yml"))
    return [p for p in files if p.is_file()]


_YAML_FILES = _config_yaml_files()


@pytest.mark.parametrize("yaml_path", _YAML_FILES, ids=[p.name for p in _YAML_FILES])
def test_no_duplicate_keys(yaml_path: Path) -> None:
    """No mapping in any config overlay may declare a key twice (silent override)."""
    with yaml_path.open(encoding="utf-8") as handle:
        root = yaml.compose(handle, Loader=yaml.SafeLoader)
    duplicates = list(_find_duplicate_keys(root))
    assert not duplicates, f"{yaml_path.name}: duplicate keys: {duplicates}"


def test_config_overlay_set_is_non_empty() -> None:
    """Guard against an rglob misconfiguration silently checking nothing."""
    assert _YAML_FILES, "no config YAML files discovered under config/"


def test_jetson_production_keeps_ultrasonic_domain_randomization() -> None:
    """Pin the PR #113 fix: the ultrasonic DR overrides must survive (they were
    lost to a duplicate ``domain_randomization`` block)."""
    data = yaml.safe_load((_CONFIG_DIR / "jetson_production.yaml").read_text(encoding="utf-8"))
    dr = data["domain_randomization"]
    assert "ultrasonic_noise_m" in dr, "ultrasonic_noise_m DR override missing"
    assert "ultrasonic_dropout_prob" in dr, "ultrasonic_dropout_prob DR override missing"


def test_deployment_record_required_keys_and_full_sha() -> None:
    """deployments/jetson-image.json must have required keys + a full 40-hex SHA
    (the config-compat gate worktrees this SHA, so it must be reachable/precise)."""
    record = json.loads(_DEPLOY_RECORD.read_text(encoding="utf-8"))
    for key in ("sha", "platform", "image_tag"):
        assert key in record, f"deployment record missing required key: {key}"
    assert _FULL_SHA_RE.match(record["sha"]), (
        f"sha must be a full 40-char lowercase hex commit: {record['sha']!r}"
    )
    assert record["platform"] == "jetson"
    assert record["image_tag"] == "mousedroid:jetson"
