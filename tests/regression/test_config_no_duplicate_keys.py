"""Regression: no YAML config file under ``config/`` contains duplicate top-level keys.

Duplicate top-level keys are silently discarded by PyYAML's ``safe_load``
(last one wins), which can hide configuration bugs where two sections disagree.
This test walks every mapping in each YAML file and asserts that no key
appears more than once within the same mapping.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"


def _find_duplicate_keys(
    path: Path,
) -> dict[str, list[str]]:
    """Walk every mapping in a YAML file; return duplicates grouped by location.

    Returns a dict mapping a human-readable YAML path string (e.g.
    ``"(root)"`` or ``"arm_training"``) to the sorted list of duplicate
    key names found within that mapping.  The dict is empty when the
    file is clean.

    Args:
        path: Path to the YAML file to inspect.

    Returns:
        Mapping of YAML location to duplicate key names.  Empty if none.
    """
    raw = path.read_text(encoding="utf-8")
    root_node = yaml.compose(raw)  # no loading — just the node tree
    if root_node is None:
        return {}

    issues: dict[str, list[str]] = {}

    def _walk(node: yaml.Node, location: str) -> None:
        if not isinstance(node, yaml.MappingNode):
            if isinstance(node, yaml.SequenceNode):
                for idx, item in enumerate(node.value):
                    _walk(item, f"{location}[{idx}]")
            return

        seen: list[str] = []
        dupes: list[str] = []
        for key_node, value_node in node.value:
            # Key nodes in safe YAML are always scalars.
            key = str(key_node.value)
            if key in seen:
                if key not in dupes:
                    dupes.append(key)
            else:
                seen.append(key)
            _walk(value_node, f"{location}.{key}" if location != "(root)" else key)

        if dupes:
            issues[location] = sorted(dupes)

    _walk(root_node, "(root)")
    return issues


def _discover_yaml_configs() -> list[Path]:
    """Return all ``.yaml`` files directly under the config directory."""
    return sorted(_CONFIG_DIR.glob("*.yaml"))


_YAML_CONFIGS = _discover_yaml_configs()


@pytest.mark.parametrize(
    "config_path",
    _YAML_CONFIGS,
    ids=[p.name for p in _YAML_CONFIGS],
)
def test_no_duplicate_keys_in_any_mapping(config_path: Path) -> None:
    """No YAML mapping in a config file may contain duplicate keys.

    Duplicate keys are silently discarded by PyYAML (last wins), which
    hides configuration bugs.  This test catches them before they cause
    subtle runtime surprises.
    """
    issues = _find_duplicate_keys(config_path)
    assert issues == {}, (
        f"{config_path.name} contains duplicate keys:\n"
        + "\n".join(f"  {loc}: {keys}" for loc, keys in sorted(issues.items()))
        + "\nRemove or merge the duplicate sections."
    )


def test_config_dir_is_non_empty() -> None:
    """Sanity check: glob must find at least one YAML file."""
    assert _YAML_CONFIGS, f"No YAML files found under {_CONFIG_DIR}"
