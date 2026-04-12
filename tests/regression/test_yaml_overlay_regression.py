"""Regression tests for YAML overlay (deep-merge) configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mousedroid.config.loader import _deep_merge, load_yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    """Write *data* as YAML to a temp file and return its path."""
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


# ---------------------------------------------------------------------------
# Scalar replacement
# ---------------------------------------------------------------------------


def test_overlay_replaces_scalar(tmp_path: Path) -> None:
    """A scalar in the overlay should overwrite the base scalar."""
    base: dict[str, Any] = {"version": 1, "name": "base"}
    overlay: dict[str, Any] = {"name": "overlay"}
    result = _deep_merge(dict(base), overlay)
    assert result["name"] == "overlay"
    assert result["version"] == 1


# ---------------------------------------------------------------------------
# Nested dict merge
# ---------------------------------------------------------------------------


def test_overlay_merges_nested_dict(tmp_path: Path) -> None:
    """Nested dicts should be merged recursively, not replaced wholesale."""
    base: dict[str, Any] = {
        "camera": {"width": 640, "height": 480, "fps": 30},
    }
    overlay: dict[str, Any] = {
        "camera": {"fps": 60},
    }
    result = _deep_merge(dict(base), overlay)
    assert result["camera"]["width"] == 640
    assert result["camera"]["height"] == 480
    assert result["camera"]["fps"] == 60


# ---------------------------------------------------------------------------
# New top-level key added
# ---------------------------------------------------------------------------


def test_overlay_adds_new_top_level_key() -> None:
    """Keys present only in the overlay must appear in the result."""
    base: dict[str, Any] = {"a": 1}
    overlay: dict[str, Any] = {"b": 2}
    result = _deep_merge(dict(base), overlay)
    assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Multiple overlays in sequence
# ---------------------------------------------------------------------------


def test_multiple_overlays_in_sequence() -> None:
    """Applying overlays sequentially should accumulate changes."""
    base: dict[str, Any] = {"x": 1, "y": {"a": 10, "b": 20}}
    overlay1: dict[str, Any] = {"x": 2, "y": {"a": 11}}
    overlay2: dict[str, Any] = {"y": {"b": 22}, "z": 99}

    result = _deep_merge(dict(base), overlay1)
    result = _deep_merge(result, overlay2)

    assert result["x"] == 2
    assert result["y"]["a"] == 11
    assert result["y"]["b"] == 22
    assert result["z"] == 99


# ---------------------------------------------------------------------------
# Empty overlay is identity
# ---------------------------------------------------------------------------


def test_empty_overlay_is_identity() -> None:
    """An empty overlay should leave the base unchanged."""
    base: dict[str, Any] = {"a": 1, "b": {"c": 3}}
    original = {"a": 1, "b": {"c": 3}}
    result = _deep_merge(dict(base), {})
    assert result == original


# ---------------------------------------------------------------------------
# load_yaml round-trip
# ---------------------------------------------------------------------------


def test_load_yaml_round_trip(tmp_path: Path) -> None:
    """Writing and reading YAML should produce the original dict."""
    data: dict[str, Any] = {"greeting": "hello", "nested": {"key": 42}}
    path = _write_yaml(tmp_path, "test.yaml", data)
    loaded = load_yaml(path)
    assert loaded == data


def test_load_yaml_missing_file(tmp_path: Path) -> None:
    """Loading a non-existent file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Overlay replaces list (not appending)
# ---------------------------------------------------------------------------


def test_overlay_replaces_list() -> None:
    """Lists in overlay should replace, not extend, the base list."""
    base: dict[str, Any] = {"items": [1, 2, 3]}
    overlay: dict[str, Any] = {"items": [4, 5]}
    result = _deep_merge(dict(base), overlay)
    assert result["items"] == [4, 5]


# ---------------------------------------------------------------------------
# Deeply nested merge (3+ levels)
# ---------------------------------------------------------------------------


def test_deeply_nested_merge() -> None:
    """Three-level nesting should merge correctly at each level."""
    base: dict[str, Any] = {"a": {"b": {"c": 1, "d": 2}}}
    overlay: dict[str, Any] = {"a": {"b": {"d": 3, "e": 4}}}
    result = _deep_merge(dict(base), overlay)
    assert result["a"]["b"] == {"c": 1, "d": 3, "e": 4}
