"""Reusable config migration helpers for backwards compatibility.

These utilities provide composable key-alias migration for nested sections,
top-level aliases, and transformed aliases (for unit conversions).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

SectionAliasMap = Mapping[str, Mapping[str, str]]
TransformAlias = tuple[str, Callable[[Any], Any]]
SectionTransformMap = Mapping[str, Mapping[str, TransformAlias]]


def get_section(
    root: MutableMapping[str, Any],
    section_name: str,
) -> MutableMapping[str, Any] | None:
    """Return a mutable section dict when present, otherwise ``None``."""
    section = root.get(section_name)
    return section if isinstance(section, dict) else None


def apply_aliases(
    target: MutableMapping[str, Any],
    aliases: Mapping[str, str],
) -> None:
    """Apply simple aliases from legacy key -> canonical key.

    The legacy key is always removed (popped) when present so the resulting
    mapping contains only canonical keys. The legacy value is promoted to the
    canonical key only when the canonical key is absent; otherwise the
    canonical value wins and the legacy value is discarded.
    """
    for legacy_key, canonical_key in aliases.items():
        if legacy_key not in target:
            continue
        legacy_value = target.pop(legacy_key)
        if canonical_key not in target:
            target[canonical_key] = legacy_value
        _log.debug("config_alias_applied", legacy_key=legacy_key, canonical_key=canonical_key)


def apply_transforms(
    target: MutableMapping[str, Any],
    transforms: Mapping[str, TransformAlias],
) -> None:
    """Apply transformed aliases from legacy key -> canonical key.

    The legacy key is always removed (popped) when present so the resulting
    mapping contains only canonical keys. The transformed value is assigned to
    the canonical key only when the canonical key is absent and the transform
    succeeds; otherwise the canonical value (or the lack of one) is preserved
    and the legacy value is discarded.
    """
    for legacy_key, (canonical_key, transform) in transforms.items():
        if legacy_key not in target:
            continue
        legacy_value = target.pop(legacy_key)
        if canonical_key in target:
            continue
        try:
            target[canonical_key] = transform(legacy_value)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        _log.debug("config_alias_applied", legacy_key=legacy_key, canonical_key=canonical_key)


def migrate_section_aliases(
    root: MutableMapping[str, Any],
    section_aliases: SectionAliasMap,
) -> None:
    """Apply per-section simple aliases to nested config sections."""
    for section_name, aliases in section_aliases.items():
        section = get_section(root, section_name)
        if section is None:
            continue
        apply_aliases(section, aliases)


def migrate_section_transforms(
    root: MutableMapping[str, Any],
    section_transforms: SectionTransformMap,
) -> None:
    """Apply per-section transformed aliases to nested config sections."""
    for section_name, transforms in section_transforms.items():
        section = get_section(root, section_name)
        if section is None:
            continue
        apply_transforms(section, transforms)


def migrate_group_sections(
    root: MutableMapping[str, Any],
    group_key: str,
    aliases: Mapping[str, str],
) -> None:
    """Lift nested group sections into top-level canonical sections.

    Example: ``robot_arm.sim`` -> top-level ``arm_sim``.
    """
    group = get_section(root, group_key)
    if group is None:
        return

    for group_section_key, canonical_key in aliases.items():
        if canonical_key in root or group_section_key not in group:
            continue
        value = group[group_section_key]
        if isinstance(value, dict):
            root[canonical_key] = value


def seconds_to_hz(value: Any) -> float:
    """Convert seconds-per-event values to Hz."""
    seconds = float(value)
    return 1.0 / seconds


def milliseconds_to_seconds(value: Any) -> float:
    """Convert milliseconds to seconds."""
    return float(value) / 1000.0  # hardcoded-ok


def seconds_to_milliseconds(value: Any) -> float:
    """Convert seconds to milliseconds."""
    return float(value) * 1000.0  # hardcoded-ok
