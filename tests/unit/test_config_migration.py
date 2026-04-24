from __future__ import annotations

import pytest

from mousedroid.config.migration import (
    apply_aliases,
    apply_transforms,
    get_section,
    migrate_group_sections,
    migrate_section_aliases,
    migrate_section_transforms,
    milliseconds_to_seconds,
    seconds_to_hz,
    seconds_to_milliseconds,
)


def test_get_section_returns_mapping_when_present() -> None:
    root: dict[str, object] = {"loop": {"perception_hz": 30.0}}
    section = get_section(root, "loop")
    assert section == {"perception_hz": 30.0}


def test_get_section_returns_none_for_missing_or_non_mapping() -> None:
    root: dict[str, object] = {"loop": "not-a-mapping"}
    assert get_section(root, "loop") is None
    assert get_section(root, "missing") is None


def test_apply_aliases_copies_legacy_value() -> None:
    target = {"legacy_hz": 30.0}
    apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert target["perception_hz"] == 30.0


def test_apply_aliases_keeps_existing_canonical_value() -> None:
    target = {"legacy_hz": 30.0, "perception_hz": 10.0}
    apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert target["perception_hz"] == 10.0


def test_apply_aliases_skips_missing_legacy_key() -> None:
    target = {"perception_hz": 10.0}
    apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert "legacy_hz" not in target
    assert target["perception_hz"] == 10.0


def test_apply_transforms_copies_transformed_value() -> None:
    target = {"interval_s": 0.5}
    apply_transforms(target, {"interval_s": ("perception_hz", seconds_to_hz)})
    assert target["perception_hz"] == 2.0


def test_apply_transforms_keeps_existing_canonical_value() -> None:
    target = {"interval_s": 0.5, "perception_hz": 42.0}
    apply_transforms(target, {"interval_s": ("perception_hz", seconds_to_hz)})
    assert target["perception_hz"] == 42.0


def test_apply_transforms_skips_missing_legacy_key() -> None:
    target = {"perception_hz": 10.0}
    apply_transforms(target, {"interval_s": ("perception_hz", seconds_to_hz)})
    assert target["perception_hz"] == 10.0


@pytest.mark.parametrize("legacy_value", ["bad", None, 0])
def test_apply_transforms_ignores_transform_failures(legacy_value: object) -> None:
    target: dict[str, object] = {"legacy": legacy_value}

    def _raise_mixed_errors(value: object) -> float:
        if value is None:
            msg = "none is unsupported"
            raise TypeError(msg)
        if value == "bad":
            msg = "cannot parse"
            raise ValueError(msg)
        return 1.0 / float(value)

    apply_transforms(target, {"legacy": ("canonical", _raise_mixed_errors)})
    assert "canonical" not in target


def test_migrate_section_aliases_applies_nested_aliases() -> None:
    root: dict[str, object] = {"loop": {"interval_s": 0.25}}
    migrate_section_aliases(root, {"loop": {"interval_s": "period_s"}})
    assert root["loop"] == {"interval_s": 0.25, "period_s": 0.25}


def test_migrate_section_aliases_skips_missing_section() -> None:
    root: dict[str, object] = {}
    migrate_section_aliases(root, {"loop": {"interval_s": "period_s"}})
    assert root == {}


def test_migrate_section_transforms_applies_nested_transforms() -> None:
    root: dict[str, object] = {"loop": {"interval_s": 0.2}}
    migrate_section_transforms(root, {"loop": {"interval_s": ("perception_hz", seconds_to_hz)}})
    assert root["loop"] == {"interval_s": 0.2, "perception_hz": 5.0}


def test_migrate_section_transforms_skips_missing_section() -> None:
    root: dict[str, object] = {}
    migrate_section_transforms(root, {"loop": {"interval_s": ("perception_hz", seconds_to_hz)}})
    assert root == {}


def test_migrate_group_sections_lifts_nested_mapping() -> None:
    root: dict[str, object] = {"robot_arm": {"sim": {"task": "hanoi"}}}
    migrate_group_sections(root, "robot_arm", {"sim": "arm_sim"})
    assert root["arm_sim"] == {"task": "hanoi"}


def test_migrate_group_sections_skips_missing_group() -> None:
    root: dict[str, object] = {}
    migrate_group_sections(root, "robot_arm", {"sim": "arm_sim"})
    assert "arm_sim" not in root


def test_migrate_group_sections_does_not_overwrite_existing_canonical() -> None:
    root: dict[str, object] = {
        "robot_arm": {"sim": {"task": "hanoi"}},
        "arm_sim": {"task": "existing"},
    }
    migrate_group_sections(root, "robot_arm", {"sim": "arm_sim"})
    assert root["arm_sim"] == {"task": "existing"}


def test_migrate_group_sections_skips_non_mapping_values() -> None:
    root: dict[str, object] = {"robot_arm": {"sim": "not-a-mapping"}}
    migrate_group_sections(root, "robot_arm", {"sim": "arm_sim"})
    assert "arm_sim" not in root


@pytest.mark.parametrize(
    ("converter", "value", "expected"),
    [
        (seconds_to_hz, 0.25, 4.0),
        (milliseconds_to_seconds, 250.0, 0.25),
        (seconds_to_milliseconds, 0.25, 250.0),
    ],
)
def test_time_conversion_helpers(converter, value: float, expected: float) -> None:
    assert converter(value) == expected
