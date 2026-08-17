from __future__ import annotations

import pytest
from structlog.testing import capture_logs

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
    assert target == {"perception_hz": 30.0}


def test_apply_aliases_keeps_existing_canonical_value() -> None:
    target = {"legacy_hz": 30.0, "perception_hz": 10.0}
    apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert target == {"perception_hz": 10.0}


def test_apply_aliases_logs_config_alias_applied_on_legacy_key() -> None:
    """A fired legacy-key alias emits a `config_alias_applied` debug breadcrumb."""
    target = {"legacy_hz": 30.0}
    with capture_logs() as logs:
        apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert any(
        entry["event"] == "config_alias_applied"
        and entry["legacy_key"] == "legacy_hz"
        and entry["canonical_key"] == "perception_hz"
        for entry in logs
    )


def test_apply_aliases_does_not_log_when_legacy_key_absent() -> None:
    target = {"perception_hz": 10.0}
    with capture_logs() as logs:
        apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert logs == []


def test_apply_aliases_does_not_log_when_canonical_already_present() -> None:
    """A discarded legacy value (canonical already set) must not log 'applied'.

    The legacy key is still popped, but nothing was actually applied — the
    canonical value silently wins. Logging `config_alias_applied` here would
    be a false positive an operator grepping logs to confirm a migration
    took effect could be misled by.
    """
    target = {"legacy_hz": 30.0, "perception_hz": 10.0}
    with capture_logs() as logs:
        apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert logs == []


def test_apply_transforms_logs_config_alias_applied_on_successful_transform() -> None:
    target = {"legacy_ms": 500.0}
    with capture_logs() as logs:
        apply_transforms(target, {"legacy_ms": ("canonical_s", milliseconds_to_seconds)})
    assert any(
        entry["event"] == "config_alias_applied"
        and entry["legacy_key"] == "legacy_ms"
        and entry["canonical_key"] == "canonical_s"
        for entry in logs
    )


def test_apply_transforms_does_not_log_on_failed_transform() -> None:
    target = {"legacy_ms": "not-a-number"}
    with capture_logs() as logs:
        apply_transforms(target, {"legacy_ms": ("canonical_s", milliseconds_to_seconds)})
    assert logs == []


def test_apply_aliases_skips_missing_legacy_key() -> None:
    target = {"perception_hz": 10.0}
    apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert "legacy_hz" not in target
    assert target["perception_hz"] == 10.0


def test_apply_transforms_copies_transformed_value() -> None:
    target = {"interval_s": 0.5}
    apply_transforms(target, {"interval_s": ("perception_hz", seconds_to_hz)})
    assert target == {"perception_hz": 2.0}


def test_apply_transforms_keeps_existing_canonical_value() -> None:
    target = {"interval_s": 0.5, "perception_hz": 42.0}
    apply_transforms(target, {"interval_s": ("perception_hz", seconds_to_hz)})
    assert target == {"perception_hz": 42.0}


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
    assert "legacy" not in target


def test_migrate_section_aliases_applies_nested_aliases() -> None:
    root: dict[str, object] = {"loop": {"interval_s": 0.25}}
    migrate_section_aliases(root, {"loop": {"interval_s": "period_s"}})
    assert root["loop"] == {"period_s": 0.25}


def test_migrate_section_aliases_skips_missing_section() -> None:
    root: dict[str, object] = {}
    migrate_section_aliases(root, {"loop": {"interval_s": "period_s"}})
    assert root == {}


def test_migrate_section_transforms_applies_nested_transforms() -> None:
    root: dict[str, object] = {"loop": {"interval_s": 0.2}}
    migrate_section_transforms(root, {"loop": {"interval_s": ("perception_hz", seconds_to_hz)}})
    assert root["loop"] == {"perception_hz": 5.0}


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


def test_apply_aliases_pops_legacy_even_when_canonical_present() -> None:
    target = {"legacy_hz": 30.0, "perception_hz": 10.0}
    apply_aliases(target, {"legacy_hz": "perception_hz"})
    assert "legacy_hz" not in target


def test_apply_transforms_pops_legacy_even_when_canonical_present() -> None:
    target = {"interval_s": 0.5, "perception_hz": 42.0}
    apply_transforms(target, {"interval_s": ("perception_hz", seconds_to_hz)})
    assert "interval_s" not in target


def test_apply_transforms_pops_legacy_on_transform_failure() -> None:
    target: dict[str, object] = {"bad": "not-a-number"}
    apply_transforms(target, {"bad": ("good", float)})
    assert "bad" not in target
    assert "good" not in target


def test_settings_migrate_legacy_fields_does_not_mutate_input() -> None:
    """Root-level validator must not leak mutations into caller-owned data."""
    from mousedroid.config.schema import Settings

    original = {
        "arm_hardware": {"driver": "mock"},
        "safety": {"max_loop_time": 50},
        "telemetry": {"publish_interval_s": 0.5},
    }
    snapshot = {
        "arm_hardware": dict(original["arm_hardware"]),
        "safety": dict(original["safety"]),
        "telemetry": dict(original["telemetry"]),
    }

    Settings.migrate_legacy_fields(original)

    assert original["arm_hardware"] == snapshot["arm_hardware"]
    assert original["safety"] == snapshot["safety"]
    assert original["telemetry"] == snapshot["telemetry"]


def test_settings_migrate_legacy_fields_pops_all_levels() -> None:
    """Legacy keys must be removed at top-level and inside nested sections."""
    from mousedroid.config.schema import Settings

    migrated = Settings.migrate_legacy_fields(
        {
            "arm_hardware": {"driver": "mock"},
            "safety": {"max_loop_time": 50, "min_clearance_m": 0.25},
            "telemetry": {"publish_interval_s": 0.5, "bind_host": "0.0.0.0"},  # noqa: S104
        }
    )

    assert isinstance(migrated, dict)
    assert "arm_hardware" not in migrated
    assert "arm" in migrated
    assert "max_loop_time" not in migrated["safety"]
    assert "min_clearance_m" not in migrated["safety"]
    assert "publish_interval_s" not in migrated["telemetry"]
    assert "bind_host" not in migrated["telemetry"]
    assert migrated["safety"]["max_loop_time_ms"] == 50
    assert migrated["safety"]["min_forward_clearance_m"] == 0.25
    assert migrated["telemetry"]["publish_hz"] == 2.0
    assert migrated["telemetry"]["host"] == "0.0.0.0"  # noqa: S104
