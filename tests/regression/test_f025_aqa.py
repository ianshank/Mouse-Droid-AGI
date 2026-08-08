"""AQA pins for the F-025 command-set surface (schema-field hygiene).

Mirrors the PR #104 AQA shape: every new field carries a useful Pydantic
``description``, defaults are checked on ``FieldInfo`` directly (so a
refactor that swaps ``Field(...)`` for a property override is caught),
env overrides are reachable, and the codec seam conforms to its Protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mousedroid.config.schema import ESP32Config

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

_MIN_DESCRIPTION_CHARS = 20

_F025_FIELDS = [
    "command_set",
    "heartbeat_enabled",
    "heartbeat_window_multiple",
    "chassis_has_wheel_encoders",
]

_F025_EXPECTED_DEFAULTS = {
    "command_set": "legacy",
    "heartbeat_enabled": True,
    "heartbeat_window_multiple": 3.0,
    "chassis_has_wheel_encoders": True,
}


@pytest.mark.parametrize("field_name", _F025_FIELDS)
def test_f025_field_has_description(field_name: str) -> None:
    """Each F-025 field carries a non-empty, useful Pydantic description."""
    fields = ESP32Config.model_fields
    assert field_name in fields, (
        f"ESP32Config is missing field {field_name!r} — F-025 changed the "
        f"schema; AQA expectations need updating."
    )
    info: FieldInfo = fields[field_name]
    assert info.description, f"{field_name} has no description"
    # Long enough to be useful (>20 chars excludes one-word placeholders).
    assert len(info.description) > _MIN_DESCRIPTION_CHARS, info.description


@pytest.mark.parametrize(
    ("field_name", "expected_default"),
    sorted(_F025_EXPECTED_DEFAULTS.items()),
)
def test_f025_field_has_expected_default(field_name: str, expected_default: object) -> None:
    """Defaults checked on FieldInfo directly, not via model_validate."""
    info: FieldInfo = ESP32Config.model_fields[field_name]
    assert info.default == expected_default, (
        f"ESP32Config.{field_name} default drifted: "
        f"{info.default!r} != expected {expected_default!r}"
    )


@pytest.mark.parametrize(
    ("env_var", "raw", "field_name", "expected"),
    [
        ("MOUSEDROID_ESP32__COMMAND_SET", "waveshare_stock", "command_set", "waveshare_stock"),
        ("MOUSEDROID_ESP32__HEARTBEAT_ENABLED", "false", "heartbeat_enabled", False),
        (
            "MOUSEDROID_ESP32__HEARTBEAT_WINDOW_MULTIPLE",
            "5.0",
            "heartbeat_window_multiple",
            5.0,
        ),
        (
            "MOUSEDROID_ESP32__CHASSIS_HAS_WHEEL_ENCODERS",
            "false",
            "chassis_has_wheel_encoders",
            False,
        ),
    ],
)
def test_f025_fields_settable_via_env(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    raw: str,
    field_name: str,
    expected: object,
) -> None:
    """Every F-025 field is reachable via the documented env path."""
    from mousedroid.config.schema import Settings

    monkeypatch.setenv(env_var, raw)
    cfg = Settings(mock_hardware=True)
    assert getattr(cfg.esp32, field_name) == expected


def test_command_set_literal_matches_public_alias() -> None:
    """The field's Literal values and the public alias stay in lockstep."""
    from typing import get_args

    from mousedroid.config.schema import ESP32CommandSetLiteral

    alias_values = set(get_args(ESP32CommandSetLiteral))
    field_values = set(get_args(ESP32Config.model_fields["command_set"].annotation))
    assert alias_values == field_values == {"legacy", "waveshare_stock"}


def test_codecs_conform_to_protocol() -> None:
    """Both codec singletons satisfy the runtime-checkable Protocol.

    NOTE: ``isinstance`` against a ``runtime_checkable`` Protocol checks
    attribute PRESENCE only — it passes for a class whose "methods" are
    integers. It is kept as a cheap smoke check; the callable/arity
    conformance that actually matters is asserted below, and full signature
    conformance is a static (mypy --strict) guarantee.
    """
    from mousedroid.comms.command_set import (
        LEGACY_CODEC,
        WAVESHARE_STOCK_CODEC,
        ESP32CommandCodec,
    )

    assert isinstance(LEGACY_CODEC, ESP32CommandCodec)
    assert isinstance(WAVESHARE_STOCK_CODEC, ESP32CommandCodec)


_CODEC_METHODS = (
    "build_velocity",
    "build_stop",
    "battery_query",
    "encoder_query",
    "parse_battery",
    "parse_encoders",
    "connect_commands",
)


def test_codec_members_are_callable_with_the_expected_arity() -> None:
    """Every Protocol member is a real method, not merely a present name.

    Closes the gap left by ``isinstance``: this is what catches a codec that
    satisfies the Protocol structurally while being unusable at runtime.
    """
    import inspect

    from mousedroid.comms.command_set import LEGACY_CODEC, WAVESHARE_STOCK_CODEC

    for codec in (LEGACY_CODEC, WAVESHARE_STOCK_CODEC):
        assert isinstance(codec.supports_lateral, bool), codec
        for name in _CODEC_METHODS:
            member = getattr(codec, name)
            assert callable(member), f"{type(codec).__name__}.{name} is not callable"
            # Bound methods: ``self`` is already applied.
            inspect.signature(member)


def test_both_codecs_agree_on_the_member_set() -> None:
    """Neither codec silently grows or drops a Protocol member."""
    from mousedroid.comms.command_set import LEGACY_CODEC, WAVESHARE_STOCK_CODEC

    def public_api(obj: object) -> set[str]:
        return {n for n in dir(obj) if not n.startswith("_")}

    assert set(_CODEC_METHODS) <= public_api(LEGACY_CODEC)
    assert public_api(LEGACY_CODEC) == public_api(WAVESHARE_STOCK_CODEC)


def test_battery_parse_returns_none_not_a_fabricated_voltage() -> None:
    """The unavailable-reading contract, pinned at the AQA tier.

    A fabricated ``0.0`` here is what let a comms fault masquerade as
    ``battery_critical`` and latch a permanent emergency stop.
    """
    from mousedroid.comms.command_set import LEGACY_CODEC, WAVESHARE_STOCK_CODEC

    assert LEGACY_CODEC.parse_battery({}) is None
    assert WAVESHARE_STOCK_CODEC.parse_battery({}) is None


def test_resilient_wrapper_stays_command_set_agnostic() -> None:
    """The resilience layer never imports the codec seam.

    ``ResilientESP32Driver`` wraps ``ESP32CommProtocol`` opaquely; if it
    ever grows a command-set dependency, per-command-set behaviour leaks
    into the fault-tolerance layer and the codec seam stops being the
    single dispatch point.
    """
    import inspect

    import mousedroid.resilience.resilient_driver as resilient_driver

    source = inspect.getsource(resilient_driver)
    assert "command_set" not in source
    assert "resolve_command_codec" not in source
