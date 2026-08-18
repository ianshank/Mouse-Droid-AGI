---
name: regression-pair-scaffold
description: Scaffold the paired tests/regression/test_<name>_aqa.py + test_<name>_backwards_compat.py files a new config field, validator, or Protocol implementation must ship with
status: active
---

# Regression Pair Scaffold

Generate the two-file regression pair this repo requires alongside a new
config field, a new validator, or a new Protocol implementation: an AQA file
pinning schema/protocol hygiene, and a backwards-compat file pinning
default-value and legacy-YAML invariants (CLAUDE.md invariant 9).

Use this whenever `add-schema-field` or `add-hardware-driver` tells you an AQA
test is needed but you want the actual file shape, or when a review asks "does
this feature have its regression pair yet?". This pattern already exists by
hand in `tests/regression/` dozens of times; this skill is the shape, not a
new convention.

## Inputs

- `$ARGUMENTS` — the feature slug used in both filenames, e.g. `pr106`,
  `f025`, `growth`. Required.

## Steps

### 1. Identify the surface being pinned

Before writing either file, name exactly what changed: the new
`Field(...)`(s) and their owning `BaseModel`, any `field_validator` /
`model_validator` that can reject input, and any new class that implements a
`@runtime_checkable Protocol`. Each of these gets its own test(s) below — a
field with no validator skips the rejection test, a config block with no
Protocol pairing skips the conformance test.

### 2. Write `tests/regression/test_<name>_aqa.py`

Check hygiene on `model_fields` (the `FieldInfo`), never by instantiating —
instantiation only proves the default is *legal*, not that it is *declared*
the way you think, and a refactor that replaces `Field(...)` with a plain
class attribute or a property override must still be caught.

```python
"""Automated Quality Assurance (AQA) — schema + protocol hygiene for <name>."""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from mousedroid.config.schema import <Config>


def test_<field>_has_description() -> None:
    """``<Config>.<field>`` carries a non-empty, explanatory description."""
    info: FieldInfo = <Config>.model_fields["<field>"]
    assert info.description
    assert len(info.description) > 20, info.description


def test_<field>_default_is_<expected>() -> None:
    """Pinned off FieldInfo, not a live instance — see module docstring."""
    info: FieldInfo = <Config>.model_fields["<field>"]
    assert info.default == <expected>


# Only when the field/model is validated, not merely declared:
def test_<bad_condition>_raises_at_load() -> None:
    """A misconfigured <field> is rejected at YAML-load time, not silently
    accepted as a no-op."""
    with pytest.raises(ValidationError, match=r"<field>"):
        <Config>(<bad_kwargs>)


def test_<good_condition>_loads_cleanly() -> None:
    """The same validator, satisfied — proves it isn't just always-raise."""
    cfg = <Config>(<good_kwargs>)
    assert cfg.<field> == <expected>


# Only when this pairs with a runtime_checkable Protocol implementation:
def test_<impl>_satisfies_<protocol>() -> None:
    """Bare ``isinstance`` against a ``runtime_checkable`` Protocol only
    proves attribute *presence* — pair it with an arity check, or a stub
    whose methods are the wrong shape would still pass."""
    instance = <impl_instance>
    assert isinstance(instance, <Protocol>)
    sig = inspect.signature(instance.<method>)
    assert list(sig.parameters) == [<expected_param_names>]
```

### 3. Write `tests/regression/test_<name>_backwards_compat.py`

```python
"""<name> backwards-compatibility regression tests.

Pins CLAUDE.md invariant 9: new config fields MUST have defaults, and
existing YAML files must load unchanged.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema import Settings


def test_<block>_defaults_to_<none_or_false>() -> None:
    """A pre-<name> YAML has no ``<block>:`` key at all."""
    cfg = Settings.model_validate({"mock_hardware": True})
    assert cfg.<block> is <expected_default>


def test_legacy_yaml_without_<block>_loads_unchanged() -> None:
    """A minimal legacy YAML loads cleanly, and an unrelated pre-existing
    field survives untouched — proves the new block is additive, not
    entangled with something it shouldn't reach."""
    legacy_yaml = """
    mock_hardware: true
    platform: mouse_droid
    <unrelated_pre_existing_field>: <value>
    """
    data = yaml.safe_load(legacy_yaml)
    cfg = Settings.model_validate(data)
    assert cfg.<block> is <expected_default>
    assert cfg.<unrelated_pre_existing_field> == <value>


def test_<block>_round_trips_when_present() -> None:
    cfg = Settings.model_validate({"mock_hardware": True, "<block>": {"enabled": True}})
    assert cfg.<block>.enabled is True


def test_default_yaml_fixture_still_loads_clean() -> None:
    """The committed ``config/default.yaml`` loads with the new block absent
    or disabled."""
    default_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    if not default_path.exists():  # pragma: no cover - safety net for moved fixture
        return
    with default_path.open() as fh:
        data = yaml.safe_load(fh)
    cfg = Settings.model_validate(data)
    if cfg.<block> is not None:
        assert cfg.<block>.enabled is False
```

### 4. Naming discipline

Always name the second file `_backwards_compat.py`, spelled out in full.
Several existing files drifted to a shortened form —
`test_domain_randomization_backcompat.py`, `test_dual_stream_compat.py` — cite
these as what not to copy, not as an alternative pattern. The full spelling is
what `test-tier-mirror` and this file both document; a drifted name still
gets collected by `tests/regression/`'s tier glob, so nothing enforces this
mechanically — consistency here is what keeps `grep _backwards_compat.py`
useful.

### 5. Add the loader-path variant only when there's an env lever

The two skeletons above construct `Settings` / `<Config>` directly. That's
sufficient for most fields. Only add a deeper test that goes through the real
`load_settings()` (not direct construction) when the feature also exposes an
operator-facing `MOUSEDROID_*` env override — the direct-construction tests
above cannot prove an env var actually reaches the field; only the loader can.

### 6. Confirm the pair can fail

Before trusting either file, temporarily revert the change under test (comment
out the new field, validator, or Protocol method) and confirm the new tests
go red. A pair that passes both before and after the revert is not testing
anything.

## Reference

| File | Tier | What it pins |
|---|---|---|
| `test_<name>_aqa.py` | `tests/regression/` (AQA) | Field description/default hygiene, validator rejection, Protocol conformance |
| `test_<name>_backwards_compat.py` | `tests/regression/` | Absent-block default, legacy-YAML isolation, present-block round-trip, shipped-YAML still parses |

Real, currently-shipping examples safe to read verbatim:
`tests/regression/test_pr106_aqa.py` + `tests/regression/test_pr106_backwards_compat.py`
(USB-C discovery — a validated field, a boot-race-guarded function, and the
default-YAML-still-parses check, all in one pair).

**Verify:** `python tools/validate_skill_commands.py` must report zero issues
(this is `tests/regression/test_skill_commands_aqa.py`'s PR gate) — it fails
on a broken front-matter, a missing description, or a backtick-wrapped repo
path that doesn't resolve. Angle-bracket placeholders (`<name>`, `<Config>`,
`<field>`) are deliberately excluded from that path-existence check, so the
skeletons above are safe to keep literal rather than further genericized.
