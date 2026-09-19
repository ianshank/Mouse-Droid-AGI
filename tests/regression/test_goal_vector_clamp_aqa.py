"""Regression: a non-finite LLM velocity field must resolve to NO motion.

Found during the 2026-09-19 positioning/safety peer review
(``docs/analysis/positioning-safety-peer-review-2026-09-19.md``, defect P-1).

Every ``LLMGatewayProtocol`` implementation bounded its velocity fields with
``max(-1.0, min(1.0, value))``. That expression returns the **upper bound**
for ``NaN``: all NaN comparisons are False, so ``min(1.0, nan)`` keeps
``1.0`` and the enclosing ``max`` passes it through. ``json.loads`` accepts
the bare ``NaN`` / ``Infinity`` / ``-Infinity`` literals by default, so a
model emitting ``{"vx": NaN}`` was translated into a *full-scale* forward
velocity target — silently, with no warning and no degraded flag, because
nothing raised. The clamp that exists to bound actuation was the thing
manufacturing the maximum command.

CHARTER section 3's cloud-egress carve-out leans on this clamp by name: the
prompt-injection filter is documented as "best-effort ... rather than a
complete defense", with the compensating control being that "actuation blast
radius stays bounded downstream by config-clamped velocity limits regardless
of filter outcome". A clamp that turns a malformed field into the maximum
permitted command is the wrong direction for that argument.

The contract pinned here: **a non-finite velocity field is a malformed
field, and malformed fields resolve to neutral** — the same outcome the
parsers already give for non-JSON, non-object and non-numeric input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mousedroid.llm_gateway.protocol import (
    GOAL_VECTOR_MAX,
    GOAL_VECTOR_MIN,
    clamp_unit,
)

_GATEWAY_DIR = Path(__file__).resolve().parents[2] / "src" / "mousedroid" / "llm_gateway"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_clamps_to_zero(bad: float) -> None:
    """NaN and both infinities resolve to no motion, not to a bound."""
    assert clamp_unit(bad) == 0.0


def test_nan_is_not_full_scale() -> None:
    """The specific defect: NaN must not become the upper bound."""
    assert clamp_unit(float("nan")) != GOAL_VECTOR_MAX
    assert clamp_unit(float("nan")) != GOAL_VECTOR_MIN


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (-0.5, -0.5),
        (1.0, 1.0),
        (-1.0, -1.0),
        (99.0, 1.0),
        (-99.0, -1.0),
    ],
)
def test_finite_clamp_is_unchanged(value: float, expected: float) -> None:
    """Finite inputs keep the pre-fix behaviour byte-for-byte."""
    assert clamp_unit(value) == expected


def test_json_accepts_bare_nan_literal() -> None:
    """The delivery mechanism is real: stdlib json decodes ``NaN`` by default.

    This is what makes the defect reachable from a model response rather
    than only from in-process arithmetic, and it is why an
    ``isinstance(doc, dict)`` guard does not catch it — the decoded payload
    is a perfectly well-formed dict.
    """
    doc = json.loads('{"vx": NaN}')
    assert isinstance(doc, dict)
    assert math.isnan(doc["vx"])


def test_no_gateway_reimplements_the_bare_clamp() -> None:
    """No ``LLMGatewayProtocol`` implementation may re-derive the clamp.

    The defect shipped in three places because three modules each carried
    their own ``max(MIN, min(MAX, value))``. Fixing one would have left the
    others live, so the shared helper is pinned as the only definition.
    """
    # The two-sided unit-clamp idiom specifically. A one-sided ``min(1.0, x)``
    # is a different operation -- ``mission_parser.py::_extract_rotation_magnitude``
    # normalises a regex-extracted ``(\d+)`` degree count, which cannot be
    # signed or non-finite -- so it is not matched here.
    idioms = ("max(-1.0, min(1.0,", "max(_GOAL_VECTOR_MIN,", "max(GOAL_VECTOR_MIN,")
    offenders = []
    for path in sorted(_GATEWAY_DIR.glob("*.py")):
        if path.name == "protocol.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(idiom in text for idiom in idioms):
            offenders.append(path.name)
    assert offenders == [], (
        f"{offenders} re-implement the unit clamp inline; import "
        "``clamp_unit`` from mousedroid.llm_gateway.protocol instead"
    )


# --------------------------------------------------------------------------
# D-21 — the compensating control the injection filter used to name
# --------------------------------------------------------------------------

_PHANTOM_BOUNDS = ("max_vx_norm_mps", "max_vy_norm_mps", "max_omega_norm_rads")
_SRC_ROOT = _GATEWAY_DIR.parent


def test_injection_filter_does_not_claim_an_unconsumed_bound() -> None:
    """The filter must not cite the ``*_norm_*`` fields as its blast-radius bound.

    ``security/injection_filter.py`` used to say parsed mission output was
    "clamped by ``LLMConfig.max_vx_norm_mps``/``max_vy_norm_mps``/
    ``max_omega_norm_rads``". Nothing reads those fields: they are declared
    in two schemas, copied into ``GatewayConfig`` by the factory, and never
    consumed. Setting ``max_vx_norm_mps=0.01`` still yields
    ``vx_target=1.0``. CHARTER section 3's cloud-egress carve-out leans on
    that sentence, so the claim must not drift back.
    """
    text = (_SRC_ROOT / "security" / "injection_filter.py").read_text(encoding="utf-8")
    claim = "parsed mission output is still\n    clamped by"
    assert claim not in text, (
        "injection_filter.py again claims the *_norm_* fields bound parsed "
        "output; they are unconsumed (D-21)"
    )


def test_phantom_bounds_are_still_unconsumed_or_the_docs_were_updated() -> None:
    """If someone wires the bounds up, this fails so the docs get revisited.

    Consumption means an attribute *read* outside the two schema definitions
    and the factory's copy into ``GatewayConfig``. The fields are declared in
    m/s and rad/s while ``GoalVector`` is normalised to ``[-1, 1]``, so wiring
    them naively would silently change actuation scale — that is a design
    decision, and this pin makes it a visible one.
    """
    readers = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(_SRC_ROOT).as_posix()
        if rel in {
            "config/schema/llm.py",
            "llm_gateway/config.py",
            "factory/llm_gateway.py",
            "security/injection_filter.py",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(f".{name}" in text for name in _PHANTOM_BOUNDS):
            readers.append(rel)
    assert readers == [], (
        f"{readers} now read the *_norm_* bounds. That is fine — but revisit "
        "security/injection_filter.py's blast-radius paragraph and the D-21 "
        "row in docs/analysis/positioning-safety-peer-review-2026-09-19.md, "
        "and check the m/s-vs-normalised unit mismatch was resolved"
    )
