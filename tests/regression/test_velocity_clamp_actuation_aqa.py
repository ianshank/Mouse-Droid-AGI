"""Regression: a non-finite velocity must never reach the motors as full scale.

Found during the 2026-09-19 peer review's final pass
(``docs/analysis/positioning-safety-peer-review-2026-09-19.md``, D-17..D-19).

``max(lo, min(hi, value))`` returns ``hi`` for ``NaN``, because every NaN
comparison is False. The first pass of that review found the idiom in the LLM
gateways and fixed it there — but tracing every ``send_velocity`` call site
showed the gateways are the one place it does **not** actuate. The same idiom
sat on three paths that do:

* ``comms/_utils.py::clamp`` — the terminal bound, shared by BOTH codecs.
  Legacy: ``int(clamp(nan, -1, 1) * MAX_PWM)`` == ``int(1.0 * 255)`` == 255,
  full-scale PWM on the wire, no exception raised. Stock: ``max_velocity_mps``
  in the ``CMD_ROS_CTRL`` frame.
* ``common/tools/motor_tools.py::_clamp`` — guards ``set_velocity``, an MCP
  tool, i.e. the only LLM-reachable ``send_velocity`` in the tree.
* ``orchestrator/_action_mixin.py::_execute_action`` — the live 30 Hz tick,
  which had no bound of its own at all and relied on an invariant its own
  docstring merely *assumed*.

The contract pinned here: **a non-finite velocity is a malformed velocity, and
a malformed velocity means no motion.** The failure direction must be toward
stopping, never toward the bound.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from mousedroid.common.tools.motor_tools import _clamp as motor_clamp
from mousedroid.comms._utils import MAX_PWM, clamp

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"

_NON_FINITE = [float("nan"), float("inf"), float("-inf")]


# --------------------------------------------------------------------------
# comms/_utils.py::clamp — the terminal bound, both codecs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_wire_clamp_non_finite_is_zero(bad: float) -> None:
    assert clamp(bad, -1.0, 1.0) == 0.0


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_wire_clamp_non_finite_is_not_the_bound(bad: float) -> None:
    """The specific defect: NaN must not resolve to ``hi``."""
    assert clamp(bad, -1.0, 1.0) != 1.0


def test_legacy_pwm_path_no_longer_emits_full_scale() -> None:
    """``int(clamp(nan, -1, 1) * MAX_PWM)`` used to be 255."""
    assert int(clamp(float("nan"), -1.0, 1.0) * MAX_PWM) == 0


def test_stock_codec_path_no_longer_emits_max_velocity() -> None:
    """The stock codec clamps in physical units, so NaN used to be max_vel."""
    max_vel = 0.5
    assert clamp(float("nan"), -max_vel, max_vel) == 0.0


@pytest.mark.parametrize(
    ("value", "lo", "hi", "expected"),
    [
        (0.2, -1.0, 1.0, 0.2),
        (9.9, -1.0, 1.0, 1.0),
        (-9.9, -1.0, 1.0, -1.0),
        (1.0, -1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0, -1.0),
        (0.0, -1.0, 1.0, 0.0),
        (0.3, -0.5, 0.5, 0.3),
    ],
)
def test_wire_clamp_finite_unchanged(value: float, lo: float, hi: float, expected: float) -> None:
    """Every finite result is byte-identical to the pre-fix behaviour."""
    assert clamp(value, lo, hi) == expected


# --------------------------------------------------------------------------
# motor_tools.py::_clamp — the only LLM-reachable send_velocity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_motor_tool_clamp_non_finite_is_zero(bad: float) -> None:
    assert motor_clamp(bad, lower=-0.5, upper=0.5) == 0.0


@pytest.mark.parametrize(
    ("value", "expected"), [(0.1, 0.1), (9.9, 0.5), (-9.9, -0.5), (0.0, 0.0), (0.5, 0.5)]
)
def test_motor_tool_clamp_finite_unchanged(value: float, expected: float) -> None:
    assert motor_clamp(value, lower=-0.5, upper=0.5) == expected


# --------------------------------------------------------------------------
# The idiom must not reappear on any actuation path
# --------------------------------------------------------------------------


# The two-sided clamp idiom, in BOTH operand orders. Order matters: builtin
# ``min``/``max`` keep the FIRST operand when the comparison is False, and
# every comparison against NaN is False — so the two forms fail *differently*:
#
#   max(-1.0, min(1.0, nan)) == 1.0   <- fabricates the UPPER BOUND
#   max(min(nan, 12.0), 0.0) == nan   <- PROPAGATES NaN
#
# An earlier version of this gate matched only the first form, which means it
# would have passed a file carrying the second. Both are matched now.
_CLAMP_IDIOM = re.compile(
    r"max\(\s*[^,()]+,\s*min\(" r"|" r"max\(\s*min\(",
)

# Every module that clamps a velocity, range or actuation quantity. Listed
# explicitly rather than globbed so that adding a new actuation module is a
# deliberate act that shows up in review.
_GUARDED_MODULES = (
    _SRC / "comms" / "_utils.py",
    _SRC / "comms" / "command_set.py",
    _SRC / "common" / "tools" / "motor_tools.py",
    _SRC / "orchestrator" / "_action_mixin.py",
    _SRC / "hardware" / "motor_controller.py",
    _SRC / "hardware" / "lidar_driver.py",
)


def _carries_unguarded_idiom(text: str) -> bool:
    """True when the clamp idiom appears with no finiteness guard in the file."""
    if not _CLAMP_IDIOM.search(text):
        return False
    return "math.isfinite" not in text and "math.isnan" not in text


def test_no_actuation_path_reimplements_the_nan_blind_clamp() -> None:
    """No module that bounds an actuation quantity may clamp without a guard.

    The defect shipped in four modules because each wrote its own
    ``max(lo, min(hi, v))``. Matching on source text (not AST) keeps this
    consistent with the suppression-budget and cancellation-hygiene gates,
    which take the same approach.

    ``hardware/motor_controller.py`` and ``hardware/lidar_driver.py`` are in
    the list because they already did this correctly — an explicit
    ``isnan``/``isinf`` check before the clamp. They are the in-repo
    precedent the ESP32 path should have followed, and pinning them keeps
    that precedent from eroding.
    """
    offenders = [
        p.name
        for p in _GUARDED_MODULES
        if _carries_unguarded_idiom(p.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == [], (
        f"{offenders} clamp an actuation quantity with no finiteness guard; "
        "depending on operand order a NaN there either resolves to the bound "
        "and is transmitted as maximum commanded speed, or propagates"
    )


def test_the_gate_itself_catches_both_orderings() -> None:
    """The gate must flag a known-bad sample — in either operand order.

    A gate that has never been shown to fail is not evidence. The earlier
    version of this test matched only the fabricating form and would have
    passed the propagating one, which is the form
    ``hardware/motor_controller.py`` uses.
    """
    fabricating = "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))\n"
    propagating = "def clamp(v, lo, hi):\n    return max(min(v, hi), lo)\n"
    guarded = (
        "import math\n"
        "def clamp(v, lo, hi):\n"
        "    if not math.isfinite(v):\n"
        "        return 0.0\n"
        "    return max(lo, min(hi, v))\n"
    )

    assert _carries_unguarded_idiom(fabricating), "gate missed the fabricating form"
    assert _carries_unguarded_idiom(propagating), "gate missed the propagating form"
    assert not _carries_unguarded_idiom(guarded), "gate false-positives on guarded code"


def test_both_orderings_really_do_differ() -> None:
    """Pin the language behaviour the gate exists because of."""
    nan = float("nan")
    assert max(-1.0, min(1.0, nan)) == 1.0  # fabricates the upper bound
    assert math.isnan(max(min(nan, 12.0), 0.0))  # propagates


def test_execute_action_has_a_finite_guard() -> None:
    """``_execute_action`` must not rely on an assumed ``[-1, 1]`` invariant."""
    text = (_SRC / "orchestrator" / "_action_mixin.py").read_text(encoding="utf-8")
    assert "math.isfinite" in text
    assert "action_non_finite_zeroed" in text


def test_math_isfinite_rejects_every_non_finite() -> None:
    """Sanity: the predicate the guards rely on covers all three cases."""
    assert not any(math.isfinite(v) for v in _NON_FINITE)
