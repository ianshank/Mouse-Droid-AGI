"""Backwards-compat pin for the NaN-safe GoalVector clamp (peer review P-1/P-2).

The fix changed two things. This file pins what it must NOT have changed,
and the one behaviour change that is deliberate.

Unchanged:
  * Every finite value clamps exactly as before.
  * ``LLMGateway._parse_response`` still returns a neutral ``GoalVector``
    for non-JSON input, still fills missing keys with 0.0, and still
    accepts a partial object — the four pre-existing unit tests in
    ``tests/unit/llm_gateway/test_llm_gateway.py`` cover those and must
    keep passing.
  * The ``[-1, 1]`` bounds themselves are untouched.

Deliberately changed:
  * ``±Infinity`` previously clamped to ``±1.0``; it now resolves to 0.0,
    like every other malformed field. Pinned here so the change is
    explicit rather than incidental.

Newly guaranteed (this is the P-2 half):
  * All three ``LLMGatewayProtocol`` implementations now uphold the
    "never raises" parser invariant that ``OpenAICompatibleLLMGateway``
    documents. The in-process ``LLMGateway`` previously raised
    ``AttributeError`` on valid-JSON-but-not-an-object input, because its
    ``except`` clause named ``KeyError`` (which ``dict.get`` cannot raise)
    and omitted ``AttributeError`` (which non-dicts always raise).
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.llm_gateway.gateway import LLMGateway
from mousedroid.llm_gateway.protocol import GoalVector, clamp_unit


@pytest.fixture
def gateway() -> LLMGateway:
    cfg = Settings(mock_hardware=True)
    return LLMGateway(cfg.llm)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (0.25, 0.25), (-0.25, -0.25), (1.0, 1.0), (-1.0, -1.0), (5.0, 1.0), (-5.0, -1.0)],
)
def test_finite_clamp_behaviour_is_byte_identical(value: float, expected: float) -> None:
    assert clamp_unit(value) == expected


@pytest.mark.parametrize("raw", ["[1, 2, 3]", "null", "7", '"go forward"', "true"])
def test_valid_json_non_object_returns_neutral_instead_of_raising(
    gateway: LLMGateway, raw: str
) -> None:
    """The P-2 defect: these five inputs used to raise AttributeError."""
    assert gateway._parse_response(raw) == GoalVector()


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        ('{"vx": NaN}', "vx_target"),
        ('{"vy": NaN}', "vy_target"),
        ('{"omega": NaN}', "omega_target"),
    ],
)
def test_nan_field_yields_no_motion_on_that_axis(gateway: LLMGateway, raw: str, field: str) -> None:
    """The P-1 defect: each of these used to yield 1.0 on its axis."""
    assert getattr(gateway._parse_response(raw), field) == 0.0


@pytest.mark.parametrize("raw", ['{"vx": Infinity}', '{"vx": -Infinity}'])
def test_infinity_now_neutral_not_bounded(gateway: LLMGateway, raw: str) -> None:
    """Deliberate change: was ±1.0 pre-fix, now 0.0."""
    assert gateway._parse_response(raw).vx_target == 0.0


def test_preexisting_parser_contract_survives(gateway: LLMGateway) -> None:
    """The four behaviours the pre-fix unit tests assert."""
    assert gateway._parse_response("not json at all") == GoalVector()
    assert gateway._parse_response("{}") == GoalVector()
    partial = gateway._parse_response('{"vx": 0.2}')
    assert (partial.vx_target, partial.vy_target, partial.omega_target) == (0.2, 0.0, 0.0)
    full = gateway._parse_response('{"vx": 0.5, "vy": -0.3, "omega": 0.8}')
    assert (full.vx_target, full.vy_target, full.omega_target) == (0.5, -0.3, 0.8)


def test_non_numeric_field_still_neutral(gateway: LLMGateway) -> None:
    """``{"vx": "fast"}`` -> ValueError inside float() -> neutral, as before."""
    assert gateway._parse_response('{"vx": "fast"}') == GoalVector()
