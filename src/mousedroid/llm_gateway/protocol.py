"""LLM Gateway protocol — NL mission to velocity command translation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ``GoalVector`` velocity-axis bounds. Not config-driven: the bounds are part
# of the ``GoalVector`` semantic contract (normalised in ``[-1, 1]``), not an
# operator-tunable knob. Defined here, beside the type they constrain, so the
# three ``LLMGatewayProtocol`` implementations share one definition instead of
# each carrying its own copy.
GOAL_VECTOR_MIN = -1.0
GOAL_VECTOR_MAX = 1.0


def clamp_unit(value: float) -> float:
    """Clamp ``value`` into the ``GoalVector`` ``[-1, 1]`` range, NaN-safe.

    A bare ``max(MIN, min(MAX, value))`` returns ``MAX`` for ``NaN``: every
    NaN comparison is False, so ``min(1.0, nan)`` keeps ``1.0`` and the
    surrounding ``max`` passes it through. ``json.loads`` accepts the bare
    ``NaN`` / ``Infinity`` / ``-Infinity`` literals by default, so a model
    emitting ``{"vx": NaN}`` was translated into a **full-scale** velocity
    target rather than a neutral one — the clamp that exists to bound
    actuation was the thing manufacturing the maximum command.

    Every non-finite input therefore resolves to ``0.0``. That matches how
    these parsers treat every other malformed field (non-JSON, non-object and
    non-numeric all yield a neutral ``GoalVector``), and it makes the failure
    direction *no motion*. Note this is a deliberate behaviour change for
    ``±Infinity``, which previously clamped to ``±1.0``.
    """
    if not math.isfinite(value):
        return 0.0
    return max(GOAL_VECTOR_MIN, min(GOAL_VECTOR_MAX, value))


@dataclass(frozen=True)
class GoalVector:
    """3D velocity target from NL mission translation.

    All values normalised to ``[-1, 1]``.
    """

    vx_target: float = 0.0
    vy_target: float = 0.0
    omega_target: float = 0.0


@runtime_checkable
class LLMGatewayProtocol(Protocol):
    """Interface for NL -> velocity command translation."""

    @property
    def is_ready(self) -> bool:
        """Whether the gateway has a loaded model ready to serve translations."""
        ...

    async def start(self) -> None:
        """Load model and warm up. Raises RuntimeError if deps missing."""
        ...

    async def translate_mission(self, nl_command: str) -> GoalVector:
        """Translate NL mission description to a GoalVector.

        Args:
            nl_command: Natural language mission (must be non-empty).

        Returns:
            ``GoalVector`` with ``(vx_target, vy_target, omega_target)`` in ``[-1, 1]``.

        Raises:
            ValueError: If nl_command is empty or missing.
        """
        ...

    async def stop(self) -> None:
        """Unload model and release GPU memory."""
        ...


@runtime_checkable
class QueryCapableLLMProtocol(Protocol):
    """Optional capability: answer a free-text operator query with text.

    Deliberately a SEPARATE protocol from :class:`LLMGatewayProtocol` rather
    than a new method on it. ``LLMGatewayProtocol`` is structurally satisfied
    by many existing test doubles that only implement ``translate_mission`` /
    ``start`` / ``stop``; adding a required method there would break every one
    of them (CLAUDE.md invariant 9 — backwards compatibility). Callers that
    want the conversational path feature-detect with
    ``isinstance(gateway, QueryCapableLLMProtocol)`` instead.

    All four shipped gateways (``llama_cpp``, ``anthropic``,
    ``openai_compatible``, and the ``FallbackLLMGateway`` composite) implement
    this, so a gateway built by :func:`mousedroid.factory.build_llm_gateway`
    always satisfies it. The query path runs OUTSIDE the 30 Hz reactive loop —
    it is operator Q&A, never a control input.
    """

    async def answer_query(self, query: str) -> str:
        """Answer a free-text operator query.

        Args:
            query: Natural language question (must be non-empty). Subject to
                the same prompt-injection filter as ``translate_mission`` on
                backends that filter (``llama_cpp`` / ``anthropic``).

        Returns:
            The model's free-text answer. An empty string signals that no
            backend could answer (gateway not started / degraded / empty
            model response) — the neutral result, mirroring the all-zero
            :class:`GoalVector` that ``translate_mission`` returns on the
            same conditions.

        Raises:
            ValueError: If ``query`` is empty, or ``InjectionRejected`` (a
                ``ValueError`` subclass, from
                :mod:`mousedroid.security.injection_filter`) when the injection
                filter rejects it. These are caller errors, not backend failures.
        """
        ...
