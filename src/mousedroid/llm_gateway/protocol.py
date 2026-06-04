"""LLM Gateway protocol — NL mission to velocity command translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
