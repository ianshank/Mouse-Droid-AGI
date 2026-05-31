"""Tier C-rover: Anthropic Claude-backed LLM gateway.

Translates a natural-language mission into a :class:`GoalVector` via the
Anthropic Messages API (``client.messages.create``). Conforms structurally
to :class:`LLMGatewayProtocol` so the existing factory + orchestrator wiring
treats it identically to the in-process llama-cpp gateway and the
OpenAI-compatible HTTP gateway. Production deployments select it with
``cfg.llm.backend == "anthropic"``.

Architecture invariants (per CLAUDE.md):

* **Asyncio-only** — uses :class:`anthropic.AsyncAnthropic`. No blocking
  I/O on the event loop.
* **Optional dependency** — the ``anthropic`` SDK is imported lazily inside
  :meth:`start` (NOT in ``__init__``) so the factory can construct the
  gateway, and the rest of the harness can import this module, without the
  SDK present. A missing SDK degrades the gateway rather than crashing the
  process — matching the llama-cpp gateway's degrade-on-missing-dep
  behaviour.
* **No hardcoded values** — every knob (model name, system prompt,
  temperature, max tokens, request timeout, API key) comes from
  :class:`LLMConfig`. The Claude model id lives in ``cfg.llm.model_name``;
  there is deliberately no baked-in default model.
* **Never raises on a backend failure** — on any transport / parse failure
  the gateway logs a structured event, sets ``_degraded=True``, and returns
  a neutral :class:`GoalVector` so the orchestrator's 30 Hz loop never
  crashes on a misbehaving LLM. (A *command* rejection — empty input or a
  prompt-injection hit — still raises :class:`ValueError`, matching the
  in-process gateway contract; that is a caller error, not a backend
  failure.)
* **Prompt-injection filtering** — unlike the OpenAI-compatible backend
  (which trusts the upstream provider's guardrails), the Anthropic backend
  runs the shared :class:`PromptInjectionFilterProtocol` locally before the
  command leaves the rover, because the command is being forwarded to a
  third-party cloud endpoint. The factory wires the same filter instance
  shared with the OpenClaw mission dispatcher.
* **Secret hygiene** — the API key is read from ``cfg.llm.api_key``
  (:class:`~pydantic.SecretStr`) via ``get_secret_value()`` only at client
  construction time and is never logged.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from mousedroid.constants import MILLISECONDS_PER_SECOND
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import (
    PromptInjectionFilterProtocol,
    RegexInjectionFilter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import LLMConfig

_log = get_logger(__name__)

# ``GoalVector`` velocity-axis bounds. Mirrors the clamp the legacy
# in-process ``LLMGateway._parse_response`` and the OpenAI-compatible
# ``_parse_goal_vector`` apply, so every backend produces equivalent
# ``GoalVector`` output for the same model response. Not config-driven
# because the bounds are part of the ``GoalVector`` semantic contract
# (normalised in ``[-1, 1]``), not an operator-tunable knob.
_GOAL_VECTOR_MIN = -1.0
_GOAL_VECTOR_MAX = 1.0


def _clamp_unit(value: float) -> float:
    """Clamp ``value`` to the GoalVector ``[-1, 1]`` velocity range."""
    return max(_GOAL_VECTOR_MIN, min(_GOAL_VECTOR_MAX, value))


class AnthropicLLMGateway:
    """Claude Messages API gateway hitting ``client.messages.create``.

    Conforms structurally to :class:`LLMGatewayProtocol`. Wired via
    :func:`mousedroid.factory.build_llm_gateway` when
    ``cfg.llm.backend == "anthropic"``.
    """

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        injection_filter: PromptInjectionFilterProtocol | None = None,
        sdk: Any | None = None,
    ) -> None:
        """Initialise the gateway.

        Args:
            cfg: LLM gateway configuration. ``cfg.model_name`` must hold a
                Claude model id (e.g. ``claude-haiku-4-5``); ``cfg.api_key``
                supplies the key (falling back to the ``ANTHROPIC_API_KEY``
                env var resolved by the SDK when ``None``).
            injection_filter: Optional shared prompt-injection filter. When
                ``None`` the gateway builds its own
                :class:`RegexInjectionFilter` from ``cfg.injection_patterns``
                and ``cfg.max_command_len``; the factory passes the shared
                instance so the rover applies one rejection envelope across
                every NL ingress.
            sdk: Optional pre-imported ``anthropic`` module (test seam). When
                ``None`` the SDK is imported lazily in :meth:`start`.
        """
        self._cfg = cfg
        self._sdk = sdk
        self._client: Any = None
        self._ready = False
        self._degraded = False
        if injection_filter is None:
            injection_filter = RegexInjectionFilter(
                cfg.injection_patterns,
                max_len=cfg.max_command_len,
            )
        self._injection_filter: PromptInjectionFilterProtocol = injection_filter

    @property
    def is_ready(self) -> bool:
        """True iff :meth:`start` constructed an async Claude client."""
        return self._ready

    @property
    def is_degraded(self) -> bool:
        """True iff the gateway hit a non-recoverable condition.

        Set when the SDK is missing, the model id / API setup is invalid, or
        a request failed. The :class:`FallbackLLMGateway` composite reads
        this flag to route around an unavailable cloud primary.
        """
        return self._degraded

    async def start(self) -> None:
        """Lazily import the SDK and build the async client.

        Degrades (logs + ``_degraded=True``) rather than raising when the
        SDK is missing, when ``model_name`` is blank, or when client
        construction fails — so an off-network / misconfigured rover keeps
        running and (when wired) falls back to a local backend.
        """
        if not self._cfg.enabled:
            _log.info("anthropic_gateway_disabled")
            return

        # Reset both flags at the top so retries reflect the latest outcome
        # (mirrors OpenAICompatibleLLMGateway.start).
        self._ready = False
        self._degraded = False

        if not self._cfg.model_name.strip():
            self._degraded = True
            _log.warning("anthropic_gateway_no_model_name")
            return

        sdk = self._sdk
        if sdk is None:
            try:
                import anthropic
            except ImportError:
                self._degraded = True
                _log.warning("anthropic_gateway_degraded_no_sdk")
                return
            sdk = anthropic
            self._sdk = sdk

        async_cls = getattr(sdk, "AsyncAnthropic", None)
        if async_cls is None:
            self._degraded = True
            _log.warning("anthropic_gateway_degraded_no_async_client")
            return

        # ``api_key=None`` lets the SDK resolve ``ANTHROPIC_API_KEY`` from the
        # environment — the standard Anthropic auth path. We pass the
        # resolved value explicitly when config supplies one so config drives
        # behaviour. ``get_secret_value()`` is the only place the secret is
        # read; it is never logged.
        api_key = self._cfg.api_key.get_secret_value() if self._cfg.api_key is not None else None
        try:
            self._client = async_cls(api_key=api_key)
        except Exception as exc:
            self._degraded = True
            _log.warning(
                "anthropic_gateway_client_init_failed",
                error=f"{type(exc).__name__}:{exc}",
            )
            return

        self._ready = True
        _log.info(
            "anthropic_gateway_started",
            model=self._cfg.model_name,
            request_timeout_s=self._cfg.request_timeout_s,
            api_key_source="config" if api_key is not None else "env",
        )

    async def translate_mission(self, nl_command: str) -> GoalVector:
        """Translate an NL mission to a clamped :class:`GoalVector`.

        Args:
            nl_command: Natural language mission. Must be non-empty.

        Returns:
            ``GoalVector`` in ``[-1, 1]``. Neutral (all-zero) on any backend
            failure (not started / degraded / API error / unparseable
            response).

        Raises:
            ValueError: If ``nl_command`` is empty, or :class:`InjectionRejected`
                (a ``ValueError`` subclass) when the prompt-injection filter
                rejects the command. These are caller errors, not backend
                failures, so they propagate (the orchestrator guards them).
        """
        if not nl_command.strip():
            msg = "nl_command must be non-empty"
            raise ValueError(msg)

        # Sanitise BEFORE the readiness check so an injection attempt is
        # rejected even while the gateway is degraded. ``InjectionRejected``
        # (ValueError subclass) propagates by design.
        nl_command = self._injection_filter.sanitize(nl_command)

        if self._client is None or not self._ready:
            _log.warning("anthropic_gateway_not_started", ready=self._ready)
            return GoalVector()

        start = time.monotonic()
        try:
            response = await self._client.messages.create(
                model=self._cfg.model_name,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                system=self._cfg.system_prompt,
                messages=[{"role": "user", "content": nl_command}],
                timeout=self._cfg.request_timeout_s,
            )
        except Exception as exc:
            self._degraded = True
            _log.warning(
                "anthropic_gateway_request_failed",
                error=f"{type(exc).__name__}:{exc}",
            )
            return GoalVector()

        elapsed_ms = (time.monotonic() - start) * MILLISECONDS_PER_SECOND
        if elapsed_ms > self._cfg.latency_target_ms:
            _log.warning(
                "anthropic_gateway_slow",
                elapsed_ms=elapsed_ms,
                target_ms=self._cfg.latency_target_ms,
            )

        text = self._extract_text(response)
        goal = self._parse_goal_vector(text)
        _log.info(
            "anthropic_gateway_translation",
            elapsed_ms=elapsed_ms,
            vx=goal.vx_target,
            vy=goal.vy_target,
            omega=goal.omega_target,
        )
        return goal

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Concatenate the ``.text`` of every text block in the response.

        The Messages API returns ``content`` as a list of blocks, each with a
        ``.text`` attribute for ``type='text'`` blocks. Defensive against a
        missing / non-list ``content`` so the "never raises on backend"
        invariant holds for a malformed response object.
        """
        content = getattr(response, "content", None) or []
        chunks: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()

    @staticmethod
    def _parse_goal_vector(content: str) -> GoalVector:
        """Parse ``content`` as JSON and build a clamped :class:`GoalVector`.

        Keys (``"vx"``, ``"vy"``, ``"omega"``) and the ``[-1, 1]`` clamp
        mirror the legacy and OpenAI-compatible parsers so swapping
        ``cfg.llm.backend`` produces equivalent ``GoalVector`` output for a
        given model response. Returns a neutral :class:`GoalVector` when
        ``content`` is empty / not JSON, when the decoded payload isn't a
        JSON object, or when a field is non-numeric.
        """
        if not content:
            _log.warning("anthropic_gateway_empty_content")
            return GoalVector()
        try:
            doc = json.loads(content)
        except json.JSONDecodeError:
            _log.warning("anthropic_gateway_non_json_content")
            return GoalVector()
        if not isinstance(doc, dict):
            _log.warning("anthropic_gateway_non_object_content")
            return GoalVector()
        try:
            return GoalVector(
                vx_target=_clamp_unit(float(doc.get("vx", 0.0))),
                vy_target=_clamp_unit(float(doc.get("vy", 0.0))),
                omega_target=_clamp_unit(float(doc.get("omega", 0.0))),
            )
        except (TypeError, ValueError):
            _log.warning("anthropic_gateway_non_numeric_fields")
            return GoalVector()

    async def stop(self) -> None:
        """Release the client reference.

        The async SDK manages its own connection pool; dropping the
        reference is sufficient. ``_ready`` is cleared so a subsequent
        :meth:`translate_mission` short-circuits to a neutral GoalVector
        until :meth:`start` runs again.
        """
        self._client = None
        self._ready = False
        _log.info("anthropic_gateway_stopped")


__all__ = ["AnthropicLLMGateway"]
