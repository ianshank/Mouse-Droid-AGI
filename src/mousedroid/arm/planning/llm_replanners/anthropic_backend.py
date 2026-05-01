"""Anthropic-backed arm replanner.

The ``anthropic`` Python SDK is an OPTIONAL dependency — install with
``pip install -e ".[anthropic]"``. The import lives inside
:meth:`AnthropicReplanner._client` so the rest of the harness can be
imported without the dependency present.

All knobs (model, max_tokens, temperature, system_prompt, request
timeout, max retries, api key env var) come from
:class:`mousedroid.config.schema.LLMReplannerConfig`. The backend never
hardcodes a model name, prompt, or timeout.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from mousedroid.arm.protocols import PlanStep
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.arm.protocols import SymbolicState
    from mousedroid.config.schema import LLMReplannerConfig

_log = get_logger(__name__)


class AnthropicSDKMissingError(RuntimeError):
    """Raised when the ``anthropic`` SDK is not installed."""


class AnthropicReplanner:
    """LLM replanner that calls the Anthropic Messages API.

    The replan method is synchronous to match the existing ``Replanner``
    surface; it submits a single non-streaming request and parses the
    JSON response into :class:`PlanStep` instances. Failures are logged
    and surfaced as an empty list so the outer ``Replanner`` can fall
    back to its symbolic planner.
    """

    name = "anthropic"

    def __init__(
        self,
        cfg: LLMReplannerConfig,
        *,
        sdk: Any | None = None,
    ) -> None:
        """Build the Anthropic replanner.

        Args:
            cfg: Replanner config — supplies model, prompt, timeouts, etc.
            sdk: Optional pre-imported ``anthropic`` module (for tests).
                When ``None``, the constructor lazily imports ``anthropic``
                and raises :class:`AnthropicSDKMissingError` if unavailable.
        """
        self._cfg = cfg
        self._sdk = sdk if sdk is not None else self._import_sdk()
        self._client = self._build_client(cfg, self._sdk)
        _log.info(
            "anthropic_replanner_initialised",
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.request_timeout_s,
        )

    # -------------------------------------------------------- public API
    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        if not self._cfg.enabled:
            _log.debug("anthropic_replanner_disabled")
            return []
        prompt = self._build_user_prompt(current_state, goal_state, error)
        last_exc: BaseException | None = None
        for attempt in range(self._cfg.max_retries + 1):
            start = time.monotonic()
            try:
                response = self._client.messages.create(
                    model=self._cfg.model,
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature,
                    system=self._cfg.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self._cfg.request_timeout_s,
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_exc = exc
                _log.warning(
                    "anthropic_replanner_request_failed",
                    attempt=attempt,
                    error=str(exc),
                )
                continue
            latency_ms = (time.monotonic() - start) * 1000.0
            text = self._extract_text(response)
            steps = self._parse_steps(text)
            _log.info(
                "anthropic_replanner_response",
                latency_ms=latency_ms,
                step_count=len(steps),
            )
            return steps
        _log.error(
            "anthropic_replanner_exhausted",
            attempts=self._cfg.max_retries + 1,
            error=str(last_exc) if last_exc else "unknown",
        )
        return []

    # ------------------------------------------------------------ helpers
    def _import_sdk(self) -> Any:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment-dependent
            msg = (
                "The 'anthropic' SDK is not installed. Install with "
                '`pip install -e ".[anthropic]"` to enable this backend.'
            )
            raise AnthropicSDKMissingError(msg) from exc
        return anthropic

    def _build_client(self, cfg: LLMReplannerConfig, sdk: Any) -> Any:
        api_key = os.environ.get(cfg.api_key_env_var, "")
        # Most Anthropic SDK versions accept api_key=None and pull from env;
        # we pass the resolved value explicitly so config drives behaviour.
        return sdk.Anthropic(api_key=api_key or None)

    def _build_user_prompt(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> str:
        return (
            "You are replanning a failed robot arm action.\n\n"
            f"Current symbolic state predicates: {sorted(current_state.predicates)}\n"
            f"Goal symbolic state predicates: {sorted(goal_state.predicates)}\n"
            f"Failure description: {error}\n\n"
            "Reply with a JSON array of plan steps. Each step is an object "
            "with keys 'action' (str) and optional 'args' (list[str]). Reply "
            "with ONLY the JSON array — no surrounding prose."
        )

    def _extract_text(self, response: Any) -> str:
        # The Messages API returns ``content`` as a list of blocks, each
        # with a ``.text`` attribute for ``type='text'`` blocks.
        content = getattr(response, "content", None) or []
        chunks: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()

    def _parse_steps(self, text: str) -> list[PlanStep]:
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            _log.warning("anthropic_replanner_non_json_response", preview=text[:120])
            return []
        if not isinstance(raw, list):
            _log.warning("anthropic_replanner_non_list_response")
            return []
        steps: list[PlanStep] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action")
            if not isinstance(action, str):
                continue
            args = entry.get("args", []) or []
            if not isinstance(args, list):
                continue
            steps.append(PlanStep(action=action, args=[str(a) for a in args]))
        return steps


__all__ = ["AnthropicReplanner", "AnthropicSDKMissingError"]
