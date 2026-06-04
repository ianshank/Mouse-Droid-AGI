"""LLM Gateway — NL mission to velocity command pipeline.

Uses llama-cpp-python for local inference on Jetson Orin Nano.
Optional dependency: ``pip install mousedroid[llm]``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from mousedroid.constants import MILLISECONDS_PER_SECOND
from mousedroid.llm_gateway._telemetry import extract_token_pair, record_round_trip_metrics
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import (
    PromptInjectionFilterProtocol,
    RegexInjectionFilter,
)

if TYPE_CHECKING:
    from mousedroid.llm_gateway.config import GatewayConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


class LLMGateway:
    """NL mission -> GoalVector translation via local LLM.

    When ``llama-cpp-python`` is not installed or the model file is missing,
    :meth:`start` logs a warning, enters degraded mode, and returns normally;
    :attr:`is_ready` will remain ``False`` until a model is loaded.
    """

    def __init__(
        self,
        cfg: GatewayConfig,
        *,
        injection_filter: PromptInjectionFilterProtocol | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        """Initialise gateway.

        Args:
            cfg: Gateway configuration.
            injection_filter: Optional shared prompt-injection filter. When
                ``None`` (the default), the gateway constructs its own
                :class:`RegexInjectionFilter` from
                ``cfg.injection_patterns`` and ``cfg.max_command_len`` so
                existing call sites stay byte-identical. The factory wires
                an external filter so the OpenClaw REST endpoint and the
                LLM gateway share the same rejection envelope.
            metrics: Optional shared :class:`MetricsRegistry`. When supplied,
                every inference records round-trip latency, token usage
                (derived from the llama-cpp ``usage`` block when present), and
                a latency-budget-exceeded counter. ``None`` (the default)
                makes every metric call a no-op, so existing callers behave
                byte-identically. Mirrors the Anthropic backend's wiring so
                ``cfg.llm.backend`` can be swapped without losing observability.
        """
        self._cfg = cfg
        self._model: Any = None
        self._degraded = False
        self._metrics = metrics
        # Prometheus ``model`` label for the token / budget counters. The GGUF
        # filename is the closest stable identifier the local backend has (it
        # carries no ``model_name``), and it is low-cardinality / operator-set.
        self._model_label = cfg.model_path.name
        if injection_filter is None:
            injection_filter = RegexInjectionFilter(
                cfg.injection_patterns,
                max_len=cfg.max_command_len,
            )
        self._injection_filter: PromptInjectionFilterProtocol = injection_filter

    @property
    def is_ready(self) -> bool:
        """Whether the gateway has a model loaded and ready."""
        return self._model is not None

    @property
    def is_degraded(self) -> bool:
        """Whether the gateway entered degraded mode during :meth:`start`.

        Degraded mode is set when llama-cpp-python is not importable or when
        the configured model file cannot be opened. In that state
        :meth:`translate_mission` will still run but returns a neutral
        :class:`GoalVector` rather than raising, allowing callers to check
        this flag and surface a user-visible warning.
        """
        return self._degraded

    async def start(self) -> None:
        """Load model and warm up.

        If ``llama-cpp-python`` is not installed or the configured model file
        cannot be opened, a warning is logged, :attr:`_degraded` is set to
        ``True``, and the method returns without raising.
        """
        if not self._cfg.enabled:
            _log.info("llm_gateway_disabled")
            return

        try:
            await asyncio.to_thread(self._load_model)
        except ImportError:
            _log.warning("llm_gateway_degraded_no_llama_cpp")
            self._degraded = True
            return
        except OSError:
            _log.warning("llm_gateway_degraded_model_not_found", path=str(self._cfg.model_path))
            self._degraded = True
            return
        _log.info("llm_gateway_started", model=str(self._cfg.model_path))

    def _load_model(self) -> None:  # pragma: no cover
        """Load GGUF model (blocking, run via to_thread)."""
        from llama_cpp import Llama

        self._model = Llama(
            model_path=str(self._cfg.model_path),
            n_ctx=self._cfg.context_length,
            n_threads=self._cfg.n_threads,
            n_gpu_layers=self._cfg.n_gpu_layers,
            n_batch=self._cfg.n_batch,
        )

    def _sanitize_command(self, text: str) -> str:
        """Sanitize NL command to mitigate prompt injection.

        Delegates to the injected :class:`PromptInjectionFilterProtocol`
        so the same rejection envelope applies across every NL ingress.
        ``InjectionRejected`` (a :class:`ValueError` subclass) propagates —
        callers that catch ``ValueError`` keep working unchanged.
        """
        return self._injection_filter.sanitize(text)

    async def translate_mission(self, nl_command: str) -> GoalVector:
        """Translate NL mission to GoalVector.

        Args:
            nl_command: Natural language mission description.

        Returns:
            GoalVector with normalised velocity targets.

        Raises:
            ValueError: If nl_command is empty or contains disallowed content.
            TimeoutError: If inference exceeds latency target.
        """
        if not nl_command.strip():
            msg = "nl_command must be non-empty"
            raise ValueError(msg)

        nl_command = self._sanitize_command(nl_command)

        if self._model is None:
            _log.warning("llm_gateway_not_started")
            return GoalVector()

        prompt = f"{self._cfg.system_prompt}\n\nMission: {nl_command}\n\nJSON:"
        raw, _elapsed_ms = await self._run_inference(prompt, self._cfg.max_tokens)
        goal = self._parse_response(raw)
        _log.info(
            "llm_translation_completed",
            vx=goal.vx_target,
            vy=goal.vy_target,
            omega=goal.omega_target,
        )
        return goal

    async def answer_query(self, query: str) -> str:
        """Answer a free-text operator query with prose (NOT a GoalVector).

        The conversational sibling of :meth:`translate_mission`: it reuses the
        same model, injection filter, and telemetry path but drives the model
        with ``cfg.query_system_prompt`` (free-text persona) and
        ``cfg.query_max_tokens`` instead of the JSON-navigation prompt. Runs
        OUTSIDE the 30 Hz control loop — operator Q&A only.

        Args:
            query: Natural language question. Must be non-empty.

        Returns:
            The model's free-text answer, or ``""`` when the gateway is not
            started (the neutral result, mirroring the all-zero GoalVector
            ``translate_mission`` returns in the same state).

        Raises:
            ValueError: If ``query`` is empty or rejected by the injection
                filter (``InjectionRejected`` is a ``ValueError`` subclass).
        """
        if not query.strip():
            msg = "query must be non-empty"
            raise ValueError(msg)

        query = self._sanitize_command(query)

        if self._model is None:
            _log.warning("llm_gateway_not_started")
            return ""

        prompt = f"{self._cfg.query_system_prompt}\n\nQuestion: {query}\n\nAnswer:"
        answer, _elapsed_ms = await self._run_inference(prompt, self._cfg.query_max_tokens)
        answer = answer.strip()
        _log.info("llm_query_answered", answer_chars=len(answer))
        return answer

    async def _run_inference(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        """Run one blocking inference off-loop, record telemetry, return text.

        Shared by :meth:`translate_mission` and :meth:`answer_query` so both
        paths get identical latency / token / budget instrumentation.

        Args:
            prompt: Fully-rendered prompt string.
            max_tokens: Generation cap (``cfg.max_tokens`` for translation,
                ``cfg.query_max_tokens`` for the query path).

        Returns:
            ``(text, elapsed_ms)`` — the model's raw completion text and the
            wall-clock latency.
        """
        start_time = time.monotonic()
        output = await asyncio.to_thread(self._infer_sync, prompt, max_tokens)
        elapsed_ms = (time.monotonic() - start_time) * MILLISECONDS_PER_SECOND

        slow = elapsed_ms > self._cfg.latency_target_ms
        if slow:
            _log.warning(
                "llm_inference_slow",
                elapsed_ms=elapsed_ms,
                target_ms=self._cfg.latency_target_ms,
            )
        input_tokens, output_tokens = extract_token_pair(
            output.get("usage"), input_key="prompt_tokens", output_key="completion_tokens"
        )
        record_round_trip_metrics(
            self._metrics,
            model=self._model_label,
            elapsed_ms=elapsed_ms,
            over_budget=slow,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return self._extract_text(output), elapsed_ms

    def _infer_sync(self, prompt: str, max_tokens: int) -> dict[str, Any]:  # pragma: no cover
        """Run blocking LLM inference (via to_thread).

        Args:
            prompt: Full prompt string.
            max_tokens: Generation token cap for this call.

        Returns:
            The raw llama-cpp output mapping (carries ``choices`` and, when the
            build reports it, a ``usage`` block consumed by :meth:`_extract_usage`).
        """
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=self._cfg.temperature,
            stop=self._cfg.stop_tokens,
        )
        return dict(output)

    @staticmethod
    def _extract_text(output: dict[str, Any]) -> str:
        """Pull ``choices[0].text`` defensively, returning ``""`` if malformed."""
        try:
            return str(output["choices"][0]["text"])
        except (KeyError, IndexError, TypeError):
            _log.warning("llm_inference_malformed_output")
            return ""

    def _parse_response(self, raw: str) -> GoalVector:
        """Parse LLM JSON response into GoalVector.

        Args:
            raw: Raw JSON string from LLM.

        Returns:
            Parsed GoalVector with clamped values.
        """
        try:
            data = json.loads(raw.strip())
            return GoalVector(
                vx_target=max(-1.0, min(1.0, float(data.get("vx", 0.0)))),
                vy_target=max(-1.0, min(1.0, float(data.get("vy", 0.0)))),
                omega_target=max(-1.0, min(1.0, float(data.get("omega", 0.0)))),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            _log.warning("llm_parse_failed", raw=raw)
            return GoalVector()

    async def stop(self) -> None:
        """Unload model and release memory."""
        self._model = None
        _log.info("llm_gateway_stopped")
