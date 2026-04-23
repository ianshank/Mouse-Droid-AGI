"""LLM Gateway — NL mission to velocity command pipeline.

Uses llama-cpp-python for local inference on Jetson Orin Nano.
Optional dependency: ``pip install mousedroid[llm]``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any

from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.llm_gateway.config import GatewayConfig

_log = get_logger(__name__)


class LLMGateway:
    """NL mission -> GoalVector translation via local LLM.

    When ``llama-cpp-python`` is not installed or the model file is missing,
    :meth:`start` logs a warning, enters degraded mode, and returns normally;
    :attr:`is_ready` will remain ``False`` until a model is loaded.
    """

    def __init__(self, cfg: GatewayConfig) -> None:
        """Initialise gateway.

        Args:
            cfg: Gateway configuration.
        """
        self._cfg = cfg
        self._model: Any = None
        self._degraded = False
        # Empty injection_patterns disables the filter; otherwise "()" would
        # compile to a pattern that matches every string and reject everything.
        if cfg.injection_patterns:
            self._injection_re: re.Pattern[str] | None = re.compile(
                "(" + "|".join(cfg.injection_patterns) + ")",
                re.IGNORECASE,
            )
        else:
            self._injection_re = None

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
        )

    def _sanitize_command(self, text: str) -> str:
        """Sanitize NL command to mitigate prompt injection."""
        text = text.strip()[: self._cfg.max_command_len]
        if self._injection_re is not None and self._injection_re.search(text):
            msg = "Mission command contains disallowed content"
            raise ValueError(msg)
        return text

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

        start_time = time.monotonic()
        raw = await asyncio.to_thread(self._infer_sync, prompt)
        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        if elapsed_ms > self._cfg.latency_target_ms:
            _log.warning(
                "llm_inference_slow",
                elapsed_ms=elapsed_ms,
                target_ms=self._cfg.latency_target_ms,
            )

        goal = self._parse_response(raw)
        _log.info(
            "llm_translation_completed",
            elapsed_ms=elapsed_ms,
            vx=goal.vx_target,
            vy=goal.vy_target,
            omega=goal.omega_target,
        )
        return goal

    def _infer_sync(self, prompt: str) -> str:  # pragma: no cover
        """Run blocking LLM inference (via to_thread).

        Args:
            prompt: Full prompt string.

        Returns:
            Raw model output text.
        """
        output = self._model(
            prompt,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            stop=self._cfg.stop_tokens,
        )
        return str(output["choices"][0]["text"])

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
