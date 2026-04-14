"""LLM Gateway — NL mission to velocity command pipeline.

Uses llama-cpp-python for local inference on Jetson Orin Nano.
Optional dependency: ``pip install mousedroid[llm]``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.llm_gateway.config import GatewayConfig

_log = get_logger(__name__)


class LLMGateway:
    """NL mission -> GoalVector translation via local LLM.

    Requires ``llama-cpp-python`` to be installed. ``start()`` raises
    ``RuntimeError`` if the dependency is missing.
    """

    def __init__(self, cfg: GatewayConfig) -> None:
        """Initialise gateway.

        Args:
            cfg: Gateway configuration.
        """
        self._cfg = cfg
        self._model: Any = None

    async def _ensure_model(self) -> None:
        """Download model if not present on disk."""
        model_path = Path(self._cfg.model_path)
        if model_path.exists():
            _log.debug("llm_model_already_present", path=str(model_path))
            return

        if not self._cfg.model_url:
            _log.warning(
                "llm_model_missing_no_url",
                path=str(model_path),
            )
            return

        _log.info(
            "llm_model_downloading",
            path=str(model_path),
            url=self._cfg.model_url,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.to_thread(self._download_model_sync, model_path)
        except Exception:
            _log.warning("llm_model_download_failed", exc_info=True)

    def _download_model_sync(self, dest: Path) -> None:
        """Download model file (blocking, run via to_thread)."""
        try:
            from huggingface_hub import hf_hub_download

            # Extract repo_id and filename from the URL
            url = self._cfg.model_url
            if "huggingface.co" in url:
                parts = url.split("/resolve/main/")
                if len(parts) == 2:
                    repo_path = parts[0].replace("https://huggingface.co/", "")
                    filename = parts[1]
                    hf_hub_download(
                        repo_id=repo_path,
                        filename=filename,
                        local_dir=str(dest.parent),
                        local_dir_use_symlinks=False,
                    )
                    if self._cfg.model_checksum:
                        self._verify_checksum(dest, self._cfg.model_checksum)
                    return
        except ImportError:
            _log.debug("huggingface_hub_not_available_using_urllib")

        import urllib.request

        urllib.request.urlretrieve(self._cfg.model_url, str(dest))  # noqa: S310
        if self._cfg.model_checksum:
            self._verify_checksum(dest, self._cfg.model_checksum)

    @staticmethod
    def _verify_checksum(path: Path, expected: str) -> None:
        """Verify SHA-256 checksum of downloaded file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual != expected:
            path.unlink(missing_ok=True)
            msg = f"Model checksum mismatch: expected {expected[:16]}..., got {actual[:16]}..."
            raise ValueError(msg)

    async def start(self) -> None:
        """Load model and warm up.

        Raises:
            RuntimeError: If llama-cpp-python is not installed or model not found.
        """
        if not self._cfg.enabled:
            _log.info("llm_gateway_disabled")
            return

        await self._ensure_model()

        try:
            await asyncio.to_thread(self._load_model)
        except ImportError as exc:
            msg = "llama-cpp-python is required: pip install mousedroid[llm]"
            raise RuntimeError(msg) from exc
        except OSError as exc:
            msg = f"Model file not found: {self._cfg.model_path}"
            raise RuntimeError(msg) from exc
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

    _INJECTION_RE = re.compile(
        r"(ignore (previous|above|all) instructions?|system prompt|you are now)",
        re.IGNORECASE,
    )

    def _sanitize_command(self, text: str) -> str:
        """Sanitize NL command to mitigate prompt injection."""
        text = text.strip()[: self._cfg.max_command_len]
        if self._INJECTION_RE.search(text):
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

        return self._parse_response(raw)

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
