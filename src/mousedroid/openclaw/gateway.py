"""OpenClaw HTTP/WS gateway — communicates with a running OpenClaw service.

Uses ``aiohttp`` for non-blocking HTTP requests. Falls back gracefully
when the service is unreachable so the orchestrator can degrade to
CognitiveCore → MCTS.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import numpy as np

from mousedroid.config.schema import OpenClawConfig
from mousedroid.logging.setup import get_logger
from mousedroid.openclaw.protocol import OpenClawActionResult

_log = get_logger(__name__)


class OpenClawGateway:
    """HTTP-based gateway to a remote or local OpenClaw service.

    Args:
        cfg: OpenClaw configuration section.
    """

    def __init__(self, cfg: OpenClawConfig) -> None:
        self._cfg = cfg
        self._session: Any = None  # aiohttp.ClientSession — lazy import
        self._connected = False
        self._current_goal: str | None = None

    async def start(self) -> None:
        """Create HTTP session and validate connectivity with retries."""
        timeout = aiohttp.ClientTimeout(total=self._cfg.api_timeout_s)
        headers: dict[str, str] = {}
        if self._cfg.api_key is not None:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"

        self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)

        # Attempt initial connection with retries
        for attempt in range(1, self._cfg.connect_retries + 1):
            try:
                async with self._session.get(f"{self._cfg.api_endpoint}/api/v1/health") as resp:
                    if resp.status == 200:
                        self._connected = True
                        _log.info(
                            "openclaw_connected",
                            endpoint=self._cfg.api_endpoint,
                            attempt=attempt,
                        )
                        return
                    _log.warning(
                        "openclaw_health_check_failed",
                        status=resp.status,
                        attempt=attempt,
                    )
            except Exception as exc:
                _log.warning(
                    "openclaw_connection_attempt_failed",
                    attempt=attempt,
                    max_retries=self._cfg.connect_retries,
                    error=str(exc),
                )
            if attempt < self._cfg.connect_retries:
                delay = self._cfg.connect_backoff_base**attempt
                await asyncio.sleep(delay)

        _log.error(
            "openclaw_connection_exhausted",
            endpoint=self._cfg.api_endpoint,
            retries=self._cfg.connect_retries,
        )
        if not self._cfg.fallback_to_cognitive:
            msg = (
                f"Failed to connect to OpenClaw at {self._cfg.api_endpoint} "
                f"after {self._cfg.connect_retries} attempts"
            )
            raise ConnectionError(msg)

    async def stop(self) -> None:
        """Close HTTP session and mark as disconnected."""
        self._connected = False
        if self._session is not None:
            await self._session.close()
            self._session = None
        _log.info("openclaw_disconnected")

    async def get_action(
        self,
        observation_dict: dict[str, Any],
    ) -> OpenClawActionResult | None:
        """POST observation to OpenClaw and parse the action response.

        Args:
            observation_dict: Current robot state.

        Returns:
            Parsed ``OpenClawActionResult`` or ``None`` on any failure.
        """
        if self._session is None or not self._connected:
            return None

        payload = self._build_payload(observation_dict)

        try:
            async with self._session.post(
                f"{self._cfg.api_endpoint}/api/v1/action",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    _log.debug("openclaw_action_http_error", status=resp.status)
                    return None
                data = await resp.json()
                return self._parse_action(data)
        except Exception as exc:
            _log.debug("openclaw_action_request_failed", error=str(exc))
            self._connected = False
            return None

    async def set_goal(self, goal: str) -> None:
        """Send a new high-level goal to OpenClaw.

        Args:
            goal: Natural-language goal description.
        """
        self._current_goal = goal
        if self._session is None or not self._connected:
            _log.warning("openclaw_set_goal_not_connected", goal=goal)
            return

        try:
            async with self._session.post(
                f"{self._cfg.api_endpoint}/api/v1/goal",
                json={"goal": goal},
            ) as resp:
                if resp.status == 200:
                    _log.info("openclaw_goal_set", goal=goal)
                else:
                    _log.warning(
                        "openclaw_set_goal_failed",
                        status=resp.status,
                        goal=goal,
                    )
        except Exception as exc:
            _log.warning("openclaw_set_goal_error", error=str(exc))

    @property
    def is_connected(self) -> bool:
        """Whether the gateway has an active connection."""
        return self._connected

    def _build_payload(self, observation_dict: dict[str, Any]) -> dict[str, Any]:
        """Filter observation dict to configured keys and serialise arrays.

        Args:
            observation_dict: Raw observation dictionary.

        Returns:
            JSON-safe payload.
        """
        payload: dict[str, Any] = {}
        for key in self._cfg.observation_keys:
            if key in observation_dict:
                val = observation_dict[key]
                # Convert numpy arrays to lists for JSON serialisation
                if hasattr(val, "tolist"):
                    val = val.tolist()
                payload[key] = val
        if self._current_goal is not None:
            payload["current_goal"] = self._current_goal
        return payload

    def _parse_action(self, data: dict[str, Any]) -> OpenClawActionResult | None:
        """Parse a JSON response into an ``OpenClawActionResult``.

        Args:
            data: Decoded JSON response body.

        Returns:
            Parsed result, or ``None`` if the response is malformed.
        """
        try:
            action_raw = data.get("action")
            if action_raw is None:
                return None
            action = np.asarray(action_raw, dtype=np.float32)
            return OpenClawActionResult(
                action=action,
                goal_id=str(data.get("goal_id", "")),
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.0)),
                timestamp=time.monotonic(),
            )
        except (TypeError, ValueError, KeyError) as exc:
            _log.debug("openclaw_parse_action_failed", error=str(exc))
            return None
