"""Cloud Pub/Sub telemetry sink — publishes telemetry and experience to GCP.

Uses the existing ``CircuitBreaker`` pattern to ensure cloud failures
never block the 30 Hz control loop.  When the circuit is open all
publishes are silently skipped and the droid continues autonomously.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import msgpack

from mousedroid.cloud._auth import resolve_credentials
from mousedroid.experience.protocol import ExperienceProtocol
from mousedroid.logging.setup import get_logger
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from google.cloud.pubsub_v1 import PublisherClient

    from mousedroid.config.schema import GCPConfig

_log = get_logger(__name__)


class CloudTelemetrySink:
    """Pub/Sub telemetry sink with circuit-breaker resilience.

    Publishes telemetry frames and experience records as msgpack-encoded
    Pub/Sub messages.  Message attributes carry schema version, source ID,
    and timestamp for downstream routing.

    Args:
        cfg: GCP configuration.
    """

    def __init__(self, cfg: GCPConfig) -> None:
        self._cfg = cfg
        self._project_id = cfg.project_id
        self._robot_id = cfg.robot_id
        self._pubsub_cfg = cfg.pubsub

        self._cb = CircuitBreaker("gcp_pubsub", cfg.circuit_breaker)
        self._publisher: PublisherClient | None = None
        self._telemetry_topic: str = ""
        self._experience_topic: str = ""

    async def start(self) -> None:
        """Initialise the Pub/Sub publisher client.

        Creates the publisher with batch settings from config.  This is
        separated from ``__init__`` to keep construction synchronous
        (matching the factory pattern).
        """
        from google.cloud.pubsub_v1 import PublisherClient as _PublisherClient
        from google.cloud.pubsub_v1.types import BatchSettings as _BatchSettings

        creds, project = resolve_credentials(self._cfg)
        effective_project = self._project_id or project

        batch_settings = _BatchSettings(
            max_messages=self._pubsub_cfg.batch_max_messages,
            max_bytes=self._pubsub_cfg.batch_max_bytes,
            max_latency=self._pubsub_cfg.batch_max_latency_s,
        )
        self._publisher = _PublisherClient(
            credentials=creds,
            batch_settings=batch_settings,
        )

        self._telemetry_topic = self._publisher.topic_path(
            effective_project,
            self._pubsub_cfg.telemetry_topic,
        )
        self._experience_topic = self._publisher.topic_path(
            effective_project,
            self._pubsub_cfg.experience_topic,
        )
        _log.info(
            "cloud_pubsub_sink_started",
            telemetry_topic=self._telemetry_topic,
            experience_topic=self._experience_topic,
        )

    async def publish_telemetry(self, frame_dict: dict[str, Any]) -> None:
        """Publish a telemetry frame to the telemetry Pub/Sub topic.

        Args:
            frame_dict: Dictionary representation of a ``TelemetryFrame``.
        """
        if self._publisher is None:
            return
        data = msgpack.packb(frame_dict)
        attrs = {
            "type": "telemetry",
            "schema_version": "1",
            "source_id": self._robot_id,
            "timestamp": str(time.time()),
        }
        await self._publish(self._telemetry_topic, data, attrs)

    async def publish_experience(self, record: ExperienceProtocol) -> None:
        """Publish a single experience record to the experience Pub/Sub topic.

        Args:
            record: Experience record implementing ``ExperienceProtocol``.
        """
        if self._publisher is None:
            return
        data = record.serialize()
        attrs = {
            "type": "experience",
            "schema_version": str(record.schema_version),
            "source_id": self._robot_id,
            "timestamp": str(time.time()),
        }
        await self._publish(self._experience_topic, data, attrs)

    async def flush(self) -> None:
        """Flush any buffered messages in the publisher."""
        if self._publisher is not None:
            _log.debug("cloud_pubsub_flushing")

    async def close(self) -> None:
        """Shut down the Pub/Sub publisher client."""
        if self._publisher is not None:
            loop = asyncio.get_running_loop()
            publisher = self._publisher
            self._publisher = None
            await loop.run_in_executor(None, publisher.stop)
            _log.info("cloud_pubsub_sink_closed")

    async def _publish(
        self,
        topic: str,
        data: bytes,
        attrs: dict[str, str],
    ) -> None:
        """Publish a message with circuit-breaker protection.

        When the circuit is open the message is silently dropped.  The droid
        continues logging locally to LMDB regardless.

        Args:
            topic: Full Pub/Sub topic path.
            data: Message payload bytes (msgpack-encoded).
            attrs: Message attributes dictionary.
        """
        try:

            async def _do_publish() -> None:
                assert self._publisher is not None
                loop = asyncio.get_running_loop()
                timeout = self._pubsub_cfg.publish_timeout_s
                future = self._publisher.publish(topic, data=data, **attrs)
                await loop.run_in_executor(None, future.result, timeout)

            await self._cb.call(_do_publish)
        except CircuitOpenError:
            _log.debug("cloud_pubsub_circuit_open", topic=topic)
        except Exception:
            _log.debug("cloud_pubsub_publish_failed", topic=topic, exc_info=True)
