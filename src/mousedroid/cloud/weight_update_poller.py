"""HuggingFace Hub weight-update poller (Tier C1).

Implements :class:`HuggingFaceWeightUpdatePoller` — a background poll loop
that watches a HuggingFace Hub repo for a new revision of a target artifact
(policy or world-model), downloads it with SHA-256 integrity verification,
and surfaces it as a :class:`PendingWeightUpdate` for the orchestrator to
swap at a tick boundary.

Design constraints (per CLAUDE.md + Tier C1 plan):

* Lazy import of ``huggingface_hub`` — mirrors VLA / B2 pattern, never
  top-level inside src/.
* No hardcoded values — every threshold / repo-id / filename / interval /
  bucket boundary comes from :class:`WeightUpdatePollConfig`.
* Structured logging only — never f-strings inside log calls.
* Default ``poll_interval_s = 0.0`` short-circuits ``start()`` so existing
  YAML files produce byte-identical pre-Tier-C1 behaviour.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.cloud.protocol import PendingWeightUpdate
from mousedroid.logging.setup import get_logger
from mousedroid.utils.weights_manager import verify_sha256

if TYPE_CHECKING:
    from mousedroid.config.schema import WeightUpdatePollConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


class HuggingFaceWeightUpdatePoller:
    """Background poller that fetches latest HF Hub revisions and verifies SHA-256.

    One instance polls one ``(repo_id, filename)`` pair tagged with one
    ``engine_type`` (``"policy"`` or ``"world_model"``). The factory builds
    two instances when both engines have non-zero poll intervals.

    Conforms to :class:`WeightUpdatePollerProtocol` structurally.
    """

    def __init__(
        self,
        cfg: WeightUpdatePollConfig,
        *,
        repo_id: str,
        filename: str,
        engine_type: str,
        metrics: MetricsRegistry | None = None,
        hf_api_factory: Any | None = None,
        hf_download: Any | None = None,
    ) -> None:
        """Construct the poller.

        Args:
            cfg: :class:`WeightUpdatePollConfig` instance (single source of
                truth for poll cadence, cache dir, SHA-256 manifest filename,
                retry counts).
            repo_id: Target HF Hub repo (typically
                ``cfg.policy_repo_id`` or ``cfg.world_model_repo_id``).
            filename: Target filename within the repo.
            engine_type: ``"policy"`` or ``"world_model"`` — propagates to
                :class:`PendingWeightUpdate.engine_type` and the Prometheus
                ``engine_type`` label on swap counters.
            metrics: Optional :class:`MetricsRegistry`. When supplied the
                poller increments download / SHA-mismatch counters and
                observes download-latency histogram samples.
            hf_api_factory: Override for ``huggingface_hub.HfApi``. When
                ``None`` (production) the implementation lazy-imports
                ``huggingface_hub``. Test stubs pass a callable returning
                a stub HfApi-like object.
            hf_download: Override for ``huggingface_hub.hf_hub_download``.
                Same lazy-import contract as ``hf_api_factory``.
        """
        self._cfg = cfg
        self._repo_id = repo_id
        self._filename = filename
        self._engine_type = engine_type
        self._metrics = metrics
        self._hf_api_factory_override = hf_api_factory
        self._hf_download_override = hf_download

        self._pending_update: PendingWeightUpdate | None = None
        self._last_known_sha: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._cache_dir = Path(cfg.cache_dir).resolve()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin the background poll loop.

        No-op when ``cfg.poll_interval_s == 0.0`` so default deployments
        skip the poller entirely (byte-identical pre-Tier-C1 behaviour).
        """
        if self._cfg.poll_interval_s <= 0.0:
            _log.info(
                "cloud_weight_update_poller_disabled",
                repo_id=self._repo_id,
                reason="poll_interval_s_zero",
            )
            return
        if self._task is not None and not self._task.done():
            _log.debug("cloud_weight_update_poller_already_running", repo_id=self._repo_id)
            return
        self._stop_event.clear()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(
            self._run(), name=f"weight_update_poller_{self._engine_type}"
        )
        _log.info(
            "cloud_weight_update_poll_started",
            repo_id=self._repo_id,
            filename=self._filename,
            engine_type=self._engine_type,
            interval_s=self._cfg.poll_interval_s,
        )

    async def stop(self) -> None:
        """Stop the poll loop + cancel any in-flight download."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        _log.info("cloud_weight_update_poll_stopped", repo_id=self._repo_id)

    # ------------------------------------------------------------------
    # Public surface for the orchestrator
    # ------------------------------------------------------------------

    @property
    def pending_update(self) -> PendingWeightUpdate | None:
        """Latest verified update awaiting orchestrator swap (``None`` if none)."""
        return self._pending_update

    def acknowledge_swap(self, update: PendingWeightUpdate) -> None:
        """Clear the pending slot once the orchestrator has applied ``update``.

        Identity check (``is``) on the slot — multiple downloads landing in
        quick succession should only clear the matching one.
        """
        if self._pending_update is update:
            self._pending_update = None
            _log.debug(
                "cloud_weight_update_swap_acknowledged",
                repo_id=update.repo_id,
                revision=update.revision,
                engine_type=update.engine_type,
            )

    # ------------------------------------------------------------------
    # Internal poll loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Background poll loop — sleeps ``poll_interval_s`` between cycles."""
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=broad-except
                _log.warning(
                    "cloud_weight_update_poll_cycle_failed",
                    repo_id=self._repo_id,
                    exc_info=True,
                )
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._cfg.poll_interval_s)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise

    async def poll_once(self) -> bool:
        """Run a single poll cycle. Returns ``True`` if a new update landed.

        Public so the operator smoke harness + unit tests can drive one
        cycle directly without spinning up the background loop.
        """
        hf_api = self._resolve_hf_api()
        hf_download = self._resolve_hf_download()

        repo_info = await asyncio.to_thread(hf_api.repo_info, self._repo_id)
        latest_sha = getattr(repo_info, "sha", None)
        if not latest_sha:
            _log.warning(
                "cloud_weight_update_repo_info_missing_sha",
                repo_id=self._repo_id,
            )
            return False

        if latest_sha == self._last_known_sha:
            _log.debug(
                "cloud_weight_update_revision_unchanged",
                repo_id=self._repo_id,
                revision=latest_sha,
            )
            return False

        _log.info(
            "cloud_weight_update_new_revision",
            repo_id=self._repo_id,
            revision=latest_sha,
        )

        # Download the SHA-256 manifest first so we can verify the artifact.
        manifest_path = await asyncio.to_thread(
            hf_download,
            repo_id=self._repo_id,
            filename=self._cfg.sha256_manifest_filename,
            revision=latest_sha,
            local_dir=str(self._cache_dir),
        )
        expected_hex = Path(manifest_path).read_text(encoding="utf-8").strip().split()[0]

        # Time the artifact download for the latency histogram.
        download_start = time.perf_counter()
        artifact_path = await asyncio.to_thread(
            hf_download,
            repo_id=self._repo_id,
            filename=self._filename,
            revision=latest_sha,
            local_dir=str(self._cache_dir),
        )
        download_seconds = time.perf_counter() - download_start

        if self._metrics is not None:
            self._metrics.observe_cloud_weight_update_download_seconds(download_seconds)

        local_path = Path(artifact_path)
        if not verify_sha256(local_path, expected_hex, log_event_prefix="cloud_weight_update"):
            if self._metrics is not None:
                self._metrics.inc_cloud_weight_update_sha256_mismatch(self._repo_id)
            # Refuse the update; do NOT mark revision as seen so a corrected
            # upstream artifact will be picked up on the next cycle.
            return False

        if self._metrics is not None:
            self._metrics.inc_cloud_weight_update_download(self._repo_id)

        self._last_known_sha = latest_sha
        self._pending_update = PendingWeightUpdate(
            repo_id=self._repo_id,
            filename=self._filename,
            revision=latest_sha,
            sha256=expected_hex.lower(),
            local_path=local_path,
            downloaded_at=time.time(),
            engine_type=self._engine_type,
        )
        _log.info(
            "cloud_weight_update_swap_pending",
            repo_id=self._repo_id,
            revision=latest_sha,
            local_path=str(local_path),
            engine_type=self._engine_type,
        )
        return True

    # ------------------------------------------------------------------
    # Lazy import resolution — kept out of module top-level so importing
    # this file in tests / on the Jetson does NOT require huggingface_hub.
    # ------------------------------------------------------------------

    def _resolve_hf_api(self) -> Any:
        if self._hf_api_factory_override is not None:
            return self._hf_api_factory_override()
        from huggingface_hub import HfApi

        return HfApi()

    def _resolve_hf_download(self) -> Any:
        if self._hf_download_override is not None:
            return self._hf_download_override
        from huggingface_hub import hf_hub_download

        return hf_hub_download
