"""Factory builders — BDI agent and metacognitive core."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from mousedroid.agents.base import AgentProtocol
    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.telemetry.metrics.registry import MetricsRegistry
    from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)

BDI_WEIGHT_FILENAMES: tuple[str, ...] = (
    "belief.npz",
    "desire.npz",
    "intention.npz",
    "affect.npz",
)
"""The four ``NeuralBDI`` weight files, in the order they are downloaded.

Module-level so the download call, the local-existence probe and the SHA-256
gate all read one list instead of three copies drifting apart. The names are
the BDI model's own on-disk contract (``cognitive/bdi_model.py``), not an
operator tunable, so they are not a config field."""


def build_agent(cfg: Settings, world_model: WorldModelProtocol) -> AgentProtocol:
    """Build navigation agent for configured platform.

    Args:
        cfg: Root settings.
        world_model: World model for planning.

    Returns:
        Agent conforming to ``AgentProtocol``.
    """
    from mousedroid.agents.navigation import MouseDroidNavigationAgent
    from mousedroid.world_model.mcts import MCTSPlanner

    planner = MCTSPlanner(cfg.mcts, world_model, action_dim=cfg.model.action_dim)
    return MouseDroidNavigationAgent(planner, cfg)


def _gate_bdi_weights(
    cfg: Settings,
    weights_dir: Path,
    *,
    metrics: MetricsRegistry | None,
) -> None:
    """Verify the BDI weight files against their SHA-256 manifest.

    This path executes on **every boot** — ``config/jetson_production.yaml``
    sets ``cognitive.enabled: true`` and ``auto_download: true`` — and until
    now performed no integrity check and pinned no revision, while the
    fail-closed machinery already existed on the OTA weight path. A resolvable
    digest is always enforced; an unresolvable one is governed by
    ``cfg.cognitive.require_sha256_manifest`` (default ``False``, preserving
    today's behaviour).

    Args:
        cfg: Root settings.
        weights_dir: Directory holding the four ``.npz`` files.
        metrics: Optional registry for the mismatch counter.

    Raises:
        ArtifactIntegrityError: Digest mismatch, or no digest under a strict
            policy.
    """
    from mousedroid.utils.artifact_integrity import enforce_artifact_digests

    enforce_artifact_digests(
        [weights_dir / name for name in BDI_WEIGHT_FILENAMES],
        manifest_path=weights_dir / cfg.cognitive.sha256_manifest_filename,
        artifact="bdi_weights",
        repo_id=cfg.cognitive.huggingface_repo,
        revision=cfg.cognitive.huggingface_revision,
        require_manifest=cfg.cognitive.require_sha256_manifest,
        metrics=metrics,
    )


def _download_bdi_weights(cfg: Settings, weights_dir: Path) -> bool:
    """Fetch the BDI weight set, then its manifest, at the pinned revision.

    The manifest fetch is best-effort and never flips the return value: a repo
    that has not published one yet must not become unbootable the moment
    verification lands. Its absence is a *policy* decision made in
    :func:`_gate_bdi_weights`. Fetching it into ``weights_dir`` also means the
    next boot verifies from the local copy with no network call.

    Args:
        cfg: Root settings.
        weights_dir: Directory the files land in.

    Returns:
        ``True`` when all four weight files downloaded successfully.
    """
    from mousedroid.utils import download_weights_from_huggingface

    cognitive = cfg.cognitive
    success = download_weights_from_huggingface(
        repo_id=cognitive.huggingface_repo,
        filenames=list(BDI_WEIGHT_FILENAMES),
        cache_dir=weights_dir,
        subfolder=cognitive.huggingface_subfolder,
        local_dir=weights_dir.parent,
        max_retries=cognitive.download_max_retries,
        backoff_base=cognitive.download_backoff_base,
        revision=cognitive.huggingface_revision,
    )
    if not success:
        return False
    manifest_fetched = download_weights_from_huggingface(
        repo_id=cognitive.huggingface_repo,
        filenames=[cognitive.sha256_manifest_filename],
        cache_dir=weights_dir,
        subfolder=cognitive.huggingface_subfolder,
        local_dir=weights_dir.parent,
        max_retries=cognitive.download_max_retries,
        backoff_base=cognitive.download_backoff_base,
        revision=cognitive.huggingface_revision,
    )
    _log.info(
        "bdi_weights_manifest_fetch",
        repo_id=cognitive.huggingface_repo,
        revision=cognitive.huggingface_revision,
        filename=cognitive.sha256_manifest_filename,
        fetched=manifest_fetched,
    )
    return True


def _resolve_bdi_weights(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> tuple[NeuralBDI, str]:
    """Resolve BDI model weights: local, HuggingFace, or random.

    Both weight-bearing branches pass through :func:`_gate_bdi_weights` before
    the weights reach ``NeuralBDI``, and the Hugging Face fetch is pinned to
    ``cfg.cognitive.huggingface_revision``.

    Args:
        cfg: Root settings.
        metrics: Optional metrics registry. Keyword-only and defaulted to
            ``None`` so every pre-existing ``_resolve_bdi_weights(cfg)`` call
            site keeps working; when supplied, a refused weight set increments
            ``mousedroid_model_artifact_sha256_mismatches_total``.

    Returns:
        Tuple of ``(NeuralBDI instance, weights_source_label)``.

    Raises:
        ArtifactIntegrityError: The local or downloaded weights failed the
            SHA-256 gate.
    """
    from pathlib import Path

    from mousedroid.cognitive.bdi_model import NeuralBDI
    from mousedroid.utils import weights_exist_locally

    weights_dir = Path(cfg.cognitive.weights_dir)

    if weights_exist_locally(weights_dir, list(BDI_WEIGHT_FILENAMES)):
        _log.info("cognitive_core_loading_local_weights", weights_dir=str(weights_dir))
        _gate_bdi_weights(cfg, weights_dir, metrics=metrics)
        return NeuralBDI(weights_dir=weights_dir), "local"

    if cfg.cognitive.auto_download and _download_bdi_weights(cfg, weights_dir):
        _log.info(
            "cognitive_core_loaded_from_huggingface",
            repo_id=cfg.cognitive.huggingface_repo,
            revision=cfg.cognitive.huggingface_revision,
            weights_dir=str(weights_dir),
        )
        _gate_bdi_weights(cfg, weights_dir, metrics=metrics)
        return NeuralBDI(weights_dir=weights_dir), "huggingface"

    _log.warning(
        "weights_not_found_using_random_initialization",
        weights_dir=str(weights_dir),
        auto_download=cfg.cognitive.auto_download,
    )
    return NeuralBDI(), "random"


def build_cognitive_core(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> CognitiveCore:
    """Build cognitive core with optional weight loading from HuggingFace.

    Args:
        cfg: Root settings.
        metrics: Optional metrics registry, threaded to the BDI weight
            resolver so a refused weight set is observable on ``/metrics``.
            Keyword-only and defaulted to ``None`` so every pre-existing
            ``build_cognitive_core(cfg)`` call site keeps working unchanged.

    Returns:
        Fully configured ``CognitiveCore``.
    """
    from mousedroid.cognitive.cognitive_core import CognitiveCore
    from mousedroid.cognitive.constitutional_rl import (
        ConstitutionalChecker,
        ConstitutionalRLConfig,
        PolicyMLP,
    )
    from mousedroid.cognitive.metacognitive import MetacognitiveModel

    _log.info(
        "cognitive_core_init_starting",
        weights_dir=str(cfg.cognitive.weights_dir),
        auto_download=cfg.cognitive.auto_download,
    )

    bdi, weights_source = _resolve_bdi_weights(cfg, metrics=metrics)

    policy = PolicyMLP(
        action_dim=cfg.model.action_dim,
        input_dim=cfg.model.belief_dim,
    )
    core = CognitiveCore(
        bdi=bdi,
        metacog=MetacognitiveModel(
            n_capabilities=cfg.metacognitive.n_capabilities,
            loop_score_scale=cfg.metacognitive.loop_score_scale,
        ),
        checker=ConstitutionalChecker(
            config=ConstitutionalRLConfig(
                speed_ceiling_mps=cfg.safety.max_velocity_mps,
                battery_min_v=cfg.safety.battery_critical_v,
            ),
        ),
        policy=policy,
    )
    _log.info(
        "cognitive_core_initialized",
        weights_source=weights_source,
        belief_dim=cfg.model.belief_dim,
        desire_dim=cfg.model.desire_dim,
        intention_classes=cfg.model.intention_classes,
    )
    return core
