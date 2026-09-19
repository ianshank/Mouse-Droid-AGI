"""Factory builders — world model, vision features, rover simulation environment.

RSSM/latent-dynamics engine, vision feature extraction, and the rover sim env.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.sim.protocols import ROVER_RSSM_PHYSICS_BACKENDS

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.hardware.camera.feature_extractor import FeatureExtractorProtocol
    from mousedroid.sim.protocols import RoverEnvProtocol
    from mousedroid.telemetry.metrics.registry import MetricsRegistry
    from mousedroid.world_model.protocol import LatentContextProtocol, WorldModelProtocol
    from mousedroid.world_model.rssm import RSSM

_log = get_logger(__name__)


def _rssm_lidar_update(cfg: Settings) -> dict[str, object]:
    """Size RSSM lidar from the backend that actually emits the scan.

    MuJoCo emits :attr:`MujocoSimConfig.lidar_num_sectors`. Isaac emits
    :attr:`RoverObservationConfig.lidar_num_sectors`. Disabled LiDAR
    keeps ``lidar_dim=0`` so the adapter's empty tensor matches the model.

    Args:
        cfg: Root settings.

    Returns:
        Partial ``ModelConfig`` update. Empty when the rover is absent or
        the backend is mock (default-OFF, byte-identical lidar_dim).
    """
    rover = cfg.rover
    if rover is None or rover.sim.backend not in ROVER_RSSM_PHYSICS_BACKENDS:
        return {}
    if not rover.observation.include_lidar_sectors:
        return {"lidar_dim": 0, "lidar_proj_dim": 0}
    if rover.sim.backend == "mujoco":
        sectors = rover.sim.mujoco.lidar_num_sectors
    else:
        sectors = rover.observation.lidar_num_sectors
    return {
        "lidar_dim": sectors,
        "lidar_proj_dim": cfg.model.lidar_proj_dim,
    }


def build_world_model(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> WorldModelProtocol:
    """Build world model for configured platform.

    Dispatch order:

    1. ``cfg.world_model.engine == "onnx_trt"`` — construct
       :class:`~mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
       backed by the exported ``.onnx`` at ``cfg.world_model.onnx_path``.
       The runtime is constructed cheaply (no ORT import at this point)
       and warms up on first ``observe_step()`` call. Requires
       ``cfc_hidden_dim > 0`` because the ONNX export is built from
       :class:`DualStreamRSSM`.
    2. ``cfg.world_model.engine == "torch"`` (default) AND
       ``cfc_hidden_dim > 0`` — construct :class:`DualStreamRSSM`.
    3. Fallback — construct the classic :class:`~mousedroid.world_model.rssm.RSSM`.

    Default behavior (``engine="torch"``) is byte-identical to pre-B2:
    existing ``config/*.yaml`` files that omit the ``world_model:`` block
    load unchanged.

    Args:
        cfg: Root settings.
        metrics: Optional metrics registry. When supplied, the constructed
            engine reports ``observe_step`` latency into
            ``mousedroid_world_model_observe_step_seconds``. Keyword-only and
            defaulted to ``None`` so every pre-existing
            ``build_world_model(cfg)`` call site keeps working unchanged; a
            ``None`` registry disables timing with no hot-path cost. Callers
            that own a registry (the orchestrator factory) should pass it —
            without it that histogram has no writer and the
            ``WorldModelObserveStepLatencyHigh`` alert cannot fire.

    Returns:
        World model conforming to ``WorldModelProtocol``.
    """
    engine = cfg.world_model.engine
    if engine == "onnx_trt":
        return _build_onnx_world_model(cfg, metrics=metrics)
    if engine != "torch":
        # Pydantic Literal["torch", "onnx_trt"] should catch this earlier,
        # but defend in depth so dynamic instantiation doesn't drift past
        # the dispatcher.
        msg = f"Unknown world_model.engine {engine!r}"
        raise ValueError(msg)

    if cfg.model.cfc_hidden_dim > 0:
        try:
            from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
        except ImportError:
            _log.warning(
                "dual_stream_unavailable_falling_back_to_rssm",
                reason="ncps package not installed (pip install ncps)",
                requested_cfc_dim=cfg.model.cfc_hidden_dim,
            )
        else:
            _log.info(
                "world_model_engine_selected",
                engine="torch",
                gru_dim=cfg.model.hidden_dim,
                cfc_dim=cfg.model.cfc_hidden_dim,
            )
            return DualStreamRSSM(cfg.model, metrics=metrics)

    from mousedroid.world_model.rssm import RSSM

    return RSSM(cfg.model, metrics=metrics)


def build_latent_context(cfg: Settings) -> LatentContextProtocol | None:
    """Build the bounded-context latent memory (F-023), or ``None`` when off.

    Returns ``None`` when the ``world_model_memory`` block is absent OR
    ``enabled=False`` — the orchestrator tick path stays byte-identical to
    pre-feature. The memory is engine-agnostic: it operates on the
    orchestrator-carried ``(h, z)`` tensors, so ``h_dim`` is the combined
    ``hidden_dim + cfc_hidden_dim`` (matching the orchestrator's carried
    state for the dual-stream engine).

    Args:
        cfg: Root settings.

    Returns:
        A :class:`LatentContextProtocol` implementation, or ``None``.
    """
    memory_cfg = cfg.world_model_memory
    if memory_cfg is None or not memory_cfg.enabled:
        return None
    from mousedroid.world_model.bounded_context import BoundedContextMemory

    h_dim = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    context = BoundedContextMemory(memory_cfg, h_dim=h_dim, z_dim=cfg.model.latent_dim)
    _log.info(
        "latent_context_enabled",
        h_dim=h_dim,
        z_dim=cfg.model.latent_dim,
        recent_size=memory_cfg.recent_size,
        blend_weight=memory_cfg.blend_weight,
    )
    return context


def build_rssm_trainable(cfg: Settings) -> RSSM:
    """Build the concrete trainable RSSM for physics-sim dynamics pretraining.

    Unlike :func:`build_world_model` (which returns a ``WorldModelProtocol``
    wrapper for deployment), this returns the concrete ``nn.Module`` so the
    pretrainer can call ``train_sequence`` + backprop. Vision is disabled
    (``vision_dim=0`` paired with ``vision_proj_dim=0`` per the schema
    validator) — the sim has no camera; the dynamics core is what gets
    pretrained. Operator pretrain knobs from :class:`TrainingConfig` are copied
    onto the model config so they live in one place (``training:``).

    ``lidar_dim`` follows the emitting backend: MuJoCo uses
    :attr:`MujocoSimConfig.lidar_num_sectors`; Isaac uses
    :attr:`RoverObservationConfig.lidar_num_sectors`. Disabled
    ``include_lidar_sectors`` keeps ``lidar_dim=0``. Mock / absent rover
    keeps the model default.

    Args:
        cfg: Root settings.

    Returns:
        A concrete :class:`~mousedroid.world_model.rssm.RSSM` with vision off.
    """
    from mousedroid.world_model.rssm import RSSM

    update: dict[str, object] = {
        "vision_dim": 0,
        "vision_proj_dim": 0,
        "kl_beta": cfg.training.kl_beta,
        "kl_free_nats": cfg.training.rssm_free_nats,
        "kl_balance_alpha": cfg.training.rssm_kl_balance_alpha,
    }
    update.update(_rssm_lidar_update(cfg))
    model_cfg = cfg.model.model_copy(update=update)
    return RSSM(model_cfg)


def build_rssm_vision_finetune(cfg: Settings, checkpoint: Path) -> RSSM:
    """Load a vision-OFF pretrained RSSM and migrate it to a vision-ON model.

    Uses :func:`~mousedroid.world_model.checkpoint_migration.load_rssm_with_migration`
    to transfer the dynamics core (gru/posterior/prior/decoder/reward) verbatim,
    copy retained-modality fusion columns, and Kaiming-init the new vision
    columns + ``vision_proj``. Vision dim = ``cfg.camera.feature_dim`` so the
    model matches the sim ``MeanPoolExtractor`` output; lidar mirrors the rover.

    Args:
        cfg: Root settings.
        checkpoint: Path to the vision-OFF pretrained RSSM checkpoint.

    Returns:
        A vision-ON :class:`~mousedroid.world_model.rssm.RSSM` ready to fine-tune.
    """
    import torch

    from mousedroid.world_model.checkpoint_migration import load_rssm_with_migration

    update: dict[str, object] = {
        "vision_dim": cfg.camera.feature_dim,
        # Use the configured projection dim directly; the ModelConfig validator
        # rejects a zero proj_dim paired with a nonzero modality dim, so a
        # misconfig surfaces explicitly instead of being patched to a literal.
        "vision_proj_dim": cfg.model.vision_proj_dim,
        "kl_beta": cfg.training.kl_beta,
        "kl_free_nats": cfg.training.rssm_free_nats,
        "kl_balance_alpha": cfg.training.rssm_kl_balance_alpha,
    }
    update.update(_rssm_lidar_update(cfg))
    model_cfg = cfg.model.model_copy(update=update)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_rssm_with_migration(checkpoint, model_cfg, device)


def build_vision_feature_extractor(cfg: Settings) -> FeatureExtractorProtocol:
    """Build the sim vision feature extractor for RSSM vision-on fine-tuning.

    Returns the same non-learned :class:`MeanPoolExtractor` the deployed
    ``mean_pool`` camera path uses (mean-pool → L2), so rendered-sim and real
    ``vision_features`` share a distribution by construction — no CNN to train.
    Dims come from :class:`CameraConfig` (invariant #3).

    Args:
        cfg: Root settings.

    Returns:
        A ``FeatureExtractorProtocol`` producing ``cfg.camera.feature_dim`` features.
    """
    from mousedroid.hardware.camera.feature_extractor import MeanPoolExtractor

    return MeanPoolExtractor(cfg.camera.feature_dim, l2_normalize=cfg.camera.l2_normalize)


def _build_onnx_world_model(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> WorldModelProtocol:
    """Construct the ONNX runtime world model.

    Resolves ``cfg.world_model.onnx_path`` (filesystem-first, HF Hub
    fallback) and hands it to :class:`DualStreamRSSMOnnx`. The runtime
    is constructed lazily — ``onnxruntime`` is not imported here.
    """
    if cfg.model.cfc_hidden_dim <= 0:
        msg = (
            "world_model.engine='onnx_trt' requires model.cfc_hidden_dim > 0 "
            "(the ONNX export is built from DualStreamRSSM, which requires "
            "the CfC stream). Set cfc_hidden_dim in your config YAML or "
            "switch to engine='torch'."
        )
        raise ValueError(msg)

    from mousedroid.world_model.composite import CompositeWorldModel
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
    from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx

    model_path = _resolve_world_model_onnx_path(cfg, metrics=metrics)

    # observe_step path: ONNX-accelerated (the hot 30Hz tick benefit).
    observe_engine = DualStreamRSSMOnnx(
        model_path=model_path,
        cfg=cfg.model,
        warmup_iterations=cfg.world_model.onnx_warmup_iterations,
        metrics=metrics,
    )
    # imagine_step path: PyTorch DualStreamRSSM. The ONNX export (B2
    # Story 1) is scoped to observe_step only, so MCTS rollouts on the
    # fallback planner path need the PyTorch graph. The composite also
    # forwards get_safety_trace here, but that method has no production
    # caller today -- mousedroid.safety.monitor never calls it, so it is
    # not a reason this engine is retained. Both engines share the same
    # ModelConfig so dimensions stay consistent across the composition
    # boundary.
    # No ``metrics=`` here on purpose: this engine serves ``imagine_step``
    # (MCTS rollouts, ~500-650 calls per plan) and ``get_safety_trace``. Timing
    # it into the observe-step histogram would swamp the per-tick signal the
    # deadline alert reads with rollout samples.
    imagine_engine = DualStreamRSSM(cfg.model)
    imagine_engine.train(False)

    _log.info(
        "world_model_engine_selected",
        engine="onnx_trt",
        model_path=str(model_path),
        cfc_dim=cfg.model.cfc_hidden_dim,
        composite=True,
        observe_engine=type(observe_engine).__name__,
        imagine_engine=type(imagine_engine).__name__,
    )
    return CompositeWorldModel(
        observe_engine=observe_engine,
        imagine_engine=imagine_engine,
    )


def _gate_world_model_onnx_artifact(
    cfg: Settings,
    model_path: Path,
    manifest_path: Path,
    *,
    metrics: MetricsRegistry | None,
) -> None:
    """Apply the artifact contract + SHA-256 gate to a resolved ``.onnx``.

    Two independent checks, in the order that fails cheapest first:

    1. Suffix contract — the resolved file must be an ``.onnx``. A digest can
       be perfectly valid for a ``.pt`` checkpoint, so this is not redundant
       with the digest check; the runtime spec requires both.
    2. SHA-256 against the manifest. A resolvable digest is always enforced;
       an unresolvable one is governed by
       ``cfg.world_model.onnx_require_sha256_manifest`` (default ``False``,
       so today's behaviour — load with a warning — is preserved).

    Args:
        cfg: Root settings.
        model_path: Resolved local artifact.
        manifest_path: Local SHA-256 manifest for that artifact. Need not
            exist; absence is a policy decision, not an error here.
        metrics: Optional registry for the mismatch counter.

    Raises:
        ArtifactContractError: Resolved file is not an ``.onnx``.
        ArtifactIntegrityError: Digest mismatch, or no digest under a strict
            policy.
    """
    from mousedroid.utils.artifact_integrity import (
        enforce_artifact_digests,
        require_artifact_suffix,
    )
    from mousedroid.world_model.onnx_io import OBSERVE_STEP_ARTIFACT_SUFFIX

    wm = cfg.world_model
    require_artifact_suffix(
        model_path,
        OBSERVE_STEP_ARTIFACT_SUFFIX,
        repo_id=wm.onnx_repo_id,
    )
    enforce_artifact_digests(
        [model_path],
        manifest_path=manifest_path,
        artifact="world_model_onnx",
        repo_id=wm.onnx_repo_id,
        revision=wm.onnx_revision,
        require_manifest=wm.onnx_require_sha256_manifest,
        metrics=metrics,
    )


def _resolve_explicit_onnx_path(
    cfg: Settings,
    explicit: Path,
    *,
    metrics: MetricsRegistry | None,
) -> Path:
    """Gate an operator-supplied ``onnx_path`` without reaching the network.

    A missing file is deliberately NOT an error here: the pre-existing
    contract is that the runtime's ``warmup()`` surfaces the clear
    ``FileNotFoundError`` (pinned by
    ``tests/unit/factory/test_factory_world_model_engine.py``), and the
    composite's PyTorch imagine engine stays usable meanwhile. We gate the
    bytes we actually have.

    The manifest is looked for beside the artifact, by the same configured
    filename used for the Hub fetch, so an operator who exports locally can
    drop a ``sha256.txt`` next to the graph and get the same enforcement.
    """
    if not explicit.is_file():
        return explicit
    manifest_path = explicit.parent / cfg.world_model.onnx_sha256_manifest_filename
    _gate_world_model_onnx_artifact(cfg, explicit, manifest_path, metrics=metrics)
    return explicit


def _fetch_world_model_onnx_manifest(cfg: Settings, cache_dir: Path) -> None:
    """Best-effort fetch of the SHA-256 manifest beside the artifact.

    Deliberately non-fatal: a repo that has not published a manifest yet must
    not become unbootable the moment digest verification lands. The absent
    manifest is then handled by the *policy* in ``enforce_artifact_digests``,
    which is where an operator can turn it into a hard failure. Fetching the
    manifest into the cache directory also means later boots verify from the
    local copy with no network call at all.
    """
    from mousedroid.utils.weights_manager import download_weights_from_huggingface

    wm = cfg.world_model
    fetched = download_weights_from_huggingface(
        repo_id=wm.onnx_repo_id,
        filenames=[wm.onnx_sha256_manifest_filename],
        cache_dir=cache_dir,
        local_dir=cache_dir,
        revision=wm.onnx_revision,
    )
    _log.info(
        "world_model_onnx_manifest_fetch",
        repo_id=wm.onnx_repo_id,
        revision=wm.onnx_revision,
        filename=wm.onnx_sha256_manifest_filename,
        fetched=fetched,
    )


def _resolve_world_model_onnx_path(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> Path:
    """Resolve + integrity-gate the .onnx artifact for the ONNX runtime engine.

    Resolution order:

    1. ``cfg.world_model.onnx_path`` when set — use it directly. If the
       file is missing, the runtime's :meth:`warmup` will raise
       ``FileNotFoundError`` so operators get a clear error from the
       runtime, not a confusing ``hf_hub_download`` traceback.
    2. HF Hub download via
       ``cfg.world_model.onnx_repo_id``/``cfg.world_model.onnx_filename``,
       pinned to ``cfg.world_model.onnx_revision``. Mirrors the [vla] pattern
       at ``_build_distilled_onnx_vla``. Cached under
       ``weights/dual_stream_rssm/`` so the same file is reused across runs
       without re-downloading.

    Every branch that ends in a usable local file — explicit path, cache hit,
    and fresh download alike — passes through
    :func:`_gate_world_model_onnx_artifact` before the path is returned. The
    pre-gate version of this function checked only ``model_path.is_file()``,
    which ADR-008's "Negative" section already named as "wrong weights =
    wrong inference, silently".

    Args:
        cfg: Root settings.
        metrics: Optional registry so a refused artifact increments
            ``mousedroid_model_artifact_sha256_mismatches_total``. ``None``
            disables only the counter, never the refusal.

    Returns:
        Local path of the integrity-gated artifact.

    Raises:
        ArtifactMissingError: The configured filename could not be fetched.
        ArtifactContractError: The resolved file is not an ``.onnx``.
        ArtifactIntegrityError: Digest mismatch, or an unverifiable artifact
            under a strict policy.
    """
    from mousedroid.utils.artifact_integrity import ArtifactMissingError

    wm = cfg.world_model
    explicit = wm.onnx_path
    if explicit is not None:
        return _resolve_explicit_onnx_path(cfg, Path(explicit), metrics=metrics)

    # HF Hub auto-download fallback. Reuses the same
    # ``download_weights_from_huggingface`` helper the VLA path uses, so
    # retries / auth tokens / progress bars work identically.
    from mousedroid.utils.weights_manager import (
        download_weights_from_huggingface,
    )

    # Operator-tunable per deployment via ``cfg.world_model.onnx_cache_dir``
    # (default ``weights/dual_stream_rssm``). Mirrors the VLA pattern at
    # ``_build_distilled_onnx_vla`` so Jetson deployments can repoint both
    # caches under ``/opt/mousedroid/weights/...`` in one place.
    cache_dir = Path(wm.onnx_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / wm.onnx_filename
    manifest_path = cache_dir / wm.onnx_sha256_manifest_filename

    if model_path.is_file():
        _log.info(
            "world_model_onnx_cache_hit",
            cache_path=str(model_path),
            repo_id=wm.onnx_repo_id,
            revision=wm.onnx_revision,
        )
        # Local manifest only — a cached artifact must stay bootable offline.
        _gate_world_model_onnx_artifact(cfg, model_path, manifest_path, metrics=metrics)
        return model_path

    _log.info(
        "world_model_onnx_download_start",
        repo_id=wm.onnx_repo_id,
        filename=wm.onnx_filename,
        cache_dir=str(cache_dir),
        revision=wm.onnx_revision,
    )
    success = download_weights_from_huggingface(
        repo_id=wm.onnx_repo_id,
        filenames=[wm.onnx_filename],
        cache_dir=cache_dir,
        # Force flat layout so model_path.is_file() check succeeds.
        # Without local_dir, hf_hub_download uses its blob/snapshot
        # cache layout and the file would not be at the expected path.
        local_dir=cache_dir,
        revision=wm.onnx_revision,
    )
    if not success or not model_path.is_file():
        msg = (
            f"failed to download world-model ONNX artifact "
            f"({wm.onnx_repo_id}/{wm.onnx_filename} at revision "
            f"{wm.onnx_revision}) into {cache_dir}. Set world_model.onnx_path "
            f"to a local path or run scripts/export_dual_stream_rssm_onnx.py "
            f"--push-to-hf to publish a fresh artifact first."
        )
        raise ArtifactMissingError(msg)
    _fetch_world_model_onnx_manifest(cfg, cache_dir)
    _gate_world_model_onnx_artifact(cfg, model_path, manifest_path, metrics=metrics)
    _log.info(
        "world_model_onnx_downloaded",
        path=str(model_path),
        repo_id=wm.onnx_repo_id,
        revision=wm.onnx_revision,
    )
    return model_path


def build_rover_env(cfg: Settings) -> RoverEnvProtocol:
    """Build the rover simulation environment selected by ``cfg.rover.sim.backend``.

    Backends:
        - ``"mock"`` (default): NumPy-only kinematic integrator. Has no
          physics or GPU dependency and is the only backend used in CI.
        - ``"isaac_lab"``: Isaac Lab env; requires
          ``pip install -e ".[isaac]"`` on a workstation with NVIDIA
          Isaac Lab prerequisites. Construction is lazy (``env.build()``).
        - ``"mujoco"``: MuJoCo skid-steer physics (CHARTER M5 CI-trainable
          backend). Wired through :class:`RoverMuJoCoEnv`.

    Args:
        cfg: Root settings. ``cfg.rover`` must be populated.

    Returns:
        Environment conforming to :class:`RoverEnvProtocol`.

    Raises:
        ValueError: If ``cfg.rover`` is ``None`` or the backend name is unknown.
    """
    if cfg.rover is None:
        msg = (
            "rover config required for build_rover_env; set the top-level "
            "'rover:' block in your YAML or pass RoverConfig() directly."
        )
        raise ValueError(msg)

    backend = cfg.rover.sim.backend
    if backend == "mock":
        from mousedroid.sim.mock_rover_env import MockRoverEnv

        _log.info(
            "rover_env_mock_built",
            mode=cfg.rover.action.mode,
            obs_keys=list(cfg.rover.observation.enabled_keys()),
        )
        return MockRoverEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
        )

    if backend == "isaac_lab":
        from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv

        env = RoverIsaacLabEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
            domain_randomization=cfg.domain_randomization,
        )
        _log.info(
            "rover_env_isaaclab_built",
            num_envs=cfg.rover.sim.num_envs,
            dr_enabled=cfg.domain_randomization.enabled,
        )
        return env

    if backend == "mujoco":
        from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv

        mj_env = RoverMuJoCoEnv(
            cfg.rover,
            wheel_radius_m=cfg.robot.wheel_radius_m,
            track_width_m=cfg.robot.track_width_m,
        )
        _log.info(
            "rover_env_mujoco_built",
            lidar_sectors=cfg.rover.sim.mujoco.lidar_num_sectors,
            dr_enabled=cfg.domain_randomization.enabled,
        )
        return mj_env

    msg = f"unknown rover sim backend: {backend!r}"
    raise ValueError(msg)
