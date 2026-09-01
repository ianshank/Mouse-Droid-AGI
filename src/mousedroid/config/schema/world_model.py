"""World-model configuration models.

RSSM/dual-stream engine selection, the F-023 bounded-context latent memory,
neural network dimensions (``ModelConfig``), MCTS planning, and dual-stream
training hyperparameters.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from mousedroid.config.schema._primitives import Self, StrictBaseModel
from mousedroid.constants import (
    DEFAULT_UCB_CANDIDATES as DEFAULT_UCB_CANDIDATES,
)
from mousedroid.constants import (
    DEFAULT_UCB_TARGET_MS as DEFAULT_UCB_TARGET_MS,
)


class WorldModelConfig(StrictBaseModel):
    """World-model runtime engine selector (Tier B2).

    Drives :func:`mousedroid.factory.build_world_model` dispatch. The
    default (``engine="torch"``) preserves byte-identical behavior with
    pre-B2 deployments: existing ``config/*.yaml`` files without a
    ``world_model:`` block load unchanged and continue to use the
    PyTorch :class:`~mousedroid.world_model.dual_stream_rssm.DualStreamRSSM`.

    Flip ``engine="onnx_trt"`` in ``config/jetson_production.yaml`` to
    serve ``observe_step`` from an exported ``.onnx`` via
    ``onnxruntime`` + TensorRT execution provider. ``imagine_step`` (MCTS
    rollouts) always runs on the PyTorch model regardless of engine; the
    factory wires both paths.
    """

    engine: Literal["torch", "onnx_trt"] = Field(
        "torch",
        description=(
            "World-model inference engine. 'torch' runs DualStreamRSSM "
            "as a PyTorch module (default, byte-identical pre-PR). "
            "'onnx_trt' loads an exported .onnx via onnxruntime with "
            "the TensorrtExecutionProvider -> CUDAExecutionProvider -> "
            "CPUExecutionProvider fallback chain."
        ),
    )
    onnx_path: str | None = Field(
        None,
        description=(
            "Filesystem path to the exported .onnx for engine='onnx_trt'. "
            "When None and engine='onnx_trt', the factory falls back to "
            "downloading from onnx_repo_id via HuggingFace Hub."
        ),
    )
    onnx_repo_id: str = Field(
        "ianshank/mousedroid-dual-stream-rssm",
        description=(
            "HuggingFace Hub repo holding the .onnx artifact for the "
            "ONNX runtime path. Mirrors the [vla] backend convention."
        ),
    )
    onnx_filename: str = Field(
        "observe_step.onnx",
        description="Filename inside the HF repo / cache directory.",
    )
    onnx_cache_dir: str = Field(
        "weights/dual_stream_rssm",
        description=(
            "Filesystem directory where the factory caches the HF-downloaded "
            ".onnx artifact. Mirrors VLAConfig.cache_dir. Operators can point "
            "this at /opt/mousedroid/weights/... on Jetson deployments so the "
            "download persists across container restarts."
        ),
    )
    onnx_warmup_iterations: int = Field(
        1,
        ge=0,
        description="Dummy inferences run during ONNX session warmup.",
    )


class WorldModelMemoryConfig(StrictBaseModel):
    """Bounded-context latent memory for the world model (F-023, default-OFF).

    Adapts the AlayaWorld sink-frame + compressed-history pattern to the rover's
    recurrent latent state: a persistent per-mission "sink" anchor plus a
    compressed rolling history (recent ring + one EMA long-summary vector) with
    a constant-size storage footprint, blended into the orchestrator-carried
    ``(h, z)`` at the observe seam. Default-OFF and backwards-compatible —
    wired as an ``Optional`` block on ``Settings`` so existing YAML loads
    byte-identically; when absent/disabled the factory returns ``None`` and the
    tick path is unchanged. The blend is a pure deterministic ``no_grad``
    tensor op (hot-loop invariant: no training, no sampling).
    """

    enabled: bool = Field(
        False,
        description="Master switch for the bounded-context latent memory (default-off)",
    )
    recent_size: int = Field(
        16,
        gt=0,
        description=(
            "Capacity of the recent-history ring (deque maxlen). Total memory "
            "footprint is recent_size + 2 vectors (ring + sink + EMA summary), "
            "constant with respect to rollout length."
        ),
    )
    stride: int = Field(
        8,
        gt=0,
        description=(
            "Fold the EMA long-summary (and emit the rate-limited "
            "latent_context_blend debug event) every N validated ticks"
        ),
    )
    long_ema_alpha: float = Field(
        0.05,
        gt=0,
        le=1,
        description=(
            "EMA weight for the long-term summary fold: long = (1 - alpha) * long + alpha * hz"
        ),
    )
    blend_weight: float = Field(
        0.1,
        ge=0,
        le=1,
        description=(
            "Lambda for the context blend h' = (1 - lambda) * h + lambda * c_h. "
            "0 disables blending (memory still observes; contextualize is the "
            "identity), letting operators A/B the retrieval without motion impact."
        ),
    )
    sink_warmup_ticks: int = Field(
        30,
        ge=0,
        description=(
            "Validated ticks to wait before freezing the sink anchor (0 = capture "
            "on the first validated tick). At 30 Hz the default captures ~1 s "
            "after boot / mission start so transient startup latents are skipped."
        ),
    )
    recapture_on_mission: bool = Field(
        True,
        description=(
            "Re-arm sink capture at the mission-completed seam so the sink is a "
            "per-mission anchor rather than a stale boot snapshot. The ring and "
            "EMA summary are retained across missions; only the sink re-captures."
        ),
    )


class ModelConfig(StrictBaseModel):
    """Neural network model dimensions."""

    vision_dim: int = Field(256, ge=0, description="Vision feature input dim (0=disabled)")
    ultrasonic_dim: int = Field(1, ge=0, description="Ultrasonic input dim (0=disabled)")
    motor_state_dim: int = Field(4, gt=0, description="Motor state dim [vx, vy, omega, battery]")
    hidden_dim: int = Field(256, gt=0, description="RNN hidden dim")
    latent_dim: int = Field(64, gt=0, description="Latent state dim")
    action_dim: int = Field(3, gt=0, description="Action dim [vx, vy, omega]")
    obs_dim: int = Field(256, gt=0, description="Fused observation embedding dim")
    vision_proj_dim: int = Field(128, ge=0, description="Vision projection dim (0=disabled)")
    ultrasonic_proj_dim: int = Field(32, ge=0, description="Ultrasonic projection dim (0=disabled)")
    motor_proj_dim: int = Field(32, gt=0, description="Motor state projection dim")
    audio_dim: int = Field(0, ge=0, description="Audio feature input dim (0=disabled)")
    audio_proj_dim: int = Field(32, ge=0, description="Audio projection dim (0=disabled)")
    lidar_dim: int = Field(0, ge=0, description="LiDAR feature input dim (0=disabled)")
    lidar_proj_dim: int = Field(32, ge=0, description="LiDAR projection dim (0=disabled)")
    belief_dim: int = Field(128, gt=0, description="BDI belief latent dim")
    desire_dim: int = Field(64, gt=0, description="BDI desire latent dim")
    intention_classes: int = Field(10, gt=0, description="BDI intention classes")
    affect_dim: int = Field(2, gt=0, description="BDI affect dim (valence, arousal)")

    # Latent state health monitoring
    latent_norm_threshold: float = Field(
        50.0,
        gt=0.0,
        description="h-state L2 norm above this value triggers a latent_saturated warning",
    )
    latent_recovery_buffer_size: int = Field(
        5,
        gt=0,
        description="Number of recent valid (h, z) pairs kept for NaN recovery",
    )

    # CfC liquid neural network stream (Dual-Stream RSSM)
    cfc_hidden_dim: int = Field(0, ge=0, description="CfC stream hidden dim (0=disabled, pure GRU)")
    cfc_backbone_units: int = Field(64, gt=0, description="CfC backbone MLP hidden units")
    cfc_backbone_layers: int = Field(1, gt=0, description="CfC backbone MLP layer count")
    cfc_mode: Literal["default", "pure", "no_gate"] = Field(
        "default", description="CfC cell mode: default, pure, no_gate"
    )
    cfc_sparsity_level: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Reserved for future AutoNCP/CfC wiring sparsity support; currently unused",
    )

    # RSSM dynamics-pretraining KL knobs (Phase 5). Read by RSSM.train_sequence;
    # build_rssm_trainable copies operator overrides from TrainingConfig onto these.
    kl_beta: float = Field(
        1.0, ge=0.0, description="KL weight in the RSSM training ELBO (recon + kl_beta*KL)."
    )
    kl_balance_alpha: float = Field(
        0.8, ge=0.0, le=1.0, description="Dreamer KL-balancing weight (prior-update term)."
    )
    kl_free_nats: float = Field(
        1.0, ge=0.0, description="Free-bits floor (nats) for the RSSM training KL."
    )
    logvar_clamp: float = Field(
        10.0,
        gt=0.0,
        description="Symmetric |logvar| clamp before exp() in the balanced-KL "
        "(fp16 AMP overflow guard). Tunable here rather than hardcoded.",
    )

    @model_validator(mode="after")
    def _validate_optional_modalities(self) -> Self:
        """Validate optional modality dimension pairs."""
        if (self.vision_dim == 0) != (self.vision_proj_dim == 0):
            msg = (
                "vision_dim and vision_proj_dim must both be zero (disabled) "
                "or both non-zero (enabled) together"
            )
            raise ValueError(msg)

        if (self.ultrasonic_dim == 0) != (self.ultrasonic_proj_dim == 0):
            msg = (
                "ultrasonic_dim and ultrasonic_proj_dim must both be zero when disabling ultrasonic"
            )
            raise ValueError(msg)

        modality_dims = {
            "ultrasonic": (self.ultrasonic_dim, self.ultrasonic_proj_dim),
            "audio": (self.audio_dim, self.audio_proj_dim),
            "lidar": (self.lidar_dim, self.lidar_proj_dim),
        }
        for modality_name, (input_dim, proj_dim) in modality_dims.items():
            if input_dim > 0 and proj_dim == 0:
                msg = f"{modality_name}_proj_dim must be > 0 when {modality_name}_dim is enabled"
                raise ValueError(msg)

        if self.ultrasonic_dim == 0 and self.lidar_dim == 0:
            msg = "at least one distance modality must be enabled: ultrasonic_dim or lidar_dim"
            raise ValueError(msg)

        return self


class MCTSConfig(StrictBaseModel):
    """Monte Carlo Tree Search configuration."""

    n_simulations_base: int = Field(50, gt=0, description="Base MCTS simulations")
    n_simulations_max: int = Field(200, gt=0, description="Max MCTS simulations")
    rollout_depth: int = Field(5, gt=0, description="Rollout depth")
    gamma: float = Field(0.97, gt=0, le=1, description="Discount factor")
    n_action_candidates: int = Field(9, gt=0, description="Action candidates per node")
    ucb_c: float = Field(1.41, gt=0, description="UCB exploration constant")
    ucb_candidates: list[float] = Field(
        default_factory=lambda: list(DEFAULT_UCB_CANDIDATES),
        description="Candidate UCB exploration constants evaluated during warm-start tuning",
    )
    ucb_target_ms: float = Field(
        DEFAULT_UCB_TARGET_MS,
        gt=0,
        description="Target median planning latency used when selecting a tuned UCB value",
    )


class DualStreamTrainingConfig(StrictBaseModel):
    """Dual-stream RSSM training configuration."""

    gru_lr: float = Field(3e-4, gt=0, description="GRU stream learning rate")
    cfc_lr: float = Field(1e-4, gt=0, description="CfC stream learning rate")
    gru_grad_clip: float = Field(10.0, gt=0, description="GRU gradient clip norm")
    cfc_grad_clip: float = Field(1.0, gt=0, description="CfC gradient clip norm")
    cfc_loss_weight_initial: float = Field(
        0.1, ge=0, le=1, description="CfC loss weight at start of training"
    )
    cfc_loss_weight_final: float = Field(
        1.0, ge=0, le=1, description="CfC loss weight at end of warmup"
    )
    cfc_loss_warmup_steps: int = Field(
        10000, ge=0, description="Steps to ramp CfC loss weight from initial to final"
    )
    fallback_check_interval: int = Field(
        1000, gt=0, description="Steps between CfC fallback quality checks"
    )
    fallback_degradation_threshold: float = Field(
        0.05, gt=0, description="Max allowed planning quality drop before CfC fallback"
    )
