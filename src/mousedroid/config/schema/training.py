"""Offline / GPU pre-training pipeline configuration models.

GPU training tunables (``GPUConfig``, e.g. Jetson Orin Nano AMP/CUDA
budget), the ADR-005 phase orchestrator, per-phase synthetic-data /
annotation / warm-start / constitutional-RL settings, LMDB replay
ingestion (including the Phase 2 sim/real mixer), the F-023 corrupted-
history drift-reduction knobs, and the top-level ``TrainingConfig``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mousedroid.config.schema._primitives import _settings_default_factory


class GPUConfig(BaseModel):
    """GPU training configuration for Jetson Orin Nano."""

    device: str | None = Field(
        None,
        description="Force torch device (e.g. 'cuda:0', 'cpu'). None = auto-detect",
    )
    require_cuda: bool = Field(
        False,
        description=(
            "Fail training when a CUDA device is unavailable or a non-CUDA device is selected"
        ),
    )
    enable_amp: bool = Field(
        True,
        description="Enable Automatic Mixed Precision for PyTorch training phases",
    )
    memory_limit_gb: float = Field(
        6.0,
        gt=0,
        description="Max GPU memory budget in GB (Jetson: 8 GB shared, leave 2 GB headroom)",
    )


class TrainingPipelineConfig(BaseModel):
    """GPU pre-training pipeline orchestrator configuration (ADR-005)."""

    phases: list[str] = Field(
        default_factory=lambda: ["rssm", "warmstart", "bdi", "constitutional_rl"],
        description="Ordered training phases to execute",
    )
    batch_sizes: dict[str, int] = Field(
        default_factory=lambda: {
            "rssm": 16,
            "warmstart": 32,
            "bdi": 32,
            "constitutional_rl": 64,
        },
        description="Per-phase base batch sizes",
    )
    thermal_limit_celsius: float = Field(
        85.0,
        gt=0,
        description="GPU temperature threshold to pause training (Celsius)",
    )
    thermal_pause_seconds: float = Field(
        30.0,
        gt=0,
        description="Seconds to wait when thermal limit exceeded",
    )
    thermal_sysfs_path: str = Field(
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        description="Sysfs path to read GPU temperature (millidegrees Celsius)",
    )
    vram_headroom_mb: int = Field(
        512,
        gt=0,
        description="VRAM headroom to reserve (MB) — batch tuner avoids using this",
    )
    checkpoint_dir: str = Field(
        "checkpoints",
        description="Directory for phase checkpoint files",
    )
    amp_enabled: bool = Field(
        True,
        description="Enable AMP (Automatic Mixed Precision) for GPU phases",
    )
    resume_from_phase: str | None = Field(
        None,
        description="Phase name to resume from (skips prior phases)",
    )


class TrainingGenerationConfig(BaseModel):
    """Synthetic data generation settings for Phase 0."""

    log_every_n_episodes: int = Field(
        100,
        gt=0,
        description="Episode logging cadence during synthetic data generation",
    )


class TrainingAnnotationConfig(BaseModel):
    """Annotation collection and heuristic labeling settings for Phase 0b."""

    n_episodes: int = Field(500, gt=0, description="Annotation collection episode count")
    max_steps: int = Field(50, gt=0, description="Max steps per annotation episode")
    log_every_n_episodes: int = Field(
        100,
        gt=0,
        description="Episode logging cadence during annotation collection",
    )
    human_safety_radius_m: float = Field(
        0.5,
        gt=0,
        description="Human proximity threshold for the protect_human label (m)",
    )
    battery_warn_v: float = Field(
        10.8,
        ge=0,
        description="Battery threshold for the charge label (V); 0 disables",
    )
    obstacle_clearance_m: float = Field(
        0.25,
        gt=0,
        description="Obstacle distance threshold for the avoid_obstacle label (m)",
    )
    idle_speed_threshold: float = Field(
        0.05,
        ge=0,
        description="Planar speed threshold below which the label may become idle",
    )
    idle_omega_threshold: float = Field(
        0.1,
        ge=0,
        description="Angular speed threshold below which the label may become idle",
    )
    wait_speed_threshold: float = Field(
        0.1,
        ge=0,
        description="Planar speed threshold below which the label may become wait",
    )
    wait_omega_threshold: float = Field(
        0.05,
        ge=0,
        description="Angular speed threshold below which the label may become wait",
    )
    turn_omega_threshold: float = Field(
        0.5,
        ge=0,
        description="Angular speed threshold above which the label becomes turn",
    )
    approach_clear_distance_m: float = Field(
        1.0,
        gt=0,
        description="Obstacle distance threshold above which the path is clear for approach",
    )
    approach_speed_threshold: float = Field(
        0.2,
        ge=0,
        description="Planar speed threshold above which the label may become approach_target",
    )
    backtrack_speed_threshold: float = Field(
        -0.2,
        le=0,
        description="Forward velocity threshold below which the label becomes backtrack",
    )


class TrainingWarmstartConfig(BaseModel):
    """Warm-start statistics and UCB tuning settings for Phase 2."""

    latent_stats_max_episodes: int = Field(
        100,
        gt=0,
        description="Maximum episodes sampled when computing latent statistics",
    )
    tuning_episodes: int = Field(
        100,
        gt=0,
        description="Episodes evaluated per UCB candidate during warm-start tuning",
    )
    rollout_steps: int = Field(
        20,
        gt=0,
        description="Imagined rollout steps per warm-start tuning episode",
    )


class TrainingConstitutionalConfig(BaseModel):
    """Constitutional RL rollout logging and validation context settings."""

    log_every_n_episodes: int = Field(
        100,
        gt=0,
        description="Training episode logging cadence for constitutional RL",
    )
    validation_battery_v: float = Field(
        12.0,
        gt=0,
        description="Battery voltage used for constitutional rollout validation context (V)",
    )
    validation_obstacle_dist_m: float = Field(
        2.0,
        gt=0,
        description="Obstacle distance used for constitutional rollout validation context (m)",
    )
    validation_mcts_sims: int = Field(
        50,
        gt=0,
        description="MCTS simulation count used for constitutional rollout validation context",
    )


class TrainingReplayConfig(BaseModel):
    """Replay-ingestion settings for RSSM and activation training flows."""

    enabled: bool = Field(
        False,
        description="Enable LMDB-backed replay ingestion alongside or instead of synthetic data",
    )
    source_path: str | None = Field(
        None,
        description="Optional LMDB replay source path (None uses experience.path)",
    )
    terminal_gap_s: float = Field(
        5.0,
        gt=0,
        description="Timestamp gap used to infer episode boundaries in LMDB replay",
    )
    real_episode_ratio: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Number of real replay episodes to include per synthetic episode. "
            "Ignored when synthetic data is absent, in which case all available replay episodes "
            "are used."
        ),
    )
    max_real_episodes: int | None = Field(
        None,
        gt=0,
        description="Optional cap on replay episodes mixed into one training dataset build",
    )
    seed: int | None = Field(
        None,
        description="Optional seed used when selecting a subset of replay episodes",
    )


class ReplayMixerConfig(BaseModel):
    """Phase 2 sim/real episode mixer configuration.

    Mirrors :class:`mousedroid.training.replay.mixer.MixerConfig` so YAML can
    drive the mixer without importing the implementation. All fields default
    to inert values — `alpha_target=0.0` means sim-only and produces the same
    behavior as legacy training paths.
    """

    alpha_target: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Target probability of drawing from the real replay source. "
            "0.0 disables mixing (sim-only); 1.0 is real-only."
        ),
    )
    alpha_ramp_steps: int = Field(
        1000,
        gt=0,
        description=("Number of mix steps to linearly ramp alpha from 0 to alpha_target."),
    )
    chunk_size: int = Field(
        64,
        gt=0,
        description="LMDB read chunk size for the async replay reader.",
    )
    seed: int | None = Field(
        None,
        description="Optional RNG seed for deterministic mixing.",
    )
    log_every_n: int = Field(
        500,
        gt=0,
        description="Emit a `mixer_ratio_check` log every N draws.",
    )
    debug_log_every_n: int = Field(
        0,
        ge=0,
        description=(
            "Emit a structlog DEBUG line ('mixer_draw' / 'replay_chunk_decoded') "
            "every N mixer/reader operations. 0 disables debug logs entirely. "
            "Useful for live triage on Jetson; set to e.g. 100 to surface state "
            "without flooding the journal. The throttle is independent of "
            "`log_every_n`, which controls INFO-level cadence."
        ),
    )


class DriftTrainingConfig(BaseModel):
    """Corrupted-history drift-reduction training knobs (F-023, default-OFF).

    Adapts the AlayaWorld corrupted-history training idea to the RSSM: a random
    prefix of each training sequence is rolled OPEN-LOOP under the model's own
    prior (its self-generated, drifted imagination) and the posterior suffix is
    trained to recover toward ground truth. Nested under ``TrainingConfig`` (the
    ``training.replay`` precedent) because this is a training-time surface only —
    the 30 Hz loop never reads it. ``enabled`` gates the production-pretraining
    integration (``RSSMPretrainer``); the offline comparison harness
    ``scripts/compare_drift.py`` is deliberately flag-independent. Applies to the
    concrete ``RSSM`` feasibility vehicle only (the sole ``train_sequence``
    engine); the ``DualStreamRSSM`` port is explicitly deferred (ADR-015).
    """

    enabled: bool = Field(
        False,
        description=(
            "Opt-in: apply corrupted-history augmentation in the RSSM pretraining "
            "path. The offline compare_drift.py harness ignores this flag."
        ),
    )
    corruption_prob: float = Field(
        0.5,
        ge=0,
        le=1,
        description=(
            "Per-batch probability of using the corrupted-history objective "
            "instead of the standard train_sequence path (0 = never, reproduces "
            "the baseline exactly)"
        ),
    )
    max_prefix_frac: float = Field(
        0.5,
        gt=0,
        le=1,
        description=(
            "Upper bound on the open-loop prefix length as a fraction of the "
            "sequence length; the prefix k is drawn uniformly from "
            "[0, floor(max_prefix_frac * T)]"
        ),
    )
    recovery_weight: float = Field(
        1.0,
        ge=0,
        description=(
            "Multiplier on the post-boundary reconstruction loss (the recovery "
            "steps immediately after the corrupted prefix). 1.0 = uniform "
            "weighting, provably inert at prefix length 0."
        ),
    )
    residual_head: bool = Field(
        True,
        description=(
            "Train the evaluation-only DriftCorrectionHead alongside the "
            "corrupted objective. The head predicts correction residuals toward "
            "ground truth and is consumed by measure_drift; it is NEVER deployed "
            "on the rover and adds no parameters to the RSSM state_dict."
        ),
    )
    eval_context_steps: int = Field(
        8,
        gt=0,
        description=(
            "Posterior warmup steps on ground truth before the open-loop drift "
            "rollout in measure_drift"
        ),
    )
    eval_horizon: int = Field(
        24,
        gt=0,
        description="Open-loop prior rollout steps scored by measure_drift",
    )
    seed: int = Field(
        42,
        description=(
            "Seed for the drift comparison harness (paired model inits, corrupted "
            "prefix draws, and the deterministic measure_drift scoring)"
        ),
    )


class TrainingConfig(BaseModel):
    """Offline training configuration."""

    batch_size: int = Field(32, gt=0, description="Training batch size")
    learning_rate: float = Field(3e-4, gt=0, description="Learning rate")
    epochs: int = Field(100, gt=0, description="Training epochs")
    checkpoint_every_n: int = Field(10, gt=0, description="Checkpoint frequency")
    gradient_scale: float = Field(2.0, gt=0, description="Gradient scale for numpy MSE losses")
    kl_beta: float = Field(1.0, gt=0, description="KL loss weight for RSSM training")
    sequence_length: int = Field(50, gt=0, description="Training sequence length")
    n_episodes: int = Field(1000, gt=0, description="Synthetic episodes to generate")
    data_dir: str = Field("training/data", description="Generated data directory")
    weights_dir: str = Field("weights", description="Checkpoint output directory")
    resume_from: str | None = Field(
        None,
        description="Path to checkpoint for resuming interrupted training",
    )
    # Phase 5 — MuJoCo->RSSM dynamics-pretraining knobs (opt-in; default OFF so
    # pre-feature behaviour is unchanged). build_rssm_trainable copies
    # rssm_free_nats / rssm_kl_balance_alpha onto the ModelConfig at build time.
    rssm_pretrain_enabled: bool = Field(
        False,
        description="Opt-in: run the MuJoCo->RSSM dynamics pretraining loop in the rssm phase.",
    )
    rssm_free_nats: float = Field(
        1.0, ge=0.0, description="Free-bits floor (nats) for the RSSM training KL."
    )
    rssm_kl_balance_alpha: float = Field(
        0.8, ge=0.0, le=1.0, description="Dreamer KL-balancing weight (prior-update term)."
    )
    rssm_grad_clip: float = Field(
        100.0, gt=0.0, description="Global grad-norm clip for RSSM pretraining."
    )
    rssm_checkpoint_name: str = Field(
        "rssm_pretrained.pt",
        min_length=1,
        description="Filename for the RSSM pretrain checkpoint (under weights_dir).",
    )
    rssm_explore_action_rad_s: float = Field(
        6.0,
        gt=0.0,
        description="Exploration wheel-command bound (rad/s) for sim seed episodes.",
    )
    rssm_explore_smoothing: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="EMA weight on the previous action for the smoothed-random explore policy.",
    )
    rssm_data_seed: int = Field(
        0, ge=0, description="Seed for the sim episode generator (exploration + reset stream)."
    )
    rssm_vision_finetune_enabled: bool = Field(
        False,
        description="Opt-in: run the vision-on RSSM fine-tune phase (renders RGB + MeanPool).",
    )
    rssm_finetune_checkpoint: str = Field(
        "",
        description="Path to the vision-OFF pretrained RSSM checkpoint to fine-tune (migrated on).",
    )
    rssm_finetune_epochs: int = Field(
        50, gt=0, description="Epochs for the vision-on fine-tune phase."
    )
    rssm_vision_checkpoint_name: str = Field(
        "rssm_vision_finetuned.pt",
        min_length=1,
        description="Filename for the vision-on fine-tuned checkpoint (under weights_dir).",
    )
    generation: TrainingGenerationConfig = Field(
        default_factory=_settings_default_factory(TrainingGenerationConfig)
    )
    annotation: TrainingAnnotationConfig = Field(
        default_factory=_settings_default_factory(TrainingAnnotationConfig)
    )
    warmstart: TrainingWarmstartConfig = Field(
        default_factory=_settings_default_factory(TrainingWarmstartConfig)
    )
    constitutional: TrainingConstitutionalConfig = Field(
        default_factory=_settings_default_factory(TrainingConstitutionalConfig)
    )
    replay: TrainingReplayConfig = Field(
        default_factory=_settings_default_factory(TrainingReplayConfig)
    )
    drift: DriftTrainingConfig | None = Field(
        None,
        description=(
            "Corrupted-history drift-reduction training block (F-023). ``None`` "
            "(default) disables — existing YAML loads byte-identical. Populate "
            "with ``enabled: true`` to apply the corrupted-prefix objective in "
            "the RSSM pretraining path."
        ),
    )
    replay_mixer: ReplayMixerConfig = Field(
        default_factory=_settings_default_factory(ReplayMixerConfig),
        description=(
            "Phase 2 sim/real interleaver configuration. Defaults are inert "
            "(alpha_target=0.0) so existing training pipelines are unchanged."
        ),
    )
    gpu: GPUConfig = Field(
        default_factory=lambda: GPUConfig(
            device=None,
            require_cuda=False,
            enable_amp=True,
            memory_limit_gb=6.0,
        ),
    )
