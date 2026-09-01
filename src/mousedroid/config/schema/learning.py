"""Continual- and incremental-learning configuration models.

Pillar 3 (EWC continual learning), the Phase-6 on-device incremental
learning loop, offline RL (CQL/IQL), and the growth-pillar knowledge
distillation block.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from mousedroid.config.schema._primitives import Self, StrictBaseModel


def _validate_relative_slot_dir(v: str, *, config_name: str) -> str:
    """Reject a ``slot_dir`` value that would escape the experience root.

    Shared by ``OnDeviceLearningConfig.slot_dir`` and ``GrowthConfig.slot_dir``:
    both are resolved as ``<ExperienceConfig.path>/<slot_dir>``, so an absolute
    path, a parent-traversal (``..``) component, or an empty / whitespace-only
    value would break that containment contract and let candidate weights land
    outside the configured experience root. Validated at YAML load so a
    misconfigured deployment fails fast with a clear, operator-actionable
    message instead of silently writing off-root.

    Args:
        v: The raw ``slot_dir`` field value.
        config_name: The owning config's dotted name (e.g. ``"on_device_learning"``),
            used only to make the error message point at the right YAML key.

    Returns:
        The stripped, validated relative path.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    slot = v.strip()
    # Check absoluteness under BOTH POSIX and Windows semantics so a
    # ``/abs/path`` (slot is resolved on the Jetson/Linux target) is caught
    # regardless of the host OS the config is validated on, and ``..``
    # traversal in either separator style is rejected.
    posix = PurePosixPath(slot)
    windows = PureWindowsPath(slot)
    is_absolute = posix.is_absolute() or windows.is_absolute()
    has_traversal = ".." in posix.parts or ".." in windows.parts
    if not slot or is_absolute or has_traversal:
        msg = (
            f"{config_name}.slot_dir must be a non-empty relative path "
            "without parent traversal (resolved under "
            "ExperienceConfig.path); got " + repr(v)
        )
        raise ValueError(msg)
    return slot


class LearningConfig(StrictBaseModel):
    """Continual learning configuration (Pillar 3)."""

    ewc_lambda: float = Field(5000.0, gt=0, description="EWC regularization strength")
    ewc_fisher_samples: int = Field(200, gt=0, description="Samples for Fisher estimation")
    ewc_fisher_batch_size: int = Field(
        1, gt=0, description="Batch size for Fisher estimation random inputs"
    )
    ewc_fallback_input_dim: int = Field(
        1,
        gt=0,
        description=(
            "Fallback input dimension when model has no Linear layers for Fisher estimation"
        ),
    )
    progressive_enabled: bool = Field(False, description="Enable progressive column growth")


class OnDeviceLearningConfig(StrictBaseModel):
    """On-device incremental-learning configuration (Phase 6).

    Lets the rover update its own policy/world-model weights *between* cloud
    retraining cycles from fresh on-device experience, gated by a
    safety-regression bound that reverts to cloud weights on underperformance.
    Default-OFF and backwards-compatible — wired as an ``Optional`` block on
    ``Settings`` so existing YAML loads byte-identically. Every value is
    config-driven; ``slot_dir`` is deliberately repo-relative (resolved by the
    factory/orchestrator under the configured experience root
    ``ExperienceConfig.path``) so NO absolute host path is hardcoded.
    """

    enabled: bool = Field(
        False,
        description="Master switch for the on-device incremental-learning loop (default-off)",
    )
    trigger_min_new_records: int = Field(
        500,
        gt=0,
        description=(
            "Minimum fresh experience records that must accumulate before an "
            "on-device update cycle is triggered"
        ),
    )
    check_interval_s: float = Field(
        300.0,
        gt=0,
        description=(
            "Slow-cadence period (seconds) between replay-trigger checks. The "
            "on-device update runs on its own background task OUTSIDE the 30 Hz "
            "hot loop; this is how often the coordinator probes the new-record "
            "count against ``trigger_min_new_records``. Defaults to 5 min so a "
            "default-on deployment never busy-polls the replay store."
        ),
    )
    update_steps: int = Field(
        50,
        gt=0,
        description="Number of bounded gradient steps per on-device update cycle",
    )
    regression_tolerance: float = Field(
        0.05,
        ge=0,
        description=(
            "Maximum allowed held-out recon+KL loss INCREASE above the live "
            "baseline before the on-device candidate is reverted (LOWER loss is "
            "better): PROMOTE iff candidate_loss <= baseline_loss + tolerance. "
            "ge=0 permits a zero-tolerance gate"
        ),
    )
    held_out_fraction: float = Field(
        0.1,
        gt=0,
        le=1,
        description=(
            "Fraction of the replay sample reserved for held-out scoring "
            "(0 < f <= 1). CONFIG SEAM, not wired into the WS-E3 recon-loss gate "
            "— the gate derives its disjoint held-out window from "
            "refine_sequence_length/refine_batch_episodes; retained for a future "
            "fraction-based held-out sizing"
        ),
    )
    ewc_lambda: float = Field(
        1.0,
        ge=0,
        description=(
            "EWC Fisher-penalty strength for the bounded online update "
            "(ge=0 permits an unregularized step)"
        ),
    )
    learning_rate: float = Field(
        1e-4,
        gt=0,
        description="Learning rate for the bounded on-device gradient steps",
    )
    slot_dir: str = Field(
        "on_device_slot",
        description=(
            "Experience-root-relative leaf for the on-device weight slot. NOT an "
            "absolute host path: the factory/orchestrator resolves it UNDER the "
            "configured experience root (``<ExperienceConfig.path>/<slot_dir>``) "
            "so any operator override of the experience path is inherited for "
            "free. On-device-updated weights land here, never overwriting the "
            "cloud-pulled slot."
        ),
    )

    @field_validator("slot_dir")
    @classmethod
    def _validate_slot_dir(cls, v: str) -> str:
        """Reject slot_dir values that escape the experience root."""
        return _validate_relative_slot_dir(v, config_name="on_device_learning")

    scoring_seed: int = Field(
        1234,
        description=(
            "Safety-gate scoring: fixed RNG seed making the held-out recon+KL loss "
            "score deterministic. Same seed + same held-out batch + same weights "
            "ALWAYS yields the identical loss, so the promote/revert decision is "
            "reproducible. Config-driven so no seed is hardcoded."
        ),
    )
    enable_hot_swap: bool = Field(
        False,
        description=(
            "Phase-6 ENABLEMENT: master switch for hot-swapping a promoted "
            "on-device candidate into the live world model through the C1 atomic-"
            "swap seam. Promotion (``slot_store.mark_active``) stays SEPARATE from "
            "activation: a candidate can pass the regression gate and be marked "
            "active without ever being swapped into the running model. Default "
            "``False`` keeps the orchestrator byte-identical to #134 — no swap ever "
            "occurs. Requires ``enabled=True`` (validated below)."
        ),
    )
    refine_sequence_length: int = Field(
        16,
        gt=0,
        description=(
            "Phase-6 ENABLEMENT: temporal length T of each ``(B, T, ...)`` sequence "
            "the RSSM refiner assembles from replay for ``train_sequence`` (WS-E2). "
            "Config-driven so the refinement window is never hardcoded."
        ),
    )
    refine_batch_episodes: int = Field(
        4,
        gt=0,
        description=(
            "Phase-6 ENABLEMENT: batch dimension B (number of replay episodes) per "
            "RSSM-refinement sequence batch (WS-E2). Config-driven so the batch "
            "size is never hardcoded."
        ),
    )

    @model_validator(mode="after")
    def _require_enabled_for_hot_swap(self) -> Self:
        """Reject ``enable_hot_swap=True`` while the master switch is off.

        Hot-swapping a candidate into the live model is meaningless unless the
        on-device learning loop that PRODUCES candidates is itself enabled.
        Catching the contradiction at YAML load gives the operator a clear,
        actionable message instead of a silently inert hot-swap flag. Mirrors the
        cross-field gate style of ``_require_endpoints_when_enabled``.
        """
        if self.enable_hot_swap and not self.enabled:
            msg = (
                "on_device_learning.enable_hot_swap=true requires "
                "on_device_learning.enabled=true (hot-swap activates a candidate "
                "the disabled learning loop never produces)"
            )
            raise ValueError(msg)
        return self


class GrowthConfig(StrictBaseModel):
    """Growth-pillar knowledge-distillation configuration.

    Distils the wired VLA teacher policy into a compact student *between* cloud
    retraining cycles, on a slow-cadence background task OUTSIDE the 30 Hz hot
    loop. Default-OFF and backwards-compatible — wired as an ``Optional`` block on
    ``Settings`` so existing YAML loads byte-identically. Every value is
    config-driven; ``slot_dir`` is deliberately repo-relative (resolved by the
    factory under the configured experience root ``ExperienceConfig.path``) so NO
    absolute host path is hardcoded. The distilled student is PERSISTED to a
    SHA-256 slot, never hot-swapped into the live policy — deployment stays a
    soak-gated operator decision.
    """

    enabled: bool = Field(
        False,
        description="Master switch for the growth-pillar distillation loop (default-off)",
    )
    trigger_min_new_records: int = Field(
        500,
        gt=0,
        description=(
            "Minimum fresh experience records that must accumulate before a "
            "distillation cycle is triggered"
        ),
    )
    check_interval_s: float = Field(
        300.0,
        gt=0,
        description=(
            "Slow-cadence period (seconds) between distillation-trigger checks. The "
            "distillation runs on its own background task OUTSIDE the 30 Hz hot "
            "loop; this is how often the coordinator probes the new-record count "
            "against ``trigger_min_new_records``. Defaults to 5 min so a default-on "
            "deployment never busy-polls."
        ),
    )
    distill_steps: int = Field(
        50,
        gt=0,
        description="Number of bounded distillation gradient steps per cycle",
    )
    batch_size: int = Field(
        32,
        gt=0,
        description="Latent-minibatch size sampled per distillation step",
    )
    temperature: float = Field(
        2.0,
        gt=0,
        description=(
            "Softmax temperature (classification-objective distillers only; ignored "
            "by the regression objective used for the continuous VLA action policy)"
        ),
    )
    alpha: float = Field(
        1.0,
        ge=0,
        le=1,
        description=(
            "Weight of the soft (teacher-matching) loss vs the hard-target loss. "
            "Default 1.0 = pure teacher-matching self-distillation (no ground-truth "
            "action labels), which is how the VLA policy is distilled."
        ),
    )
    learning_rate: float = Field(
        1e-4,
        gt=0,
        description="Learning rate for the bounded distillation gradient steps",
    )
    student_hidden_dim: int = Field(
        64,
        gt=0,
        description=(
            "Hidden-layer width of the compact student MLP. Keep it small — "
            "compression is the point of the growth pillar."
        ),
    )
    slot_dir: str = Field(
        "growth_slot",
        description=(
            "Experience-root-relative leaf for the distilled-student weight slot. "
            "NOT an absolute host path: the factory resolves it UNDER the configured "
            "experience root (``<ExperienceConfig.path>/<slot_dir>``) so any operator "
            "override of the experience path is inherited for free. Distilled "
            "students land here, never overwriting the cloud-pulled slot."
        ),
    )

    @field_validator("slot_dir")
    @classmethod
    def _validate_slot_dir(cls, v: str) -> str:
        """Reject slot_dir values that escape the experience root.

        Mirrors ``OnDeviceLearningConfig._validate_slot_dir`` via the shared
        ``_validate_relative_slot_dir`` helper.
        """
        return _validate_relative_slot_dir(v, config_name="growth")


class OfflineRLConfig(StrictBaseModel):
    """Offline RL training configuration (CQL / IQL)."""

    algorithm: Literal["cql", "iql"] = Field("cql", description="Offline RL algorithm")
    hidden_dim: int = Field(256, gt=0, description="Hidden layer dimension")
    gamma: float = Field(0.99, gt=0, le=1, description="Discount factor")
    tau: float = Field(0.005, gt=0, le=1, description="Soft target update coefficient")
    learning_rate: float = Field(3e-4, gt=0, description="Learning rate")
    epochs: int = Field(100, gt=0, description="Training epochs")
    batch_size: int = Field(64, gt=0, description="Training batch size")
    terminal_gap_s: float = Field(
        5.0,
        gt=0,
        description="Timestamp gap to mark episode boundaries (s)",
    )
    log_every_n_epochs: int = Field(10, gt=0, description="Log summary every N epochs")
    checkpoint_every_n_epochs: int = Field(20, gt=0, description="Save checkpoint every N epochs")
    cql_alpha: float = Field(1.0, gt=0, description="CQL regularization weight")
    cql_n_random_actions: int = Field(10, gt=0, description="Random actions for CQL logsumexp")
    iql_expectile: float = Field(0.7, gt=0, lt=1, description="IQL expectile for asymmetric loss")
    iql_beta: float = Field(
        3.0,
        gt=0,
        description="IQL inverse temperature for advantage weighting",
    )
    real_supervised_weight: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Phase 2: weight on the auxiliary BC-style supervised loss applied to real "
            "replay batches drawn from the LMDB store. 0.0 disables the BC term — "
            "training is then byte-identical to the pre-Phase-2 path. See also "
            "``bc_lr`` and ``bc_batch_size`` for optional dedicated BC optimizer tuning, "
            "and ``use_replay_mixer`` to source batches from the sim/real mixer."
        ),
    )
    bc_lr: float | None = Field(
        None,
        gt=0,
        description=(
            "Phase 2.1: optional dedicated learning rate for the BC auxiliary loss. "
            "When ``None`` the policy optimizer (and its learning_rate) is reused — "
            "byte-identical to the pre-Phase-2.1 path. When set, a separate "
            "``bc_optimizer`` is built over policy parameters."
        ),
    )
    bc_batch_size: int | None = Field(
        None,
        gt=0,
        description=(
            "Phase 2.1: optional dedicated mini-batch size for the BC step. "
            "When ``None`` the main ``batch_size`` is reused. Reserved for future "
            "use when the BC update is decoupled from the actor-critic batch; "
            "currently consumed by the trainer constructor and exposed in checkpoints."
        ),
    )
    use_replay_mixer: bool = Field(
        False,
        description=(
            "Phase 2.1: when True, draw training batches from the sim/real "
            "``ReplayMixer`` instead of the single ``OfflineRLDataset`` LMDB. "
            "Default False preserves byte-identical behavior to the single-LMDB path."
        ),
    )
