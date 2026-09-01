"""Cognitive-pillar configuration models.

Pillar 2 (dual-cadence BDI + metacognitive loop), Pillar 8 (curiosity), and
Pillar 4 (layered memory) tunables, plus the surprise / anomaly detector.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from mousedroid.config.schema._primitives import StrictBaseModel


class CognitiveConfig(StrictBaseModel):
    """Cognitive core configuration (Pillar 2 — dual-cadence BDI + constitutional RL)."""

    enabled: bool = Field(
        False,
        description="Enable cognitive core injection into orchestrator (requires trained weights)",
    )

    weights_dir: Path = Field(
        Path("weights/bdi/"),
        description="Directory containing BDI model weights (.npz files)",
    )
    huggingface_repo: str = Field(
        "ianshank/mousedroid-weights",
        pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$",
        description="HuggingFace repository ID (format: owner/repo-name)",
    )
    huggingface_subfolder: str = Field(
        "bdi",
        pattern=r"^$|^[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*$",
        description="Subfolder within the HF repo containing BDI .npz files (no '..' allowed)",
    )
    auto_download: bool = Field(
        True,
        description="Auto-download weights from HuggingFace if local weights missing",
    )
    fallback_to_mcts: bool = Field(
        True,
        description="Fall back to MCTS agent if cognitive core initialization fails",
    )
    download_max_retries: int = Field(
        3,
        gt=0,
        description="Max retry attempts for HuggingFace weight downloads",
    )
    download_backoff_base: float = Field(
        2.0,
        gt=0,
        description="Exponential backoff base for download retries (wait = base ^ attempt)",
    )


class MetacognitiveConfig(StrictBaseModel):
    """Metacognitive loop configuration (self-monitoring capabilities)."""

    n_capabilities: int = Field(
        8,
        gt=0,
        description="Number of self-assessed capability dimensions",
    )
    loop_score_scale: float = Field(
        100.0,
        gt=0,
        description="Scaling factor for metacognitive loop scores",
    )


class CuriosityConfig(StrictBaseModel):
    """Curiosity-driven exploration configuration (Pillar 8)."""

    intrinsic_reward_scale: float = Field(
        0.1,
        gt=0,
        description="Intrinsic reward scaling factor",
    )
    forward_model_hidden: int = Field(256, gt=0, description="Forward model hidden dim")
    inverse_model_hidden: int = Field(256, gt=0, description="Inverse model hidden dim")
    novelty_decay_enabled: bool = Field(
        False,
        description="Enable novelty decay to reduce curiosity for familiar states",
    )
    novelty_decay_rate: float = Field(
        0.01,
        gt=0,
        description="Exponential decay rate per state visit",
    )
    novelty_min_scale: float = Field(
        0.01,
        gt=0,
        le=1,
        description="Minimum curiosity scale after decay",
    )
    novelty_n_bins: int = Field(
        32,
        gt=0,
        description="Discretisation bins per dimension for novelty decay",
    )


class MemoryConfig(StrictBaseModel):
    """Layered memory system configuration (Pillar 4)."""

    enabled: bool = Field(False, description="Enable memory tier (episodic, semantic, working)")
    working_context_size: int = Field(8192, gt=0, description="Working memory context tokens")
    episodic_capacity: int = Field(50_000, gt=0, description="Episodic replay buffer size")
    semantic_dim: int = Field(256, gt=0, description="Semantic embedding dimension")
    consolidation_batch_size: int = Field(32, gt=0, description="Offline consolidation batch")
    consolidation_interval_s: float = Field(60.0, gt=0, description="Consolidation period (s)")
    semantic_retrieve_k: int = Field(
        1,
        gt=0,
        description="Number of nearest neighbours to retrieve from semantic index per tick",
    )
    min_episodic_priority: float = Field(
        1e-6,
        gt=0,
        description="Minimum episodic replay priority (floor above zero for FAISS)",
    )
    replay_seed: int | None = Field(
        None,
        description="Random seed for episodic replay sampling (None = non-deterministic)",
    )


class SurpriseConfig(StrictBaseModel):
    """Surprise / anomaly detection configuration."""

    ema_alpha: float = Field(0.1, gt=0, le=1, description="EMA smoothing factor")
    high_threshold: float = Field(2.0, gt=0, description="High surprise threshold")
    critical_threshold: float = Field(5.0, gt=0, description="Critical surprise threshold")
