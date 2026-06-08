"""RSSM checkpoint migration — handles encoder modality changes between runs.

When a model is retrained with a different sensor configuration (e.g. dropping
ultrasonic in favour of LiDAR), the ``encoder.fusion.weight`` tensor changes
shape.  This module provides a pure-function migration path that surgically
rebuilds the fusion weight by retaining columns for unchanged modalities and
Kaiming-initialising columns for newly added ones.

Typical usage::

    from mousedroid.world_model.checkpoint_migration import load_rssm_with_migration

    rssm = load_rssm_with_migration(
        Path("weights/rssm/epoch_100.pt"),
        new_cfg=cfg.model,
        device=torch.device("cuda"),
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.world_model.rssm import RSSM

_log = get_logger(__name__)

StateDict = dict[str, Tensor]
"""Type alias for a PyTorch module state dict."""

_KAIMING_LINEAR_A: float = math.sqrt(5)
"""Kaiming-uniform gain that matches ``nn.Linear.__init__`` weight initialisation."""


# ---------------------------------------------------------------------------
# Internal helpers — must stay in sync with MultimodalEncoder.forward()
# ---------------------------------------------------------------------------


def _infer_old_parts(sd: StateDict) -> list[tuple[str, int]]:
    """Infer ordered encoder projection parts from an existing state dict.

    The ordering MUST match ``MultimodalEncoder.forward()``'s ``parts`` list
    construction: vision, (ultrasonic,) motor, (audio,) (lidar,).

    Args:
        sd: State dict from a saved RSSM checkpoint.

    Returns:
        Ordered list of ``(modality_name, proj_dim)`` tuples present in *sd*.
    """
    parts: list[tuple[str, int]] = []
    if "encoder.vision_proj.weight" in sd:
        parts.append(("vision", int(sd["encoder.vision_proj.weight"].shape[0])))
    if "encoder.ultrasonic_proj.weight" in sd:
        parts.append(("ultrasonic", int(sd["encoder.ultrasonic_proj.weight"].shape[0])))
    if "encoder.motor_proj.weight" in sd:
        parts.append(("motor", int(sd["encoder.motor_proj.weight"].shape[0])))
    if "encoder.audio_proj.weight" in sd:
        parts.append(("audio", int(sd["encoder.audio_proj.weight"].shape[0])))
    if "encoder.lidar_proj.weight" in sd:
        parts.append(("lidar", int(sd["encoder.lidar_proj.weight"].shape[0])))
    return parts


def _build_new_parts(cfg: ModelConfig) -> list[tuple[str, int]]:
    """Build ordered encoder projection parts from a target ``ModelConfig``.

    Args:
        cfg: Target model configuration.

    Returns:
        Ordered list of ``(modality_name, proj_dim)`` tuples matching
        ``MultimodalEncoder.forward()``'s ``parts`` construction order.
    """
    parts: list[tuple[str, int]] = []
    if cfg.vision_dim > 0 and cfg.vision_proj_dim > 0:
        parts.append(("vision", cfg.vision_proj_dim))
    if cfg.ultrasonic_dim > 0 and cfg.ultrasonic_proj_dim > 0:
        parts.append(("ultrasonic", cfg.ultrasonic_proj_dim))
    parts.append(("motor", cfg.motor_proj_dim))
    if cfg.audio_dim > 0 and cfg.audio_proj_dim > 0:
        parts.append(("audio", cfg.audio_proj_dim))
    if cfg.lidar_dim > 0 and cfg.lidar_proj_dim > 0:
        parts.append(("lidar", cfg.lidar_proj_dim))
    return parts


def _migrate_fusion_weight(
    old_w: Tensor,
    old_parts: list[tuple[str, int]],
    new_parts: list[tuple[str, int]],
) -> Tensor:
    """Rebuild ``encoder.fusion.weight`` for a new modality layout.

    Columns for retained modalities are copied verbatim.  Columns for newly
    added modalities are Kaiming-uniform initialised to match PyTorch's
    ``nn.Linear`` default.

    Args:
        old_w: Old fusion weight tensor, shape ``(obs_dim, old_fused_dim)``.
        old_parts: Ordered ``(name, proj_dim)`` list from the old checkpoint.
        new_parts: Ordered ``(name, proj_dim)`` list for the new config.

    Returns:
        New fusion weight tensor, shape ``(obs_dim, new_fused_dim)``.

    Raises:
        ValueError: If a retained modality has different ``proj_dim`` values
            in the old and new configs (incompatible projection sizes).
    """
    obs_dim = old_w.shape[0]

    # Build a column-offset lookup for every modality in the old weight.
    old_offsets: dict[str, tuple[int, int]] = {}
    col = 0
    for name, dim in old_parts:
        old_offsets[name] = (col, col + dim)
        col += dim

    new_cols: list[Tensor] = []
    for name, new_dim in new_parts:
        if name in old_offsets:
            start, end = old_offsets[name]
            old_dim = end - start
            if old_dim != new_dim:
                raise ValueError(
                    f"Projection dim mismatch for '{name}': "
                    f"old={old_dim}, new={new_dim}. "
                    "Cannot auto-migrate incompatible projection sizes."
                )
            new_cols.append(old_w[:, start:end].clone())
        else:
            # New modality — Kaiming-uniform init matching nn.Linear default.
            cols = torch.empty(obs_dim, new_dim, device=old_w.device, dtype=old_w.dtype)
            nn.init.kaiming_uniform_(cols, a=_KAIMING_LINEAR_A)
            new_cols.append(cols)

    return torch.cat(new_cols, dim=1)


def _new_proj_tensors(cfg: ModelConfig, modality: str) -> dict[str, Tensor]:
    """Create Kaiming-initialised weight and bias for a new projection layer.

    The initialisation matches the defaults applied by ``nn.Linear.__init__``:
    Kaiming-uniform for weight, uniform ``±1/√fan_in`` for bias.

    Args:
        cfg: Target model configuration.
        modality: One of ``"vision"``, ``"lidar"``, ``"ultrasonic"``, or ``"audio"``.

    Returns:
        Dict with ``"encoder.<modality>_proj.weight"`` and
        ``"encoder.<modality>_proj.bias"`` tensors.

    Raises:
        ValueError: If *modality* is not recognised.
    """
    dim_map: dict[str, tuple[int, int]] = {
        "vision": (cfg.vision_proj_dim, cfg.vision_dim),
        "lidar": (cfg.lidar_proj_dim, cfg.lidar_dim),
        "ultrasonic": (cfg.ultrasonic_proj_dim, cfg.ultrasonic_dim),
        "audio": (cfg.audio_proj_dim, cfg.audio_dim),
    }
    if modality not in dim_map:
        raise ValueError(f"Unknown modality for projection init: {modality!r}")
    out_dim, in_dim = dim_map[modality]
    if in_dim <= 0:
        # NOT assert: stripped under PYTHONOPTIMIZE=1 (the Jetson Docker entrypoint).
        msg = f"Cannot initialise projection from zero-dim input for {modality!r}"
        raise ValueError(msg)
    w = torch.empty(out_dim, in_dim)
    nn.init.kaiming_uniform_(w, a=_KAIMING_LINEAR_A)
    bound = 1.0 / math.sqrt(in_dim)
    b = torch.empty(out_dim).uniform_(-bound, bound)
    return {
        f"encoder.{modality}_proj.weight": w,
        f"encoder.{modality}_proj.bias": b,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def migrate_state_dict(
    old_sd: StateDict,
    new_cfg: ModelConfig,
) -> tuple[StateDict, dict[str, object]]:
    """Migrate an RSSM state dict to a new modality configuration.

    Only encoder-level changes (projection layers and fusion input columns)
    are handled.  All other sub-modules (GRU, posterior/prior, reward head,
    observation decoder) are carried over verbatim.

    Args:
        old_sd: State dict from a previously saved RSSM checkpoint.
        new_cfg: Target :class:`~mousedroid.config.schema.ModelConfig`.

    Returns:
        A tuple ``(migrated_sd, report)`` where *migrated_sd* is ready to be
        passed to :meth:`RSSM.load_state_dict` and *report* contains
        ``dropped_modalities``, ``added_modalities``, and
        ``old_fusion_shape`` / ``new_fusion_shape`` lists suitable for
        structured logging.

    Raises:
        ValueError: If a retained modality has incompatible projection dims.
    """
    sd: StateDict = dict(old_sd)  # shallow copy — tensor data not duplicated

    old_parts = _infer_old_parts(sd)
    new_parts = _build_new_parts(new_cfg)
    old_modalities: set[str] = {name for name, _ in old_parts}
    new_modalities: set[str] = {name for name, _ in new_parts}

    dropped = old_modalities - new_modalities
    added = new_modalities - old_modalities

    report: dict[str, object] = {
        "dropped_modalities": sorted(dropped),
        "added_modalities": sorted(added),
        "old_fusion_shape": list(sd["encoder.fusion.weight"].shape),
    }

    # Remove projection keys for dropped modalities.
    for modality in sorted(dropped):
        for suffix in ("weight", "bias"):
            sd.pop(f"encoder.{modality}_proj.{suffix}", None)

    # Surgically rebuild the fusion weight.
    sd["encoder.fusion.weight"] = _migrate_fusion_weight(
        sd["encoder.fusion.weight"], old_parts, new_parts
    )

    # Add Kaiming-initialised projection keys for newly added modalities.
    for modality in sorted(added):
        sd.update(_new_proj_tensors(new_cfg, modality))

    report["new_fusion_shape"] = list(sd["encoder.fusion.weight"].shape)
    report["added_keys"] = sorted(k for k in sd if any(f"encoder.{m}_proj" in k for m in added))

    return sd, report


def load_rssm_with_migration(
    path: Path,
    cfg: ModelConfig,
    device: torch.device | None = None,
    *,
    strict: bool = True,
) -> RSSM:
    """Load an RSSM checkpoint, migrating encoder modality changes automatically.

    Handles both *full* training checkpoints (containing a
    ``"model_state_dict"`` key, as written by ``train_rssm._save_checkpoint``)
    and *bare* model state dicts (weights only).

    Args:
        path: Path to the ``.pt`` checkpoint file.
        cfg: Target :class:`~mousedroid.config.schema.ModelConfig`.
        device: Target device; defaults to CPU when ``None``.
        strict: Passed to :meth:`torch.nn.Module.load_state_dict`.

    Returns:
        A fully initialised :class:`~mousedroid.world_model.rssm.RSSM`
        placed on *device*.

    Raises:
        TypeError: If the checkpoint file does not deserialise to a ``dict``.
        ValueError: If the state dict is incompatible with *cfg*.
    """
    # Lazy import avoids a circular-import cycle (rssm → encoder → this module).
    from mousedroid.world_model.rssm import RSSM

    _d = device if device is not None else torch.device("cpu")
    raw = torch.load(path, map_location=_d, weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(
            f"Unsupported checkpoint format at {path!s}: expected dict, got {type(raw).__name__}"
        )

    # Accept both full training checkpoints and bare state dicts.
    old_sd: StateDict = raw.get("model_state_dict", raw)

    new_sd, report = migrate_state_dict(old_sd, cfg)
    _log.info(
        "checkpoint_migrated",
        path=str(path),
        dropped=report["dropped_modalities"],
        added=report["added_modalities"],
        old_fusion_shape=report["old_fusion_shape"],
        new_fusion_shape=report["new_fusion_shape"],
    )

    rssm = RSSM(cfg).to(_d)
    rssm.load_state_dict(new_sd, strict=strict)
    return rssm
