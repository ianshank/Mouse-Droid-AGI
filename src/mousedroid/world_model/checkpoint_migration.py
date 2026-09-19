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
    if "encoder.imu_proj.weight" in sd:
        parts.append(("imu", int(sd["encoder.imu_proj.weight"].shape[0])))
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
    if cfg.imu_dim > 0 and cfg.imu_proj_dim > 0:
        parts.append(("imu", cfg.imu_proj_dim))
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
        "imu": (cfg.imu_proj_dim, cfg.imu_dim),
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


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint failed its integrity precondition and was not loaded.

    Raised when ``expected_sha256`` is supplied and does not match, and when
    ``allow_unsafe_pickle`` is requested without a digest to pin the bytes it
    would execute.
    """


def _load_checkpoint_payload(
    path: Path,
    device: torch.device,
    *,
    expected_sha256: str | None,
    allow_unsafe_pickle: bool,
) -> object:
    """Deserialise ``path`` under the safest mode its preconditions allow.

    ``torch.load(..., weights_only=False)`` executes arbitrary pickle opcodes
    from the file. This loader therefore defaults to ``weights_only=True`` and
    admits the unsafe mode only behind an explicit, named precondition:

    * ``expected_sha256`` given — the bytes are verified with the same
      :func:`~mousedroid.utils.weights_manager.verify_sha256` helper the OTA
      path uses *before* anything is deserialised, mirroring
      ``growth/slot_store.py``'s verify-then-``weights_only=True`` order.
    * ``allow_unsafe_pickle`` requires ``expected_sha256`` as well. An
      un-pinned request to run pickle is refused rather than honoured, so the
      unsafe mode cannot be reached by omission — only by an operator stating
      a trusted digest.

    Args:
        path: Checkpoint file.
        device: ``map_location`` target.
        expected_sha256: Hex digest the file must match, or ``None``.
        allow_unsafe_pickle: Permit ``weights_only=False`` for a legacy
            checkpoint holding non-tensor objects. Requires a digest.

    Returns:
        The deserialised object, unvalidated.

    Raises:
        CheckpointIntegrityError: Digest mismatch, or unsafe pickle requested
            without a digest.
    """
    from mousedroid.utils.weights_manager import verify_sha256

    if allow_unsafe_pickle and expected_sha256 is None:
        msg = (
            f"refusing to load '{path}' with weights_only=False: that mode "
            f"executes arbitrary pickle opcodes from the file, so it requires "
            f"expected_sha256 naming the exact bytes you trust. Re-save the "
            f"checkpoint as a plain state dict, or pass the digest."
        )
        raise CheckpointIntegrityError(msg)
    if expected_sha256 is not None and not verify_sha256(
        path, expected_sha256, log_event_prefix="rssm_checkpoint"
    ):
        msg = (
            f"refusing to load '{path}': SHA-256 verification failed against "
            f"the expected digest. Re-fetch the checkpoint or correct the "
            f"recorded digest."
        )
        raise CheckpointIntegrityError(msg)
    return torch.load(path, map_location=device, weights_only=not allow_unsafe_pickle)


def load_rssm_with_migration(
    path: Path,
    cfg: ModelConfig,
    device: torch.device | None = None,
    *,
    strict: bool = True,
    expected_sha256: str | None = None,
    allow_unsafe_pickle: bool = False,
) -> RSSM:
    """Load an RSSM checkpoint, migrating encoder modality changes automatically.

    Handles both *full* training checkpoints (containing a
    ``"model_state_dict"`` key, as written by ``train_rssm._save_checkpoint``)
    and *bare* model state dicts (weights only).

    The checkpoint is deserialised with ``weights_only=True``. That is a
    behaviour change from the pre-gate version, which passed
    ``weights_only=False`` with no preceding digest check — an arbitrary-code
    path on any checkpoint that reached the rover. Checkpoints written by this
    project (``train_rssm._save_checkpoint``, ``model.state_dict()``) contain
    only tensors and plain containers and load unchanged under the safe mode;
    a checkpoint that genuinely needs pickle must state a digest and opt in.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        cfg: Target :class:`~mousedroid.config.schema.ModelConfig`.
        device: Target device; defaults to CPU when ``None``.
        strict: Passed to :meth:`torch.nn.Module.load_state_dict`.
        expected_sha256: Hex SHA-256 the file must match before it is
            deserialised. ``None`` skips verification (unchanged default) but
            the safe load mode still applies.
        allow_unsafe_pickle: Opt into ``weights_only=False`` for a legacy
            checkpoint carrying non-tensor objects. Requires
            ``expected_sha256``; a bare request is refused.

    Returns:
        A fully initialised :class:`~mousedroid.world_model.rssm.RSSM`
        placed on *device*.

    Raises:
        CheckpointIntegrityError: Digest mismatch, or unsafe pickle requested
            without a digest.
        TypeError: If the checkpoint file does not deserialise to a ``dict``.
        ValueError: If the state dict is incompatible with *cfg*.
    """
    # Lazy import avoids a circular-import cycle (rssm → encoder → this module).
    from mousedroid.world_model.rssm import RSSM

    _d = device if device is not None else torch.device("cpu")
    raw = _load_checkpoint_payload(
        path,
        _d,
        expected_sha256=expected_sha256,
        allow_unsafe_pickle=allow_unsafe_pickle,
    )
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
