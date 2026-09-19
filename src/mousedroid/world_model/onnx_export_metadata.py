"""Export-metadata sidecar for the ``observe_step`` ONNX artifact.

A digest proves an artifact is the bytes someone recorded. It does not say
*what* those bytes are: which checkpoint trained them, which config shaped the
graph, which commit produced them, or which modality set the graph declares.
The runtime spec makes that gap explicit — "A SHA-256 check does not detect
this: the digest can be perfectly valid for the wrong graph" — so every export
writes this sidecar beside the ``.onnx``.

The I/O contract is **not** re-derived here. Input and output names come from
:mod:`mousedroid.world_model.onnx_io`, the module whose docstring exists to
stop exactly that drift, and which
``tests/unit/world_model/test_onnx_io.py`` pins. Shapes come from the same
:class:`~mousedroid.config.schema.ModelConfig` fields
``build_example_inputs`` uses, so a modality toggled in YAML moves the
metadata and the graph together.

Everything here is pure and dependency-light: no ``onnxruntime``, no ``onnx``,
no ``torch``. :func:`build_export_metadata` is a total function of its
arguments, so the export script can be tested in the blocking ``test`` job
(which installs no ONNX extras) and an operator can rebuild the record for an
artifact without a runtime installed.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Final

from mousedroid.config.schema import ModelConfig
from mousedroid.constants import N_SENSOR_MODALITIES_WITH_IMU
from mousedroid.world_model.onnx_io import (
    OBSERVE_STEP_BATCH_DIM_NAME,
    OBSERVE_STEP_INPUT_AUDIO,
    OBSERVE_STEP_INPUT_H,
    OBSERVE_STEP_INPUT_IMU,
    OBSERVE_STEP_INPUT_LIDAR,
    OBSERVE_STEP_INPUT_MOTOR,
    OBSERVE_STEP_INPUT_PREV_ACTION,
    OBSERVE_STEP_INPUT_ULTRASONIC,
    OBSERVE_STEP_INPUT_VALID_MASK,
    OBSERVE_STEP_INPUT_VISION,
    OBSERVE_STEP_INPUT_Z,
    OBSERVE_STEP_OUTPUT_NAMES,
    OBSERVE_STEP_OUTPUT_NEW_H,
    OBSERVE_STEP_OUTPUT_NEW_Z,
    OBSERVE_STEP_OUTPUT_OBS_EMBED,
    OBSERVE_STEP_OUTPUT_SURPRISE,
    all_input_names_for_cfg,
)

EXPORT_METADATA_SCHEMA_VERSION: Final[int] = 1
"""Sidecar schema version. Bump when a field's meaning changes, so a reader
can refuse a record it does not understand instead of guessing."""

EXPORT_TENSOR_DTYPE: Final[str] = "float32"
"""Element type of every graph tensor.

``build_example_inputs`` constructs every example tensor as
``dtype=torch.float32`` and ``observation_packer.pack_observation`` builds
every live tensor the same way, so the whole graph is single-precision today.
Recorded explicitly rather than assumed: an FP16 overlay changes this value,
and a reader comparing an artifact against a config needs to see which one it
got.
"""

EXPORT_TOOL_PACKAGES: Final[tuple[str, ...]] = (
    "torch",
    "onnx",
    "onnxruntime",
    "numpy",
)
"""Packages whose versions decide whether an artifact reproduces.

``torch`` and ``onnx`` produce the graph; ``onnxruntime`` consumes it and
picks execution providers; ``numpy`` is the array boundary on the portable
path. A package that is not installed is recorded as absent rather than
omitted, so a reader can tell "not installed" from "not recorded".
"""

_NOT_INSTALLED: Final[str] = "not_installed"
"""Recorded in place of a version for a package absent from the environment."""


def config_digest(cfg: ModelConfig) -> str:
    """Return a stable SHA-256 over ``cfg``'s canonical JSON form.

    Canonicalised with sorted keys and no insignificant whitespace, so the
    digest depends on the config's *values* and not on field declaration
    order, YAML formatting or Pydantic version. Two exports that agree on this
    digest were shaped by the same model configuration.

    Args:
        cfg: The model config the export was produced from.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(
        cfg.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _modality_dims(cfg: ModelConfig) -> dict[str, int]:
    """Return the per-input feature width keyed by graph input name."""
    return {
        OBSERVE_STEP_INPUT_VISION: cfg.vision_dim,
        OBSERVE_STEP_INPUT_MOTOR: cfg.motor_state_dim,
        OBSERVE_STEP_INPUT_VALID_MASK: N_SENSOR_MODALITIES_WITH_IMU,
        OBSERVE_STEP_INPUT_PREV_ACTION: cfg.action_dim,
        OBSERVE_STEP_INPUT_H: cfg.hidden_dim + cfg.cfc_hidden_dim,
        OBSERVE_STEP_INPUT_Z: cfg.latent_dim,
        OBSERVE_STEP_INPUT_ULTRASONIC: cfg.ultrasonic_dim,
        OBSERVE_STEP_INPUT_AUDIO: cfg.audio_dim,
        OBSERVE_STEP_INPUT_LIDAR: cfg.lidar_dim,
        OBSERVE_STEP_INPUT_IMU: cfg.imu_dim,
    }


def input_specs_for_cfg(cfg: ModelConfig) -> dict[str, dict[str, object]]:
    """Return ``{input_name: {"shape": [...], "dtype": ...}}`` for ``cfg``.

    Names and their order come from
    :func:`mousedroid.world_model.onnx_io.all_input_names_for_cfg`, so a
    disabled modality is absent here exactly as it is absent from the graph.
    Axis 0 is the symbolic batch dimension the exporter declares.

    Args:
        cfg: The model config the export was produced from.

    Returns:
        Insertion-ordered mapping matching the graph's declared input list.
    """
    dims = _modality_dims(cfg)
    return {
        name: {
            "shape": [OBSERVE_STEP_BATCH_DIM_NAME, dims[name]],
            "dtype": EXPORT_TENSOR_DTYPE,
        }
        for name in all_input_names_for_cfg(cfg)
    }


def output_specs_for_cfg(cfg: ModelConfig) -> dict[str, dict[str, object]]:
    """Return ``{output_name: {"shape": [...], "dtype": ...}}`` for ``cfg``.

    ``surprise`` is a **scalar**: ``observe_step_traceable`` returns the
    reduced KL divergence, which the runtime flattens and reads element zero
    of. Recording it as ``[]`` rather than ``[batch, 1]`` keeps the sidecar
    honest about what the graph actually publishes.

    Args:
        cfg: The model config the export was produced from.

    Returns:
        Mapping keyed in :data:`OBSERVE_STEP_OUTPUT_NAMES` order.
    """
    batched: dict[str, int] = {
        OBSERVE_STEP_OUTPUT_NEW_H: cfg.hidden_dim + cfg.cfc_hidden_dim,
        OBSERVE_STEP_OUTPUT_NEW_Z: cfg.latent_dim,
        OBSERVE_STEP_OUTPUT_OBS_EMBED: cfg.obs_dim,
    }
    specs: dict[str, dict[str, object]] = {}
    for name in OBSERVE_STEP_OUTPUT_NAMES:
        shape: list[object] = (
            []
            if name == OBSERVE_STEP_OUTPUT_SURPRISE
            else [OBSERVE_STEP_BATCH_DIM_NAME, batched[name]]
        )
        specs[name] = {"shape": shape, "dtype": EXPORT_TENSOR_DTYPE}
    return specs


def collect_tool_versions(
    packages: tuple[str, ...] = EXPORT_TOOL_PACKAGES,
) -> dict[str, str]:
    """Return installed versions for ``packages`` plus the interpreter version.

    Uses :mod:`importlib.metadata` rather than importing each package, so
    collecting versions never pulls ``onnxruntime`` into the process — the
    blocking ``test`` job has no ORT installed and must still exercise this.

    Args:
        packages: Distribution names to look up.

    Returns:
        Mapping of name to version string. A package absent from the
        environment maps to :data:`_NOT_INSTALLED`; the interpreter is always
        present under the ``"python"`` key.
    """
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = _NOT_INSTALLED
    versions["python"] = platform.python_version()
    return versions


def build_export_metadata(
    *,
    cfg: ModelConfig,
    opset: int,
    artifact_filename: str,
    artifact_sha256: str,
    checkpoint_path: str | None,
    checkpoint_sha256: str | None,
    git_sha: str | None,
    tool_versions: Mapping[str, str],
) -> dict[str, object]:
    """Assemble the export-metadata record.

    Pure: no filesystem, no subprocess, no network, no optional imports. The
    caller resolves the impure inputs (digests, git SHA, tool versions) and
    hands them in, which is what makes the record reproducible in a test.

    A ``None`` for ``checkpoint_path`` / ``checkpoint_sha256`` is preserved
    rather than dropped, because "exported from a freshly-initialised model"
    is the single most important thing a promotion reviewer can learn from
    this file — the export script's own ``--checkpoint`` help calls that case
    "never for production deployment", and a missing key would read as an
    oversight instead of a finding.

    Args:
        cfg: Model config the graph was exported from.
        opset: ONNX opset version passed to ``torch.onnx.export``.
        artifact_filename: Basename of the ``.onnx`` this record describes.
        artifact_sha256: Digest of that artifact.
        checkpoint_path: Checkpoint the weights came from, or ``None``.
        checkpoint_sha256: Digest of that checkpoint, or ``None``.
        git_sha: Commit the export ran from, or ``None`` when unresolvable.
        tool_versions: Versions of the tools that produced the artifact.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        "schema_version": EXPORT_METADATA_SCHEMA_VERSION,
        "artifact": {
            "filename": artifact_filename,
            "sha256": artifact_sha256,
        },
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
        },
        "config": {
            "sha256": config_digest(cfg),
            "modality_dims": {
                "vision_dim": cfg.vision_dim,
                "ultrasonic_dim": cfg.ultrasonic_dim,
                "audio_dim": cfg.audio_dim,
                "lidar_dim": cfg.lidar_dim,
                "imu_dim": cfg.imu_dim,
                "motor_state_dim": cfg.motor_state_dim,
            },
        },
        "git_sha": git_sha,
        "opset": opset,
        "inputs": input_specs_for_cfg(cfg),
        "outputs": output_specs_for_cfg(cfg),
        "tool_versions": dict(tool_versions),
    }


def write_export_metadata(metadata: Mapping[str, object], path: Path) -> None:
    """Write ``metadata`` to ``path`` as sorted, newline-terminated JSON.

    Sorted keys and a trailing newline make the sidecar diffable and let a
    promotion review see a one-field change as a one-line change.

    Args:
        metadata: Record from :func:`build_export_metadata`.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(metadata), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")


__all__ = [
    "EXPORT_METADATA_SCHEMA_VERSION",
    "EXPORT_TENSOR_DTYPE",
    "EXPORT_TOOL_PACKAGES",
    "build_export_metadata",
    "collect_tool_versions",
    "config_digest",
    "input_specs_for_cfg",
    "output_specs_for_cfg",
    "write_export_metadata",
]
