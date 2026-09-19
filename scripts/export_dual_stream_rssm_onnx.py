"""Export ``DualStreamRSSM.observe_step`` to ONNX.

Tier B Track B2 Story 1 — produces a portable ``.onnx`` artifact from
a trained PyTorch checkpoint. The exported graph runs via
``onnxruntime`` with TensorRT / CUDA / CPU execution provider fallback
on the Jetson Orin Nano (see :class:`DualStreamRSSMOnnx` runtime class
in Story 2).

The CfC ONNX exportability spike at ``tools/spikes/cfc_onnx_spike.py``
proved this path works (numerical equivalence 4.47e-08, deterministic).
This script generalises that result to the full ``observe_step``
(GRU + encoder + StreamFusion + CfC + posterior + prior) in one fused
ONNX graph.

CLI usage::

    python scripts/export_dual_stream_rssm_onnx.py \\
        --checkpoint weights/dual_stream_rssm/final.pt \\
        --config config/jetson_dual_stream.yaml \\
        --output weights/dual_stream_rssm/observe_step.onnx \\
        --opset 17

``--config`` must name an overlay that sets ``model.cfc_hidden_dim > 0``
(the export is built from ``DualStreamRSSM``, which requires the CfC
stream). Only ``config/jetson_dual_stream.yaml`` does;
``config/jetson_production.yaml`` carries no ``model:`` block, so its
``cfc_hidden_dim`` is the schema default ``0``. The same constraint
applies to the runtime: ``world_model.engine = "onnx_trt"`` cannot be
enabled from ``config/jetson_production.yaml`` -- the factory raises
``ValueError`` at boot. ``config/jetson_dual_stream.yaml`` self-gates
the CfC stream behind human review ("Only enable (64+) after human
review of training metrics").

This script also exposes ``build_export_shim``, ``build_example_inputs``,
and ``run_export`` as library entry points so unit tests can exercise
the export logic in-process (much faster than spawning a subprocess
per test case).
"""

from __future__ import annotations

import argparse
import inspect
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.constants import N_SENSOR_MODALITIES_WITH_IMU
from mousedroid.logging.setup import get_logger
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
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
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default tracing parameters — keep here, not at the call site
# ---------------------------------------------------------------------------
_DEFAULT_OPSET = 17
_EXPORT_BATCH = 1


class _ObserveStepExportShim(nn.Module):
    """Wraps ``DualStreamRSSM.observe_step_traceable`` for ONNX tracing.

    The CfC spike (Story 0) established that ``torch.onnx.export`` cannot
    trace the keyword-only ``dt`` parameter on ``CfCWrapper.forward``
    directly. The same principle applies to ``observe_step_traceable``'s
    keyword-only arguments — the tracer treats them as positional. This
    shim narrows the API to a single positional-only ``forward(...)``
    signature that ``torch.onnx.export`` traces cleanly.

    The shim's forward signature is determined by which modalities are
    enabled in ``cfg``. Disabled modalities (e.g. ``cfg.audio_dim == 0``)
    are NOT exposed as inputs — keeping the ONNX graph minimal so the
    runtime class doesn't have to feed phantom tensors.
    """

    def __init__(self, rssm: DualStreamRSSM) -> None:
        super().__init__()
        self._rssm = rssm
        cfg = rssm._cfg
        self._ultrasonic_enabled = cfg.ultrasonic_dim > 0
        self._audio_enabled = cfg.audio_dim > 0
        self._lidar_enabled = cfg.lidar_dim > 0
        self._imu_enabled = cfg.imu_dim > 0

    def forward(
        self,
        vision: Tensor,
        motor: Tensor,
        valid_mask: Tensor,
        prev_action: Tensor,
        h: Tensor,
        z: Tensor,
        ultrasonic: Tensor | None = None,
        audio: Tensor | None = None,
        lidar: Tensor | None = None,
        imu: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run one observe step on tensors only.

        Args:
            vision: ``(batch, cfg.vision_dim)``.
            motor: ``(batch, cfg.motor_state_dim)``.
            valid_mask: ``(batch, n_modalities)``.
            prev_action: ``(batch, cfg.action_dim)``.
            h: ``(batch, hidden_dim + cfc_hidden_dim)``.
            z: ``(batch, cfg.latent_dim)``.
            ultrasonic: ``(batch, cfg.ultrasonic_dim)`` when enabled.
            audio: ``(batch, cfg.audio_dim)`` when enabled.
            lidar: ``(batch, cfg.lidar_dim)`` when enabled.
            imu: ``(batch, cfg.imu_dim)`` when enabled.

        Returns:
            ``(new_h, new_z, obs_embed, surprise)`` — all ``Tensor``.
        """
        # ``observe_step_traceable`` takes keyword args; supplying ``None``
        # for disabled modalities is the contract the encoder reads.
        return self._rssm.observe_step_traceable(
            vision=vision,
            motor=motor,
            valid_mask=valid_mask,
            ultrasonic=ultrasonic if self._ultrasonic_enabled else None,
            audio=audio if self._audio_enabled else None,
            lidar=lidar if self._lidar_enabled else None,
            imu=imu if self._imu_enabled else None,
            prev_action=prev_action,
            h=h,
            z=z,
        )


def build_export_shim(model: DualStreamRSSM) -> nn.Module:
    """Construct the export shim around a ``DualStreamRSSM`` instance.

    Args:
        model: A constructed (optionally checkpoint-loaded) ``DualStreamRSSM``.

    Returns:
        An ``nn.Module`` whose ``forward`` consumes flat tensors and is
        directly traceable by ``torch.onnx.export``.
    """
    model.train(False)
    return _ObserveStepExportShim(model)


def build_example_inputs(
    cfg: ModelConfig,
    *,
    device: torch.device,
    batch_size: int = _EXPORT_BATCH,
) -> dict[str, Tensor]:
    """Build the example input dict the export shim's forward expects.

    Required keys: ``vision``, ``motor``, ``valid_mask``, ``prev_action``,
    ``h``, ``z``. Optional keys are added based on enabled modalities:
    ``ultrasonic`` when ``cfg.ultrasonic_dim > 0``, ``audio`` when
    ``cfg.audio_dim > 0``, ``lidar`` when ``cfg.lidar_dim > 0``.

    Args:
        cfg: Model configuration — drives all tensor shapes.
        device: Target device for the example tensors.
        batch_size: Batch dimension for the example tensors. Defaults to
            ``1`` for inference-style export. The exported ``.onnx`` still
            supports dynamic batch at runtime via ``dynamic_axes``.

    Returns:
        Dict ready for ``shim(**inputs)`` and
        ``torch.onnx.export(... , tuple(inputs.values()), ...)``.
    """
    combined_h_dim = cfg.hidden_dim + cfg.cfc_hidden_dim
    # Modality slot count in the valid_mask vector. Single source of truth
    # lives in ``mousedroid.constants.SENSOR_SLOT_MAP`` / the explicit
    # ``N_SENSOR_MODALITIES_WITH_IMU`` constant so the export contract
    # tracks the encoder's slot layout automatically — operators adding a
    # new modality update one place. Note this is SLOTS, not enabled count:
    # disabled modalities still occupy their slot for stable ordering.
    n_modalities = N_SENSOR_MODALITIES_WITH_IMU
    inputs: dict[str, Tensor] = {
        OBSERVE_STEP_INPUT_VISION: torch.zeros(
            batch_size, cfg.vision_dim, dtype=torch.float32, device=device
        ),
        OBSERVE_STEP_INPUT_MOTOR: torch.zeros(
            batch_size, cfg.motor_state_dim, dtype=torch.float32, device=device
        ),
        OBSERVE_STEP_INPUT_VALID_MASK: torch.ones(
            batch_size, n_modalities, dtype=torch.float32, device=device
        ),
        OBSERVE_STEP_INPUT_PREV_ACTION: torch.zeros(
            batch_size, cfg.action_dim, dtype=torch.float32, device=device
        ),
        OBSERVE_STEP_INPUT_H: torch.zeros(
            batch_size, combined_h_dim, dtype=torch.float32, device=device
        ),
        OBSERVE_STEP_INPUT_Z: torch.zeros(
            batch_size, cfg.latent_dim, dtype=torch.float32, device=device
        ),
    }
    if cfg.ultrasonic_dim > 0:
        inputs[OBSERVE_STEP_INPUT_ULTRASONIC] = torch.zeros(
            batch_size, cfg.ultrasonic_dim, dtype=torch.float32, device=device
        )
    if cfg.audio_dim > 0:
        inputs[OBSERVE_STEP_INPUT_AUDIO] = torch.zeros(
            batch_size, cfg.audio_dim, dtype=torch.float32, device=device
        )
    if cfg.lidar_dim > 0:
        inputs[OBSERVE_STEP_INPUT_LIDAR] = torch.zeros(
            batch_size, cfg.lidar_dim, dtype=torch.float32, device=device
        )
    if cfg.imu_dim > 0:
        inputs[OBSERVE_STEP_INPUT_IMU] = torch.zeros(
            batch_size, cfg.imu_dim, dtype=torch.float32, device=device
        )
    return inputs


def _dynamic_axes_for_inputs(
    input_names: list[str], output_names: list[str]
) -> dict[str, dict[int, str]]:
    """Mark axis 0 as the dynamic ``batch`` dimension for every tensor.

    Both inputs and outputs vary along axis 0, so future training-time
    use of the same ``.onnx`` doesn't require a separate export per batch
    size — see ``CFC_ONNX_SPIKE_REPORT.md`` for the rationale.
    """
    axes: dict[str, dict[int, str]] = {}
    for name in input_names + output_names:
        axes[name] = {0: OBSERVE_STEP_BATCH_DIM_NAME}
    return axes


def run_export(
    *,
    model: DualStreamRSSM,
    cfg: ModelConfig,
    output_path: Path,
    opset: int = _DEFAULT_OPSET,
) -> None:
    """Export the model's ``observe_step`` to an ``.onnx`` file.

    Args:
        model: A constructed (optionally checkpoint-loaded) ``DualStreamRSSM``.
        cfg: Model configuration — drives example-input shapes and the
            list of enabled modalities.
        output_path: Filesystem path for the produced ``.onnx``.
        opset: ONNX opset version. Default 17 — established by the
            Story 0 spike as working on torch 2.5.1.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shim = build_export_shim(model)
    inputs = build_example_inputs(cfg, device=torch.device("cpu"))
    input_names = list(inputs.keys())
    output_names = list(OBSERVE_STEP_OUTPUT_NAMES)

    # Build (positional, kwargs) ``args`` so torch.onnx.export binds inputs
    # to the shim's forward by NAME, not by position. Without this, when
    # ``cfg.ultrasonic_dim == 0`` but ``cfg.audio_dim > 0`` (or any other
    # gap in the optional-modality run), ``tuple(inputs.values())`` would
    # shift the later optional into the slot of the disabled earlier one
    # — e.g. the audio tensor would land in the ``ultrasonic`` parameter
    # and the lidar tensor would land in ``audio``. torch.onnx.export
    # accepts ``args = (positional_tuple, kwargs_dict)`` to bind by name.
    required_names = {
        OBSERVE_STEP_INPUT_VISION,
        OBSERVE_STEP_INPUT_MOTOR,
        OBSERVE_STEP_INPUT_VALID_MASK,
        OBSERVE_STEP_INPUT_PREV_ACTION,
        OBSERVE_STEP_INPUT_H,
        OBSERVE_STEP_INPUT_Z,
    }
    positional_args = tuple(inputs[name] for name in input_names if name in required_names)
    keyword_args: dict[str, Tensor] = {
        name: inputs[name] for name in input_names if name not in required_names
    }
    export_args: tuple[Tensor | dict[str, Tensor], ...] = (
        (*positional_args, keyword_args) if keyword_args else positional_args
    )

    _log.info(
        "world_model_export_started",
        output=str(output_path),
        opset=opset,
        input_names=input_names,
        output_names=output_names,
        ultrasonic_enabled=cfg.ultrasonic_dim > 0,
        audio_enabled=cfg.audio_dim > 0,
        lidar_enabled=cfg.lidar_dim > 0,
        n_required=len(positional_args),
        n_optional=len(keyword_args),
    )
    started = time.perf_counter()

    export_kwargs: dict[str, Any] = {
        "opset_version": opset,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": _dynamic_axes_for_inputs(input_names, output_names),
        "do_constant_folding": True,
    }
    # torch >= 2.4 added the ``dynamo`` kwarg; torch >= 2.6 flipped its
    # default from False to True (new dynamo exporter, requires onnxscript).
    # Pin to the legacy TorchScript exporter that the CfC spike (B2 Story 0)
    # validated against — same exporter the CfC numerical-equivalence test
    # was tuned for. Older torch (<2.4) doesn't recognise the kwarg, so we
    # introspect the signature and pass it only when supported. Project's
    # minimum is ``torch>=2.1`` per pyproject.toml.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
        _log.debug("world_model_export_using_legacy_exporter")
    torch.onnx.export(
        shim,
        export_args,
        str(output_path),
        **export_kwargs,
    )

    elapsed_s = time.perf_counter() - started
    _log.info(
        "world_model_export_finished",
        output=str(output_path),
        elapsed_s=elapsed_s,
        artifact_bytes=output_path.stat().st_size,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_checkpoint(
    model: DualStreamRSSM,
    ckpt_path: Path,
    *,
    expected_sha256: str | None = None,
) -> None:
    """Load model weights from a checkpoint file, verifying digest first.

    Supports both raw ``state_dict()`` checkpoints (``torch.save(model.state_dict())``)
    and dict-wrapped checkpoints (``torch.save({"model_state_dict": ...})``).
    The exporter never trains — ``map_location='cpu'`` is unconditional.

    ``weights_only=True`` is unconditional. The previous version wrapped it in
    ``except TypeError`` and silently retried with the legacy (arbitrary
    pickle) loader — a fail-open fallback whose trigger condition cannot occur
    here: ``pyproject.toml`` requires ``torch>=2.1`` and the kwarg has existed
    since 1.13, so the fallback could only ever fire on an unsupported
    interpreter, where running unverified pickle is the worst response. A real
    ``TypeError`` now surfaces instead of being swallowed.

    Args:
        model: Model whose ``state_dict`` is replaced.
        ckpt_path: ``.pt`` checkpoint to load.
        expected_sha256: Hex digest the checkpoint must match before it is
            read. ``None`` skips verification — the published artifact records
            whichever digest was actually loaded, so an unverified export is
            visible in the sidecar rather than indistinguishable from a
            verified one.

    Raises:
        FileNotFoundError: Checkpoint absent.
        ValueError: Digest supplied and did not match.
    """
    from mousedroid.utils.weights_manager import verify_sha256

    if not ckpt_path.exists():
        msg = f"checkpoint not found: {ckpt_path}"
        raise FileNotFoundError(msg)
    if expected_sha256 is not None and not verify_sha256(
        ckpt_path, expected_sha256, log_event_prefix="world_model_checkpoint"
    ):
        msg = (
            f"refusing to export from '{ckpt_path}': SHA-256 verification "
            f"failed against the digest passed via --checkpoint-sha256."
        )
        raise ValueError(msg)
    state: object = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        payload = cast(dict[str, Tensor], state["model_state_dict"])
    else:
        payload = cast(dict[str, Tensor], state)
    model.load_state_dict(payload)
    _log.info("world_model_checkpoint_loaded", path=str(ckpt_path))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        default=None,
        help=(
            "Path to a .pt checkpoint to load before export. When omitted, "
            "exports a freshly-initialised model — useful for CI smoke tests "
            "but never for production deployment."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        default=None,
        help=(
            "Path to a Settings YAML. When omitted, the script uses "
            "ModelConfig defaults — note this requires cfc_hidden_dim > 0, "
            "which the schema default does NOT satisfy. Pass --config "
            "for production exports."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .onnx path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=_DEFAULT_OPSET,
        help=f"ONNX opset version (default {_DEFAULT_OPSET}).",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        type=str,
        default=None,
        help=(
            "Hex SHA-256 the --checkpoint must match before it is loaded. "
            "Verified with the same helper the OTA weight path uses; a "
            "mismatch aborts the export. Omit to export without verifying — "
            "the metadata sidecar records the checkpoint digest either way, "
            "so a promotion reviewer can tell the two cases apart."
        ),
    )
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        help=(
            "After successful export, upload the .onnx to HuggingFace Hub. "
            "Requires HUGGINGFACE_TOKEN env var or prior `huggingface-cli login`. "
            "Repo defaults to --hf-repo (or cfg.world_model.onnx_repo_id when "
            "--config is provided)."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help=(
            "HuggingFace Hub repo to upload to. When omitted, falls back to "
            "cfg.world_model.onnx_repo_id (when --config is provided) or "
            "'ianshank/mousedroid-dual-stream-rssm'."
        ),
    )
    return parser.parse_args(argv)


def _build_model_from_cli(args: argparse.Namespace) -> tuple[DualStreamRSSM, ModelConfig]:
    """Resolve config + construct the model from CLI arguments."""
    if args.config is not None:
        from mousedroid.config.loader import load_settings

        settings = load_settings(args.config)
        cfg = settings.model
    else:
        # ``ModelConfig()`` would work at runtime (all fields have defaults)
        # but mypy --strict doesn't see Pydantic's field defaults as keyword
        # defaults; ``model_validate`` is the Pydantic-recommended fully-typed
        # construction path for an empty input.
        cfg = ModelConfig.model_validate({})
    if cfg.cfc_hidden_dim <= 0:
        config_arg = "--config <path-to-yaml>" if args.config is None else f"--config {args.config}"
        msg = (
            f"DualStreamRSSM requires cfc_hidden_dim > 0; got "
            f"{cfg.cfc_hidden_dim}. The schema default is 0 (pure-GRU mode); "
            f"pass {config_arg} pointing at a YAML that sets "
            "model.cfc_hidden_dim to a positive value (e.g. "
            "config/jetson_production.yaml)."
        )
        raise ValueError(msg)
    model = DualStreamRSSM(cfg)
    if args.checkpoint is not None:
        _load_checkpoint(model, args.checkpoint, expected_sha256=args.checkpoint_sha256)
    return model, cfg


def _push_to_hf(
    onnx_path: Path,
    repo_id: str,
    filename: str,
) -> None:
    """Upload the freshly-exported ``.onnx`` to HuggingFace Hub.

    Imports ``huggingface_hub`` lazily — the dependency is only required
    when ``--push-to-hf`` is set. Errors are surfaced upward so CI sees
    the failure clearly; no silent retry / no exception swallowing.

    Args:
        onnx_path: Local ``.onnx`` to upload.
        repo_id: Target HF repo (e.g. ``ianshank/mousedroid-dual-stream-rssm``).
        filename: Destination filename inside the repo.
    """
    from huggingface_hub import HfApi  # lazy import

    _log.info(
        "world_model_onnx_push_start",
        repo_id=repo_id,
        filename=filename,
        local_path=str(onnx_path),
    )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(onnx_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    _log.info(
        "world_model_onnx_push_finished",
        repo_id=repo_id,
        filename=filename,
    )


def _resolve_git_sha() -> str | None:
    """Return the current commit SHA, or ``None`` when it cannot be resolved.

    Shells out to a fixed ``git rev-parse HEAD`` argv — no shell, no
    interpolation, nothing operator-supplied. The executable is resolved to an
    absolute path through :func:`shutil.which` rather than relying on argv[0]
    PATH lookup, so the invocation carries no partial executable path.
    Returns ``None`` rather than raising when git is absent or the tree is not
    a repository: a sidecar with ``git_sha: null`` is honest and still useful,
    whereas aborting a successful export over provenance metadata is not.
    """
    git_binary = shutil.which("git")
    if git_binary is None:
        _log.warning("world_model_export_git_sha_unresolved", reason="git_not_on_path")
        return None
    try:
        proc = subprocess.run(
            [git_binary, "rev-parse", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        _log.warning("world_model_export_git_sha_unresolved", error_type=type(exc).__name__)
        return None
    if proc.returncode != 0:
        _log.warning(
            "world_model_export_git_sha_unresolved",
            returncode=proc.returncode,
        )
        return None
    return proc.stdout.strip() or None


def _resolve_metadata_filename(args: argparse.Namespace) -> str:
    """Resolve the sidecar filename from config, never from a literal here.

    ``cfg.world_model.onnx_metadata_filename`` owns the name. Without
    ``--config`` the schema default is read from ``WorldModelConfig`` itself,
    so the two paths cannot disagree.
    """
    from mousedroid.config.schema import WorldModelConfig

    if args.config is not None:
        from mousedroid.config.loader import load_settings

        return load_settings(args.config).world_model.onnx_metadata_filename
    return WorldModelConfig.model_validate({}).onnx_metadata_filename


def _write_metadata_sidecar(args: argparse.Namespace, cfg: ModelConfig) -> Path:
    """Write the export-metadata sidecar beside the produced ``.onnx``.

    Args:
        args: Parsed CLI namespace (``--output``, ``--opset``, ``--checkpoint``,
            ``--config``).
        cfg: Model config the graph was exported from.

    Returns:
        Path of the written sidecar.
    """
    from mousedroid.common.hashing import digest_file_sha256
    from mousedroid.world_model.onnx_export_metadata import (
        build_export_metadata,
        collect_tool_versions,
        write_export_metadata,
    )

    output_path: Path = args.output
    checkpoint: Path | None = args.checkpoint
    metadata = build_export_metadata(
        cfg=cfg,
        opset=args.opset,
        artifact_filename=output_path.name,
        artifact_sha256=digest_file_sha256(output_path),
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        checkpoint_sha256=(digest_file_sha256(checkpoint) if checkpoint is not None else None),
        git_sha=_resolve_git_sha(),
        tool_versions=collect_tool_versions(),
    )
    sidecar_path = output_path.parent / _resolve_metadata_filename(args)
    write_export_metadata(metadata, sidecar_path)
    _log.info(
        "world_model_export_metadata_written",
        path=str(sidecar_path),
        artifact_sha256=metadata["artifact"],
        checkpoint_present=checkpoint is not None,
    )
    return sidecar_path


def _resolve_hf_repo(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve (repo_id, filename) for HF upload.

    Order of preference:
    1. --hf-repo CLI arg
    2. cfg.world_model.onnx_repo_id when --config is provided
    3. Default 'ianshank/mousedroid-dual-stream-rssm' /
       'observe_step.onnx'
    """
    repo_id: str
    filename: str
    if args.hf_repo is not None:
        repo_id = args.hf_repo
        filename = args.output.name
    elif args.config is not None:
        from mousedroid.config.loader import load_settings

        settings = load_settings(args.config)
        repo_id = settings.world_model.onnx_repo_id
        filename = settings.world_model.onnx_filename
    else:
        repo_id = "ianshank/mousedroid-dual-stream-rssm"
        filename = args.output.name
    return repo_id, filename


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    model, cfg = _build_model_from_cli(args)
    run_export(model=model, cfg=cfg, output_path=args.output, opset=args.opset)
    _write_metadata_sidecar(args, cfg)
    if args.push_to_hf:
        repo_id, filename = _resolve_hf_repo(args)
        _push_to_hf(args.output, repo_id=repo_id, filename=filename)
    return 0


if __name__ == "__main__":
    sys.exit(main())
