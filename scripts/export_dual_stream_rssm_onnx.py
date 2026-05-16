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
        --config config/jetson_production.yaml \\
        --output weights/dual_stream_rssm/observe_step.onnx \\
        --opset 17

This script also exposes ``build_export_shim``, ``build_example_inputs``,
and ``run_export`` as library entry points so unit tests can exercise
the export logic in-process (much faster than spawning a subprocess
per test case).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.config.schema import ModelConfig
from mousedroid.logging.setup import get_logger
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default tracing parameters — keep here, not at the call site
# ---------------------------------------------------------------------------
_DEFAULT_OPSET = 17
_EXPORT_BATCH = 1
_BATCH_DIM_NAME = "batch"


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
    n_modalities = 5  # vision, ultrasonic, motor, audio, lidar slots
    inputs: dict[str, Tensor] = {
        "vision": torch.zeros(batch_size, cfg.vision_dim, dtype=torch.float32, device=device),
        "motor": torch.zeros(batch_size, cfg.motor_state_dim, dtype=torch.float32, device=device),
        "valid_mask": torch.ones(batch_size, n_modalities, dtype=torch.float32, device=device),
        "prev_action": torch.zeros(batch_size, cfg.action_dim, dtype=torch.float32, device=device),
        "h": torch.zeros(batch_size, combined_h_dim, dtype=torch.float32, device=device),
        "z": torch.zeros(batch_size, cfg.latent_dim, dtype=torch.float32, device=device),
    }
    if cfg.ultrasonic_dim > 0:
        inputs["ultrasonic"] = torch.zeros(
            batch_size, cfg.ultrasonic_dim, dtype=torch.float32, device=device
        )
    if cfg.audio_dim > 0:
        inputs["audio"] = torch.zeros(batch_size, cfg.audio_dim, dtype=torch.float32, device=device)
    if cfg.lidar_dim > 0:
        inputs["lidar"] = torch.zeros(batch_size, cfg.lidar_dim, dtype=torch.float32, device=device)
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
        axes[name] = {0: _BATCH_DIM_NAME}
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
    output_names = ["new_h", "new_z", "obs_embed", "surprise"]

    _log.info(
        "world_model_export_started",
        output=str(output_path),
        opset=opset,
        input_names=input_names,
        output_names=output_names,
        ultrasonic_enabled=cfg.ultrasonic_dim > 0,
        audio_enabled=cfg.audio_dim > 0,
        lidar_enabled=cfg.lidar_dim > 0,
    )
    started = time.perf_counter()

    torch.onnx.export(
        shim,
        tuple(inputs.values()),
        str(output_path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=_dynamic_axes_for_inputs(input_names, output_names),
        do_constant_folding=True,
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


def _load_checkpoint(model: DualStreamRSSM, ckpt_path: Path) -> None:
    """Load model weights from a checkpoint file.

    Supports both raw ``state_dict()`` checkpoints (``torch.save(model.state_dict())``)
    and dict-wrapped checkpoints (``torch.save({"model_state_dict": ...})``).
    The exporter never trains — ``map_location='cpu'`` is unconditional.
    """
    if not ckpt_path.exists():
        msg = f"checkpoint not found: {ckpt_path}"
        raise FileNotFoundError(msg)
    # weights_only=True is the supported safe-load mode on torch 2.5+; older
    # torch raises TypeError and we fall back to the legacy loader.
    state: object
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover — older torch
        state = torch.load(ckpt_path, map_location="cpu")
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
        help="Path to a Settings YAML. When omitted, uses ModelConfig() defaults.",
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
        msg = (
            "DualStreamRSSM requires cfc_hidden_dim > 0; "
            f"got {cfg.cfc_hidden_dim}. Set ModelConfig.cfc_hidden_dim "
            "in your config YAML."
        )
        raise ValueError(msg)
    model = DualStreamRSSM(cfg)
    if args.checkpoint is not None:
        _load_checkpoint(model, args.checkpoint)
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
    if args.push_to_hf:
        repo_id, filename = _resolve_hf_repo(args)
        _push_to_hf(args.output, repo_id=repo_id, filename=filename)
    return 0


if __name__ == "__main__":
    sys.exit(main())
