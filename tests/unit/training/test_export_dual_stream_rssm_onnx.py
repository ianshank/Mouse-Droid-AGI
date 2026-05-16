"""Unit tests for ``scripts/export_dual_stream_rssm_onnx.py``.

The export script wraps :meth:`DualStreamRSSM.observe_step_traceable` in
an :class:`nn.Module` shim and runs ``torch.onnx.export``. These tests
verify the script's library-level entry points (``build_export_shim``,
``run_export``) without spinning up a subprocess for each test case —
much faster, and lets us assert on the in-memory model output too.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("ncps")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _load_export_module() -> Any:
    """Import the export script as a module without invoking its CLI.

    ``scripts/`` is intentionally not on ``sys.path`` (it's a CLI directory,
    not a package). Tests import via importlib to avoid spreading
    side-effecting CLI imports across the test suite.
    """
    script_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "export_dual_stream_rssm_onnx.py"
    )
    spec = importlib.util.spec_from_file_location("export_dual_stream_rssm_onnx", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_dual_stream_rssm_onnx"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def export_module() -> Any:
    return _load_export_module()


def _make_cfg():  # type: ignore[no-untyped-def]
    """Tiny ModelConfig — keeps the export under 1 second."""
    from mousedroid.config.schema import ModelConfig

    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        ultrasonic_proj_dim=4,
        motor_state_dim=4,
        hidden_dim=32,
        latent_dim=8,
        action_dim=2,
        obs_dim=16,
        vision_proj_dim=8,
        motor_proj_dim=4,
        cfc_hidden_dim=16,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


class TestExportShim:
    """The shim exposes the encoded tensor signature ONNX needs."""

    def test_shim_constructs_with_a_rssm(self, export_module: Any) -> None:
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        shim = export_module.build_export_shim(model)
        assert isinstance(shim, torch.nn.Module)

    def test_shim_forward_returns_4_tensors(self, export_module: Any) -> None:
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        model.train(False)
        shim = export_module.build_export_shim(model)
        inputs = export_module.build_example_inputs(cfg, device=torch.device("cpu"))
        result = shim(**inputs)
        assert len(result) == 4
        new_h, new_z, obs_embed, surprise = result
        assert isinstance(new_h, torch.Tensor)
        assert isinstance(new_z, torch.Tensor)
        assert isinstance(obs_embed, torch.Tensor)
        assert isinstance(surprise, torch.Tensor)


class TestRunExport:
    """run_export() produces a loadable, runnable, numerically-equivalent .onnx."""

    def test_run_export_writes_valid_onnx_file(
        self,
        export_module: Any,
        tmp_path: Path,
    ) -> None:
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        model.train(False)

        output_path = tmp_path / "observe_step.onnx"
        export_module.run_export(
            model=model,
            cfg=cfg,
            output_path=output_path,
            opset=17,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_exported_onnx_loads_and_runs(
        self,
        export_module: Any,
        tmp_path: Path,
    ) -> None:
        import onnxruntime as ort

        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        model.train(False)

        output_path = tmp_path / "observe_step.onnx"
        export_module.run_export(
            model=model,
            cfg=cfg,
            output_path=output_path,
            opset=17,
        )

        # Re-load and run — proves the file is a valid ORT-consumable graph.
        sess = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"],
        )
        # Build numpy inputs matching the schema declared at export.
        torch_inputs = export_module.build_example_inputs(cfg, device=torch.device("cpu"))
        np_inputs: dict[str, np.ndarray] = {}
        for name, tensor in torch_inputs.items():
            np_inputs[name] = tensor.detach().cpu().numpy()
        outputs = sess.run(None, np_inputs)
        assert len(outputs) == 4

    def test_torch_onnx_numerical_equivalence(
        self,
        export_module: Any,
        tmp_path: Path,
    ) -> None:
        """``np.allclose`` on deterministic outputs across engines (atol=1e-4).

        This is the cross-engine equivalence guarantee that
        ``cfg.world_model.engine`` ships on — flipping ``torch`` →
        ``onnx_trt`` MUST NOT change downstream behaviour beyond float32
        round-off **for deterministic outputs**.

        ``new_z`` (index 1) is intentionally excluded: it samples from the
        posterior Gaussian via ``torch.randn_like``, which is stochastic
        and uses a different RNG path in torch vs onnxruntime. Operators
        consuming ``new_z`` see a different sample per engine, but the
        underlying distribution (mean, logvar) is identical — the same
        guarantee that justifies swapping engines without retraining.
        """
        import onnxruntime as ort

        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        model.train(False)
        torch_inputs = export_module.build_example_inputs(cfg, device=torch.device("cpu"))

        # Reference torch output (call the traceable variant directly).
        torch_outputs = model.observe_step_traceable(**torch_inputs)

        # Export + run via ORT.
        output_path = tmp_path / "observe_step.onnx"
        export_module.run_export(
            model=model,
            cfg=cfg,
            output_path=output_path,
            opset=17,
        )
        sess = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"],
        )
        np_inputs = {name: tensor.detach().cpu().numpy() for name, tensor in torch_inputs.items()}
        ort_outputs = sess.run(None, np_inputs)

        # Compare deterministic outputs (skip index 1 = new_z, stochastic).
        # Output order: (new_h, new_z, obs_embed, surprise) per run_export().
        deterministic_indices = (0, 2, 3)
        for idx in deterministic_indices:
            torch_np = torch_outputs[idx].detach().cpu().numpy()
            onnx_out = ort_outputs[idx]
            assert torch_np.shape == onnx_out.shape, (
                f"shape mismatch at output[{idx}]: "
                f"torch={torch_np.shape}, onnx={onnx_out.shape}"
            )
            max_abs_diff = float(np.abs(torch_np - onnx_out).max())
            assert (
                max_abs_diff < 1e-4
            ), f"output[{idx}] max_abs_diff={max_abs_diff:.6f} exceeds 1e-4"

        # new_z (index 1) — shape must still match even though the sample diverges.
        assert torch_outputs[1].detach().cpu().numpy().shape == ort_outputs[1].shape

    def test_export_uses_dynamic_batch_axis(
        self,
        export_module: Any,
        tmp_path: Path,
    ) -> None:
        """The exported .onnx accepts batch sizes other than 1 (future training-time use)."""
        import onnxruntime as ort

        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        model.train(False)

        output_path = tmp_path / "observe_step.onnx"
        export_module.run_export(
            model=model,
            cfg=cfg,
            output_path=output_path,
            opset=17,
        )

        sess = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"],
        )
        # Try batch=3 inputs (instead of the batch=1 used at export time).
        torch_inputs = export_module.build_example_inputs(
            cfg, device=torch.device("cpu"), batch_size=3
        )
        np_inputs = {name: tensor.detach().cpu().numpy() for name, tensor in torch_inputs.items()}
        outputs = sess.run(None, np_inputs)
        # new_h is outputs[0]; batch dim should now be 3.
        assert outputs[0].shape[0] == 3
