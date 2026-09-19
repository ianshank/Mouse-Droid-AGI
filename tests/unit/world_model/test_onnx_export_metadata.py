"""Unit tests for the ONNX export-metadata sidecar (Phase 7, task 7.2).

The point of the sidecar is that a digest alone cannot tell you *what* an
artifact is. These tests therefore pin the two things that make the record
trustworthy:

1. Names come from ``world_model.onnx_io`` — the single source of truth the
   export and the runtime already share — not from a second derivation here.
   ``test_input_names_match_the_onnx_io_accessor`` fails if anyone re-derives
   them, which is the drift ``onnx_io``'s docstring exists to prevent.
2. The builder is a total function of its arguments, so the record is
   reproducible: no filesystem, no subprocess, no ONNX import.

No ``onnxruntime`` and no ``onnx``: this file runs in the blocking ``test``
job, which installs neither.
"""

from __future__ import annotations

import json
from pathlib import Path

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.onnx_export_metadata import (
    EXPORT_METADATA_SCHEMA_VERSION,
    EXPORT_TENSOR_DTYPE,
    EXPORT_TOOL_PACKAGES,
    build_export_metadata,
    collect_tool_versions,
    config_digest,
    input_specs_for_cfg,
    output_specs_for_cfg,
)
from mousedroid.world_model.onnx_io import (
    OBSERVE_STEP_BATCH_DIM_NAME,
    OBSERVE_STEP_OUTPUT_NAMES,
    all_input_names_for_cfg,
)

_OPSET = 17


def _cfg(**overrides: int) -> ModelConfig:
    payload: dict[str, int] = {
        "vision_dim": 16,
        "vision_proj_dim": 8,
        "ultrasonic_dim": 1,
        "ultrasonic_proj_dim": 4,
        "motor_state_dim": 4,
        "motor_proj_dim": 4,
        "hidden_dim": 32,
        "latent_dim": 8,
        "action_dim": 2,
        "obs_dim": 16,
        "cfc_hidden_dim": 16,
    }
    payload.update(overrides)
    return ModelConfig.model_validate(payload)


class TestConfigDigest:
    def test_is_stable_across_calls(self) -> None:
        assert config_digest(_cfg()) == config_digest(_cfg())

    def test_changes_when_a_dimension_changes(self) -> None:
        assert config_digest(_cfg()) != config_digest(_cfg(latent_dim=9))

    def test_is_a_64_char_hex_digest(self) -> None:
        digest = config_digest(_cfg())
        assert len(digest) == 64
        assert all(char in "0123456789abcdef" for char in digest)


class TestInputSpecs:
    def test_input_names_match_the_onnx_io_accessor(self) -> None:
        cfg = _cfg(lidar_dim=8, lidar_proj_dim=4, imu_dim=3, imu_proj_dim=4)
        assert tuple(input_specs_for_cfg(cfg)) == all_input_names_for_cfg(cfg)

    def test_a_disabled_modality_is_absent(self) -> None:
        cfg = _cfg()
        assert "lidar" not in input_specs_for_cfg(cfg)
        assert "audio" not in input_specs_for_cfg(cfg)
        assert "imu" not in input_specs_for_cfg(cfg)

    def test_an_enabled_modality_carries_its_configured_width(self) -> None:
        cfg = _cfg(lidar_dim=12, lidar_proj_dim=4)
        assert input_specs_for_cfg(cfg)["lidar"]["shape"] == [OBSERVE_STEP_BATCH_DIM_NAME, 12]

    def test_the_recurrent_state_width_is_the_combined_stream_width(self) -> None:
        cfg = _cfg(hidden_dim=32, cfc_hidden_dim=16)
        assert input_specs_for_cfg(cfg)["h"]["shape"] == [OBSERVE_STEP_BATCH_DIM_NAME, 48]

    def test_every_input_records_its_dtype(self) -> None:
        specs = input_specs_for_cfg(_cfg())
        assert {spec["dtype"] for spec in specs.values()} == {EXPORT_TENSOR_DTYPE}


class TestOutputSpecs:
    def test_output_names_match_the_onnx_io_tuple(self) -> None:
        assert tuple(output_specs_for_cfg(_cfg())) == OBSERVE_STEP_OUTPUT_NAMES

    def test_surprise_is_recorded_as_a_scalar(self) -> None:
        """``observe_step_traceable`` returns the reduced KL, not a batch vector."""
        assert output_specs_for_cfg(_cfg())["surprise"]["shape"] == []

    def test_batched_outputs_carry_their_configured_widths(self) -> None:
        specs = output_specs_for_cfg(_cfg(hidden_dim=32, cfc_hidden_dim=16, latent_dim=8))
        assert specs["new_h"]["shape"] == [OBSERVE_STEP_BATCH_DIM_NAME, 48]
        assert specs["new_z"]["shape"] == [OBSERVE_STEP_BATCH_DIM_NAME, 8]
        assert specs["obs_embed"]["shape"] == [OBSERVE_STEP_BATCH_DIM_NAME, 16]


class TestCollectToolVersions:
    def test_records_the_interpreter_version(self) -> None:
        assert collect_tool_versions()["python"]

    def test_covers_every_declared_package(self) -> None:
        versions = collect_tool_versions()
        assert set(EXPORT_TOOL_PACKAGES) <= set(versions)

    def test_an_absent_package_is_recorded_rather_than_omitted(self) -> None:
        """ "not recorded" and "not installed" must not look the same."""
        versions = collect_tool_versions(("definitely-not-a-real-distribution",))
        assert versions["definitely-not-a-real-distribution"] == "not_installed"

    def test_reports_versions_without_importing_the_packages(self) -> None:
        """Uses ``importlib.metadata``, so ORT's version is readable without ORT."""
        versions = collect_tool_versions(("numpy",))
        assert versions["numpy"] != "not_installed"


class TestModuleNeutrality:
    """The metadata module must stay importable in the blocking ``test`` job.

    Asserted at the source level rather than via ``sys.modules``: the advisory
    ``onnx-world-model-extras`` job DOES install ORT and other tests in the
    same session import it, so a ``sys.modules`` probe would pass for the wrong
    reason there and flake.
    """

    def test_imports_neither_onnxruntime_nor_onnx_nor_torch(self) -> None:
        import ast

        from mousedroid.world_model import onnx_export_metadata

        source = Path(onnx_export_metadata.__file__).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.split(".")[0])
        assert not imported & {"onnxruntime", "onnx", "torch", "numpy"}


def _metadata(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "cfg": _cfg(),
        "opset": _OPSET,
        "artifact_filename": "observe_step.onnx",
        "artifact_sha256": "ab" * 32,
        "checkpoint_path": "weights/dual_stream_rssm/final.pt",
        "checkpoint_sha256": "cd" * 32,
        "git_sha": "ef" * 20,
        "tool_versions": {"torch": "2.5.1"},
    }
    payload.update(overrides)
    return build_export_metadata(**payload)  # type: ignore[arg-type]


class TestBuildExportMetadata:
    def test_records_the_schema_version(self) -> None:
        assert _metadata()["schema_version"] == EXPORT_METADATA_SCHEMA_VERSION

    def test_records_the_artifact_digest_and_filename(self) -> None:
        assert _metadata()["artifact"] == {
            "filename": "observe_step.onnx",
            "sha256": "ab" * 32,
        }

    def test_records_the_checkpoint_digest(self) -> None:
        checkpoint = _metadata()["checkpoint"]
        assert isinstance(checkpoint, dict)
        assert checkpoint["sha256"] == "cd" * 32

    def test_a_weightless_export_is_visible_not_omitted(self) -> None:
        """ "exported from a random init" is the headline finding, not a gap."""
        record = _metadata(checkpoint_path=None, checkpoint_sha256=None)
        assert record["checkpoint"] == {"path": None, "sha256": None}

    def test_records_the_git_sha_and_opset(self) -> None:
        record = _metadata()
        assert record["git_sha"] == "ef" * 20
        assert record["opset"] == _OPSET

    def test_an_unresolvable_git_sha_is_recorded_as_null(self) -> None:
        assert _metadata(git_sha=None)["git_sha"] is None

    def test_records_the_config_digest(self) -> None:
        config = _metadata()["config"]
        assert isinstance(config, dict)
        assert config["sha256"] == config_digest(_cfg())

    def test_records_the_modality_dims_the_graph_was_built_from(self) -> None:
        cfg = _cfg(lidar_dim=8, lidar_proj_dim=4)
        config = _metadata(cfg=cfg)["config"]
        assert isinstance(config, dict)
        assert config["modality_dims"]["lidar_dim"] == 8

    def test_records_tool_versions(self) -> None:
        assert _metadata()["tool_versions"] == {"torch": "2.5.1"}

    def test_is_json_serialisable(self) -> None:
        json.dumps(_metadata())

    def test_is_a_pure_function_of_its_arguments(self) -> None:
        assert _metadata() == _metadata()

    def test_inputs_and_outputs_come_from_the_io_contract(self) -> None:
        cfg = _cfg(audio_dim=4, audio_proj_dim=4)
        record = _metadata(cfg=cfg)
        assert record["inputs"] == input_specs_for_cfg(cfg)
        assert record["outputs"] == output_specs_for_cfg(cfg)


class TestWriteExportMetadata:
    def test_writes_sorted_newline_terminated_json(self, tmp_path: Path) -> None:
        from mousedroid.world_model.onnx_export_metadata import write_export_metadata

        target = tmp_path / "nested" / "observe_step.metadata.json"
        write_export_metadata(_metadata(), target)
        text = target.read_text(encoding="utf-8")
        assert text.endswith("\n")
        reloaded = json.loads(text)
        assert reloaded["opset"] == _OPSET
        keys = list(reloaded)
        assert keys == sorted(keys)
