"""AQA — F-050 observe-step instrumentation seam.

Pins the behaviour this feature adds, so it cannot silently regress:

* ``build_world_model`` accepts a metrics registry and hands it to the engine.
* Both PyTorch engines report into
  ``mousedroid_world_model_observe_step_seconds``, so the family finally has a
  production writer and ``WorldModelObserveStepLatencyHigh`` can evaluate.
* The timing boundary records on success only (telemetry invariant 5).
* No new histogram bucket literal leaks into code — boundaries stay in config.

Torch-only by design: the blocking ``test`` job installs ``.[dev,telemetry,mcp]``
with no onnxruntime, and ``scripts/validations/F-050.sh`` fails on a skip-all
collection, so every assertion here must run without ORT.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from pydantic_core import PydanticUndefined

from mousedroid.config.schema import MetricsConfig, ModelConfig, Settings
from mousedroid.factory.world_model import build_world_model
from mousedroid.telemetry.metrics.registry import MetricsRegistry
from mousedroid.world_model.observe_step_timing import (
    ObserveStepLatencySink,
    observe_step_latency,
)
from mousedroid.world_model.rssm import RSSM

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAMILY = "mousedroid_world_model_observe_step_seconds"


@dataclass
class _Obs:
    """Minimal ``ObservationProtocol`` stub."""

    timestamp: float = 0.0
    distance_m: float = 1.5
    n_modalities: int = 4
    vision_features: NDArray[np.float32] = field(default_factory=lambda: np.zeros(16, np.float32))
    motor_state: NDArray[np.float32] = field(default_factory=lambda: np.zeros(4, np.float32))
    audio_chunk: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    valid_mask: NDArray[np.float32] = field(default_factory=lambda: np.ones(4, np.float32))
    lidar_features: NDArray[np.float32] | None = None


def _cfg() -> ModelConfig:
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
        cfc_hidden_dim=0,
    )


def _count(registry: MetricsRegistry) -> int | None:
    for line in registry.render_prometheus().splitlines():
        if line.startswith(f"{_FAMILY}_count"):
            return int(float(line.split()[-1]))
    return None


def test_build_world_model_exposes_a_keyword_only_metrics_seam() -> None:
    parameter = inspect.signature(build_world_model).parameters["metrics"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_registry_reaches_the_constructed_engine() -> None:
    settings = Settings(mock_hardware=True, model=_cfg())
    registry = MetricsRegistry(settings.metrics)
    model = build_world_model(settings, metrics=registry)
    assert model._metrics is registry  # type: ignore[attr-defined]


def test_torch_engine_populates_the_observe_step_histogram() -> None:
    """The regression this feature fixes: the family had no production writer."""
    cfg = _cfg()
    settings = Settings(mock_hardware=True, model=cfg)
    registry = MetricsRegistry(settings.metrics)
    model = build_world_model(settings, metrics=registry)

    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    action = torch.zeros(1, cfg.action_dim)
    for _ in range(2):
        h, z, _, _ = model.observe_step(_Obs(), action, h, z)

    assert _count(registry) == 2


def test_timing_records_on_success_only() -> None:
    """telemetry/CLAUDE.md invariant 5 — never record on an error path."""
    recorded: list[float] = []

    class _Sink:
        def observe_world_model_observe_step_seconds(self, value: float) -> None:
            recorded.append(value)

    try:
        with observe_step_latency(_Sink(), engine="aqa"):
            msg = "inference failed"
            raise ValueError(msg)
    except ValueError:
        pass
    assert recorded == []


def test_metrics_registry_satisfies_the_latency_sink_protocol() -> None:
    """Engines depend on the Protocol, not the concrete registry."""
    registry = MetricsRegistry(Settings(mock_hardware=True).metrics)
    assert isinstance(registry, ObserveStepLatencySink)


def test_both_torch_engines_accept_the_metrics_keyword() -> None:
    """DualStreamRSSM is checked by signature so ncps stays optional here."""
    from mousedroid.world_model import dual_stream_rssm as dual_stream_module

    for target in (RSSM.__init__, dual_stream_module.DualStreamRSSM.__init__):
        parameter = inspect.signature(target).parameters["metrics"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_orchestrator_factory_builds_the_registry_before_the_world_model() -> None:
    """Ordering is the whole fix — a later registry cannot reach the engine."""
    source = (_REPO_ROOT / "src" / "mousedroid" / "factory" / "orchestrator.py").read_text(
        encoding="utf-8"
    )
    registry_at = source.index("metrics_registry = build_metrics_registry(cfg)")
    world_model_at = source.index("wm = build_world_model(")
    assert registry_at < world_model_at, (
        "build_metrics_registry must precede build_world_model; otherwise the "
        "engine is constructed before a writer exists"
    )
    assert source.count("metrics_registry = build_metrics_registry(cfg)") == 1, (
        "the registry must be built exactly once — two registries would split "
        "the exposition between one that is served and one that is not"
    )


def test_histogram_buckets_stay_in_config_not_code() -> None:
    """CLAUDE.md invariant 2 — no threshold literals in runtime code."""
    assert "world_model_observe_step_seconds_buckets" in MetricsConfig.model_fields
    helper = (
        _REPO_ROOT / "src" / "mousedroid" / "world_model" / "observe_step_timing.py"
    ).read_text(encoding="utf-8")
    assert "0.001" not in helper, "bucket literals belong in MetricsConfig, not the helper"
    assert "0.033" not in helper, "deadline literals belong in config, not the helper"


def test_timing_helper_uses_no_assert_statements() -> None:
    """ruff S101 is blocking in src/; PYTHONOPTIMIZE=1 strips asserts on the rover."""
    helper = (
        _REPO_ROOT / "src" / "mousedroid" / "world_model" / "observe_step_timing.py"
    ).read_text(encoding="utf-8")
    for line in helper.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("assert "), f"assert in src/: {stripped}"


# ---------------------------------------------------------------------------
# Phase 7 — artifact integrity (tasks 7.1, 7.1b, 7.2)
#
# Appended to this file rather than a second f050 regression module so
# scripts/validations/F-050.sh keeps one entry point for the feature. Every
# assertion is torch-only and reads config or parsed source, so the file stays
# runnable without onnxruntime.
# ---------------------------------------------------------------------------

_SRC = _REPO_ROOT / "src" / "mousedroid"

_NEW_WORLD_MODEL_FIELDS = (
    "onnx_revision",
    "onnx_sha256_manifest_filename",
    "onnx_require_sha256_manifest",
    "onnx_metadata_filename",
)
_NEW_COGNITIVE_FIELDS = (
    "huggingface_revision",
    "sha256_manifest_filename",
    "require_sha256_manifest",
)


def _keyword_names_of_calls_to(source: str, func_name: str) -> list[set[str]]:
    """Return one set of keyword names per call to ``func_name`` in ``source``."""
    calls: list[set[str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == func_name:
            calls.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return calls


class TestIntegrityFieldsAreSchemaDriven:
    """CLAUDE.md invariants 2 and 6 for every field Phase 7 adds."""

    def test_world_model_fields_exist_with_defaults_and_descriptions(self) -> None:
        from mousedroid.config.schema import WorldModelConfig

        for name in _NEW_WORLD_MODEL_FIELDS:
            info = WorldModelConfig.model_fields[name]
            assert info.default is not PydanticUndefined, f"{name} must carry a default"
            assert info.description, f"{name} must carry a description"

    def test_cognitive_fields_exist_with_defaults_and_descriptions(self) -> None:
        from mousedroid.config.schema import CognitiveConfig

        for name in _NEW_COGNITIVE_FIELDS:
            info = CognitiveConfig.model_fields[name]
            assert info.default is not PydanticUndefined, f"{name} must carry a default"
            assert info.description, f"{name} must carry a description"

    def test_the_strict_switches_default_off_so_todays_behaviour_is_preserved(self) -> None:
        """The published repo carries no manifest yet; boot must not break."""
        from mousedroid.config.schema import CognitiveConfig, WorldModelConfig

        assert WorldModelConfig.model_validate({}).onnx_require_sha256_manifest is False
        assert CognitiveConfig.model_validate({}).require_sha256_manifest is False

    def test_the_revision_is_a_config_field_not_a_branch_name_in_code(self) -> None:
        from mousedroid.config.schema import CognitiveConfig, WorldModelConfig

        assert WorldModelConfig.model_validate({}).onnx_revision
        assert CognitiveConfig.model_validate({}).huggingface_revision
        for relative in ("factory/world_model.py", "factory/cognitive.py"):
            source = (_SRC / relative).read_text(encoding="utf-8")
            assert 'revision="' not in source, f"{relative} hardcodes a revision"

    def test_every_hub_fetch_on_both_boot_paths_pins_a_revision(self) -> None:
        for relative in ("factory/world_model.py", "factory/cognitive.py"):
            source = (_SRC / relative).read_text(encoding="utf-8")
            calls = _keyword_names_of_calls_to(source, "download_weights_from_huggingface")
            assert calls, f"{relative}: expected at least one Hugging Face fetch"
            for keywords in calls:
                assert "revision" in keywords, f"{relative}: unpinned Hugging Face fetch"


class TestOneSha256Implementation:
    """Task 7.1b reuses ``verify_sha256``; a second hasher would be the defect."""

    def test_the_gate_delegates_to_the_existing_helper(self) -> None:
        source = (_SRC / "utils" / "artifact_integrity.py").read_text(encoding="utf-8")
        assert "verify_sha256" in source

    def test_the_gate_adds_no_second_hasher(self) -> None:
        source = (_SRC / "utils" / "artifact_integrity.py").read_text(encoding="utf-8")
        assert "hashlib" not in source, "hash through verify_sha256, not a new hasher"

    def test_the_new_modules_use_no_assert_statements(self) -> None:
        """ruff S101 is blocking in src/; PYTHONOPTIMIZE=1 strips asserts."""
        for relative in ("utils/artifact_integrity.py", "world_model/onnx_export_metadata.py"):
            source = (_SRC / relative).read_text(encoding="utf-8")
            asserts = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assert)]
            assert not asserts, f"assert statement in src/{relative}"


class TestArtifactContract:
    """A digest can be valid for the wrong kind of file."""

    def test_a_checkpoint_is_never_substituted_for_the_graph(self, tmp_path: Path) -> None:
        from mousedroid.utils.artifact_integrity import (
            ArtifactContractError,
            require_artifact_suffix,
        )
        from mousedroid.world_model.onnx_io import OBSERVE_STEP_ARTIFACT_SUFFIX

        with pytest.raises(ArtifactContractError):
            require_artifact_suffix(
                tmp_path / "final.pt",
                OBSERVE_STEP_ARTIFACT_SUFFIX,
                repo_id="ianshank/mousedroid-dual-stream-rssm",
            )

    def test_a_missing_artifact_is_a_named_failure(self) -> None:
        from mousedroid.utils.artifact_integrity import ArtifactMissingError

        assert issubclass(ArtifactMissingError, FileNotFoundError)


class TestMismatchCounterHygiene:
    """Telemetry invariants 4 and 6 for the new counter."""

    def test_the_registry_field_name_omits_the_render_suffix(self) -> None:
        registry = MetricsRegistry(MetricsConfig.model_validate({}))
        assert not registry._name_model_artifact_sha256_mismatches.endswith("_total")

    def test_the_runtime_guard_mirrors_the_compile_time_literal(self) -> None:
        from typing import get_args

        from mousedroid.config.schema._primitives import ModelArtifactLiteral
        from mousedroid.telemetry.metrics.primitives import _MODEL_ARTIFACT_KINDS

        assert set(get_args(ModelArtifactLiteral)) == set(_MODEL_ARTIFACT_KINDS)

    def test_the_family_is_seeded_in_the_promtool_sample(self) -> None:
        from mousedroid.telemetry.metrics import generate_metrics_sample

        sample = generate_metrics_sample()
        for kind in ("world_model_onnx", "bdi_weights"):
            assert f'artifact="{kind}"' in sample


class TestExportMetadataGoesThroughOnnxIo:
    """Task 7.2 — the IO contract has one owner, not three derivations."""

    def test_input_names_come_from_the_accessor(self) -> None:
        from mousedroid.world_model.onnx_export_metadata import input_specs_for_cfg
        from mousedroid.world_model.onnx_io import all_input_names_for_cfg

        cfg = _cfg()
        assert tuple(input_specs_for_cfg(cfg)) == all_input_names_for_cfg(cfg)

    def test_output_names_come_from_the_shared_tuple(self) -> None:
        from mousedroid.world_model.onnx_export_metadata import output_specs_for_cfg
        from mousedroid.world_model.onnx_io import OBSERVE_STEP_OUTPUT_NAMES

        assert tuple(output_specs_for_cfg(_cfg())) == OBSERVE_STEP_OUTPUT_NAMES

    def test_the_metadata_module_imports_no_onnx_runtime(self) -> None:
        """Buildable in the blocking ``test`` job, which installs no ORT."""
        source = (_SRC / "world_model" / "onnx_export_metadata.py").read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.split(".")[0])
        assert not imported & {"onnxruntime", "onnx", "torch"}

    def test_the_builder_is_callable_without_touching_the_filesystem(self) -> None:
        from mousedroid.world_model.onnx_export_metadata import build_export_metadata

        record = build_export_metadata(
            cfg=_cfg(),
            opset=17,
            artifact_filename="observe_step.onnx",
            artifact_sha256="ab" * 32,
            checkpoint_path=None,
            checkpoint_sha256=None,
            git_sha=None,
            tool_versions={},
        )
        assert record["checkpoint"] == {"path": None, "sha256": None}


class TestUnsafeLoadersAreFenced:
    """Task 7.1b — neither loader may reach arbitrary pickle by omission."""

    def test_the_migration_loader_defaults_to_the_safe_mode(self) -> None:
        from mousedroid.world_model.checkpoint_migration import load_rssm_with_migration

        signature = inspect.signature(load_rssm_with_migration)
        assert signature.parameters["allow_unsafe_pickle"].default is False
        assert signature.parameters["expected_sha256"].default is None

    def test_no_loader_hardcodes_the_unsafe_mode(self) -> None:
        """Parsed, not grepped, so the docstring explaining the fence cannot pass it."""
        for relative in ("world_model/checkpoint_migration.py",):
            source = (_SRC / relative).read_text(encoding="utf-8")
            unsafe = [
                kw.value
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                for kw in node.keywords
                if kw.arg == "weights_only"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
            ]
            assert not unsafe, f"{relative} hardcodes weights_only=False"

    def test_the_export_script_has_no_silent_legacy_fallback(self) -> None:
        source = (_REPO_ROOT / "scripts" / "export_dual_stream_rssm_onnx.py").read_text(
            encoding="utf-8"
        )
        loader = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint"
        )
        assert not [n for n in ast.walk(loader) if isinstance(n, ast.Try)]
        values = [
            kw.value
            for node in ast.walk(loader)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "weights_only"
        ]
        assert values
        assert all(isinstance(v, ast.Constant) and v.value is True for v in values)
