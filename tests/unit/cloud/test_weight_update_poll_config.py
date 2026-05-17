"""Backwards-compat tests for the C1.2 world_model_enabled toggle."""

from __future__ import annotations

from mousedroid.config.schema import (
    _WORLD_MODEL_DEFAULT_REPO_ID,
    Settings,
    WeightUpdatePollConfig,
)


def test_world_model_enabled_defaults_false_for_backwards_compat() -> None:
    """New field defaults to False to preserve pre-C1.2 byte-identical behaviour."""
    cfg = WeightUpdatePollConfig()
    assert cfg.world_model_enabled is False


def test_world_model_enabled_can_be_toggled_on() -> None:
    """world_model_enabled can be set True for the second OTA poller."""
    cfg = WeightUpdatePollConfig(
        world_model_enabled=True,
        world_model_repo_id="myteam/custom-rssm",
    )
    assert cfg.world_model_enabled is True


def test_existing_yaml_loads_without_world_model_enabled() -> None:
    """Existing YAML files (pre-C1.2) must still load with the default value."""
    settings = Settings(mock_hardware=True)
    assert settings.cloud.weight_update.world_model_enabled is False


def test_upload_extensions_defaults_include_onnx_and_safetensors() -> None:
    """Cloud-trainer extensions default to a tuple that covers ONNX + HF native."""
    cfg = WeightUpdatePollConfig()
    assert isinstance(cfg.upload_extensions, tuple)
    assert ".onnx" in cfg.upload_extensions
    assert ".safetensors" in cfg.upload_extensions
    assert ".pt" in cfg.upload_extensions


def test_upload_extensions_can_be_overridden() -> None:
    """Operators can override the extension filter via YAML / env / kwarg."""
    cfg = WeightUpdatePollConfig(upload_extensions=(".bin",))
    assert cfg.upload_extensions == (".bin",)


def test_gcs_artifact_prefix_defaults_to_trained_slash() -> None:
    """CLI ``--from-gcs`` resolves the prefix from this schema field."""
    cfg = WeightUpdatePollConfig()
    assert cfg.gcs_artifact_prefix == "trained/"


def test_gcs_artifact_prefix_can_be_overridden() -> None:
    """Operators can override the prefix."""
    cfg = WeightUpdatePollConfig(gcs_artifact_prefix="custom/path/")
    assert cfg.gcs_artifact_prefix == "custom/path/"


def test_world_model_default_repo_constant_matches_field_default() -> None:
    """The module-level constant and the Field default must not drift."""
    cfg = WeightUpdatePollConfig()
    assert cfg.world_model_repo_id == _WORLD_MODEL_DEFAULT_REPO_ID


def test_validator_warns_when_world_model_enabled_with_default_repo(capsys) -> None:
    """Footgun guard: enabling the WM poller with the default repo logs a warning."""
    WeightUpdatePollConfig(world_model_enabled=True)
    captured = capsys.readouterr()
    assert "world_model_poller_using_default_repo" in (captured.out + captured.err)


def test_validator_silent_when_world_model_enabled_with_override(capsys) -> None:
    """No warning when the operator has explicitly overridden the repo."""
    WeightUpdatePollConfig(
        world_model_enabled=True,
        world_model_repo_id="myteam/custom-rssm",
    )
    captured = capsys.readouterr()
    assert "world_model_poller_using_default_repo" not in (captured.out + captured.err)


def test_validator_silent_when_world_model_disabled(capsys) -> None:
    """No warning when the world-model poller is disabled (default)."""
    WeightUpdatePollConfig()
    captured = capsys.readouterr()
    assert "world_model_poller_using_default_repo" not in (captured.out + captured.err)
