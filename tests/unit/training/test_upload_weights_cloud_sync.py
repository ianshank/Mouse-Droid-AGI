"""C1.1: GCS->HF Hub closed-loop weight publication leg."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from training.upload_weights import main, sync_gcs_to_hf


def test_sync_gcs_to_hf_downloads_then_uploads(tmp_path: Path) -> None:
    """sync_gcs_to_hf downloads all blobs under prefix then calls upload_weights."""
    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    fake_blob = MagicMock()
    fake_blob.name = "trained/policy.onnx"
    fake_blob.download_to_filename = MagicMock(
        side_effect=lambda dest: Path(dest).write_bytes(b"weights")
    )
    fake_bucket.list_blobs.return_value = [fake_blob]
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True) as upload:
        ok = sync_gcs_to_hf(
            gcs_bucket="mousedroid-weights",
            gcs_prefix="trained/",
            repo_id="ianshank/mousedroid-policy-v2",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
        )

    assert ok is True
    upload.assert_called_once()
    assert (tmp_path / "policy.onnx").read_bytes() == b"weights"


def test_sync_gcs_to_hf_returns_false_when_bucket_empty(tmp_path: Path) -> None:
    """Empty prefix short-circuits — returns False and logs a warning."""
    fake_gcs_client = MagicMock()
    fake_gcs_client.bucket.return_value.list_blobs.return_value = []
    ok = sync_gcs_to_hf(
        gcs_bucket="mousedroid-weights",
        gcs_prefix="trained/",
        repo_id="ianshank/mousedroid-policy-v2",
        local_dir=tmp_path,
        gcs_client=fake_gcs_client,
    )
    assert ok is False


def test_sync_gcs_to_hf_skips_prefix_itself_blob(tmp_path: Path) -> None:
    """list_blobs may surface the prefix itself as a zero-byte blob — skip it."""
    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    prefix_blob = MagicMock()
    prefix_blob.name = "trained/"  # the prefix directory itself
    real_blob = MagicMock()
    real_blob.name = "trained/policy.onnx"
    real_blob.download_to_filename = MagicMock(
        side_effect=lambda dest: Path(dest).write_bytes(b"weights")
    )
    fake_bucket.list_blobs.return_value = [prefix_blob, real_blob]
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True):
        ok = sync_gcs_to_hf(
            gcs_bucket="b",
            gcs_prefix="trained/",
            repo_id="ianshank/x",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
        )

    assert ok is True
    prefix_blob.download_to_filename.assert_not_called()
    real_blob.download_to_filename.assert_called_once()
    assert (tmp_path / "policy.onnx").read_bytes() == b"weights"


def test_sync_gcs_to_hf_constructs_default_client_when_none(tmp_path: Path) -> None:
    """``gcs_client=None`` triggers a lazy ``google.cloud.storage.Client()`` build.

    Production callers never inject a client; this path needs explicit
    coverage so a regression in the lazy import wouldn't silently break
    the cloud trainer at boot.
    """
    fake_storage_module = ModuleType("google.cloud.storage")
    fake_client_cls = MagicMock()
    fake_client = MagicMock()
    fake_client.bucket.return_value.list_blobs.return_value = []
    fake_client_cls.return_value = fake_client
    fake_storage_module.Client = fake_client_cls  # type: ignore[attr-defined]

    fake_google = ModuleType("google")
    fake_google_cloud = ModuleType("google.cloud")
    fake_google_cloud.storage = fake_storage_module  # type: ignore[attr-defined]
    fake_google.cloud = fake_google_cloud  # type: ignore[attr-defined]

    with patch.dict(
        sys.modules,
        {
            "google": fake_google,
            "google.cloud": fake_google_cloud,
            "google.cloud.storage": fake_storage_module,
        },
    ):
        ok = sync_gcs_to_hf(
            gcs_bucket="b",
            gcs_prefix="trained/",
            repo_id="ianshank/x",
            local_dir=tmp_path,
            gcs_client=None,
        )

    assert ok is False  # empty list_blobs short-circuits
    fake_client_cls.assert_called_once_with()
    fake_client.bucket.assert_called_once_with("b")


def test_sync_gcs_to_hf_skips_blob_with_empty_filename(tmp_path: Path) -> None:
    """Pathological blob name ``trained//`` is skipped, sync returns False.

    The trailing-slash guard catches the bogus blob, and the post-iteration
    ``downloaded_count == 0`` check short-circuits before ``upload_weights``
    is called — there's nothing to upload, so calling ``upload_weights``
    on an empty staging dir would be pointless cost + a misleading
    ``no_weight_files_found`` warning.
    """
    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    bogus_blob = MagicMock()
    bogus_blob.name = "trained//"  # pathological — endswith("/") catches this
    fake_bucket.list_blobs.return_value = [bogus_blob]
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True) as upload:
        ok = sync_gcs_to_hf(
            gcs_bucket="b",
            gcs_prefix="trained/",
            repo_id="ianshank/x",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
        )

    bogus_blob.download_to_filename.assert_not_called()
    # No downloads → sync returns False AND ``upload_weights`` is NOT
    # called (nothing to upload).
    assert ok is False
    upload.assert_not_called()


def test_sync_gcs_to_hf_preserves_nested_directory_structure(tmp_path: Path) -> None:
    """Nested blobs land in matching local subdirs (regression for dir flattening).

    ``trained/policy/model.pt`` and ``trained/world_model/model.pt`` previously
    both resolved to ``local_dir/model.pt`` via ``Path(blob.name).name``,
    silently overwriting each other before ``upload_weights`` ran. The fix
    computes a prefix-relative path and mkdirs parents so both files
    survive the round-trip.
    """
    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    policy_blob = MagicMock()
    policy_blob.name = "trained/policy/model.pt"
    policy_blob.download_to_filename = MagicMock(
        side_effect=lambda d: Path(d).write_bytes(b"policy-bytes")
    )
    world_blob = MagicMock()
    world_blob.name = "trained/world_model/model.pt"
    world_blob.download_to_filename = MagicMock(
        side_effect=lambda d: Path(d).write_bytes(b"world-bytes")
    )
    fake_bucket.list_blobs.return_value = [policy_blob, world_blob]
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True):
        ok = sync_gcs_to_hf(
            gcs_bucket="b",
            gcs_prefix="trained/",
            repo_id="ianshank/x",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
        )

    assert ok is True
    # Both files exist under their original subpath — no overwrite.
    assert (tmp_path / "policy" / "model.pt").read_bytes() == b"policy-bytes"
    assert (tmp_path / "world_model" / "model.pt").read_bytes() == b"world-bytes"


def test_sync_gcs_to_hf_tolerates_prefix_without_trailing_slash(tmp_path: Path) -> None:
    """``gcs_prefix='trained'`` (no slash) still strips the prefix from blob names.

    Operators may wire ``cfg.cloud.weight_update.gcs_artifact_prefix`` with or
    without a trailing slash; the prefix-relative path computation must be
    tolerant of both forms.
    """
    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    blob = MagicMock()
    blob.name = "trained/policy/model.pt"
    blob.download_to_filename = MagicMock(side_effect=lambda d: Path(d).write_bytes(b"x"))
    fake_bucket.list_blobs.return_value = [blob]
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True):
        ok = sync_gcs_to_hf(
            gcs_bucket="b",
            gcs_prefix="trained",  # no trailing slash
            repo_id="ianshank/x",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
        )

    assert ok is True
    assert (tmp_path / "policy" / "model.pt").read_bytes() == b"x"


def test_sync_gcs_to_hf_streams_listing_lazily(tmp_path: Path) -> None:
    """``bucket.list_blobs`` result is iterated, never wrapped in ``list()``.

    Regression net for the previous eager ``list(...)`` call that
    materialised the entire blob listing before downloading anything.
    Pins streaming behaviour by using a generator that raises if exhausted
    more than once (which would indicate an eager materialisation).
    """
    blob = MagicMock()
    blob.name = "trained/policy.onnx"
    blob.download_to_filename = MagicMock(side_effect=lambda d: Path(d).write_bytes(b"w"))

    iteration_count = 0

    def _one_shot_iter() -> object:
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count > 1:
            raise AssertionError("list_blobs iterated more than once — caller wrapped in list()")
        yield blob

    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    fake_bucket.list_blobs.return_value = _one_shot_iter()
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True):
        ok = sync_gcs_to_hf(
            gcs_bucket="b",
            gcs_prefix="trained/",
            repo_id="ianshank/x",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
        )

    assert ok is True
    assert iteration_count == 1


def test_sync_gcs_to_hf_forwards_custom_upload_extensions(tmp_path: Path) -> None:
    """Caller-supplied ``upload_extensions`` is forwarded to ``upload_weights``."""
    fake_gcs_client = MagicMock()
    fake_bucket = MagicMock()
    blob = MagicMock()
    blob.name = "trained/weights.safetensors"
    blob.download_to_filename = MagicMock(side_effect=lambda d: Path(d).write_bytes(b"w"))
    fake_bucket.list_blobs.return_value = [blob]
    fake_gcs_client.bucket.return_value = fake_bucket

    with patch("training.upload_weights.upload_weights", return_value=True) as upload:
        sync_gcs_to_hf(
            gcs_bucket="b",
            gcs_prefix="trained/",
            repo_id="ianshank/x",
            local_dir=tmp_path,
            gcs_client=fake_gcs_client,
            upload_extensions=(".safetensors",),
        )

    upload.assert_called_once()
    _args, kwargs = upload.call_args
    assert kwargs["extensions"] == {".safetensors"}


# ---------------------------------------------------------------------------
# CLI main() coverage — schema-driven defaults, parser.error guards
# ---------------------------------------------------------------------------


def test_main_legacy_mode_uses_default_repo(tmp_path: Path) -> None:
    """No ``--from-gcs`` flag → calls ``upload_weights`` with the legacy default repo."""
    with patch("training.upload_weights.upload_weights", return_value=True) as upload:
        rc = main(
            [
                "--weights-dir",
                str(tmp_path),
            ]
        )
    assert rc == 0
    _args, kwargs = upload.call_args
    assert kwargs["repo_id"] == "ianshank/mousedroid-weights"


def test_main_legacy_mode_returns_1_on_failure(tmp_path: Path) -> None:
    """Failed ``upload_weights`` → CLI exit code 1."""
    with patch("training.upload_weights.upload_weights", return_value=False):
        rc = main(["--weights-dir", str(tmp_path)])
    assert rc == 1


def test_main_from_gcs_resolves_settings_defaults(tmp_path: Path) -> None:
    """``--from-gcs`` with no overrides resolves bucket/prefix/repo from Settings."""
    fake_settings = MagicMock()
    fake_settings.gcp = MagicMock()
    fake_settings.gcp.training = MagicMock()
    fake_settings.gcp.training.training_bucket = "test-bucket"
    fake_settings.cloud.weight_update.policy_repo_id = "test/policy"
    fake_settings.cloud.weight_update.gcs_artifact_prefix = "trained/"
    fake_settings.cloud.weight_update.upload_extensions = (".onnx", ".pt")

    with (
        patch(
            "mousedroid.config.loader.load_settings",
            return_value=fake_settings,
        ),
        patch(
            "training.upload_weights.sync_gcs_to_hf",
            return_value=True,
        ) as sync,
    ):
        rc = main(
            [
                "--weights-dir",
                str(tmp_path),
                "--from-gcs",
            ]
        )

    assert rc == 0
    _args, kwargs = sync.call_args
    assert kwargs["gcs_bucket"] == "test-bucket"
    assert kwargs["gcs_prefix"] == "trained/"
    assert kwargs["repo_id"] == "test/policy"
    assert kwargs["upload_extensions"] == (".onnx", ".pt")


def test_main_from_gcs_explicit_args_bypass_settings(tmp_path: Path) -> None:
    """All overrides on the CLI → ``load_settings`` is NOT called."""
    with (
        patch("mousedroid.config.loader.load_settings") as load_settings,
        patch("training.upload_weights.sync_gcs_to_hf", return_value=True) as sync,
    ):
        rc = main(
            [
                "--weights-dir",
                str(tmp_path),
                "--from-gcs",
                "--gcs-bucket",
                "cli-bucket",
                "--gcs-prefix",
                "cli-prefix/",
                "--repo",
                "cli/repo",
            ]
        )

    assert rc == 0
    load_settings.assert_not_called()
    _args, kwargs = sync.call_args
    assert kwargs["gcs_bucket"] == "cli-bucket"
    assert kwargs["gcs_prefix"] == "cli-prefix/"
    assert kwargs["repo_id"] == "cli/repo"
    # No Settings → upload_extensions defaults to None (sync_gcs_to_hf resolves internally).
    assert kwargs["upload_extensions"] is None


def test_main_from_gcs_errors_when_gcp_training_disabled(tmp_path: Path) -> None:
    """``--from-gcs`` with no override + no ``cfg.gcp.training`` → ``parser.error`` exits."""
    fake_settings = MagicMock()
    fake_settings.gcp = None  # GCP disabled entirely
    fake_settings.cloud.weight_update.policy_repo_id = "test/policy"
    fake_settings.cloud.weight_update.gcs_artifact_prefix = "trained/"
    fake_settings.cloud.weight_update.upload_extensions = (".onnx",)

    with (
        patch(
            "mousedroid.config.loader.load_settings",
            return_value=fake_settings,
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        main(["--weights-dir", str(tmp_path), "--from-gcs"])

    # argparse's parser.error() exits with code 2.
    assert excinfo.value.code == 2


def test_main_from_gcs_returns_1_when_sync_fails(tmp_path: Path) -> None:
    """``sync_gcs_to_hf`` returns False → CLI exit code 1."""
    with (
        patch("training.upload_weights.sync_gcs_to_hf", return_value=False),
    ):
        rc = main(
            [
                "--weights-dir",
                str(tmp_path),
                "--from-gcs",
                "--gcs-bucket",
                "b",
                "--gcs-prefix",
                "p/",
                "--repo",
                "r/x",
            ]
        )
    assert rc == 1


def test_create_model_card_includes_training_metadata(tmp_path: Path) -> None:
    """``_create_model_card`` appends JSON metadata when ``mcts/tuned_config.json`` exists.

    Covers the previously-untested metadata branch — important because the
    cloud trainer emits this file via the MCTS tuning sweep and operators
    rely on it landing in the HF model card for provenance.
    """
    import json

    from training.upload_weights import _create_model_card

    (tmp_path / "mcts").mkdir()
    (tmp_path / "mcts" / "tuned_config.json").write_text(
        json.dumps({"exploration_constant": 1.41, "rollout_depth": 32}),
        encoding="utf-8",
    )

    card = _create_model_card(tmp_path, "ianshank/test-repo")
    assert "## Training Metadata" in card
    assert "exploration_constant" in card
    assert "1.41" in card
