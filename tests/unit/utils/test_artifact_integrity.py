"""Unit tests for the boot-path artifact SHA-256 gate.

Covers ``mousedroid.utils.artifact_integrity`` plus the two manifest helpers it
delegates to in ``mousedroid.utils.weights_manager``. Everything here is pure
filesystem + hashing: no onnxruntime, no torch, no network, so the whole file
runs in the blocking ``test`` job.

The two-level policy is what these tests pin:

* a resolvable digest is *always* enforced, whatever ``require_manifest`` says;
* an unresolvable digest is a warning by default and a hard failure under a
  strict policy.

Getting that backwards in either direction is the regression this file exists
to catch — a gate that only fires under a non-default flag is not a gate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from mousedroid.utils.artifact_integrity import (
    ArtifactContractError,
    ArtifactIntegrityError,
    ArtifactMissingError,
    enforce_artifact_digests,
    require_artifact_suffix,
)
from mousedroid.utils.weights_manager import (
    UNNAMED_MANIFEST_ENTRY,
    expected_digest_for,
    parse_sha256_manifest,
)

_REPO = "ianshank/mousedroid-dual-stream-rssm"
_REVISION = "main"


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class TestParseSha256Manifest:
    def test_parses_sha256sum_lines_keyed_by_basename(self, tmp_path: Path) -> None:
        manifest = tmp_path / "sha256.txt"
        manifest.write_text(
            "aa" * 32 + "  belief.npz\n" + "bb" * 32 + "  bdi/desire.npz\n",
            encoding="utf-8",
        )
        entries = parse_sha256_manifest(manifest)
        assert entries == {"belief.npz": "aa" * 32, "desire.npz": "bb" * 32}

    def test_parses_a_bare_digest_under_the_unnamed_key(self, tmp_path: Path) -> None:
        manifest = tmp_path / "sha256.txt"
        manifest.write_text("CC" * 32 + "\n", encoding="utf-8")
        entries = parse_sha256_manifest(manifest)
        assert entries == {UNNAMED_MANIFEST_ENTRY: "cc" * 32}

    def test_strips_the_binary_mode_star(self, tmp_path: Path) -> None:
        manifest = tmp_path / "sha256.txt"
        manifest.write_text("dd" * 32 + " *observe_step.onnx\n", encoding="utf-8")
        assert parse_sha256_manifest(manifest) == {"observe_step.onnx": "dd" * 32}

    def test_skips_blank_lines_and_keeps_the_first_entry_per_name(self, tmp_path: Path) -> None:
        manifest = tmp_path / "sha256.txt"
        manifest.write_text(
            "\n" + "ee" * 32 + "  a.onnx\n\n" + "ff" * 32 + "  a.onnx\n",
            encoding="utf-8",
        )
        assert parse_sha256_manifest(manifest) == {"a.onnx": "ee" * 32}

    def test_missing_file_returns_empty_without_raising(self, tmp_path: Path) -> None:
        assert parse_sha256_manifest(tmp_path / "absent.txt") == {}

    def test_whitespace_only_file_returns_empty(self, tmp_path: Path) -> None:
        manifest = tmp_path / "sha256.txt"
        manifest.write_text("   \n\n", encoding="utf-8")
        assert parse_sha256_manifest(manifest) == {}


class TestExpectedDigestFor:
    def test_prefers_the_named_entry_over_the_bare_one(self) -> None:
        entries = {"a.onnx": "aa" * 32, UNNAMED_MANIFEST_ENTRY: "bb" * 32}
        assert expected_digest_for(entries, "a.onnx") == "aa" * 32

    def test_falls_back_to_the_bare_entry(self) -> None:
        assert expected_digest_for({UNNAMED_MANIFEST_ENTRY: "bb" * 32}, "a.onnx") == "bb" * 32

    def test_reduces_a_path_to_its_basename(self) -> None:
        assert expected_digest_for({"a.onnx": "aa" * 32}, "weights/a.onnx") == "aa" * 32

    def test_returns_none_when_uncovered(self) -> None:
        assert expected_digest_for({"other.onnx": "aa" * 32}, "a.onnx") is None


class TestRequireArtifactSuffix:
    def test_accepts_the_expected_suffix_case_insensitively(self, tmp_path: Path) -> None:
        require_artifact_suffix(tmp_path / "graph.ONNX", ".onnx", repo_id=_REPO)

    def test_refuses_a_checkpoint_substituted_for_the_graph(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactContractError, match=r"expected a '\.onnx' file"):
            require_artifact_suffix(tmp_path / "final.pt", ".onnx", repo_id=_REPO)

    def test_the_error_names_the_repo_so_the_operator_knows_which_config_line(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ArtifactContractError, match=_REPO):
            require_artifact_suffix(tmp_path / "final.pt", ".onnx", repo_id=_REPO)


class _RecordingMetrics:
    """Minimal stand-in for the mismatch-counter seam."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def inc_model_artifact_sha256_mismatch(self, artifact: str, amount: int = 1) -> None:
        self.calls.append(artifact)


def _gate(
    artifact_paths: list[Path],
    manifest_path: Path,
    *,
    require_manifest: bool = False,
    metrics: Any = None,
) -> dict[str, str]:
    return enforce_artifact_digests(
        artifact_paths,
        manifest_path=manifest_path,
        artifact="world_model_onnx",
        repo_id=_REPO,
        revision=_REVISION,
        require_manifest=require_manifest,
        metrics=metrics,
    )


class TestEnforceArtifactDigests:
    def test_matching_digest_is_accepted_and_returned(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        digest = _write(artifact, b"graph-bytes")
        (tmp_path / "sha256.txt").write_text(digest + "\n", encoding="utf-8")
        assert _gate([artifact], tmp_path / "sha256.txt") == {"observe_step.onnx": digest}

    def test_mismatch_raises_regardless_of_the_permissive_policy(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        (tmp_path / "sha256.txt").write_text("00" * 32 + "\n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError, match="SHA-256 verification failed"):
            _gate([artifact], tmp_path / "sha256.txt", require_manifest=False)

    def test_mismatch_increments_the_counter_once(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        (tmp_path / "sha256.txt").write_text("00" * 32 + "\n", encoding="utf-8")
        metrics = _RecordingMetrics()
        with pytest.raises(ArtifactIntegrityError):
            _gate([artifact], tmp_path / "sha256.txt", metrics=metrics)
        assert metrics.calls == ["world_model_onnx"]

    def test_mismatch_still_refuses_without_a_registry(self, tmp_path: Path) -> None:
        """``metrics=None`` disables the counter, never the refusal."""
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        (tmp_path / "sha256.txt").write_text("00" * 32 + "\n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError):
            _gate([artifact], tmp_path / "sha256.txt", metrics=None)

    def test_absent_manifest_is_permitted_by_default(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        assert _gate([artifact], tmp_path / "sha256.txt") == {}

    def test_absent_manifest_fails_closed_under_a_strict_policy(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        with pytest.raises(ArtifactIntegrityError, match="manifest_missing"):
            _gate([artifact], tmp_path / "sha256.txt", require_manifest=True)

    def test_unparseable_manifest_fails_closed_under_a_strict_policy(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        (tmp_path / "sha256.txt").write_text("\n \n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError, match="manifest_unparseable"):
            _gate([artifact], tmp_path / "sha256.txt", require_manifest=True)

    def test_unparseable_manifest_is_permitted_by_default(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        (tmp_path / "sha256.txt").write_text("\n", encoding="utf-8")
        assert _gate([artifact], tmp_path / "sha256.txt") == {}

    def test_manifest_covering_another_file_fails_closed_under_strict(self, tmp_path: Path) -> None:
        artifact = tmp_path / "observe_step.onnx"
        _write(artifact, b"graph-bytes")
        (tmp_path / "sha256.txt").write_text("11" * 32 + "  other.onnx\n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError, match="digest_absent"):
            _gate([artifact], tmp_path / "sha256.txt", require_manifest=True)

    def test_uncovered_file_is_skipped_under_the_permissive_policy(self, tmp_path: Path) -> None:
        covered = tmp_path / "belief.npz"
        uncovered = tmp_path / "desire.npz"
        digest = _write(covered, b"belief")
        _write(uncovered, b"desire")
        (tmp_path / "sha256.txt").write_text(digest + "  belief.npz\n", encoding="utf-8")
        assert _gate([covered, uncovered], tmp_path / "sha256.txt") == {"belief.npz": digest}

    def test_every_file_in_a_multi_file_set_is_verified(self, tmp_path: Path) -> None:
        names = ("belief.npz", "desire.npz", "intention.npz", "affect.npz")
        lines: list[str] = []
        paths: list[Path] = []
        for name in names:
            path = tmp_path / name
            lines.append(_write(path, name.encode()) + "  " + name)
            paths.append(path)
        (tmp_path / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert set(_gate(paths, tmp_path / "sha256.txt")) == set(names)

    def test_one_bad_file_in_a_set_refuses_the_whole_set(self, tmp_path: Path) -> None:
        good = tmp_path / "belief.npz"
        bad = tmp_path / "desire.npz"
        good_digest = _write(good, b"belief")
        _write(bad, b"desire")
        (tmp_path / "sha256.txt").write_text(
            good_digest + "  belief.npz\n" + "00" * 32 + "  desire.npz\n",
            encoding="utf-8",
        )
        with pytest.raises(ArtifactIntegrityError):
            _gate([good, bad], tmp_path / "sha256.txt")

    def test_a_missing_artifact_file_never_passes_the_gate(self, tmp_path: Path) -> None:
        """A digest recorded for a file that is not there must not read as verified."""
        (tmp_path / "sha256.txt").write_text("22" * 32 + "  observe_step.onnx\n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError):
            _gate([tmp_path / "observe_step.onnx"], tmp_path / "sha256.txt")


class TestNamedExceptionBases:
    """The bases are part of the contract, not incidental."""

    def test_missing_artifact_error_is_a_file_not_found_error(self) -> None:
        assert issubclass(ArtifactMissingError, FileNotFoundError)

    def test_contract_error_is_a_value_error(self) -> None:
        assert issubclass(ArtifactContractError, ValueError)

    def test_integrity_error_is_not_a_file_not_found_error(self) -> None:
        """A caller catching a failed download must not swallow a bad digest."""
        assert not issubclass(ArtifactIntegrityError, FileNotFoundError)
