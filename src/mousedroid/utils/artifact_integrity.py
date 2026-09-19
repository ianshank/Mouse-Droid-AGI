"""Boot-path SHA-256 gate for downloaded model artifacts.

The fail-closed digest machinery already existed on the OTA weight path
(``cloud/weight_update_poller.py`` + ``CloudWeightUpdateConfig.
sha256_manifest_filename`` + ``inc_cloud_weight_update_sha256_mismatch``) but
none of it reached the two seams that run at *boot*: the world-model ``.onnx``
resolved by :mod:`mousedroid.factory.world_model` and the BDI weight set
resolved by :mod:`mousedroid.factory.cognitive`. Both checked only that a file
existed. ADR-008's own "Negative" section states the consequence — "wrong
weights = wrong inference, silently".

This module is the one gate both seams call. It owns the *policy*; the
mechanics stay in :mod:`mousedroid.utils.weights_manager`
(:func:`~mousedroid.utils.weights_manager.verify_sha256`,
:func:`~mousedroid.utils.weights_manager.parse_sha256_manifest`) so there is a
single SHA-256 implementation in the tree.

The policy has two levels, deliberately:

* A digest that **is** resolvable is always enforced. A mismatch increments
  ``mousedroid_model_artifact_sha256_mismatches_total{artifact}`` and raises
  :class:`ArtifactIntegrityError`. No config can switch that off — an artifact
  whose recorded digest disagrees with its bytes is never loaded.
* A digest that is **not** resolvable (no manifest published yet) is governed
  by the caller's ``require_manifest`` flag, which comes from a schema field
  defaulting to ``False``. Default-``False`` keeps today's behaviour for the
  repos that carry no manifest yet, and logs the gap instead of hiding it.
  Operators flip it to ``True`` once artifact and manifest are both published,
  giving the fail-closed shape ``onnx_require_primary_provider`` has.

No ``assert`` anywhere: ``Dockerfile.jetson`` sets ``PYTHONOPTIMIZE=1``, so an
assert-based guard is no guard on the rover.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger
from mousedroid.utils.weights_manager import (
    expected_digest_for,
    parse_sha256_manifest,
    verify_sha256,
)

if TYPE_CHECKING:
    from mousedroid.config.schema._primitives import ModelArtifactLiteral
    from mousedroid.telemetry.metrics.registry import MetricsRegistry

_log = get_logger(__name__)


class ArtifactContractError(ValueError):
    """The requested artifact violates its declared contract.

    Raised when the resolved artifact is not the *kind* of file the caller
    asked for — the case the runtime spec singles out as "a ``.pt`` checkpoint
    SHALL NEVER be substituted for an ``.onnx`` filename". A digest check
    cannot catch this: a checkpoint's digest can be perfectly valid while
    still being the wrong artifact for the consumer.

    A ``ValueError`` subclass because it is a configuration mistake, caught at
    resolve time, not an I/O or integrity failure.
    """


class ArtifactMissingError(FileNotFoundError):
    """A requested artifact could not be fetched from its source repo.

    Subclasses :class:`FileNotFoundError` so every pre-existing caller and
    test that catches the broad error keeps working, while the name makes the
    contract failure explicit: the repo does not carry the filename the config
    asked for. Verified against the Hub, the world-model repo currently holds
    only ``.gitattributes`` and ``README.md``, so this is the live path today.
    """


class ArtifactIntegrityError(RuntimeError):
    """A model artifact failed its SHA-256 gate and MUST NOT be loaded.

    Raised for a digest mismatch (always) and for an unresolvable manifest
    when the caller's policy requires one. Named rather than a bare
    ``RuntimeError`` so a caller can distinguish "the bytes are wrong" from
    "the download failed" (:class:`FileNotFoundError`) without string
    matching, and so an operator reading a traceback sees the category
    immediately.
    """


def require_artifact_suffix(path: Path, expected_suffix: str, *, repo_id: str) -> None:
    """Refuse an artifact whose extension is not ``expected_suffix``.

    Args:
        path: Resolved artifact path.
        expected_suffix: Required extension including the dot, sourced from the
            consumer's own contract module (e.g.
            :data:`mousedroid.world_model.onnx_io.OBSERVE_STEP_ARTIFACT_SUFFIX`)
            rather than spelled at the call site.
        repo_id: Source repo, named in the error so an operator can see which
            config line to fix.

    Raises:
        ArtifactContractError: When the suffix does not match (case-insensitive).
    """
    if path.suffix.lower() != expected_suffix.lower():
        msg = (
            f"refusing artifact '{path.name}' from {repo_id}: expected a "
            f"'{expected_suffix}' file. A checkpoint or archive is never a "
            f"substitute for the exported graph — the consumer would fail at "
            f"session creation on the rover instead of here at resolve time."
        )
        raise ArtifactContractError(msg)


def _report_unverifiable(
    *,
    artifact: ModelArtifactLiteral,
    reason: str,
    require_manifest: bool,
    repo_id: str,
    revision: str,
    detail: str,
) -> None:
    """Apply the absent-digest half of the policy: raise, or warn and allow.

    Args:
        artifact: Artifact kind, used as the structured-log event prefix.
        reason: Machine-readable cause (``"manifest_missing"``,
            ``"manifest_unparseable"``, ``"digest_absent"``).
        require_manifest: Caller's fail-closed switch.
        repo_id: Source repo, for the operator-facing message.
        revision: Pinned revision, for the operator-facing message.
        detail: Human-readable specifics (path or filename).

    Raises:
        ArtifactIntegrityError: When ``require_manifest`` is true.
    """
    if require_manifest:
        msg = (
            f"refusing to load {artifact}: no SHA-256 digest could be resolved "
            f"({reason}; {detail}) for {repo_id}@{revision} and the active "
            f"config requires one. Publish the manifest beside the artifact, "
            f"or clear the require-manifest switch to accept an unverified "
            f"artifact."
        )
        raise ArtifactIntegrityError(msg)
    _log.warning(
        artifact + "_manifest_unavailable",
        reason=reason,
        repo_id=repo_id,
        revision=revision,
        detail=detail,
    )


def enforce_artifact_digests(
    artifact_paths: Sequence[Path],
    *,
    manifest_path: Path,
    artifact: ModelArtifactLiteral,
    repo_id: str,
    revision: str,
    require_manifest: bool,
    metrics: MetricsRegistry | None = None,
) -> dict[str, str]:
    """Verify every artifact in ``artifact_paths`` against ``manifest_path``.

    Args:
        artifact_paths: Local files to verify. Each must already exist —
            callers resolve "did the download work?" before asking "are these
            the right bytes?", so a missing file here is reported as an
            unverifiable artifact rather than silently passing.
        manifest_path: Local path of the SHA-256 manifest (a bare digest, or
            ``sha256sum`` lines naming each file).
        artifact: Which artifact set this is. Doubles as the metric label and
            the structured-log event prefix.
        repo_id: Hugging Face repo the artifacts came from (for messages).
        revision: Revision the fetch was pinned to (for messages).
        require_manifest: Fail closed when no digest can be resolved.
        metrics: Optional registry. When supplied, a mismatch increments
            ``mousedroid_model_artifact_sha256_mismatches_total{artifact}``.
            ``None`` disables only the counter — never the refusal.

    Returns:
        Mapping of verified filename to its confirmed lowercase digest. Empty
        when no digest was resolvable and the policy allowed proceeding.

    Raises:
        ArtifactIntegrityError: On any digest mismatch, or on an unresolvable
            digest when ``require_manifest`` is true.
    """
    if not manifest_path.is_file():
        _report_unverifiable(
            artifact=artifact,
            reason="manifest_missing",
            require_manifest=require_manifest,
            repo_id=repo_id,
            revision=revision,
            detail=str(manifest_path),
        )
        return {}

    entries = parse_sha256_manifest(manifest_path, log_event_prefix=artifact)
    if not entries:
        _report_unverifiable(
            artifact=artifact,
            reason="manifest_unparseable",
            require_manifest=require_manifest,
            repo_id=repo_id,
            revision=revision,
            detail=str(manifest_path),
        )
        return {}

    verified: dict[str, str] = {}
    for path in artifact_paths:
        expected = expected_digest_for(entries, path.name)
        if expected is None:
            _report_unverifiable(
                artifact=artifact,
                reason="digest_absent",
                require_manifest=require_manifest,
                repo_id=repo_id,
                revision=revision,
                detail=path.name,
            )
            continue
        if not verify_sha256(path, expected, log_event_prefix=artifact):
            if metrics is not None:
                metrics.inc_model_artifact_sha256_mismatch(artifact)
            msg = (
                f"refusing to load {artifact}: SHA-256 verification failed for "
                f"'{path}' against '{manifest_path}' "
                f"({repo_id}@{revision}). Delete the cached file and re-fetch, "
                f"or correct the published manifest."
            )
            raise ArtifactIntegrityError(msg)
        verified[path.name] = expected.strip().lower()

    if verified:
        _log.info(
            artifact + "_digest_verified",
            repo_id=repo_id,
            revision=revision,
            file_count=len(verified),
        )
    return verified


__all__ = [
    "ArtifactContractError",
    "ArtifactIntegrityError",
    "ArtifactMissingError",
    "enforce_artifact_digests",
    "require_artifact_suffix",
]
