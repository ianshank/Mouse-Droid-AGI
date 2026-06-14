"""SHA-256-stamped candidate weight slot store (Phase 6 WS3).

Persists the candidate state-dict produced by the WS2 on-device learner to a
SHA-256-stamped file UNDER the configured experience root, and loads it back
with the same C1 OTA integrity check (``utils.weights_manager.verify_sha256``,
ADR-010). This is the seam where the WS0 de-hardcode is realized: the slot
directory is the repo-relative ``OnDeviceLearningConfig.slot_dir`` leaf
resolved under ``ExperienceConfig.path`` — NEVER an absolute host path or CWD.

Contract:

* :meth:`OnDeviceSlotStore.persist` writes ``torch.save`` of the candidate to
  ``<experience.path>/<slot_dir>/<digest>.pt`` (digest stamps the filename so
  concurrent candidates never collide), creating parents defensively, and
  returns a :class:`CandidateSlot` carrying the path + digest.
* :meth:`OnDeviceSlotStore.load` re-verifies the on-disk digest via the
  REUSED C1 helper before deserialising; a mismatch / missing file raises
  :class:`SlotIntegrityError` so WS4 can map it to the ``integrity_mismatch``
  revert reason.

WS3 PRODUCES + PERSISTS the candidate. It never activates/swaps it into the
live policy — promotion is WS4's safety-gated decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.utils.weights_manager import verify_sha256

if TYPE_CHECKING:
    from mousedroid.config.schema import ExperienceConfig, OnDeviceLearningConfig

_log = get_logger(__name__)

# Chunk size for the SHA-256 stamp digest. Mirrors the C1 OTA helper's
# 64 KiB streaming read so a multi-MB checkpoint never loads fully into RAM
# just to be hashed.
_SHA256_CHUNK_BYTES: int = 64 * 1024
# Structured-log event prefix reused by ``verify_sha256`` on load so a slot
# integrity failure greps under the same ``on_device_slot_*`` family.
_LOG_EVENT_PREFIX: str = "on_device_slot"
# Content-addressed slot file extension (torch.save blob). Named once so the
# final and temp filenames never drift apart.
_SLOT_SUFFIX: str = ".pt"
# Temp filename used during the write-then-rename so an interrupted write never
# leaves a mis-stamped slot behind. Renamed to ``<digest>{_SLOT_SUFFIX}``.
_TMP_SLOT_NAME: str = f"candidate{_SLOT_SUFFIX}.tmp"
# Active-slot pointer manifest. WS4's safety gate writes the blessed candidate's
# digest here on PROMOTE; the live policy (WS5) reads it to know which slot to
# hot-swap. A JSON dict (not a bare string) so future fields (e.g. promoted_at,
# scores) are purely additive. Lives alongside the content-addressed slots.
_ACTIVE_MANIFEST_NAME: str = "active.json"
# JSON key holding the active candidate's SHA-256 digest.
_ACTIVE_DIGEST_KEY: str = "active_digest"
# A valid slot digest is a 64-char lowercase-hex SHA-256 (matches the
# ``hexdigest()`` shape stamped onto every persisted slot filename). Anything
# else in the manifest is a corrupt/tampered pointer and must fail safe.
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class SlotIntegrityError(RuntimeError):
    """Raised when a candidate slot fails its SHA-256 integrity check on load.

    WS4 maps this to the ``integrity_mismatch`` revert reason — a slot that
    does not verify must NEVER be promoted into the live policy.
    """


@dataclass(frozen=True, slots=True)
class CandidateSlot:
    """A persisted, SHA-256-stamped candidate weight slot.

    Attributes:
        path: Absolute filesystem path of the stamped ``.pt`` blob.
        digest: Lowercase 64-char hex SHA-256 digest of the blob. Stamps the
            filename and is re-verified on :meth:`OnDeviceSlotStore.load`.
    """

    path: Path
    digest: str


class OnDeviceSlotStore:
    """Persist + load SHA-256-stamped on-device candidate weight slots.

    The slot directory is resolved as
    ``<experience_cfg.path>/<on_device_cfg.slot_dir>`` so any operator
    override of the experience root is inherited for free and no absolute
    host path is hardcoded (the WS0 de-hardcode).

    Args:
        experience_cfg: Experience-storage config supplying the root path.
        on_device_cfg: On-device-learning config supplying the repo-relative
            ``slot_dir`` leaf.
    """

    def __init__(
        self,
        *,
        experience_cfg: ExperienceConfig,
        on_device_cfg: OnDeviceLearningConfig,
    ) -> None:
        self._slot_dir = (Path(experience_cfg.path) / on_device_cfg.slot_dir).resolve()

    @property
    def slot_dir(self) -> Path:
        """Resolved slot directory ``<experience.path>/<slot_dir>``."""
        return self._slot_dir

    def persist(self, candidate_state_dict: Mapping[str, Tensor]) -> CandidateSlot:
        """Write ``candidate_state_dict`` to a SHA-256-stamped slot file.

        The blob is written first, then digested (streaming, off the in-RAM
        path) so the filename carries the digest. Parent directories are
        created defensively.

        Args:
            candidate_state_dict: The WS2 candidate parameters to persist.

        Returns:
            A :class:`CandidateSlot` with the stamped path and its digest.
        """
        self._slot_dir.mkdir(parents=True, exist_ok=True)

        # Write to a temp name first, digest it, then rename to <digest>.pt so
        # the final filename is content-addressed and an interrupted write
        # never leaves a mis-stamped slot behind.
        tmp_path = self._slot_dir / _TMP_SLOT_NAME
        torch.save(dict(candidate_state_dict), tmp_path)
        digest = self._digest_file(tmp_path)
        final_path = self._slot_dir / f"{digest}{_SLOT_SUFFIX}"
        tmp_path.replace(final_path)

        _log.info(
            "on_device_slot_persisted",
            slot_dir=str(self._slot_dir),
            digest=digest,
            n_params=len(candidate_state_dict),
        )
        return CandidateSlot(path=final_path, digest=digest)

    def load(self, slot: CandidateSlot) -> dict[str, Tensor]:
        """Verify ``slot``'s digest, then deserialise + return its state-dict.

        Args:
            slot: The :class:`CandidateSlot` to load.

        Returns:
            The candidate state-dict mapping parameter names to tensors.

        Raises:
            SlotIntegrityError: If the file is missing or its SHA-256 digest
                does not match ``slot.digest`` (the C1 OTA integrity check,
                reused). WS4 maps this to ``integrity_mismatch``.
        """
        if not verify_sha256(slot.path, slot.digest, log_event_prefix=_LOG_EVENT_PREFIX):
            msg = f"candidate slot integrity check failed for '{slot.path}'"
            raise SlotIntegrityError(msg)
        loaded: dict[str, Tensor] = torch.load(slot.path, weights_only=True)
        return loaded

    def mark_active(self, slot: CandidateSlot) -> None:
        """Mark ``slot`` as the ACTIVE (blessed) candidate.

        WS4's safety-regression gate calls this on a PROMOTE decision. It writes
        an ``active.json`` manifest holding the slot's SHA-256 digest so the
        live policy (WS5) knows which content-addressed slot to hot-swap. The
        actual swap into the running policy is deliberately NOT done here —
        promotion only records the pointer; activation is WS5's job.

        The write is atomic (temp + replace) so a crash never leaves a torn
        manifest, and idempotent (re-pointing simply overwrites).

        Args:
            slot: The :class:`CandidateSlot` to bless as active.
        """
        self._slot_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._slot_dir / _ACTIVE_MANIFEST_NAME
        tmp = self._slot_dir / f"{_ACTIVE_MANIFEST_NAME}.tmp"
        tmp.write_text(json.dumps({_ACTIVE_DIGEST_KEY: slot.digest}), encoding="utf-8")
        tmp.replace(manifest)
        _log.info("on_device_slot_marked_active", slot_dir=str(self._slot_dir), digest=slot.digest)

    def load_active(self) -> str | None:
        """Return the active (blessed) candidate's digest, or ``None``.

        Reads the ``active.json`` manifest written by :meth:`mark_active`.
        Returns ``None`` when no slot has been blessed (no manifest), when the
        manifest is missing/malformed, or when the stored digest is not a
        64-char lowercase-hex SHA-256 — a corrupt/tampered pointer must fail
        safe to "no active slot" rather than hand a bogus content-address to
        the live-policy load path.

        Returns:
            The 64-char hex SHA-256 digest of the active slot, or ``None``.
        """
        manifest = self._slot_dir / _ACTIVE_MANIFEST_NAME
        if not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _log.warning("on_device_slot_active_manifest_corrupt", path=str(manifest))
            return None
        digest = data.get(_ACTIVE_DIGEST_KEY) if isinstance(data, dict) else None
        if not isinstance(digest, str) or not _SHA256_HEX_RE.match(digest):
            _log.warning("on_device_slot_active_digest_malformed", path=str(manifest))
            return None
        return digest

    @staticmethod
    def _digest_file(path: Path) -> str:
        """Stream-hash ``path`` with SHA-256 (mirrors the C1 OTA digest)."""
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(_SHA256_CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()


__all__ = [
    "CandidateSlot",
    "OnDeviceSlotStore",
    "SlotIntegrityError",
]
