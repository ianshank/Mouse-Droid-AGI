"""SHA-256-stamped distilled-student slot store (growth pillar).

Persists the compact student produced by :class:`GrowthDistillationCoordinator`
to a SHA-256-stamped file UNDER the configured experience root, mirroring the
Phase-6 on-device :class:`~mousedroid.learning.on_device.slot_store.OnDeviceSlotStore`
contract and reusing the same C1 OTA integrity primitive
(:func:`mousedroid.utils.weights_manager.verify_sha256`, ADR-010).

The growth pillar PRODUCES + PERSISTS a distilled student; it never activates or
hot-swaps it into the live VLA policy — deployment of a distilled student stays a
soak-gated operator decision, exactly like on-device WS4 promotion. The slot
directory is the repo-relative ``GrowthConfig.slot_dir`` leaf resolved under
``ExperienceConfig.path`` — NEVER an absolute host path or CWD.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mousedroid.common.hashing import digest_file_sha256
from mousedroid.logging.setup import get_logger
from mousedroid.utils.weights_manager import verify_sha256

if TYPE_CHECKING:
    from mousedroid.config.schema import ExperienceConfig, GrowthConfig

_log = get_logger(__name__)

_LOG_EVENT_PREFIX: str = "growth_slot"
_SLOT_SUFFIX: str = ".pt"
_TMP_SLOT_NAME: str = f"student{_SLOT_SUFFIX}.tmp"


class GrowthSlotIntegrityError(RuntimeError):
    """Raised when a distilled-student slot fails its SHA-256 check on load."""


@dataclass(frozen=True, slots=True)
class StudentSlot:
    """A persisted, SHA-256-stamped distilled-student weight slot.

    Attributes:
        path: Absolute filesystem path of the stamped ``.pt`` blob.
        digest: Lowercase 64-char hex SHA-256 digest; stamps the filename and is
            re-verified on :meth:`GrowthSlotStore.load`.
    """

    path: Path
    digest: str


class GrowthSlotStore:
    """Persist + load SHA-256-stamped distilled-student weight slots.

    The slot directory is resolved as
    ``<experience_cfg.path>/<growth_cfg.slot_dir>`` so any operator override of
    the experience root is inherited for free and no absolute host path is
    hardcoded.

    Args:
        experience_cfg: Experience-storage config supplying the root path.
        growth_cfg: Growth config supplying the repo-relative ``slot_dir`` leaf.
    """

    def __init__(
        self,
        *,
        experience_cfg: ExperienceConfig,
        growth_cfg: GrowthConfig,
    ) -> None:
        self._slot_dir = (Path(experience_cfg.path) / growth_cfg.slot_dir).resolve()

    @property
    def slot_dir(self) -> Path:
        """Resolved slot directory ``<experience.path>/<slot_dir>``."""
        return self._slot_dir

    def persist(self, student_state_dict: Mapping[str, Tensor]) -> StudentSlot:
        """Write ``student_state_dict`` to a SHA-256-stamped slot file.

        Written to a temp name first, digested (streaming), then renamed to
        ``<digest>.pt`` so the final filename is content-addressed and an
        interrupted write never leaves a mis-stamped slot behind.

        Args:
            student_state_dict: The distilled student parameters to persist.

        Returns:
            A :class:`StudentSlot` with the stamped path and its digest.
        """
        self._slot_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._slot_dir / _TMP_SLOT_NAME
        torch.save(dict(student_state_dict), tmp_path)
        digest = digest_file_sha256(tmp_path)
        final_path = self._slot_dir / f"{digest}{_SLOT_SUFFIX}"
        tmp_path.replace(final_path)
        _log.info(
            "growth_slot_persisted",
            slot_dir=str(self._slot_dir),
            digest=digest,
            n_params=len(student_state_dict),
        )
        return StudentSlot(path=final_path, digest=digest)

    def load(self, slot: StudentSlot) -> dict[str, Tensor]:
        """Verify ``slot``'s digest, then deserialise + return its state-dict.

        Raises:
            GrowthSlotIntegrityError: If the file is missing or its SHA-256
                digest does not match ``slot.digest``.
        """
        if not verify_sha256(slot.path, slot.digest, log_event_prefix=_LOG_EVENT_PREFIX):
            msg = f"distilled-student slot integrity check failed for '{slot.path}'"
            raise GrowthSlotIntegrityError(msg)
        loaded: dict[str, Tensor] = torch.load(slot.path, weights_only=True)
        return loaded


__all__ = [
    "GrowthSlotIntegrityError",
    "GrowthSlotStore",
    "StudentSlot",
]
