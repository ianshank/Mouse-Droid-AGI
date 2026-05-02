"""VLM-derived dense progress reward (Phase 4).

This module provides a VLM-backed reward head that produces a scalar
"progress toward instruction" signal in ``[0, 1]`` for an
``(prev_obs, curr_obs, instruction)`` triple. Calls are cached by a bounded
:class:`cachetools.LRUCache` keyed on a stable hash of the inputs so the
same triple is never scored twice (VLM calls are expensive).

The head is wired into :class:`mousedroid.reward.model.MultiObjectiveRewardModel`
as an additive term that is gated by the Three Laws Law-1 multiplicative
sigmoid — a contrived high progress score cannot override a harm violation.

Architecture invariants:
    * Pure ``cachetools.LRUCache`` (NOT :func:`functools.lru_cache` — needs
      explicit ``maxsize`` to keep memory bounded across long training runs).
    * Backend is a :class:`Protocol` so the real VLM call site is pluggable;
      :class:`MockVLMProgress` returns a configured constant for tests and
      default-off operation.
    * No hardcoded values — every tunable comes from
      :class:`mousedroid.config.schema.VLMProgressConfig`.
    * All inference paths run under :func:`torch.no_grad`.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn as nn
from cachetools import LRUCache
from torch import Tensor

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VLMProgressConfig

_log = get_logger(__name__)


@runtime_checkable
class VLMProgressBackend(Protocol):
    """Pluggable VLM scoring backend.

    Implementations should be pure (no side-effects beyond optional logging)
    and deterministic for identical inputs so the LRU cache stays consistent.
    """

    def score(self, prev_obs: Tensor, curr_obs: Tensor, instruction: str) -> float:
        """Return progress toward ``instruction`` in ``[0, 1]``.

        Args:
            prev_obs: Previous observation tensor (any shape — backend-specific).
            curr_obs: Current observation tensor (same shape as ``prev_obs``).
            instruction: Natural-language goal description.

        Returns:
            Scalar progress estimate in ``[0, 1]``.
        """
        ...


class MockVLMProgress:
    """Constant-value VLM backend used for tests and default-off operation.

    Args:
        value: Constant progress value in ``[0, 1]`` returned for every call.
    """

    def __init__(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"MockVLMProgress value must be in [0, 1], got {value}")
        self._value = value

    def score(self, prev_obs: Tensor, curr_obs: Tensor, instruction: str) -> float:
        """Return the configured constant value."""
        del prev_obs, curr_obs, instruction
        return self._value


def _hash_tensor(tensor: Tensor, decimals: int) -> str:
    """Return a stable content hash for ``tensor`` rounded to ``decimals``.

    Rounding is required because floating-point round-trips through the
    world-model decoder are not bit-stable across devices/optimizations.
    The decimal precision is configurable via
    :attr:`VLMProgressConfig.hash_decimals`.

    Args:
        tensor: Tensor to hash.
        decimals: Number of decimal places to round to before hashing.

    Returns:
        Hex SHA-1 digest (40 chars).
    """
    rounded = torch.round(tensor.detach().cpu() * (10**decimals))  # hardcoded-ok: decimal base
    payload = rounded.to(torch.int64).numpy().tobytes()
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


class VLMProgressHead(nn.Module):
    """Cached VLM progress reward head.

    Wraps a :class:`VLMProgressBackend` with a bounded LRU cache keyed on
    ``(hash(prev_obs), hash(curr_obs), hash(instruction))`` so repeated
    rollout segments do not trigger redundant VLM calls.

    Args:
        cfg: VLM progress configuration.
        backend: Pluggable scoring backend. When ``None``, a
            :class:`MockVLMProgress` backend is built from
            ``cfg.mock_progress_value``.
    """

    def __init__(
        self,
        cfg: VLMProgressConfig,
        backend: VLMProgressBackend | None = None,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._backend: VLMProgressBackend = backend or MockVLMProgress(cfg.mock_progress_value)
        self._cache: LRUCache[tuple[str, str, str], float] = LRUCache(maxsize=cfg.cache_size)
        # Identity-based pre-cache: avoids re-hashing tensor contents on
        # repeat calls with the same physical tensor objects (common in RL
        # rollouts that reuse a buffer). Keyed on
        # ``(data_ptr, _version, data_ptr, _version, instr_hash)`` so any
        # in-place mutation invalidates the entry via the version counter.
        # Bounded to a small constant — purely a hot-path shortcut, the
        # content cache below is the source of truth.
        self._id_cache: LRUCache[tuple[int, int, int, int, str], float] = LRUCache(
            maxsize=cfg.cache_size
        )
        self._instr_hashes: dict[str, str] = {}
        self._hits = 0
        self._misses = 0
        _log.info(
            "vlm_progress_head_init",
            cache_size=cfg.cache_size,
            backend=type(self._backend).__name__,
            instruction=cfg.instruction,
        )

    def _hash_instruction(self, instr: str) -> str:
        """Memoize instruction hashes (typical training reuses one string)."""
        cached = self._instr_hashes.get(instr)
        if cached is not None:
            return cached
        h = hashlib.sha1(instr.encode("utf-8"), usedforsecurity=False).hexdigest()
        self._instr_hashes[instr] = h
        return h

    @property
    def cache_info(self) -> dict[str, int]:
        """Cache statistics (hits, misses, current size, max size)."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            # ``LRUCache.maxsize`` is typed loosely (float in some stub
            # versions); cast to int since we always configure it as one.
            "maxsize": int(self._cache.maxsize),
        }

    def score(
        self,
        prev_obs: Tensor,
        curr_obs: Tensor,
        *,
        instruction: str | None = None,
    ) -> Tensor:
        """Compute progress reward for a transition.

        Supports both single (``(D,)`` or ``(1, D)``) and batched
        (``(B, D)``) inputs. The returned tensor has shape ``(B, 1)`` so it
        broadcasts safely against ``state`` rewards in
        :class:`MultiObjectiveRewardModel`.

        Args:
            prev_obs: Previous observation tensor.
            curr_obs: Current observation tensor (same shape as
                ``prev_obs``).
            instruction: Override for ``cfg.instruction``. ``None`` uses the
                configured default.

        Returns:
            Progress tensor of shape ``(B, 1)`` with values in ``[0, 1]``.
        """
        if prev_obs.shape != curr_obs.shape:
            raise ValueError(
                f"prev_obs/curr_obs shape mismatch: {prev_obs.shape} vs {curr_obs.shape}"
            )

        instr = instruction if instruction is not None else self._cfg.instruction

        # Batched dispatch: score each transition independently and stack.
        # 2 = "shape is at least (B, D)"; 1 = "more than one row → batch path".
        if prev_obs.dim() >= 2 and prev_obs.shape[0] > 1:  # hardcoded-ok
            rows = [
                self._score_single(prev_obs[i : i + 1], curr_obs[i : i + 1], instr)
                for i in range(prev_obs.shape[0])
            ]
            return torch.cat(rows, dim=0)

        return self._score_single(prev_obs, curr_obs, instr)

    def _score_single(self, prev_obs: Tensor, curr_obs: Tensor, instr: str) -> Tensor:
        instr_h = self._hash_instruction(instr)

        # Identity fast-path: same tensor objects, no in-place mutation
        # since last call → bypass content hashing entirely.
        prev_v = int(getattr(prev_obs, "_version", 0))
        curr_v = int(getattr(curr_obs, "_version", 0))
        id_key = (prev_obs.data_ptr(), prev_v, curr_obs.data_ptr(), curr_v, instr_h)
        cached_id = self._id_cache.get(id_key)
        if cached_id is not None:
            self._hits += 1
            return torch.tensor([[cached_id]], dtype=torch.float32, device=curr_obs.device)

        # Content cache (cross-tensor-instance reuse).
        key = (
            _hash_tensor(prev_obs, self._cfg.hash_decimals),
            _hash_tensor(curr_obs, self._cfg.hash_decimals),
            instr_h,
        )

        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            value = cached
        else:
            with torch.no_grad():
                value = float(self._backend.score(prev_obs, curr_obs, instr))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"VLM backend returned out-of-range score {value}; expected [0, 1]"
                )
            self._cache[key] = value
            self._misses += 1
            _log.debug(
                "vlm_progress_cache_miss",
                value=value,
                cache_size=len(self._cache),
                cache_max=self._cache.maxsize,
            )

        self._id_cache[id_key] = value
        return torch.tensor([[value]], dtype=torch.float32, device=curr_obs.device)

    def forward(
        self,
        prev_obs: Tensor,
        curr_obs: Tensor,
        *,
        instruction: str | None = None,
    ) -> Tensor:
        """:class:`nn.Module` alias for :meth:`score`."""
        return self.score(prev_obs, curr_obs, instruction=instruction)
