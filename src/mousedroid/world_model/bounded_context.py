"""Bounded-context latent memory — sink anchor + compressed rolling history.

Adapts the AlayaWorld sink-frame + compressed-history pattern (F-023, ADR-015)
to the rover's recurrent latent state. Three constant-size stores over the
concatenated ``hz = cat(h, z)`` vector:

- **sink** — one frozen anchor captured after ``sink_warmup_ticks`` validated
  ticks; re-armed by :meth:`reset` (OTA weight swap) and :meth:`rearm_sink`
  (mission boundary), making it a per-mission anchor rather than a stale boot
  snapshot.
- **ring** — ``deque(maxlen=recent_size)`` of detached recent states.
- **long summary** — a single EMA vector folded every ``stride`` observes.

Total footprint is ``recent_size + 2`` vectors — constant with respect to
rollout length. Retrieval is scaled dot-product softmax attention (the
``WorkingMemory.attend`` math) over the populated stores, blended as
``h' = (1 - λ)·h + λ·c_h``.

Cold-start contract: an uncaptured sink and a never-folded EMA are EXCLUDED
from the key set, and an empty key set makes :meth:`contextualize` the exact
identity. This deliberately diverges from ``WorkingMemory.attend``'s
zero-vector return for an empty buffer — mirroring that here would blend
toward zero and damp ``h`` by ``(1 - λ)`` on every warmup tick.

NaN contract: :meth:`observe` drops non-finite inputs with a warning, and
:meth:`contextualize` isfinite-checks the blended output and falls back to the
identity — a transiently-NaN tick can never poison the EMA/sink or the carried
state (the orchestrator additionally skips both calls on unhealthy ticks).
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from mousedroid.config.schema import WorldModelMemoryConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class BoundedContextMemory:
    """Constant-size latent context: sink + recent ring + EMA long-summary.

    Args:
        cfg: The ``world_model_memory`` config block.
        h_dim: Dimension of the carried hidden state ``h`` (the combined
            ``hidden_dim + cfc_hidden_dim`` for the dual-stream engine).
        z_dim: Dimension of the stochastic latent ``z``.
    """

    def __init__(self, cfg: WorldModelMemoryConfig, h_dim: int, z_dim: int) -> None:
        if h_dim <= 0 or z_dim <= 0:
            msg = f"h_dim and z_dim must be positive; got h_dim={h_dim}, z_dim={z_dim}"
            raise ValueError(msg)
        self._cfg = cfg
        self._h_dim = h_dim
        self._z_dim = z_dim
        self._dim = h_dim + z_dim
        # Ring capacity comes from config (invariant #8: deque(maxlen=N)).
        self._ring: deque[Tensor] = deque(maxlen=cfg.recent_size)
        self._sink: Tensor | None = None
        self._long: Tensor | None = None
        # Validated observes since the last sink (re-)arm — drives sink capture.
        self._ticks_since_arm = 0
        # Total validated observes — drives the EMA fold + debug-event cadence.
        self._observe_count = 0
        self._ctx_count = 0
        _log.info(
            "bounded_context_init",
            h_dim=h_dim,
            z_dim=z_dim,
            recent_size=cfg.recent_size,
            stride=cfg.stride,
            blend_weight=cfg.blend_weight,
            sink_warmup_ticks=cfg.sink_warmup_ticks,
        )

    @torch.no_grad()
    def observe(self, h: Tensor, z: Tensor) -> None:
        """Store a validated latent state.

        The caller passes the RAW post-validation state (pre-blend), so every
        stored entry is posterior-corrected — re-anchored to a real
        observation. Non-finite inputs are dropped defensively (the
        orchestrator already skips unhealthy ticks; this guards direct
        callers such as evaluation harnesses).

        Args:
            h: Hidden state, shape ``(1, h_dim)``.
            z: Stochastic latent, shape ``(1, z_dim)``.
        """
        hz = torch.cat([h, z], dim=-1).detach()
        if not bool(torch.isfinite(hz).all()):
            _log.warning("bounded_context_nonfinite_dropped", observe_count=self._observe_count)
            return
        flat = hz.reshape(-1).clone()
        if self._sink is None and self._ticks_since_arm >= self._cfg.sink_warmup_ticks:
            self._sink = flat.clone()
            _log.info(
                "bounded_context_sink_captured",
                ticks_since_arm=self._ticks_since_arm,
                observe_count=self._observe_count,
            )
        self._ticks_since_arm += 1
        self._ring.append(flat)
        self._observe_count += 1
        if self._observe_count % self._cfg.stride == 0:
            if self._long is None:
                # First fold seeds the EMA with the observed state — a
                # zero-initialised summary would drag attention toward zero.
                self._long = flat.clone()
            else:
                alpha = self._cfg.long_ema_alpha
                self._long = (1.0 - alpha) * self._long + alpha * flat

    @torch.no_grad()
    def contextualize(self, h: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """Blend attention-retrieved context into ``(h, z)``.

        Exact identity when ``blend_weight == 0`` or no store is populated.
        The blended output is isfinite-guarded with fallback to the identity.

        Device-agnostic: stored keys follow the device they were observed on
        and are harmonised to the QUERY's device here (``Tensor.to`` is a
        no-op reference return when devices already match, so the steady-state
        hot path pays nothing). This keeps the blend working across a
        CPU↔CUDA engine swap where stale entries could otherwise
        cross-device-crash the matmul.

        Args:
            h: Hidden state, shape ``(1, h_dim)``.
            z: Stochastic latent, shape ``(1, z_dim)``.

        Returns:
            ``(h', z')`` with the same shapes as the inputs.
        """
        if self._cfg.blend_weight == 0.0:
            return h, z
        keys_list: list[Tensor] = []
        if self._sink is not None:
            keys_list.append(self._sink)
        if self._long is not None:
            keys_list.append(self._long)
        keys_list.extend(self._ring)
        if not keys_list:
            return h, z
        hz = torch.cat([h, z], dim=-1).reshape(-1)
        keys = torch.stack([key.to(hz.device) for key in keys_list])  # (n, dim)
        scale = float(self._dim) ** 0.5
        weights = torch.softmax((keys @ hz) / scale, dim=0)  # (n,)
        context = weights @ keys  # (dim,)
        lam = self._cfg.blend_weight
        blended = (1.0 - lam) * hz + lam * context
        if not bool(torch.isfinite(blended).all()):
            _log.warning("bounded_context_blend_nonfinite_identity", ctx_count=self._ctx_count)
            return h, z
        self._ctx_count += 1
        if self._ctx_count % self._cfg.stride == 0:
            # Rate-limited observability for runaway self-reinforcement: how far
            # the retrieved context sits from the current state, and how aligned.
            delta = float(torch.linalg.norm(context - hz))
            cos = float(
                torch.nn.functional.cosine_similarity(
                    context.unsqueeze(0), hz.unsqueeze(0)
                ).squeeze(0)
            )
            _log.debug(
                "latent_context_blend",
                context_delta_norm=round(delta, 4),
                context_cosine=round(cos, 4),
                n_keys=len(keys_list),
                ctx_count=self._ctx_count,
            )
        new_h = blended[: self._h_dim].reshape(1, self._h_dim)
        new_z = blended[self._h_dim :].reshape(1, self._z_dim)
        return new_h, new_z

    def rearm_sink(self) -> None:
        """Clear the sink and restart warmup (mission-boundary seam).

        The ring and EMA summary are retained — only the anchor re-captures,
        making the sink a per-mission anchor.
        """
        self._sink = None
        self._ticks_since_arm = 0
        _log.info("bounded_context_sink_rearmed", observe_count=self._observe_count)

    def reset(self) -> None:
        """Clear every store and re-arm sink warmup (OTA weight-swap seam).

        A sink frozen under pre-swap weights would be stale under the new
        weights, so the swap seam clears everything AND restarts warmup.
        """
        self._ring.clear()
        self._long = None
        self._sink = None
        self._ticks_since_arm = 0
        _log.info("bounded_context_reset", observe_count=self._observe_count)

    def __len__(self) -> int:
        """Number of stored context vectors (ring + sink + EMA summary)."""
        return (
            len(self._ring)
            + (1 if self._sink is not None else 0)
            + (1 if self._long is not None else 0)
        )
