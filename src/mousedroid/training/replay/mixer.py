"""Deterministic sim/real episode mixer.

Wraps a synthetic ("sim") iterable and a real-replay iterable behind a
single iterator that draws from each according to a ramped target ratio.
A single seeded :class:`numpy.random.Generator` powers the choice so two
runs at the same seed produce identical interleavings (RL-Co two-stage
curriculum).

Defaults are off — ``alpha_target=0.0`` makes the mixer pull only from the
sim source, preserving byte-identical training when this module is
imported but not flag-flipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T")

_log = get_logger(__name__)

# Default ramp horizon (steps) for alpha 0 -> alpha_target.
DEFAULT_RAMP_STEPS: int = 1000
# Default cadence (steps) for emitting `mixer_ratio_check` telemetry.
DEFAULT_LOG_INTERVAL: int = 500
# The mixer interleaves exactly two sources: sim and real.
_NUM_SOURCES: int = 2  # hardcoded-ok: structural invariant — two sources only
# Decimal precision used when logging realized/current alpha.
_LOG_ALPHA_PRECISION: int = 4  # hardcoded-ok: log formatting precision


class MixerConfig(BaseModel):
    """Configuration for :class:`RealSimMixer`.

    All fields are opt-in: the defaults make the mixer behave as a pass-through
    over the sim source.
    """

    alpha_target: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Target probability of drawing from the real replay source. "
            "0.0 disables mixing (sim-only); 1.0 is real-only."
        ),
    )
    alpha_ramp_steps: int = Field(
        DEFAULT_RAMP_STEPS,
        gt=0,
        description=(
            "Number of mix steps over which alpha linearly ramps from 0 "
            "to alpha_target. Mirrors the RL-Co two-stage curriculum."
        ),
    )
    seed: int | None = Field(
        None,
        description="Optional RNG seed for deterministic mixing.",
    )
    log_every_n: int = Field(
        DEFAULT_LOG_INTERVAL,
        gt=0,
        description="Emit a `mixer_ratio_check` log every N draws.",
    )
    debug_log_every_n: int = Field(
        0,
        ge=0,
        description=(
            "Emit a structlog DEBUG `mixer_draw` line every N draws "
            "(0 disables). Mirrors "
            "`mousedroid.config.schema.ReplayMixerConfig.debug_log_every_n`."
        ),
    )

    @classmethod
    def from_settings(cls, replay_mixer_cfg: object) -> MixerConfig:
        """Build a :class:`MixerConfig` from a YAML-loaded ``ReplayMixerConfig``.

        The arg is typed as ``object`` to avoid a circular import with
        :mod:`mousedroid.config.schema`. Only the field names are required to
        match.

        Args:
            replay_mixer_cfg: A ``ReplayMixerConfig`` (or any object with the
                same field names).

        Returns:
            Equivalent :class:`MixerConfig`.
        """
        return cls(
            alpha_target=getattr(replay_mixer_cfg, "alpha_target", 0.0),
            alpha_ramp_steps=getattr(replay_mixer_cfg, "alpha_ramp_steps", DEFAULT_RAMP_STEPS),
            seed=getattr(replay_mixer_cfg, "seed", None),
            log_every_n=getattr(replay_mixer_cfg, "log_every_n", DEFAULT_LOG_INTERVAL),
            debug_log_every_n=getattr(replay_mixer_cfg, "debug_log_every_n", 0),
        )


class RealSimMixer(Generic[T]):
    """Iterate over ``(sim_iter, real_iter)`` with a ramped real-fraction.

    The mixer is deterministic at a fixed seed and exposes the realized
    real fraction via :attr:`stats` for tests and telemetry.

    Args:
        sim_source: Iterable yielding synthetic items.
        real_source: Iterable yielding real-replay items. Pass an empty
            iterable to disable real draws.
        cfg: Mixer configuration.

    Notes:
        When the real source is exhausted, draws fall back to sim
        (``stats["real_exhausted"]`` increments). When the sim source is
        exhausted, draws fall back to real. When both are exhausted, the
        mixer stops.
    """

    def __init__(
        self,
        sim_source: Iterable[T],
        real_source: Iterable[T],
        cfg: MixerConfig,
    ) -> None:
        self._sim: Iterator[T] = iter(sim_source)
        self._real: Iterator[T] = iter(real_source)
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._step = 0
        self._real_drawn = 0
        self._sim_drawn = 0
        self._real_exhausted = 0
        self._sim_exhausted = 0
        self._stopped = False

    @property
    def stats(self) -> dict[str, float]:
        """Realized counters and the empirical real fraction."""
        total = self._real_drawn + self._sim_drawn
        realized_alpha = self._real_drawn / total if total else 0.0
        return {
            "step": float(self._step),
            "real_drawn": float(self._real_drawn),
            "sim_drawn": float(self._sim_drawn),
            "real_exhausted": float(self._real_exhausted),
            "sim_exhausted": float(self._sim_exhausted),
            "realized_alpha": realized_alpha,
            "current_alpha": self._current_alpha(),
        }

    def _current_alpha(self) -> float:
        """Linearly ramped alpha clamped at ``alpha_target``."""
        ramp = self._step / self._cfg.alpha_ramp_steps
        return min(self._cfg.alpha_target, self._cfg.alpha_target * ramp)

    def _draw_real(self) -> T | None:
        try:
            item = next(self._real)
        except StopIteration:
            self._real_exhausted += 1
            return None
        self._real_drawn += 1
        return item

    def _draw_sim(self) -> T | None:
        try:
            item = next(self._sim)
        except StopIteration:
            self._sim_exhausted += 1
            return None
        self._sim_drawn += 1
        return item

    def __iter__(self) -> RealSimMixer[T]:
        """Return self — the mixer is its own iterator."""
        return self

    def __next__(self) -> T:
        """Draw the next item, ramping alpha toward target."""
        if self._stopped:
            raise StopIteration

        # Bounded loop over the two sources: avoids ambiguous fall-through.
        for _ in range(_NUM_SOURCES):
            alpha = self._current_alpha()
            pick_real = bool(self._rng.random() < alpha)
            primary = self._draw_real if pick_real else self._draw_sim
            fallback = self._draw_sim if pick_real else self._draw_real

            item = primary()
            if item is None:
                item = fallback()
            if item is not None:
                self._step += 1
                if self._step % self._cfg.log_every_n == 0:
                    realized = round(self.stats["realized_alpha"], _LOG_ALPHA_PRECISION)
                    current = round(alpha, _LOG_ALPHA_PRECISION)
                    _log.info(
                        "mixer_ratio_check",
                        step=self._step,
                        realized_alpha=realized,
                        current_alpha=current,
                        real_drawn=self._real_drawn,
                        sim_drawn=self._sim_drawn,
                    )
                # Throttled DEBUG-level live triage (operator opt-in via
                # cfg.training.replay_mixer.debug_log_every_n; 0 disables).
                if (
                    self._cfg.debug_log_every_n > 0
                    and self._step % self._cfg.debug_log_every_n == 0
                ):
                    _log.debug(
                        "mixer_draw",
                        step=self._step,
                        source="real" if pick_real else "sim",
                        alpha=round(alpha, _LOG_ALPHA_PRECISION),
                        realized=round(self.stats["realized_alpha"], _LOG_ALPHA_PRECISION),
                    )
                return item

        # Both sources exhausted on this step.
        self._stopped = True
        raise StopIteration
