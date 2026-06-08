"""In-process sim episode generation -> batched tensors for RSSM pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from mousedroid.logging.setup import get_logger
from mousedroid.sim.protocols import RoverEnvProtocol
from mousedroid.training.domain_randomization import DomainRandomizer
from mousedroid.training.rover_obs_adapter import RoverObsAdapter

if TYPE_CHECKING:
    from mousedroid.hardware.camera.feature_extractor import FeatureExtractorProtocol

_log = get_logger(__name__)


@dataclass(frozen=True)
class EpisodeBatch:
    """Batched ``(B, T, ...)`` tensors consumed by ``RSSM.train_sequence``.

    A new in-memory container — NOT ``MouseDroidExperienceRecord`` (that schema
    cannot hold the rover modalities). LMDB persistence is a deferred follow-on.
    """

    motor: Tensor
    ultrasonic: Tensor
    lidar: Tensor
    valid_mask: Tensor
    action: Tensor
    reward: Tensor
    vision: Tensor  # (B, T, feature_dim) when a feature extractor is used, else (B, T, 0)


class SimEpisodeGenerator:
    """Roll N episodes of T steps under a smoothed-random policy; adapt + stack."""

    def __init__(
        self,
        env: RoverEnvProtocol,
        adapter: RoverObsAdapter,
        *,
        n_episodes: int,
        seq_len: int,
        seed: int,
        explore_action_rad_s: float = 6.0,
        explore_smoothing: float = 0.7,
        domain_randomizer: DomainRandomizer | None = None,
        feature_extractor: FeatureExtractorProtocol | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            env: A rover env conforming to ``RoverEnvProtocol``.
            adapter: Maps rover obs -> RSSM encoder inputs.
            n_episodes: Number of episodes (batch dimension).
            seq_len: Steps per episode (time dimension).
            seed: Seed for the deterministic exploration + reset stream.
            explore_action_rad_s: Bound (rad/s) on the random wheel-command target.
            explore_smoothing: EMA weight on the previous action (temporal correlation).
            domain_randomizer: Optional per-episode physics randomizer. When provided
                and enabled, its sampled chassis params are applied to the env before
                each episode (no-op for envs without ``apply_domain_params``).
            feature_extractor: Optional vision feature extractor. When provided,
                each step renders an RGB frame (``env.render_rgb()``) and extracts
                ``vision_features``, populating ``EpisodeBatch.vision``. ``None``
                (Phase 5 pretrain) leaves vision empty.
        """
        self._env = env
        self._adapter = adapter
        self._n = n_episodes
        self._t = seq_len
        self._rng = np.random.default_rng(seed)
        self._action_dim = env.action_dim
        self._explore_bound = explore_action_rad_s
        self._explore_smoothing = explore_smoothing
        self._dr = domain_randomizer
        self._extractor = feature_extractor

    def _sample_action(self, prev: NDArray[np.float32]) -> NDArray[np.float32]:
        # Smoothed uniform-random wheel commands (Dreamer seed-episode policy);
        # bound + smoothing are config-driven (the env clips to its own cap).
        bound = self._explore_bound
        alpha = self._explore_smoothing
        target = self._rng.uniform(-bound, bound, size=self._action_dim).astype(np.float32)
        out: NDArray[np.float32] = (alpha * prev + (1.0 - alpha) * target).astype(np.float32)
        return out

    def _maybe_randomize(self) -> None:
        """Apply one fresh domain-randomization sample to the env, if enabled."""
        if self._dr is None or not self._dr.enabled:
            return
        apply_dr = getattr(self._env, "apply_domain_params", None)
        if not callable(apply_dr):
            return
        chassis = self._dr.sample(self._rng).chassis
        apply_dr(
            friction=float(chassis["friction"]),
            slip=float(chassis["slip"]),
            mass_kg=float(chassis["mass_kg"]),
            motor_gain=float(chassis["motor_gain"]),
        )

    def generate(self) -> EpisodeBatch:
        """Roll the configured episodes and stack them into an ``EpisodeBatch``."""
        motors: list[list[NDArray[np.float32]]] = []
        ultras: list[list[NDArray[np.float32]]] = []
        lidars: list[list[NDArray[np.float32]]] = []
        masks: list[list[NDArray[np.float32]]] = []
        actions: list[list[NDArray[np.float32]]] = []
        rewards: list[list[np.float32]] = []
        visions: list[list[NDArray[np.float32]]] = []
        render_rgb = getattr(self._env, "render_rgb", None)

        for _ep in range(self._n):
            self._maybe_randomize()
            obs, info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
            # Annotate explicitly: numpy stubs on some versions type np.zeros(scalar)
            # as a strict 1-D shape, which then rejects the looser NDArray from
            # _sample_action (3.10 vs 3.11 stub drift). The loose type is correct.
            prev: NDArray[np.float32] = np.zeros(self._action_dim, dtype=np.float32)
            em, eu, el, ek, ea, er, ev = ([] for _ in range(7))
            for _step in range(self._t):
                vis = self._extract_vision(render_rgb)
                adapted = self._adapter.adapt(obs, info, vision_features=vis)
                action = self._sample_action(prev)
                # Pad 2-DoF wheel action to the RSSM's 3-DoF [vx, vy=0, omega] space.
                padded = np.asarray([float(action[0]), 0.0, float(action[-1])], dtype=np.float32)
                em.append(adapted["motor"])
                eu.append(adapted["ultrasonic"])
                el.append(adapted.get("lidar", np.zeros(0, dtype=np.float32)))
                ek.append(adapted["valid_mask"])
                ea.append(padded)
                ev.append(adapted.get("vision", np.zeros(0, dtype=np.float32)))
                obs, reward, terminated, truncated, info = self._env.step(action)
                er.append(np.float32(reward))
                prev = action
                if terminated or truncated:
                    obs, info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
                    prev = np.zeros(self._action_dim, dtype=np.float32)
            motors.append(em)
            ultras.append(eu)
            lidars.append(el)
            masks.append(ek)
            actions.append(ea)
            rewards.append(er)
            visions.append(ev)
        _log.info(
            "sim_episodes_generated",
            n_episodes=self._n,
            seq_len=self._t,
            vision=self._extractor is not None,
        )

        def _stack(x: list[list[NDArray[np.float32]]]) -> Tensor:
            return torch.as_tensor(np.asarray(x, dtype=np.float32))

        return EpisodeBatch(
            motor=_stack(motors),
            ultrasonic=_stack(ultras),
            lidar=_stack(lidars),
            valid_mask=_stack(masks),
            action=_stack(actions),
            reward=torch.as_tensor(np.asarray(rewards, dtype=np.float32)),
            vision=_stack(visions),
        )

    def _extract_vision(self, render_rgb: object) -> NDArray[np.float32] | None:
        """Render + extract vision features for the current step, or ``None``."""
        if self._extractor is None or not callable(render_rgb):
            return None
        rgb = render_rgb()
        return self._extractor.extract(rgb)
