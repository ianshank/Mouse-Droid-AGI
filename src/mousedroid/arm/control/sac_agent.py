"""SAC + HER goal-conditioned policy for robot arm control.

Wraps Stable-Baselines3 SAC with Hindsight Experience Replay for
sparse-reward manipulation tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTrainingConfig

_log = get_logger(__name__)


class SACAgent:
    """SAC + HER agent for goal-conditioned arm manipulation.

    Combines Soft Actor-Critic with Hindsight Experience Replay to
    handle sparse reward signals in long-horizon tasks.

    Args:
        training_cfg: Training hyperparameter configuration.
    """

    def __init__(self, training_cfg: ArmTrainingConfig) -> None:
        """Initialise SAC+HER agent.

        Args:
            training_cfg: Training config with algorithm, LR, buffer size, etc.
        """
        self._cfg = training_cfg
        self._model: Any = None
        self._is_trained = False
        _log.info(
            "sac_agent_init",
            algorithm=training_cfg.algorithm,
            lr=training_cfg.learning_rate,
            buffer_size=training_cfg.buffer_size,
        )

    def build(self, env: Any) -> None:
        """Build the SAC+HER model for a given environment.

        Args:
            env: Gymnasium-compatible environment (must support GoalEnv).

        Raises:
            ImportError: If stable-baselines3 is not installed.
        """
        try:
            from stable_baselines3 import SAC, HerReplayBuffer
        except ImportError as exc:
            msg = "stable-baselines3 required for SAC+HER training"
            raise ImportError(msg) from exc

        self._model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=self._cfg.learning_rate,
            batch_size=self._cfg.batch_size,
            buffer_size=self._cfg.buffer_size,
            gamma=self._cfg.gamma,
            tau=self._cfg.tau,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": self._cfg.her_n_sampled_goal,
                "goal_selection_strategy": self._cfg.her_goal_selection,
            },
            seed=self._cfg.seed,
            verbose=0,
        )
        _log.info("sac_model_built")

    def train(self, total_timesteps: int | None = None) -> None:
        """Train the SAC+HER policy.

        Args:
            total_timesteps: Override total training timesteps (None = use config).

        Raises:
            RuntimeError: If model has not been built yet.
        """
        if self._model is None:
            msg = "Model not built. Call build(env) first."
            raise RuntimeError(msg)

        steps = total_timesteps or self._cfg.total_timesteps
        _log.info("training_start", total_timesteps=steps)
        self._model.learn(total_timesteps=steps)
        self._is_trained = True
        _log.info("training_complete")

    def predict(self, observation: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Predict action for given observation.

        Args:
            observation: Goal-conditioned observation dict.

        Returns:
            Action vector.

        Raises:
            RuntimeError: If model has not been built.
        """
        if self._model is None:
            msg = "Model not built. Call build(env) first."
            raise RuntimeError(msg)

        action, _ = self._model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    def save(self, path: str | None = None) -> Path:
        """Save model checkpoint.

        Args:
            path: Save path (None = use config weights_dir).

        Returns:
            Path where model was saved.

        Raises:
            RuntimeError: If model has not been built.
        """
        if self._model is None:
            msg = "Model not built."
            raise RuntimeError(msg)

        save_path = Path(path or self._cfg.weights_dir) / "sac_her_checkpoint"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(save_path))
        _log.info("model_saved", path=str(save_path))
        return save_path

    def load(self, path: str) -> None:
        """Load model from checkpoint.

        Args:
            path: Path to saved model checkpoint.
        """
        try:
            from stable_baselines3 import SAC

            self._model = SAC.load(path)
            self._is_trained = True
            _log.info("model_loaded", path=path)
        except ImportError as exc:
            msg = "stable-baselines3 required for model loading"
            raise ImportError(msg) from exc

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained
