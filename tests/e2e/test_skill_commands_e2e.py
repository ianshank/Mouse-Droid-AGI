"""E2E validation of ``.claude/skills`` skill *outputs*.

Each test runs the workflow a command-skill documents and asserts on what it
produces. The three skills (``sim-test``, ``train-policy``,
``robot-arm-trainer``) all target the arm subsystem, so every test here is
gated on the optional ``[arm]`` extras (``mujoco`` / ``stable_baselines3``):
CI hosts without those deps SKIP cleanly rather than fail, because robot-arm is
the deferred baseline (``docs/planning/NEXT_STEPS.md``).

Selection is by the ``tests/e2e/`` directory, not a custom marker — ``e2e`` is
not a registered pytest marker in this repo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# tests/e2e/<file> -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sim_test_skill_runs_arm_suite() -> None:
    """``sim-test`` output is a passing arm test run (smallest documented scope).

    Runs exactly the command ``.claude/skills/sim-test/SKILL.md`` documents under
    "Specific category" — ``pytest tests/unit/arm/ -k "env"`` — in a child
    process. The child inherits ``os.environ`` (so a ``PYTHONPATH`` set by the
    caller is honoured) and runs from the repo root; no path is hardcoded.
    """
    pytest.importorskip("mujoco")  # arm extras absent -> skip, not fail
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/arm/",
            "-k",
            "env",
            "-q",
            "--import-mode=importlib",
            "--no-cov",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"sim-test 'env' scope failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.slow
def test_train_policy_skill_emits_checkpoint(tmp_path: Path) -> None:
    """``train-policy`` output is a SAC+HER checkpoint artifact.

    Smoke (not convergence): proves the documented wiring — load the arm YAML
    via the repo's runtime config loader, build the real arm env via the arm
    factory, build the real :class:`SACAgent`, run a handful of training steps,
    and ``save()`` — genuinely writes a Stable-Baselines3 checkpoint to disk.

    The repo's arm env (``ArmEnvironmentBase``) exposes ``reset``/``step`` but
    is *not* a ``gymnasium.Env`` (no ``observation_space``/``action_space``), so
    SB3's ``SAC`` constructor rejects it directly. A minimal in-test Gymnasium
    GoalEnv adapter — built from the *real* env's actual observation/action
    shapes — bridges to SB3. The checkpoint asserted on is genuine SB3 output
    from the real ``SACAgent.save()``; only the gym-space bridge is test-local.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from gymnasium import spaces

    from mousedroid.arm.control.sac_agent import SACAgent
    from mousedroid.factory import build_arm_environment
    from mousedroid.validation.runtime import load_runtime_settings

    cfg = load_runtime_settings([str(_REPO_ROOT / "config/robot_arm_training.yaml")])
    assert cfg.arm is not None, "arm config must load from YAML, not env"
    assert cfg.arm_training is not None, "arm_training config must load from YAML"

    # Real arm env via the real factory (proves the documented build step).
    arm_env = build_arm_environment(cfg)
    sample_obs, _ = arm_env.reset(seed=cfg.arm_training.seed)
    obs_dim = int(sample_obs["observation"].shape[0])
    goal_dim = int(sample_obs["achieved_goal"].shape[0])
    dof = int(cfg.arm.dof)

    class _GymGoalAdapter(gym.Env):  # type: ignore[type-arg]
        """Thin Gymnasium GoalEnv shim over the repo's arm env for SB3.

        Declares the gym spaces SB3 requires; all dynamics come from the wrapped
        real arm env. ``compute_reward`` mirrors HER's sparse goal distance so
        ``HerReplayBuffer`` can relabel transitions.
        """

        def __init__(self) -> None:
            box = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float64)
            goal_box = spaces.Box(-np.inf, np.inf, (goal_dim,), dtype=np.float64)
            self.observation_space = spaces.Dict(
                {"observation": box, "achieved_goal": goal_box, "desired_goal": goal_box}
            )
            self.action_space = spaces.Box(-1.0, 1.0, (dof,), dtype=np.float64)

        def reset(self, *, seed: int | None = None, options: object = None):  # type: ignore[no-untyped-def]
            return arm_env.reset(seed=seed)

        def step(self, action):  # type: ignore[no-untyped-def]
            return arm_env.step(np.asarray(action, dtype=np.float64))

        def compute_reward(self, achieved_goal, desired_goal, info):  # type: ignore[no-untyped-def]
            return -np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)

    # Test-local smoke budget: tiny buffer/batch keep memory + time bounded; a
    # handful of steps prove the wiring. These are smoke constants, not config.
    smoke_cfg = cfg.arm_training.model_copy(update={"buffer_size": 500, "batch_size": 8})
    agent = SACAgent(smoke_cfg)
    agent.build(_GymGoalAdapter())
    agent.train(total_timesteps=10)
    # ``SACAgent.save`` treats its argument as a *directory*, appends the
    # ``sac_her_checkpoint`` stem, and SB3 writes ``<stem>.zip``. Save into an
    # explicit subdir of tmp_path and assert the concrete artifact named by the
    # returned path — robust, not an rglob guess.
    save_dir = tmp_path / "weights"
    returned = agent.save(str(save_dir))

    assert returned.parent == save_dir, "checkpoint not written under the requested directory"
    artifact = returned.with_suffix(".zip")
    assert artifact.is_file(), f"SACAgent.save() did not write {artifact}"


def test_robot_arm_trainer_milestone1_env_smoke() -> None:
    """``robot-arm-trainer`` milestone 1 = a MuJoCo scene + Gymnasium wrapper.

    Asserts the milestone-1 deliverable the skill names: the arm env imports,
    ``reset()`` yields a goal-conditioned observation dict, and one ``step()``
    returns a valid Gymnasium-shaped tuple with finite observations. Mirrors the
    existing ``tests/unit/arm/test_tower_of_hanoi_env.py`` construction pattern.
    """
    pytest.importorskip("mujoco")
    from mousedroid.arm.environments.tower_of_hanoi import TowerOfHanoiEnv
    from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig

    task_cfg = ArmTaskConfig(num_disks=3, max_episode_steps=50)
    training_cfg = ArmTrainingConfig()
    env = TowerOfHanoiEnv(task_cfg, training_cfg, dof=6)

    obs, info = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert isinstance(info, dict)
    assert np.all(np.isfinite(obs["observation"]))

    action = np.zeros(6, dtype=np.float64)
    step_obs, reward, terminated, truncated, step_info = env.step(action)
    assert isinstance(step_obs, dict)
    assert np.all(np.isfinite(step_obs["observation"]))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)
