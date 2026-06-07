"""Integration: OfflineRLTrainer (CQL + IQL) logs per-step metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")
torch = pytest.importorskip("torch")

from mlflow import MlflowClient

from mousedroid.learning.offline_rl import CQLTrainer, IQLTrainer
from mousedroid.training.observability.mlflow_logger import (
    MlflowExperimentLogger,
)


@pytest.fixture
def logger(tmp_path: Path) -> MlflowExperimentLogger:
    uri = f"file:{tmp_path / 'mlruns'}"
    return MlflowExperimentLogger(tracking_uri=uri, experiment_name="trainer-test")


def _batch(batch_size: int = 4, state_dim: int = 4, action_dim: int = 2) -> dict:
    return {
        "states": torch.zeros(batch_size, state_dim),
        "actions": torch.zeros(batch_size, action_dim),
        "rewards": torch.zeros(batch_size),
        "next_states": torch.zeros(batch_size, state_dim),
        "dones": torch.zeros(batch_size),
    }


def test_cql_trainer_logs_q_bellman_cql_policy_losses_per_step(
    logger: MlflowExperimentLogger,
) -> None:
    logger.start_run(run_name="cql")
    ctx = logger.start_phase(phase="cql")
    trainer = CQLTrainer(state_dim=4, action_dim=2, experiment_logger=logger, log_phase=ctx)
    batch = _batch()
    for _ in range(3):
        trainer.update_step(**batch)
    logger.end_phase(ctx)
    logger.end_run()

    client = MlflowClient(tracking_uri=logger._tracking_uri)
    expected_keys = {"q_loss", "bellman_loss", "cql_loss", "policy_loss"}
    for key in expected_keys:
        history = client.get_metric_history(ctx.run_id, key)
        assert [m.step for m in history] == [0, 1, 2]


def test_iql_trainer_logs_q_value_policy_losses_per_step(
    logger: MlflowExperimentLogger,
) -> None:
    logger.start_run(run_name="iql")
    ctx = logger.start_phase(phase="iql")
    trainer = IQLTrainer(state_dim=4, action_dim=2, experiment_logger=logger, log_phase=ctx)
    batch = _batch()
    for _ in range(3):
        trainer.update_step(**batch)
    logger.end_phase(ctx)
    logger.end_run()

    client = MlflowClient(tracking_uri=logger._tracking_uri)
    expected_keys = {"q_loss", "value_loss", "policy_loss"}
    for key in expected_keys:
        history = client.get_metric_history(ctx.run_id, key)
        assert [m.step for m in history] == [0, 1, 2]


def test_trainer_without_logger_is_byte_identical_default() -> None:
    """A trainer built without an experiment_logger arg runs unchanged."""
    trainer = CQLTrainer(state_dim=4, action_dim=2)
    out = trainer.update_step(**_batch())
    assert set(out.keys()) == {"q_loss", "bellman_loss", "cql_loss", "policy_loss"}
    for v in out.values():
        assert isinstance(v, float)
