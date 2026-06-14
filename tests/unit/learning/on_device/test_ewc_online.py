"""Unit tests for the bounded EWC-regularized on-device online update (WS2).

Pins the WS2 contract: exactly ``cfg.update_steps`` steps at
``cfg.learning_rate``, the EWC penalty actually bites (loss differs with
``ewc_lambda>0`` vs ``0``), a typed result is returned, and structlog
start/complete events are emitted. A tiny deterministic ``nn.Module`` with a
fixed seed keeps every assertion reproducible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from structlog.testing import capture_logs

from mousedroid.config.schema import OnDeviceLearningConfig
from mousedroid.learning.on_device.ewc_online import EWCOnlineLearner
from mousedroid.learning.on_device.protocol import (
    OnDeviceLearner,
    OnDeviceUpdateResult,
)

_INPUT_DIM = 4
_OUTPUT_DIM = 3
_BATCH = 8


def _make_model() -> nn.Module:
    """Build a tiny deterministic two-layer model."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(_INPUT_DIM, 6),
        nn.ReLU(),
        nn.Linear(6, _OUTPUT_DIM),
    )


def _make_batch() -> torch.Tensor:
    """Build a fixed input batch."""
    torch.manual_seed(1)
    return torch.randn(_BATCH, _INPUT_DIM)


def test_runs_configured_number_of_steps() -> None:
    """The result reports exactly ``cfg.update_steps`` steps."""
    cfg = OnDeviceLearningConfig(enabled=True, update_steps=7, ewc_lambda=1.0)
    learner = EWCOnlineLearner(cfg, _make_model())

    result = learner.update(_make_batch())

    assert isinstance(result, OnDeviceUpdateResult)
    assert result.n_steps == 7


def test_returns_candidate_state_dict_with_model_keys() -> None:
    """The candidate state-dict covers the model's parameters."""
    model = _make_model()
    cfg = OnDeviceLearningConfig(enabled=True, update_steps=3, ewc_lambda=1.0)
    learner = EWCOnlineLearner(cfg, model)

    result = learner.update(_make_batch())

    assert set(result.candidate_state_dict) == set(model.state_dict())


def test_ewc_penalty_changes_loss() -> None:
    """A non-zero ``ewc_lambda`` yields a different loss than the unregularized run."""
    batch = _make_batch()

    cfg_off = OnDeviceLearningConfig(enabled=True, update_steps=5, ewc_lambda=0.0)
    loss_off = EWCOnlineLearner(cfg_off, _make_model()).update(batch).train_loss

    cfg_on = OnDeviceLearningConfig(enabled=True, update_steps=5, ewc_lambda=500.0)
    loss_on = EWCOnlineLearner(cfg_on, _make_model()).update(batch).train_loss

    assert loss_on != loss_off


def test_learning_rate_is_respected() -> None:
    """A larger learning rate moves the candidate weights farther."""
    model = _make_model()
    batch = _make_batch()
    base_first = next(iter(model.state_dict().values())).clone()

    small = EWCOnlineLearner(
        OnDeviceLearningConfig(enabled=True, update_steps=3, learning_rate=1e-4, ewc_lambda=0.0),
        model,
    ).update(batch)
    large = EWCOnlineLearner(
        OnDeviceLearningConfig(enabled=True, update_steps=3, learning_rate=1e-1, ewc_lambda=0.0),
        model,
    ).update(batch)

    first_key = next(iter(model.state_dict()))
    drift_small = (small.candidate_state_dict[first_key] - base_first).abs().sum()
    drift_large = (large.candidate_state_dict[first_key] - base_first).abs().sum()

    assert drift_large > drift_small


def test_emits_structlog_events() -> None:
    """Start and complete events are emitted with step + loss context."""
    cfg = OnDeviceLearningConfig(enabled=True, update_steps=2, ewc_lambda=1.0)
    learner = EWCOnlineLearner(cfg, _make_model())

    with capture_logs() as logs:
        learner.update(_make_batch())

    events = {entry["event"] for entry in logs}
    assert "on_device_update_start" in events
    assert "on_device_update_complete" in events


def test_concrete_satisfies_protocol() -> None:
    """The concrete learner is an instance of the runtime-checkable protocol."""
    learner = EWCOnlineLearner(
        OnDeviceLearningConfig(enabled=True, update_steps=1, ewc_lambda=1.0),
        _make_model(),
    )
    assert isinstance(learner, OnDeviceLearner)
