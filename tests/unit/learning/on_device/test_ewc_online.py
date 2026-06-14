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


class _TrainModeProbe(nn.Module):
    """A module recording its ``training`` flag at every forward call.

    Used to prove the candidate optimises in train mode even after the EWC
    consolidation step (which internally calls ``model.eval()``). A real
    BN/dropout layer would silently corrupt under eval-mode gradient steps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_INPUT_DIM, _OUTPUT_DIM)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(_OUTPUT_DIM)
        self.training_flags: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record train-mode then apply dropout + batch-norm + linear."""
        self.training_flags.append(self.training)
        return self.bn(self.dropout(self.linear(x)))


def test_candidate_optimises_in_train_mode_after_consolidate() -> None:
    """Every optimisation-loop forward on the candidate is in train mode.

    ``EWCAgent.consolidate`` puts the model in eval mode; the learner must
    restore train mode before the optimisation loop so BN/dropout layers are
    not silently corrupted.
    """
    from typing import Any

    captured: dict[str, _TrainModeProbe] = {}
    cfg = OnDeviceLearningConfig(enabled=True, update_steps=3, ewc_lambda=1.0)

    probe = _TrainModeProbe()
    learner = EWCOnlineLearner(cfg, probe)
    # Patch deepcopy so we can grab the actual candidate the learner optimises.
    import mousedroid.learning.on_device.ewc_online as mod

    real_deepcopy = mod.copy.deepcopy

    def _capture(obj: Any, memo: Any = None) -> Any:
        clone = real_deepcopy(obj, memo) if memo is not None else real_deepcopy(obj)
        if isinstance(clone, _TrainModeProbe):
            captured["candidate"] = clone
        return clone

    mod.copy.deepcopy = _capture  # type: ignore[assignment]
    try:
        learner.update(_make_batch())
    finally:
        mod.copy.deepcopy = real_deepcopy  # type: ignore[assignment]

    candidate = captured["candidate"]
    # consolidate() forward(s) run in eval mode; the optimisation-loop forwards
    # (the last ``update_steps`` calls) must all be in train mode.
    optimisation_flags = candidate.training_flags[-3:]
    assert optimisation_flags == [True, True, True]


def test_custom_task_loss_fn_is_used() -> None:
    """An injected ``task_loss_fn`` overrides the default stand-in criterion."""
    calls: list[torch.Tensor] = []

    def _custom_loss(output: torch.Tensor) -> torch.Tensor:
        calls.append(output)
        return output.abs().mean()

    cfg = OnDeviceLearningConfig(enabled=True, update_steps=2, ewc_lambda=0.0)
    learner = EWCOnlineLearner(cfg, _make_model(), task_loss_fn=_custom_loss)

    learner.update(_make_batch())

    assert len(calls) == 2  # called once per step


def test_default_task_loss_fn_is_byte_identical() -> None:
    """Omitting ``task_loss_fn`` reproduces the legacy squared-mean stand-in."""
    batch = _make_batch()
    cfg = OnDeviceLearningConfig(enabled=True, update_steps=4, ewc_lambda=0.0)

    default_loss = EWCOnlineLearner(cfg, _make_model()).update(batch).train_loss
    explicit_loss = (
        EWCOnlineLearner(
            cfg,
            _make_model(),
            task_loss_fn=lambda out: out.pow(2).mean(),
        )
        .update(batch)
        .train_loss
    )

    assert default_loss == explicit_loss
