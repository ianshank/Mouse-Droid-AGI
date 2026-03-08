"""Tests for Elastic Weight Consolidation (EWC) agent."""

from __future__ import annotations

import torch
import torch.nn as nn

from mousedroid.config.schema import LearningConfig
from mousedroid.learning.ewc import EWCAgent


def _simple_model() -> nn.Module:
    return nn.Linear(4, 2)


def _make_agent(ewc_lambda: float = 100.0, fisher_samples: int = 5) -> EWCAgent:
    cfg = LearningConfig(ewc_lambda=ewc_lambda, ewc_fisher_samples=fisher_samples)
    return EWCAgent(cfg, _simple_model())


def test_constructor_stores_config():
    agent = _make_agent(ewc_lambda=42.0, fisher_samples=10)
    assert agent._lambda == 42.0
    assert agent._fisher_samples == 10


def test_initial_fisher_empty():
    agent = _make_agent()
    assert agent._fisher == {}
    assert agent._star_params == {}


def test_compute_penalty_without_consolidation_returns_zero():
    agent = _make_agent()
    penalty = agent.compute_penalty()
    assert penalty.item() == 0.0


def test_consolidate_snapshots_params():
    agent = _make_agent()
    agent.consolidate()
    assert len(agent._star_params) > 0
    for _name, param in agent._star_params.items():
        assert isinstance(param, torch.Tensor)


def test_consolidate_creates_fisher():
    agent = _make_agent()
    agent.consolidate()
    assert len(agent._fisher) > 0
    for _name, fisher_val in agent._fisher.items():
        assert isinstance(fisher_val, torch.Tensor)


def test_penalty_after_consolidation_is_zero_when_no_drift():
    model = _simple_model()
    cfg = LearningConfig(ewc_lambda=100.0, ewc_fisher_samples=5)
    agent = EWCAgent(cfg, model)
    agent.consolidate()
    # No parameter change => penalty should be 0
    penalty = agent.compute_penalty()
    assert penalty.item() == 0.0


def test_penalty_increases_after_param_drift():
    model = _simple_model()
    cfg = LearningConfig(ewc_lambda=100.0, ewc_fisher_samples=5)
    agent = EWCAgent(cfg, model)

    # Manually set fisher to nonzero so penalty is nonzero after drift
    agent.consolidate()
    # Force fisher to be nonzero
    for name in agent._fisher:
        agent._fisher[name] = torch.ones_like(agent._fisher[name])

    # Drift the model params
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    penalty = agent.compute_penalty()
    assert penalty.item() > 0.0


def test_named_parameters_yields_only_grad_params():
    model = _simple_model()
    # Freeze bias
    model.bias.requires_grad = False
    cfg = LearningConfig(ewc_lambda=1.0, ewc_fisher_samples=1)
    agent = EWCAgent(cfg, model)
    names = [n for n, _ in agent._named_parameters()]
    assert "weight" in names
    assert "bias" not in names
