"""Property test: an on-device update NEVER mutates the base weights in place.

This is the load-bearing safety invariant for Phase 6 WS2 — the cloud-pulled
base policy must survive a bounded on-device update bitwise-unchanged so the
regression gate (WS3/WS5) can always fall back to it. Hypothesis fuzzes the
seed, model width, batch size, learning rate and EWC weight; for every draw we
snapshot every base parameter, run the update, and assert the base is still
``torch.equal`` to its snapshot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import OnDeviceLearningConfig
from mousedroid.learning.on_device.ewc_online import EWCOnlineLearner


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**16),
    input_dim=st.integers(min_value=1, max_value=6),
    hidden_dim=st.integers(min_value=1, max_value=6),
    output_dim=st.integers(min_value=1, max_value=4),
    batch_size=st.integers(min_value=1, max_value=8),
    steps=st.integers(min_value=1, max_value=5),
    learning_rate=st.floats(min_value=1e-4, max_value=1.0),
    ewc_lambda=st.floats(min_value=0.0, max_value=1000.0),
)
def test_base_weights_unchanged_after_update(
    seed: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    ewc_lambda: float,
) -> None:
    """Base parameters are bitwise-identical before and after the update."""
    torch.manual_seed(seed)
    base_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )

    # Snapshot every base parameter BEFORE the update.
    before = {name: p.detach().clone() for name, p in base_model.named_parameters()}

    cfg = OnDeviceLearningConfig(
        enabled=True,
        update_steps=steps,
        learning_rate=learning_rate,
        ewc_lambda=ewc_lambda,
    )
    batch = torch.randn(batch_size, input_dim)

    result = EWCOnlineLearner(cfg, base_model).update(batch)

    # The base must be untouched in place...
    for name, param in base_model.named_parameters():
        assert torch.equal(param.detach(), before[name]), f"base param '{name}' mutated"

    # ...and the candidate must be a SEPARATE object (not the base's tensors).
    for name in before:
        assert (
            result.candidate_state_dict[name].data_ptr()
            != dict(base_model.named_parameters())[name].data_ptr()
        )
