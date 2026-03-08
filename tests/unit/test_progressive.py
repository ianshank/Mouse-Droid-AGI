"""Tests for Progressive Neural Network."""

from __future__ import annotations

import torch

from mousedroid.learning.progressive import ProgressiveNetwork


def test_constructor_creates_one_column():
    net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
    assert len(net.columns) == 1


def test_forward_returns_correct_shape():
    net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(2, 8)
    out = net(x)
    assert out.shape == (2, 4)


def test_grow_adds_column():
    net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
    net.grow()
    assert len(net.columns) == 2


def test_grow_freezes_previous_columns():
    net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
    net.grow()
    # First column should be frozen
    for param in net.columns[0].parameters():
        assert not param.requires_grad
    # Second column should be trainable
    for param in net.columns[1].parameters():
        assert param.requires_grad


def test_forward_after_grow():
    net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
    net.grow()
    x = torch.randn(3, 8)
    out = net(x)
    assert out.shape == (3, 4)


def test_multiple_grows():
    net = ProgressiveNetwork(input_dim=4, hidden_dim=8, output_dim=2)
    net.grow()
    net.grow()
    assert len(net.columns) == 3
    x = torch.randn(1, 4)
    out = net(x)
    assert out.shape == (1, 2)


def test_column_lateral_connections():
    net = ProgressiveNetwork(input_dim=4, hidden_dim=8, output_dim=2)
    # First column: 0 laterals
    assert len(net.columns[0].laterals) == 0
    net.grow()
    # Second column: 1 lateral
    assert len(net.columns[1].laterals) == 1
    net.grow()
    # Third column: 2 laterals
    assert len(net.columns[2].laterals) == 2
