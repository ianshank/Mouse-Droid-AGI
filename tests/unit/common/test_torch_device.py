"""Unit tests for the shared ``resolve_device`` helper."""

from __future__ import annotations

import torch

from mousedroid.common.torch_device import resolve_device


def test_none_and_auto_resolve_to_available_hardware() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(None).type == expected
    assert resolve_device("auto").type == expected


def test_explicit_cpu_is_honoured() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device(torch.device("cpu")).type == "cpu"


def test_device_object_passthrough() -> None:
    dev = torch.device("cpu")
    assert resolve_device(dev) == dev
