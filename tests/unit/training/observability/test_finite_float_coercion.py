"""Property test for ``to_finite_float`` — MLflow metric value coercion.

MLflow ``log_metric`` accepts a Python ``float`` but training loops produce
``torch.Tensor`` / numpy scalars; logging NaN/Inf silently corrupts metric
curves. The helper centralises both concerns at the logger boundary.
"""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from mousedroid.training.observability.protocol import to_finite_float


@given(value=st.floats(allow_nan=False, allow_infinity=False, width=32))
@settings(max_examples=80, deadline=None)
def test_python_float_passes_through_finite(value: float) -> None:
    assert to_finite_float(value) == pytest.approx(value, rel=1e-6, abs=1e-9)


@given(value=st.floats(allow_nan=False, allow_infinity=False, width=32))
@settings(max_examples=80, deadline=None)
def test_numpy_scalar_collapses_to_float(value: float) -> None:
    assert to_finite_float(np.float32(value)) == pytest.approx(
        float(np.float32(value)), rel=1e-6, abs=1e-9
    )


def test_torch_zero_dim_tensor_collapses_via_item() -> None:
    torch = pytest.importorskip("torch")
    t = torch.tensor(3.14)
    assert to_finite_float(t) == pytest.approx(3.14, rel=1e-6)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_returns_none(bad: float) -> None:
    assert to_finite_float(bad) is None


def test_numpy_nan_returns_none() -> None:
    assert to_finite_float(np.float32("nan")) is None


def test_torch_nan_tensor_returns_none() -> None:
    torch = pytest.importorskip("torch")
    assert to_finite_float(torch.tensor(float("nan"))) is None


def test_unsupported_type_returns_none() -> None:
    """A string / dict / None is not a metric value — coerce to None, not raise."""
    assert to_finite_float("not a number") is None
    assert to_finite_float(None) is None
    assert to_finite_float({"key": 1.0}) is None


def test_int_collapses_to_float() -> None:
    """Python int is a valid scalar; collapse to float."""
    assert to_finite_float(42) == 42.0
    assert isinstance(to_finite_float(42), float)
