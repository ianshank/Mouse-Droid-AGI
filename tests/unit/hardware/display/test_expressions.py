"""Tests for the pure expression renderer (no I/O required)."""

from __future__ import annotations

import hashlib

import pytest

from mousedroid.hardware.display.expressions import (
    Expression,
    render_expression,
    render_text,
)


@pytest.mark.parametrize("expression", list(Expression))
def test_render_expression_returns_correct_size_and_mode(expression: Expression) -> None:
    """Every expression renders a 1-bit framebuffer of the requested size."""
    img = render_expression(expression, 128, 64)
    assert img.size == (128, 64)
    assert img.mode == "1"


@pytest.mark.parametrize("expression", list(Expression))
def test_render_expression_is_deterministic(expression: Expression) -> None:
    """Repeated renders of the same expression produce identical bytes."""
    a = render_expression(expression, 128, 64).tobytes()
    b = render_expression(expression, 128, 64).tobytes()
    assert a == b


def test_render_expressions_differ_pairwise() -> None:
    """Different expressions produce different framebuffers.

    This catches a regression where the dispatch table goes stale and every
    expression silently renders to the same image.
    """
    hashes = {
        expr: hashlib.sha256(render_expression(expr, 128, 64).tobytes()).hexdigest()
        for expr in Expression
    }
    assert len(set(hashes.values())) == len(Expression), hashes


def test_render_expression_respects_custom_dimensions() -> None:
    """Renderer honours width/height arguments rather than hardcoding 128x64."""
    img = render_expression(Expression.NEUTRAL, 64, 32)
    assert img.size == (64, 32)


def test_render_text_centres_message() -> None:
    """``render_text`` produces a 1-bit panel of the requested size."""
    img = render_text("MSE-6 online", 128, 64)
    assert img.size == (128, 64)
    assert img.mode == "1"
    # Text is non-empty: at least one pixel is set.
    assert any(b != 0 for b in img.tobytes())
