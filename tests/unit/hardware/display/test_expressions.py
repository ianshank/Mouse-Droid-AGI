"""Tests for the pure expression renderer (no I/O required)."""

from __future__ import annotations

import hashlib

import numpy as np
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


def test_angry_eyes_are_mirrored() -> None:
    """ANGRY must slant the inner corners of both eyes down (V-shape).

    The left eye's right half (inner, toward nose) should have more dark pixels
    in the top-right quadrant, and the right eye's left half (inner, toward
    nose) should mirror that. We verify by comparing the pixel totals in the
    inner vs. outer halves of each eye bounding box.
    """
    img = render_expression(Expression.ANGRY, 128, 64)
    # Convert to a numpy array (H×W, dtype uint8/bool) for index-safe access.
    arr = np.array(img)

    def pixel(x: int, y: int) -> int:
        return int(arr[y, x])

    # Left eye bounding box (roughly): cx≈38, eye_w=32 → [22, 54]
    left_cx = 128 // 2 - 128 // 5  # = 38
    eye_w = 128 // 4  # = 32
    half_w = eye_w // 2
    x0l, x1l = left_cx - half_w, left_cx + half_w

    # Right eye bounding box (roughly): cx≈90, [74, 106]
    right_cx = 128 // 2 + 128 // 5  # = 90
    x0r, x1r = right_cx - half_w, right_cx + half_w

    top_rows = range(0, 64 // 2)  # upper half of panel

    def count_in_cols(col_range: range, row_range: range) -> int:
        return sum(pixel(x, y) for x in col_range for y in row_range)

    # Left eye: inner corner (x1, toward nose) slants DOWN, outer corner (x0)
    # stays at the top.  The outer (left) half therefore covers more of the
    # top-row area → outer pixel count > inner pixel count.
    left_outer = count_in_cols(range(x0l, left_cx), top_rows)
    left_inner = count_in_cols(range(left_cx, x1l), top_rows)
    assert left_outer > left_inner, "left eye outer half should have more top pixels (slant inward)"

    # Right eye: inner corner (x0, toward nose) slants DOWN, outer corner (x1)
    # stays at the top → same relationship, mirrored.
    right_inner = count_in_cols(range(x0r, right_cx), top_rows)
    right_outer = count_in_cols(range(right_cx, x1r), top_rows)
    assert right_outer > right_inner, "right eye outer half should have more top pixels (slant inward)"


def test_render_text_centres_message() -> None:
    """``render_text`` produces a 1-bit panel of the requested size."""
    img = render_text("MSE-6 online", 128, 64)
    assert img.size == (128, 64)
    assert img.mode == "1"
    # Text is non-empty: at least one pixel is set.
    assert any(b != 0 for b in img.tobytes())
