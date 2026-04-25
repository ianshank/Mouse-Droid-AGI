"""Pure expression renderer for the MSE-6 face display.

This module is intentionally I/O-free: it only renders 1-bit
:class:`PIL.Image.Image` framebuffers from an :class:`Expression` enum and
configurable dimensions. Real and mock drivers both consume these images,
which lets unit tests cover the entire rendering path without a physical
panel.

The numeric constants below describe the *artwork* (arc angles, stroke
widths, polygon proportions). They are deliberately not exposed in
``FaceDisplayConfig`` because they are not runtime-tunable — they
parameterise the drawn shapes, not behaviour. ``FaceDisplayConfig`` owns
every behaviour-affecting threshold (panel size, blink timing, expression
mapping cut-offs, etc.).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# --- Artwork geometry constants (pixel-art parameters, not runtime tunables) -
DEFAULT_HALF_DIVISOR: int = 2  # used to halve dimensions when centring
DEFAULT_EYE_PUPIL_HEIGHT_FACTOR: int = 2  # angry pupil reaches twice its radius below cy
DEFAULT_PUPIL_DIVISOR: int = 8
DEFAULT_PUPIL_MIN_RADIUS_PX: int = 2
DEFAULT_CLOSED_INSET_PX: int = 2
DEFAULT_CLOSED_STROKE_PX: int = 2
DEFAULT_HAPPY_ARC_START_DEG: int = 200
DEFAULT_HAPPY_ARC_END_DEG: int = 340
DEFAULT_SAD_ARC_START_DEG: int = 20
DEFAULT_SAD_ARC_END_DEG: int = 160
DEFAULT_CURVE_STROKE_PX: int = 3
DEFAULT_ANGRY_SLANT_DIVISOR: int = 3
DEFAULT_ANGRY_SLANT_MIN_PX: int = 4
DEFAULT_WIDE_PAD_DIVISOR: int = 16
DEFAULT_WIDE_PAD_MIN_PX: int = 2
DEFAULT_SLEEPY_ARC_START_DEG: int = 0
DEFAULT_SLEEPY_ARC_END_DEG: int = 180
DEFAULT_SLEEPY_STROKE_PX: int = 2
DEFAULT_CROSS_STROKE_PX: int = 2
DEFAULT_LABEL_PADDING_PX: int = 2
DEFAULT_EYE_WIDTH_DIVISOR: int = 4
DEFAULT_EYE_OFFSET_DIVISOR: int = 5


class Expression(str, Enum):
    """Discrete facial expressions mapped from affect + safety state.

    The string-valued enum keeps log lines and YAML config readable.
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ALERT = "alert"
    SLEEPY = "sleepy"
    BLINK = "blink"
    EMERGENCY = "emergency"
    BOOT = "boot"


_DEFAULT_FONT: object | None = None


def _get_default_font() -> object:
    """Return a cached PIL default font, loading it lazily on first use.

    Avoids re-parsing the bitmap font data on every text render. The cache
    is module-global; PIL's default font is immutable.
    """
    global _DEFAULT_FONT
    if _DEFAULT_FONT is None:
        from PIL import ImageFont

        _DEFAULT_FONT = ImageFont.load_default()
    return _DEFAULT_FONT


def _new_canvas(width: int, height: int) -> tuple[PILImage, object]:
    """Create a 1-bit PIL canvas + ImageDraw, lazy-importing PIL."""
    from PIL import Image, ImageDraw

    img = Image.new("1", (width, height), 0)
    return img, ImageDraw.Draw(img)


def _eye_geometry(width: int, height: int) -> tuple[int, int, int, int, int]:
    """Return ``(left_cx, right_cx, cy, eye_w, eye_h)`` for the panel."""
    eye_w = width // DEFAULT_EYE_WIDTH_DIVISOR
    eye_h = height // DEFAULT_HALF_DIVISOR
    cy = height // DEFAULT_HALF_DIVISOR
    eye_offset = width // DEFAULT_EYE_OFFSET_DIVISOR
    left_cx = width // DEFAULT_HALF_DIVISOR - eye_offset
    right_cx = width // DEFAULT_HALF_DIVISOR + eye_offset
    return left_cx, right_cx, cy, eye_w, eye_h


def _draw_eye(
    draw: object,
    cx: int,
    cy: int,
    eye_w: int,
    eye_h: int,
    shape: str,
    *,
    side: str = "left",
) -> None:
    """Render a single eye in the given shape inside its bounding box.

    Args:
        draw: PIL ImageDraw instance.
        cx: Centre x of the eye bounding box.
        cy: Centre y of the eye bounding box.
        eye_w: Bounding-box width.
        eye_h: Bounding-box height.
        shape: Shape key from ``_SHAPE_BY_EXPR``.
        side: ``"left"`` or ``"right"`` — used only by asymmetric shapes
            (e.g. ``"angry"``) to mirror the slant toward the nose.
    """
    half_w = eye_w // DEFAULT_HALF_DIVISOR
    half_h = eye_h // DEFAULT_HALF_DIVISOR
    x0, y0 = cx - half_w, cy - half_h
    x1, y1 = cx + half_w, cy + half_h
    pupil_r = max(eye_h // DEFAULT_PUPIL_DIVISOR, DEFAULT_PUPIL_MIN_RADIUS_PX)

    if shape == "closed":
        draw.line(  # type: ignore[attr-defined]
            (x0 + DEFAULT_CLOSED_INSET_PX, cy, x1 - DEFAULT_CLOSED_INSET_PX, cy),
            fill=1,
            width=DEFAULT_CLOSED_STROKE_PX,
        )
        return
    if shape == "round":
        draw.ellipse((x0, y0, x1, y1), outline=1, fill=1)  # type: ignore[attr-defined]
        draw.ellipse(  # type: ignore[attr-defined]
            (cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r),
            fill=0,
        )
        return
    if shape == "happy":
        # Upward-curved arc (smiling eyes).
        draw.arc(  # type: ignore[attr-defined]
            (x0, y0, x1, y1),
            start=DEFAULT_HAPPY_ARC_START_DEG,
            end=DEFAULT_HAPPY_ARC_END_DEG,
            fill=1,
            width=DEFAULT_CURVE_STROKE_PX,
        )
        return
    if shape == "sad":
        # Downward-curved arc.
        draw.arc(  # type: ignore[attr-defined]
            (x0, y0, x1, y1),
            start=DEFAULT_SAD_ARC_START_DEG,
            end=DEFAULT_SAD_ARC_END_DEG,
            fill=1,
            width=DEFAULT_CURVE_STROKE_PX,
        )
        return
    if shape == "angry":
        # Filled trapezoid with the *inner* corner (nearest the nose)
        # slanted down and the outer corner raised, mirrored per side.
        slant = max(eye_h // DEFAULT_ANGRY_SLANT_DIVISOR, DEFAULT_ANGRY_SLANT_MIN_PX)
        if side == "left":
            # Left eye: inner corner is x1 (right side, toward centre).
            pts = [(x0, y0), (x1, y0 + slant), (x1, y1), (x0, y1)]
        else:
            # Right eye: inner corner is x0 (left side, toward centre).
            pts = [(x0, y0 + slant), (x1, y0), (x1, y1), (x0, y1)]
        draw.polygon(pts, outline=1, fill=1)  # type: ignore[attr-defined]
        draw.ellipse(  # type: ignore[attr-defined]
            (
                cx - pupil_r,
                cy,
                cx + pupil_r,
                cy + pupil_r * DEFAULT_EYE_PUPIL_HEIGHT_FACTOR,
            ),
            fill=0,
        )
        return
    if shape == "wide":
        pad = max(eye_h // DEFAULT_WIDE_PAD_DIVISOR, DEFAULT_WIDE_PAD_MIN_PX)
        draw.ellipse(  # type: ignore[attr-defined]
            (x0 - pad, y0 - pad, x1 + pad, y1 + pad),
            outline=1,
            fill=1,
        )
        draw.ellipse(  # type: ignore[attr-defined]
            (cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r),
            fill=0,
        )
        return
    if shape == "sleepy":
        draw.arc(  # type: ignore[attr-defined]
            (x0, y0, x1, y1),
            start=DEFAULT_SLEEPY_ARC_START_DEG,
            end=DEFAULT_SLEEPY_ARC_END_DEG,
            fill=1,
            width=DEFAULT_SLEEPY_STROKE_PX,
        )
        return
    if shape == "cross":
        draw.line((x0, y0, x1, y1), fill=1, width=DEFAULT_CROSS_STROKE_PX)  # type: ignore[attr-defined]
        draw.line((x0, y1, x1, y0), fill=1, width=DEFAULT_CROSS_STROKE_PX)  # type: ignore[attr-defined]
        return
    # Fallback: open round eye.
    draw.ellipse((x0, y0, x1, y1), outline=1, fill=1)  # type: ignore[attr-defined]


_SHAPE_BY_EXPR: dict[Expression, str] = {
    Expression.NEUTRAL: "round",
    Expression.HAPPY: "happy",
    Expression.SAD: "sad",
    Expression.ANGRY: "angry",
    Expression.ALERT: "wide",
    Expression.SLEEPY: "sleepy",
    Expression.BLINK: "closed",
    Expression.EMERGENCY: "cross",
    Expression.BOOT: "round",
}


def render_expression(expression: Expression, width: int, height: int) -> PILImage:
    """Render an expression to a fresh 1-bit framebuffer.

    Args:
        expression: Target expression to render.
        width: Panel width in pixels (must match ``FaceDisplayConfig.width``).
        height: Panel height in pixels (must match ``FaceDisplayConfig.height``).

    Returns:
        A 1-bit-mode :class:`PIL.Image.Image` of size ``(width, height)``.
    """
    img, draw = _new_canvas(width, height)
    left_cx, right_cx, cy, eye_w, eye_h = _eye_geometry(width, height)
    shape = _SHAPE_BY_EXPR[expression]
    _draw_eye(draw, left_cx, cy, eye_w, eye_h, shape, side="left")
    _draw_eye(draw, right_cx, cy, eye_w, eye_h, shape, side="right")

    if expression in {Expression.EMERGENCY, Expression.BOOT}:
        label = "EMERGENCY" if expression is Expression.EMERGENCY else "BOOT"
        font = _get_default_font()
        bbox = draw.textbbox((0, 0), label, font=font)  # type: ignore[attr-defined]
        text_w = bbox[2] - bbox[0]
        draw.text(  # type: ignore[attr-defined]
            (
                (width - text_w) // DEFAULT_HALF_DIVISOR,
                DEFAULT_LABEL_PADDING_PX,
            ),
            label,
            font=font,
            fill=1,
        )

    return img


def render_text(message: str, width: int, height: int) -> PILImage:
    """Render a short single-line status message centred on the panel.

    Args:
        message: Text to render (truncated by the panel size).
        width: Panel width in pixels.
        height: Panel height in pixels.

    Returns:
        A 1-bit-mode :class:`PIL.Image.Image`.
    """
    img, draw = _new_canvas(width, height)
    font = _get_default_font()
    bbox = draw.textbbox((0, 0), message, font=font)  # type: ignore[attr-defined]
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max((width - text_w) // DEFAULT_HALF_DIVISOR, 0)
    y = max((height - text_h) // DEFAULT_HALF_DIVISOR, 0)
    draw.text((x, y), message, font=font, fill=1)  # type: ignore[attr-defined]
    return img
