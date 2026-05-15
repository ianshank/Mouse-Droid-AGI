"""P12b — Operator-workstation dashboard canvas-diff smoke test.

Runs Playwright (headless Chromium) from the operator workstation against
a live rover or a port-forwarded URL. Opens ``/lidar`` and ``/camera``,
waits ``settle_s`` for the canvas to populate, then captures two
screenshots ``capture_gap_s`` apart and asserts the per-pixel diff
percentage exceeds ``min_canvas_diff_pct``. This proves the canvas is
actually redrawing from live data, not rendering a static placeholder.

Why this test does NOT run on the Jetson:
    Installing Chromium on the Orin Nano adds ~150 MB to the container
    or host for a once-per-deploy check. The dashboard is intended for
    laptops on the same Wi-Fi; testing it from the operator workstation
    matches actual usage. The on-Jetson :mod:`scripts.jetson_probe_dashboard_e2e`
    probe gives us the rover-side data-flow guarantee at a fraction of
    the cost.

Why this test is decoupled from the Jetson conftest:
    The Jetson conftest in ``tests/e2e/conftest.py`` autouse-fixtures
    ``MOUSEDROID_MOCK_HARDWARE=false`` and skips on non-Jetson hosts.
    That's the wrong contract for canvas-diff: it should run on the
    operator workstation (not the Jetson) against the rover URL. We
    therefore override the autouse fixture locally.

Skip conditions:
    * ``--no-playwright`` CLI flag → SKIP with explicit reason
      (CI environments without Chromium installed).
    * ``MOUSEDROID_DASHBOARD_URL`` env var unset → SKIP with explicit
      reason (test target unknown).
    * ``playwright.sync_api`` not importable → SKIP.

Environment:
    MOUSEDROID_DASHBOARD_URL          base URL, e.g. http://mousedroid-telemetry.local:8080
    MOUSEDROID_DASHBOARD_TOKEN        bearer token (sent as ``?token=...`` query param)
    MOUSEDROID_DASHBOARD_SETTLE_S     wait after page load before first capture (default 2.0)
    MOUSEDROID_DASHBOARD_GAP_S        gap between captures (default 2.0)
    MOUSEDROID_DASHBOARD_MIN_DIFF_PCT min % pixel diff threshold (default 1.0)
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page


# Override the Jetson conftest's autouse fixture for this module ONLY.
# The canvas-diff test runs on the OPERATOR WORKSTATION, not the Jetson,
# so it must NOT clobber MOUSEDROID_MOCK_HARDWARE or skip on non-Jetson hosts.
@pytest.fixture(autouse=True)
def _no_jetson_env_override() -> None:
    """No-op replacement for the parent autouse env fixture."""
    return None


def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover - hook
    """Register the ``--no-playwright`` flag for CI environments.

    Args:
        parser: The pytest CLI parser.
    """
    parser.addoption(
        "--no-playwright",
        action="store_true",
        default=False,
        help="Skip Playwright-dependent tests (operator workstation CI fallback).",
    )


@pytest.fixture
def dashboard_url() -> str:
    """Return the operator-configured dashboard base URL or skip."""
    url = os.environ.get("MOUSEDROID_DASHBOARD_URL", "").strip()
    if not url:
        pytest.skip(
            "MOUSEDROID_DASHBOARD_URL unset; set to e.g. "
            "http://mousedroid-telemetry.local:8080 to run the canvas-diff probe."
        )
    return url.rstrip("/")


@pytest.fixture
def dashboard_token() -> str | None:
    """Return the bearer token, if any."""
    token = os.environ.get("MOUSEDROID_DASHBOARD_TOKEN", "").strip()
    return token or None


def _maybe_skip_no_playwright(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("--no-playwright"):
        pytest.skip("Playwright disabled via --no-playwright flag")
    pytest.importorskip(
        "playwright.sync_api",
        reason=(
            "playwright not installed; run "
            "`pip install playwright && playwright install chromium` to enable"
        ),
    )


def _settle_seconds() -> float:
    return float(os.environ.get("MOUSEDROID_DASHBOARD_SETTLE_S", "2.0"))


def _gap_seconds() -> float:
    return float(os.environ.get("MOUSEDROID_DASHBOARD_GAP_S", "2.0"))


def _min_diff_pct() -> float:
    return float(os.environ.get("MOUSEDROID_DASHBOARD_MIN_DIFF_PCT", "1.0"))


def _build_page_url(base: str, path: str, token: str | None) -> str:
    if token:
        return f"{base}{path}?token={token}"
    return f"{base}{path}"


def _capture_canvas_png(page: Page, canvas_selector: str) -> bytes:
    """Take a screenshot of the canvas element only.

    Args:
        page: The Playwright page after navigation + settle.
        canvas_selector: CSS selector for the canvas element.

    Returns:
        Raw PNG bytes of the canvas.
    """
    element = page.locator(canvas_selector).first
    element.wait_for(state="visible", timeout=10_000)
    return element.screenshot(type="png")


def _diff_pct(png_a: bytes, png_b: bytes) -> float:
    """Compute the per-pixel difference percentage between two PNGs.

    Uses Pillow for cross-platform pixel access. The comparison is
    deliberately coarse (any pixel difference counts) — we are testing
    "did the canvas redraw at all", not "did it draw the right thing".

    Args:
        png_a: First capture.
        png_b: Second capture.

    Returns:
        Percentage of differing pixels in ``[0.0, 100.0]``.
    """
    # PIL submodule names are CapWords by convention; ruff N806 would
    # normally flag these local names, but renaming would obscure the
    # well-known PIL idiom.
    Image = pytest.importorskip(  # noqa: N806
        "PIL.Image", reason="Pillow not installed; pip install pillow"
    )
    ImageChops = pytest.importorskip("PIL.ImageChops")  # noqa: N806

    img_a = Image.open(io.BytesIO(png_a)).convert("RGBA")
    img_b = Image.open(io.BytesIO(png_b)).convert("RGBA")
    if img_a.size != img_b.size:
        # Window resized between captures — treat as 100% diff so the
        # test does not silently pass on a corner case we cannot
        # disambiguate.
        return 100.0
    diff = ImageChops.difference(img_a, img_b)
    bbox = diff.getbbox()
    if bbox is None:
        return 0.0
    total_pixels = img_a.size[0] * img_a.size[1]
    # Convert the RGBA difference to grayscale ('L') so a single
    # histogram lookup tells us how many pixels differ in ANY channel.
    # The prior per-band ``max(... - hist[0])`` was a lower bound that
    # undercounted when channel-differences were spread across pixels
    # (Gemini review on PR #83).
    diff_l = diff.convert("L")
    hist = diff_l.histogram()
    changed = total_pixels - hist[0]
    return float(100.0 * changed / total_pixels)


@pytest.mark.parametrize(
    ("page_path", "canvas_selector"),
    [
        ("/lidar", "canvas"),
        ("/camera", "canvas"),
    ],
)
def test_dashboard_canvas_redraws(
    request: pytest.FixtureRequest,
    dashboard_url: str,
    dashboard_token: str | None,
    tmp_path: Path,
    page_path: str,
    canvas_selector: str,
) -> None:
    """Each dashboard canvas must redraw between two captures 2s apart.

    Args:
        request: pytest fixture-request handle, used for ``--no-playwright``.
        dashboard_url: Base URL of the deployed telemetry server.
        dashboard_token: Bearer token (passed as ``?token=`` query param).
        tmp_path: pytest-provided temp dir for capture artefacts.
        page_path: ``/lidar`` or ``/camera``.
        canvas_selector: CSS selector inside the page for the redrawing canvas.
    """
    _maybe_skip_no_playwright(request)
    from playwright.sync_api import sync_playwright

    url = _build_page_url(dashboard_url, page_path, dashboard_token)
    settle = _settle_seconds()
    gap = _gap_seconds()
    threshold = _min_diff_pct()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            # Wait for any canvas to appear AND for the first paint window.
            page.locator(canvas_selector).first.wait_for(state="visible", timeout=10_000)
            page.wait_for_timeout(int(settle * 1000))

            png_a = _capture_canvas_png(page, canvas_selector)
            (tmp_path / f"{page_path.strip('/')}_a.png").write_bytes(png_a)

            page.wait_for_timeout(int(gap * 1000))

            png_b = _capture_canvas_png(page, canvas_selector)
            (tmp_path / f"{page_path.strip('/')}_b.png").write_bytes(png_b)
        finally:
            browser.close()

    pct = _diff_pct(png_a, png_b)
    assert pct >= threshold, (
        f"{page_path}: canvas diff {pct:.3f}% < threshold {threshold:.3f}% "
        f"(gap={gap}s, settle={settle}s). Canvas appears static — either the "
        "WS stream is silent or the dashboard JS is not redrawing. "
        f"Artefacts in {tmp_path}."
    )
