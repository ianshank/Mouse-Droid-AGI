"""Regression: pure validation modules stay decoupled from the sensor stack.

``mousedroid.validation.latency_stats`` and ``report_store`` are pure helpers.
They must import without dragging in the heavy sensor-runtime dependencies
(``numpy``/``cv2``/``pyaudio``) that ``mousedroid.validation.runtime`` pulls.
The package re-exports those runtime symbols **lazily** (:pep:`562`), so:

1. Importing a pure module must NOT import numpy/cv2 (import-cost + modularity).
2. The lazy re-exports must STILL resolve (backwards compatibility — existing
   ``from mousedroid.validation import capture_camera_frame`` callers).

Run in a subprocess so the assertion sees a clean import graph regardless of
what the surrounding pytest session already imported.
"""

from __future__ import annotations

import subprocess
import sys


def _run_snippet(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_importing_pure_modules_does_not_import_numpy_or_cv2() -> None:
    code = (
        "import sys\n"
        "import mousedroid.validation.latency_stats\n"
        "import mousedroid.validation.report_store\n"
        "assert 'numpy' not in sys.modules, 'latency_stats/report_store pulled numpy'\n"
        "assert 'cv2' not in sys.modules, 'latency_stats/report_store pulled cv2'\n"
        "print('OK')\n"
    )
    result = _run_snippet(code)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_lazy_runtime_reexports_still_resolve() -> None:
    code = (
        "from mousedroid.validation import (\n"
        "    capture_camera_frame,\n"
        "    resolve_runtime_config_paths,\n"
        "    LidarScanDiagnostics,\n"
        ")\n"
        "assert callable(capture_camera_frame)\n"
        "assert callable(resolve_runtime_config_paths)\n"
        "assert LidarScanDiagnostics is not None\n"
        "print('OK')\n"
    )
    result = _run_snippet(code)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_unknown_attribute_still_raises_attribute_error() -> None:
    code = (
        "import mousedroid.validation as v\n"
        "try:\n"
        "    v.does_not_exist\n"
        "except AttributeError:\n"
        "    print('OK')\n"
        "else:\n"
        "    raise SystemExit('expected AttributeError')\n"
    )
    result = _run_snippet(code)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
