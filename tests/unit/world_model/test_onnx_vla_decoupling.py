"""Import-neutrality test: ``world_model.dual_stream_rssm_onnx`` is VLA-independent.

PR #140 extracted the shared ONNX session lifecycle into the neutral
:mod:`mousedroid.common.onnx_session` but deliberately left the
``DEFAULT_ORT_PROVIDERS`` constant in :mod:`mousedroid.vla.policy`, so the
world-model ONNX runtime still imported the ``vla`` package for that one
default. The follow-up relocated the constant to ``common/onnx_session``;
this test pins the resulting decoupling so a future edit that re-introduces a
``world_model -> vla`` import fails loudly.

Mirrors
``tests/unit/common/test_onnx_session.py::test_module_is_neutral_and_lazy``:
the import is exercised in a fresh subprocess so a real / stub ``vla`` module
left in the parent ``sys.modules`` by an earlier test cannot mask a
regression, and ``PYTHONPATH`` is propagated so the spawned interpreter
resolves ``mousedroid`` (pytest's ``pythonpath`` does not survive the
subprocess boundary). The check is deliberately **not** gated behind
``onnxruntime`` — the whole point is that the runtime module imports cleanly,
and VLA-free, on a host without ORT installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_dual_stream_rssm_onnx_does_not_import_vla() -> None:
    """Importing the world-model ONNX runtime must not pull in ``mousedroid.vla``.

    The runtime sources its ``DEFAULT_ORT_PROVIDERS`` default from the neutral
    :mod:`mousedroid.common.onnx_session` helper (not ``vla.policy``), so the
    ``world_model`` package is fully independent of the ``vla`` package.
    ``onnxruntime`` stays lazy (imported only inside ``warmup``), so it must be
    absent from ``sys.modules`` after a bare module import too.
    """
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    extra = os.pathsep.join([str(repo / "src"), str(repo)])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extra + (os.pathsep + existing if existing else "")

    script = textwrap.dedent(
        """
        import sys
        import mousedroid.world_model.dual_stream_rssm_onnx  # noqa: F401
        for mod in ('mousedroid.vla', 'mousedroid.vla.policy', 'onnxruntime'):
            assert mod not in sys.modules, (
                'mousedroid.world_model.dual_stream_rssm_onnx must not import '
                f'{mod} at module load'
            )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert (
        result.returncode == 0
    ), f"vla-decoupling check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
