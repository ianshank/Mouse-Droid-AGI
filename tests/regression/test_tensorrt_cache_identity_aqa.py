"""AQA: the engine cache names its runtime, and never unpickles a loose one.

Two defects in ``efficiency/tensorrt.py``, found together because each
concealed the other (peer review D-7 and D-25).

**D-7.** ``_model_fingerprint`` described only the *model* -- class name,
architecture string, input shapes, precision, parameter count. Nothing about
the runtime that produced the engine. ``compile_model`` treats a matching
``engine_<fingerprint>.pth`` as a cache hit, and ``docker-compose.jetson.yml``
bind-mounts the cache directory from the host, so an engine built under one
TensorRT survives a base-image bump, a JetPack upgrade or a GPU swap and is
loaded as current.

**D-25.** ``load_compiled``'s fallback reaches ``torch.load(weights_only=
False)`` -- arbitrary pickle, i.e. code execution on load -- behind a comment
asserting the cache directory "should have restricted permissions (0700)".
Nothing enforced that: ``mkdir`` passed no ``mode`` (executed: the directory
landed at ``0o755``), there was no ``chmod`` anywhere in ``src/``, and no
validator on the config field.

**Rated latent, not live, and the rating is the point.** Nothing constructs
``OptimizedInference``, the only caller of ``compile_model``, and nothing
calls ``build_tensorrt_compiler`` -- ``vulture`` reports both as unused. A
first draft of this work called D-25 a live RCE, which repeated the D-0
misrating from the first batch of this review: rating a mechanism without
tracing it to its consumers. The pin in the backwards-compat half is what
makes the rating revisitable the day someone wires the seam.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from mousedroid.efficiency.tensorrt import (
    UntrustedEngineCacheError,
    _model_fingerprint,
    _runtime_identity,
    cache_dir_is_private,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"
_TENSORRT = _SRC / "efficiency" / "tensorrt.py"

#: Any unpickling call. Matched loosely on purpose: the point is to catch a
#: NEW one appearing, whatever it is named or however it is spelled.
_UNPICKLE = re.compile(r"weights_only\s*=\s*False")


def _guarded_unpickle_sites(source: str) -> list[tuple[int, bool]]:
    """Locate ``weights_only=False`` calls and whether a privacy guard precedes.

    Structural rather than textual: an ``ast`` walk finds the enclosing
    function of each call and asks whether ``cache_dir_is_private`` is
    consulted anywhere in it. A substring gate would have been coupled to
    the current variable names -- the mistake the first version of this
    review's NaN-clamp gate made, which is why this one carries a self-test.

    Args:
        source: Module source text.

    Returns:
        ``(lineno, guarded)`` for every unpickling call found.
    """
    tree = ast.parse(source)
    results: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.get_source_segment(source, node) or ""
        if not _UNPICKLE.search(body):
            continue
        guarded = "cache_dir_is_private" in body
        results.append((node.lineno, guarded))
    return results


# -- the gate ---------------------------------------------------------------


def test_the_module_still_has_an_unpickling_call_to_guard() -> None:
    """A gate over zero sites is a gate that cannot fail.

    If the pickle path is ever removed outright, this test says so loudly
    rather than passing vacuously -- and the rest of this file should then be
    deleted, not left as decoration.
    """
    assert _guarded_unpickle_sites(_TENSORRT.read_text(encoding="utf-8"))


def test_every_unpickling_call_sits_behind_the_privacy_guard() -> None:
    source = _TENSORRT.read_text(encoding="utf-8")
    unguarded = [line for line, guarded in _guarded_unpickle_sites(source) if not guarded]
    assert not unguarded, (
        f"weights_only=False reached without a cache_dir_is_private check "
        f"at line(s) {unguarded} of {_TENSORRT}"
    )


def test_the_gate_flags_a_known_bad_sample() -> None:
    """Self-test. The gate's predecessor in this review matched literal
    substrings tied to two files' variable names and flagged neither of its
    own synthetic samples. A gate that has never been shown to fail is a
    comment."""
    bad = "def _load():\n    return torch.load(p, weights_only=False)\n"
    assert _guarded_unpickle_sites(bad) == [(1, False)]


def test_the_gate_accepts_a_known_good_sample() -> None:
    """...and is not simply always-flag."""
    good = (
        "def _load():\n"
        "    if not cache_dir_is_private(p.parent):\n"
        "        raise UntrustedEngineCacheError(p)\n"
        "    return torch.load(p, weights_only=False)\n"
    )
    assert _guarded_unpickle_sites(good) == [(1, True)]


def test_no_chmod_free_mkdir_of_the_cache_remains() -> None:
    """``mkdir(mode=...)`` is masked by the umask, so mode alone is not enough.

    This is the exact shape of the original defect: a permission intent
    expressed in a way that does not take effect.
    """
    source = _TENSORRT.read_text(encoding="utf-8")
    assert "os.chmod" in source, "mkdir(mode=) alone is umask-masked; chmod is required"


# -- the runtime identity ---------------------------------------------------


def test_identity_is_ordered_and_deterministic() -> None:
    assert _runtime_identity() == _runtime_identity()


def test_identity_covers_every_documented_component() -> None:
    """Named rather than counted, so dropping one names itself on failure."""
    labels = {component.split("=", 1)[0] for component in _runtime_identity()}
    assert labels == {"torch", "cuda", "tensorrt", "torch2trt", "compute"}


def test_identity_reaches_the_fingerprint() -> None:
    """Computing the tuple but forgetting to hash it is the silent failure."""
    import torch.nn as nn

    model = nn.Linear(4, 2)
    shapes = {"x": [1, 4]}
    baseline = _model_fingerprint(model, shapes, "fp16")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("mousedroid.efficiency.tensorrt._runtime_identity", lambda: ("torch=0",))
        assert _model_fingerprint(model, shapes, "fp16") != baseline


# -- the exception ----------------------------------------------------------


def test_the_error_is_importable_from_the_package_facade() -> None:
    """A caller cannot catch what it cannot import."""
    from mousedroid.efficiency import UntrustedEngineCacheError as Exported

    assert Exported is UntrustedEngineCacheError


def test_the_error_is_catchable_as_runtime_error() -> None:
    """Existing ``except RuntimeError`` handlers keep working (invariant 6)."""
    assert issubclass(UntrustedEngineCacheError, RuntimeError)


def test_no_fail_open_knob_exists() -> None:
    """A configurable permission would only be a way to configure this back
    open -- the same reasoning as ``EmergencyLatchConfig``'s absent fail-open
    knob and ``LidarScan.sensor_responding`` being a bool rather than a
    threshold.

    Matched against a named list rather than the substring ``"mode"``: the
    first version of this assertion flagged ``JetsonConfig.power_mode``, which
    is the Jetson's 15W/7W power profile and has nothing to do with file
    permissions. A gate coupled to an incidental word is the same mistake as
    the NaN-clamp gate earlier in this review.
    """
    from mousedroid.config.schema import JetsonConfig

    forbidden = {
        "cache_mode",
        "cache_dir_mode",
        "tensorrt_cache_mode",
        "cache_permissions",
        "umask",
        "allow_world_readable_cache",
        "allow_untrusted_cache",
        "fail_open",
    }
    assert not (forbidden & set(JetsonConfig.model_fields))


def test_the_privacy_predicate_is_not_always_true(tmp_path: Path) -> None:
    """Guards against a refactor that makes the check vacuous."""
    import os
    import stat
    import sys

    if sys.platform == "win32":  # pragma: no cover - POSIX-only assertion
        pytest.skip("POSIX permission bits")
    loose = tmp_path / "loose"
    loose.mkdir()
    # Group-READ only (0o740). Enough to fail the privacy check without
    # ruff S103 flagging a genuinely permissive mask in a test fixture.
    os.chmod(loose, stat.S_IRWXU | stat.S_IRGRP)
    assert cache_dir_is_private(loose) is False
