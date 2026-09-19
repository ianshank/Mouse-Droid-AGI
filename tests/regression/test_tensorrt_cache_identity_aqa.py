"""AQA: the engine cache names its runtime, and never unpickles a loose one.

Two defects in ``efficiency/tensorrt.py``, found together because each
concealed the other (peer review D-7 and D-25).

**D-7.** ``_model_fingerprint`` described only the *model* -- class name,
architecture string, input shapes, precision, parameter count. Nothing about
the runtime that produced the engine. ``compile_model`` treats a matching
``engine_<fingerprint>.pth`` as a cache hit, and ``docker-compose.jetson.yml``
mounts the cache directory from outside the image (the named volume
``mousedroid_tensorrt_cache`` since F-051), so an engine built under one
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

#: The guard whose presence makes an unpickling call defensible.
_GUARD_NAME = "cache_dir_is_private"

#: The keyword that turns a load into arbitrary pickle deserialization.
#: Matched on the keyword rather than on ``torch.load`` so that renaming or
#: re-exporting the callee does not evade the gate.
_UNPICKLE_KWARG = "weights_only"

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _is_unpickling_call(node: ast.AST) -> bool:
    """Whether ``node`` is a call passing ``weights_only=False``."""
    if not isinstance(node, ast.Call):
        return False
    return any(
        keyword.arg == _UNPICKLE_KWARG
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in node.keywords
    )


def _calls_the_guard(node: ast.AST) -> bool:
    """Whether ``node``'s subtree contains a real CALL to the guard.

    A call, not a mention. ``cache_dir_is_private`` appearing in a comment,
    a docstring, or a bare name reference does not protect anything.
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name == _GUARD_NAME:
            return True
    return False


def _guarded_unpickle_sites(source: str) -> list[tuple[int, bool]]:
    """Locate ``weights_only=False`` calls and whether a privacy guard applies.

    Fully structural. Both halves are read off the AST:

    * the unpickling site is an :class:`ast.Call` carrying
      ``weights_only=False`` as a literal keyword, so the text
      ``weights_only=False`` inside a comment or a string is *not* a site;
    * the guard is an :class:`ast.Call` to ``cache_dir_is_private`` in an
      enclosing function, so the *name* appearing in a comment or docstring
      does not satisfy it.

    That second point is the whole reason this helper was rewritten. Its
    first version walked the AST only to find enclosing functions and then
    did ``"cache_dir_is_private" in source_segment`` -- so a guard deleted
    but *mentioned in a comment* passed the gate. In a change whose entire
    thesis is that a security claim written in a comment is not a control
    (peer review D-21/D-25), a gate satisfiable by a comment was the same
    defect one level up. Caught in review on #234; the self-tests below now
    pin both directions of it.

    The line reported is the **call site**, not the enclosing ``def``, so a
    failure points at the code to fix.

    Args:
        source: Module source text.

    Returns:
        ``(lineno, guarded)`` for every unpickling call found.
    """
    tree = ast.parse(source)

    # Nearest-enclosing-function chain for every node, so a guard in an outer
    # function still covers a call in a nested one (``load_compiled`` guards
    # the call inside its own ``_load_sync``).
    enclosing: dict[ast.AST, list[_FunctionNode]] = {tree: []}
    for parent in ast.walk(tree):
        chain = enclosing[parent]
        for child in ast.iter_child_nodes(parent):
            enclosing[child] = [child, *chain] if isinstance(child, _FunctionNode) else chain

    results: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if not _is_unpickling_call(node):
            continue
        guarded = any(_calls_the_guard(fn) for fn in enclosing.get(node, []))
        results.append((node.lineno, guarded))
    return sorted(results)


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
    assert _guarded_unpickle_sites(bad) == [(2, False)]


def test_the_gate_accepts_a_known_good_sample() -> None:
    """...and is not simply always-flag."""
    good = (
        "def _load():\n"
        "    if not cache_dir_is_private(p.parent):\n"
        "        raise UntrustedEngineCacheError(p)\n"
        "    return torch.load(p, weights_only=False)\n"
    )
    assert _guarded_unpickle_sites(good) == [(4, True)]


def test_a_guard_that_exists_only_in_a_comment_does_not_satisfy_the_gate() -> None:
    """The regression this gate's first version shipped with.

    It did ``"cache_dir_is_private" in source_segment``, so deleting the call
    and leaving the comment passed. In a change arguing that a security claim
    in a comment is not a control, that was the same defect one level up.
    Raised by review on #234.
    """
    commented_out = (
        "def _load():\n"
        "    # cache_dir_is_private(p.parent) used to be checked here\n"
        "    return torch.load(p, weights_only=False)\n"
    )
    assert _guarded_unpickle_sites(commented_out) == [(3, False)]


def test_a_docstring_mentioning_the_guard_does_not_satisfy_the_gate() -> None:
    """The same false positive, in the form this module would really take."""
    documented = (
        "def _load():\n"
        '    """Loads only when cache_dir_is_private(path) holds."""\n'
        "    return torch.load(p, weights_only=False)\n"
    )
    assert _guarded_unpickle_sites(documented) == [(3, False)]


def test_the_keyword_in_a_comment_is_not_a_call_site() -> None:
    """The other direction: text is not a call, so it must not be flagged."""
    mentioned = "def _load():\n    # never pass weights_only=False here\n    return 1\n"
    assert _guarded_unpickle_sites(mentioned) == []


def test_a_guard_in_an_enclosing_function_still_counts() -> None:
    """``load_compiled`` guards the call inside its own nested ``_load_sync``."""
    nested = (
        "def outer(path):\n"
        "    if not cache_dir_is_private(path.parent):\n"
        "        raise UntrustedEngineCacheError(path)\n"
        "\n"
        "    def inner():\n"
        "        return torch.load(path, weights_only=False)\n"
        "\n"
        "    return inner()\n"
    )
    assert _guarded_unpickle_sites(nested) == [(6, True)]


def test_the_reported_line_is_the_call_site_not_the_def() -> None:
    """A failure must point at the code to fix."""
    offset = "import os\n\n\ndef _load():\n    return torch.load(p, weights_only=False)\n"
    assert _guarded_unpickle_sites(offset) == [(5, False)]


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
