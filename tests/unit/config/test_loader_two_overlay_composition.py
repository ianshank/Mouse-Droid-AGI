"""F-006 remote-LLM sprint: ``load_settings(overlay_a, overlay_b)`` deep-merge tests.

The architect peer-review on the F-006 plan flagged that ``_deep_merge`` in
``src/mousedroid/config/loader.py:23-38`` was exercised only transitively —
``tests/unit/config/`` contained no ``test_loader*.py`` covering the
multi-overlay path. The F-006 remote-LLM sprint depends on
``load_settings(jetson_production.yaml, jetson_production_remote_llm.yaml)``
correctly composing a partial second overlay onto a full first overlay
(the second only sets ``llm.backend`` / ``base_url`` / ``model_name`` /
``request_timeout_s`` / ``latency_target_ms`` and MUST preserve unrelated
``llm.*`` fields from the first like ``llm.n_gpu_layers``).

These tests pin the composition contract so a future refactor of
``_deep_merge`` or the nested-env-strip cannot silently break it.

Architecture invariants exercised:

* Deep-merge is recursive (nested dicts merge field-by-field, not whole-block replace).
* Last overlay wins on shared leaf keys.
* Nested env precedence (PR #101 commit ``d4dab14``) still beats both YAML overlays.
* Empty overlay-path list is equivalent to no overlays.
* Overlay order is significant (left-to-right reduce).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings


def _write_yaml(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_partial_overlay_preserves_unrelated_fields_in_same_section(
    tmp_path: Path,
) -> None:
    """A partial ``llm.*`` overlay must NOT nuke unrelated ``llm.*`` fields.

    Pins the deep-merge contract that the F-006 remote-LLM design relies on:
    operator drops ``jetson_production_remote_llm.yaml`` setting only
    ``llm.backend`` + ``llm.base_url`` on top of ``jetson_production.yaml``
    which sets the full ``llm`` block including ``llm.n_gpu_layers: -1``.
    The merged ``llm.n_gpu_layers`` MUST come from the first overlay.
    """
    overlay_a = _write_yaml(
        tmp_path,
        "overlay_a.yaml",
        """
mock_hardware: true
llm:
  enabled: true
  model_path: /opt/mousedroid/models/Phi-3-mini-q4.gguf
  n_gpu_layers: -1
  latency_target_ms: 500.0
  backend: llama_cpp
""",
    )
    overlay_b = _write_yaml(
        tmp_path,
        "overlay_b.yaml",
        """
llm:
  backend: openai_compatible
  base_url: http://192.168.55.100:11434
""",
    )

    cfg = load_settings(overlay_a, overlay_b)

    # Second overlay wins for shared keys.
    assert cfg.llm.backend == "openai_compatible"
    assert cfg.llm.base_url == "http://192.168.55.100:11434"
    # First overlay's unrelated keys survive — this is the core regression net.
    assert cfg.llm.n_gpu_layers == -1
    assert cfg.llm.latency_target_ms == 500.0


def test_second_overlay_overrides_first_for_shared_keys(tmp_path: Path) -> None:
    """When both overlays set the same leaf, the LAST one wins."""
    overlay_a = _write_yaml(
        tmp_path,
        "a.yaml",
        """
mock_hardware: true
llm:
  enabled: true
  backend: llama_cpp
""",
    )
    overlay_b = _write_yaml(
        tmp_path,
        "b.yaml",
        """
llm:
  backend: openai_compatible
""",
    )

    cfg = load_settings(overlay_a, overlay_b)
    assert cfg.llm.backend == "openai_compatible"


def test_nested_env_precedence_wins_over_both_overlays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MOUSEDROID_LLM__BACKEND=…`` beats every YAML overlay.

    Regression for PR #101 commit d4dab14 — the nested-env-strip
    extension. Without it, pydantic-settings would silently drop the
    env value because the YAML-merged init kwargs took priority.
    """
    overlay_a = _write_yaml(
        tmp_path,
        "a.yaml",
        """
mock_hardware: true
llm:
  enabled: true
  backend: llama_cpp
""",
    )
    overlay_b = _write_yaml(
        tmp_path,
        "b.yaml",
        """
llm:
  backend: llama_cpp
""",
    )

    monkeypatch.setenv("MOUSEDROID_LLM__BACKEND", "openai_compatible")
    cfg = load_settings(overlay_a, overlay_b)
    assert cfg.llm.backend == "openai_compatible"


def test_empty_overlay_path_list_returns_default_only(tmp_path: Path) -> None:
    """``load_settings()`` with no overlays equals ``load_settings()`` of the default.

    Defensive coverage: an unused varargs parameter must not change behaviour.
    The schema default for ``llm.backend`` is ``"llama_cpp"`` per
    ``src/mousedroid/config/schema.py:656``.
    """
    cfg_no_overlays = load_settings()
    # Default schema value is llama_cpp; this is the contract for any
    # deployment that loads no operator overlay.
    assert cfg_no_overlays.llm.backend == "llama_cpp"


def test_overlay_order_matters(tmp_path: Path) -> None:
    """Apply overlays in different orders → the last one's leaf wins each time.

    ``_deep_merge`` reduces left-to-right (later overlays win on
    conflicts), so swapping the argument order swaps the resolved value.
    """
    overlay_remote = _write_yaml(
        tmp_path,
        "remote.yaml",
        """
mock_hardware: true
llm:
  enabled: true
  backend: openai_compatible
""",
    )
    overlay_local = _write_yaml(
        tmp_path,
        "local.yaml",
        """
mock_hardware: true
llm:
  enabled: true
  backend: llama_cpp
""",
    )

    # remote applied LAST → wins
    cfg_remote_last = load_settings(overlay_local, overlay_remote)
    assert cfg_remote_last.llm.backend == "openai_compatible"

    # local applied LAST → wins
    cfg_local_last = load_settings(overlay_remote, overlay_local)
    assert cfg_local_last.llm.backend == "llama_cpp"
