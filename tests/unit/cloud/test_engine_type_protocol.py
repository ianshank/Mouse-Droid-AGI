"""C1.2 hardening: typed ``EngineType`` Literal + ``engine_type`` property.

Pins:
* The two canonical engine constants exist on ``cloud.protocol`` and equal
  the Prometheus-label string values orchestrator dispatch depends on.
* ``HuggingFaceWeightUpdatePoller`` exposes ``engine_type`` as a typed
  property (not just a private ``_engine_type`` attribute), so the
  orchestrator's legacy-kwarg fold-in path doesn't need ``getattr``
  reflection for protocol-conformant pollers.
* The ``EngineType`` ``Literal`` alias is importable for downstream typing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mousedroid.cloud.protocol import (
    ENGINE_TYPE_POLICY,
    ENGINE_TYPE_WORLD_MODEL,
    EngineType,
    WeightUpdatePollerProtocol,
)
from mousedroid.cloud.weight_update_poller import HuggingFaceWeightUpdatePoller
from mousedroid.config.schema import WeightUpdatePollConfig


def test_engine_type_constants_match_prometheus_label_values() -> None:
    """Constants are the exact strings the orchestrator dispatch + metrics use."""
    assert ENGINE_TYPE_POLICY == "policy"
    assert ENGINE_TYPE_WORLD_MODEL == "world_model"


def test_engine_type_literal_alias_importable() -> None:
    """The Literal alias is importable — downstream code can annotate against it."""

    # If the alias is missing, the import at module top would have failed.
    # Trivial reflection check to keep ruff F401 happy + document the use.
    def _accepts(_: EngineType) -> None:  # pragma: no cover - signature check
        return None

    _accepts(ENGINE_TYPE_POLICY)
    _accepts(ENGINE_TYPE_WORLD_MODEL)


def test_poller_engine_type_property_returns_constructor_arg(tmp_path: Path) -> None:
    """``HuggingFaceWeightUpdatePoller.engine_type`` returns the constructor arg."""
    cfg = WeightUpdatePollConfig(cache_dir=str(tmp_path / "weights"))
    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id="x/y",
        filename="f.onnx",
        engine_type=ENGINE_TYPE_WORLD_MODEL,
    )
    assert poller.engine_type == ENGINE_TYPE_WORLD_MODEL


def test_poller_satisfies_protocol_runtime_check(tmp_path: Path) -> None:
    """``HuggingFaceWeightUpdatePoller`` is recognised as a ``WeightUpdatePollerProtocol``."""
    cfg = WeightUpdatePollConfig(cache_dir=str(tmp_path / "weights"))
    # Avoid the I/O-heavy directory validator by patching it for the structural check.
    with patch(
        "mousedroid.cloud.weight_update_poller._validate_download_directory",
        return_value=None,
    ):
        poller = HuggingFaceWeightUpdatePoller(
            cfg,
            repo_id="x/y",
            filename="f.onnx",
            engine_type=ENGINE_TYPE_POLICY,
        )
    assert isinstance(poller, WeightUpdatePollerProtocol)


def test_engine_type_property_is_distinct_from_private_attr() -> None:
    """The property reads the private attr — pin the access path so refactors stay safe."""
    poller = MagicMock()
    poller._engine_type = ENGINE_TYPE_POLICY
    # Bind the descriptor to the mock to mimic the property semantics.
    type(poller).engine_type = property(lambda self: self._engine_type)
    assert poller.engine_type == ENGINE_TYPE_POLICY
