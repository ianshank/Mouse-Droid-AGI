"""AQA pins for F-034 — mlflow tracking_uri default is sqlite, extras are present.

Complements ``tests/unit/training/observability/test_mlflow_logger.py`` and
``tests/unit/factory/test_factory_observability.py``, which prove *behaviour*
against a real ``MlflowClient``. These pins prove *shape* — the specific
values a refactor could quietly drift without any behavioural test going red
(e.g. reverting the default back to ``file:./mlruns``, or a dependency-pin
edit accidentally dropping ``sqlalchemy``/``alembic`` from the ``[mlflow]``
extra).

Type A (revert-provable via ``scripts/prove_pin_fails.sh``): the two pins
below. Type B (behavioural pin for previously-untested-but-unchanged
behaviour): ``_resolve_tracking_uri``'s passthrough for non-``file:`` URIs,
including the new default's ``sqlite:`` scheme.
"""

from __future__ import annotations

from packaging.requirements import Requirement

from mousedroid.config.schema.telemetry import ExperimentLoggerConfig
from mousedroid.factory import _resolve_tracking_uri
from tests._pyproject import load_pyproject


def test_tracking_uri_default_is_sqlite() -> None:
    """The schema default is the sqlite backend, not the legacy file store.

    mlflow 3.x rejects the plain file-store backend outright; sqlite is
    mlflow's own recommended local backend and requires no explicit
    operator opt-in env var.
    """
    assert ExperimentLoggerConfig().tracking_uri == "sqlite:///mlflow.db"


def test_mlflow_extra_includes_sqlite_tracking_store_deps() -> None:
    """The [mlflow] extra carries sqlalchemy + alembic alongside mlflow-skinny.

    mlflow-skinny alone raises ``UnsupportedModelRegistryStoreURIException``
    for a ``sqlite:///`` tracking_uri without these two -- confirmed by
    constructing a real ``MlflowClient`` against a ``sqlite:///`` URI with
    only ``mlflow-skinny`` installed before this fix.
    """
    data = load_pyproject()
    project = data["project"]
    assert isinstance(project, dict)
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    mlflow_extra = optional["mlflow"]
    assert isinstance(mlflow_extra, list)
    names = {Requirement(d).name for d in mlflow_extra if isinstance(d, str)}
    assert "mlflow-skinny" in names
    assert "sqlalchemy" in names
    assert "alembic" in names


def test_resolve_tracking_uri_passes_sqlite_uris_through_unchanged() -> None:
    """The new default's sqlite:/// scheme is not mistaken for a file: URI.

    ``_resolve_tracking_uri`` only resolves ``file:`` URIs to an absolute
    path (so they survive a trainer's ``chdir()``); every other scheme,
    including the new default, must pass through byte-identical.
    """
    assert _resolve_tracking_uri("sqlite:///mlflow.db") == "sqlite:///mlflow.db"


def test_resolve_tracking_uri_passes_http_and_absolute_sqlite_uris_through() -> None:
    """Belt-and-braces: remote and absolute-path URI schemes are also inert."""
    assert _resolve_tracking_uri("http://host:5000") == "http://host:5000"
    assert _resolve_tracking_uri("sqlite:////opt/mousedroid/mlflow.db") == (
        "sqlite:////opt/mousedroid/mlflow.db"
    )


def test_resolve_tracking_uri_still_resolves_file_uris_to_absolute_path() -> None:
    """Unchanged behaviour check: the file: URI resolution path still works.

    Proves the sqlite-default change did not accidentally touch this
    function's pre-existing behaviour for operators who explicitly opt
    back into the legacy file-store backend.
    """
    resolved = _resolve_tracking_uri("file:./mlruns")
    assert resolved.startswith("file:")
    assert resolved.endswith("/mlruns")
    assert "./mlruns" not in resolved  # relative segment must be gone
