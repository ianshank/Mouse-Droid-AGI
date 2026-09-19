"""AQA — the GitHub Actions trust boundary is pinned, not changed (F-051).

The delivery spec's position on this is unusual and worth restating, because it
determines what this file is: the boundary is **already satisfied**, so this is
a pin that keeps it satisfied, not a test for new behaviour. Verified against
the tree as it stands:

* all five workflows declare top-level ``permissions: contents: read``
  (``ci.yml:49``, ``config-compat.yml:28``, ``harness.yml:35``,
  ``jetson-nightly.yml:41``, ``release.yml:28``);
* no workflow references ``secrets.*`` anywhere;
* no workflow has a ``pull_request_target`` trigger;
* the only self-hosted runner is ``jetson-nightly.yml``'s
  ``runs-on: [self-hosted, jetson]`` — a validation workflow, not a deploy one,
  and this change adds no job to it;
* the only protected environment is ``pypi``.

Two real deviations from "read-only everywhere" exist and are pinned BY NAME
rather than papered over, because an allowlist that quietly tolerates any
write scope is not a trust boundary:

* ``release.yml``'s ``publish`` job holds ``id-token: write`` for PyPI trusted
  publishing, scoped to the ``pypi`` environment;
* ``ci.yml``'s ``gitleaks`` and ``vulture-audit`` jobs re-declare
  ``contents: read`` at job level (a no-op narrowing, not a widening).

Anything else — a new workflow, a new job on the Jetson runner, a
``secrets.*`` reference, a widened permission — turns this file red on purpose.

Regression tier per ``.claude/skills/test-tier-mirror/SKILL.md``: it pins a
property of files on disk that must not drift, with no runtime involved.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

# The five workflows the spec enumerates. A sixth must be a conscious decision
# that adds it here, having checked it against everything below.
_WORKFLOWS = (
    "ci.yml",
    "config-compat.yml",
    "harness.yml",
    "jetson-nightly.yml",
    "release.yml",
)

_SELF_HOSTED_WORKFLOW = "jetson-nightly.yml"
_SELF_HOSTED_RUNS_ON = ["self-hosted", "jetson"]

# jetson-nightly.yml's complete job set. The spec: "This change SHALL NOT add a
# job to it. If a Jetson deploy job is ever added, it SHALL require a protected
# `jetson-production` environment with reviewers, branch restrictions and
# concurrency of one." Pinning the set is what makes that a decision rather
# than a diff nobody noticed.
_SELF_HOSTED_JOBS = ("ten-pillars",)

# Job-level permission overrides, by (workflow, job). Everything not listed
# here must inherit the read-only top level.
_ALLOWED_JOB_PERMISSIONS: dict[tuple[str, str], dict[str, str]] = {
    ("ci.yml", "gitleaks"): {"contents": "read"},
    ("ci.yml", "vulture-audit"): {"contents": "read"},
    ("release.yml", "publish"): {"id-token": "write"},
}

# (workflow, job) -> environment name. `pypi` is the only protected environment.
_ALLOWED_ENVIRONMENTS: dict[tuple[str, str], str] = {
    ("release.yml", "publish"): "pypi",
}

# ``secrets`` inside a ${{ }} expression. Matching the bare word would trip on
# prose like "no job receives rover credentials"; this matches the real thing.
_SECRETS_REF = re.compile(r"secrets\s*\.\s*[A-Za-z_]")


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((_WORKFLOW_DIR / name).read_text(encoding="utf-8"))
    return data


def _triggers(workflow: dict[str, Any]) -> dict[str, Any] | list[Any] | str:
    """Return the ``on:`` block.

    PyYAML resolves the bare key ``on`` to the boolean ``True`` (YAML 1.1
    truthiness), so a plain ``workflow["on"]`` misses it and every assertion
    built on it would pass vacuously.
    """
    if "on" in workflow:
        found: dict[str, Any] | list[Any] | str = workflow["on"]
        return found
    return workflow[True]


def _jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = workflow.get("jobs", {})
    return jobs


def test_the_pinned_workflow_set_is_the_real_one() -> None:
    """Premise for every other test here: these five files exist and are all of
    them. Without this, a renamed workflow would simply escape the sweep."""
    on_disk = sorted(p.name for p in _WORKFLOW_DIR.glob("*.y*ml"))
    assert on_disk == sorted(_WORKFLOWS), (
        "workflow inventory changed; re-audit the trust boundary before "
        f"updating this list. on disk: {on_disk}"
    )


@pytest.mark.parametrize("name", _WORKFLOWS)
def test_top_level_permissions_are_contents_read(name: str) -> None:
    """The default token for every job must be read-only."""
    workflow = _load(name)
    assert workflow.get("permissions") == {"contents": "read"}, (
        f"{name} must declare top-level `permissions: contents: read`; "
        f"got {workflow.get('permissions')!r}"
    )


@pytest.mark.parametrize("name", _WORKFLOWS)
def test_no_workflow_references_secrets(name: str) -> None:
    """No job receives rover credentials, a deploy key or a registry token.

    Asserted on the raw text, not the parsed tree: a ``secrets.*`` reference can
    appear inside any string value at any depth, including ``with:`` inputs and
    multi-line ``run:`` bodies.
    """
    text = (_WORKFLOW_DIR / name).read_text(encoding="utf-8")
    hits = [line.strip() for line in text.splitlines() if _SECRETS_REF.search(line)]
    assert hits == [], f"{name} references secrets: {hits}"


@pytest.mark.parametrize("name", _WORKFLOWS)
def test_no_pull_request_target_trigger(name: str) -> None:
    """``pull_request_target`` runs fork PR code with a writable token against
    the base repo. Nothing here may use it."""
    triggers = _triggers(_load(name))
    if isinstance(triggers, (dict, list)):
        trigger_names = {str(key) for key in triggers}
    else:
        trigger_names = {str(triggers)}
    assert "pull_request_target" not in trigger_names, (
        f"{name} must not use pull_request_target; triggers: {sorted(trigger_names)}"
    )
    # Belt and braces: the string must not appear at all, so a commented-out or
    # partially-migrated trigger is also caught.
    text = (_WORKFLOW_DIR / name).read_text(encoding="utf-8")
    assert "pull_request_target" not in text, f"{name} mentions pull_request_target"


def test_only_jetson_nightly_uses_the_self_hosted_runner() -> None:
    """The self-hosted Jetson runner is reachable from exactly one workflow, and
    it is a validation workflow."""
    for name in _WORKFLOWS:
        for job_name, job in _jobs(_load(name)).items():
            runs_on = job.get("runs-on")
            labels = runs_on if isinstance(runs_on, list) else [runs_on]
            if "self-hosted" in labels:
                assert name == _SELF_HOSTED_WORKFLOW, (
                    f"{name}:{job_name} runs on the self-hosted runner; only "
                    f"{_SELF_HOSTED_WORKFLOW} may"
                )
                assert labels == _SELF_HOSTED_RUNS_ON, (
                    f"{name}:{job_name} runs-on must be {_SELF_HOSTED_RUNS_ON}; got {labels}"
                )


def test_the_self_hosted_runner_label_appears_in_one_file_only() -> None:
    """The literal label list, as the spec words it. ``harness.yml`` mentions
    "self-hosted" in a comment, which is why this is asserted on the label
    syntax rather than on the bare word."""
    carriers = sorted(
        name
        for name in _WORKFLOWS
        if "[self-hosted, jetson]" in (_WORKFLOW_DIR / name).read_text(encoding="utf-8")
    )
    assert carriers == [_SELF_HOSTED_WORKFLOW], carriers


def test_no_job_was_added_to_the_self_hosted_workflow() -> None:
    """Adding a promotion job to the Jetson runner is exactly what this change
    does not do. If one is ever added it needs a protected
    ``jetson-production`` environment with reviewers and concurrency of one —
    update this pin only alongside that."""
    jobs = tuple(_jobs(_load(_SELF_HOSTED_WORKFLOW)))
    assert jobs == _SELF_HOSTED_JOBS, (
        f"{_SELF_HOSTED_WORKFLOW} job set changed: {jobs}. A deploy job here "
        "requires a protected jetson-production environment first."
    )


def test_job_level_permission_overrides_are_the_allowlisted_ones() -> None:
    """A job may narrow its token, never widen it unaudited."""
    found: dict[tuple[str, str], dict[str, str]] = {}
    for name in _WORKFLOWS:
        for job_name, job in _jobs(_load(name)).items():
            if "permissions" in job:
                found[(name, job_name)] = job["permissions"]
    assert found == _ALLOWED_JOB_PERMISSIONS, (
        "job-level permissions changed; every write scope must be justified. "
        f"found {found!r}"
    )


def test_pypi_is_the_only_protected_environment() -> None:
    found: dict[tuple[str, str], str] = {}
    for name in _WORKFLOWS:
        for job_name, job in _jobs(_load(name)).items():
            env = job.get("environment")
            if env is not None:
                found[(name, job_name)] = env if isinstance(env, str) else env.get("name", "")
    assert found == _ALLOWED_ENVIRONMENTS, f"protected environments changed: {found!r}"


def test_the_id_token_write_override_stays_scoped_to_the_pypi_publish_job() -> None:
    """The one write scope in the repo. It must keep its environment gate: an
    ``id-token: write`` job without one can mint an OIDC token from any branch
    that can trigger the workflow."""
    publish = _jobs(_load("release.yml"))["publish"]
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["environment"] == "pypi"
    assert "contents" not in publish["permissions"], (
        "the publish job must not also grant contents write"
    )
