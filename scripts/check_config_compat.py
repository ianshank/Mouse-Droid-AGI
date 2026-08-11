#!/usr/bin/env python3
"""Detect config YAML changes that would require an image rebuild before deploy.

The Jetson container's running image embeds a specific version of the
``Settings`` schema. ``config/*.yaml`` files in a PR may add fields that
this older schema rejects (``extra="forbid"`` is the default on most
config models), which produces a silent crash on deploy. Operators have
observed this exact failure mode in production: a yaml-only PR merges,
the rover pulls main, and the container crash-loops with
``ValidationError: Extra inputs are not permitted``.

This script runs in CI on any PR that touches ``config/*.yaml``. It:

1. Reads ``deployments/<platform>-image.json`` to learn the SHA of the
   code that built the currently-deployed image.
2. Worktrees that SHA into a temporary directory.
3. Loads each changed (or added) yaml against the deployed SHA's
   ``Settings`` schema.
4. Reports any validation error with an actionable message: "this PR
   adds fields the deployed image's schema rejects; rebuild and update
   deployments/<platform>-image.json before merge."

Usage:
    python3 scripts/check_config_compat.py \\
        --platform jetson \\
        --changed-files config/jetson_production.yaml \\
        [--deployments-dir deployments] \\
        [--base-ref origin/main]

Exit codes:
    0 — all changed yamls load cleanly against the deployed schema
    1 — at least one validation failure (PR must rebuild image)
    2 — invocation error (missing arg, bootstrap file absent, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEPLOYMENT_FILE_TEMPLATE = "{platform}-image.json"
REQUIRED_KEYS = ("sha", "platform", "image_tag")
# Marker shared with ``scripts/validate_configs.py``: files carrying this
# comment in their first 5 lines are NOT Settings overlays and must be
# excluded from schema-compatibility checking.
SKIP_MARKER = "# config-validator: skip"
_SKIP_SCAN_LINES = 5


def _has_skip_marker(path: Path) -> bool:
    """Return *True* when *path* carries the skip marker in its header."""
    try:
        with path.open(encoding="utf-8") as fh:
            for _, line in zip(range(_SKIP_SCAN_LINES), fh, strict=False):
                if SKIP_MARKER in line:
                    return True
    except OSError:
        pass
    return False


@dataclass(frozen=True)
class DeployedImage:
    """Pointer to the source state that built the deployed image."""

    sha: str
    platform: str
    image_tag: str


def load_deployment(deployments_dir: Path, platform: str) -> DeployedImage:
    """Read ``deployments/<platform>-image.json`` and validate its shape.

    Args:
        deployments_dir: Directory containing the deployment JSON files.
        platform: Platform name (``jetson``, etc.).

    Returns:
        Parsed deployment record.

    Raises:
        SystemExit: When the file is missing or malformed (exit code 2).
    """
    path = deployments_dir / DEPLOYMENT_FILE_TEMPLATE.format(platform=platform)
    if not path.is_file():
        sys.stderr.write(
            f"error: {path} not found. Bootstrap a deployment record before enabling this gate.\n",
        )
        raise SystemExit(2)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"error: {path} is not valid JSON: {exc}\n")
        raise SystemExit(2) from exc
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        sys.stderr.write(f"error: {path} missing required keys: {missing}\n")
        raise SystemExit(2)
    return DeployedImage(
        sha=str(data["sha"]),
        platform=str(data["platform"]),
        image_tag=str(data["image_tag"]),
    )


def worktree_at_sha(sha: str) -> Path:
    """Create a temporary git worktree at ``sha`` and return its path.

    The caller is responsible for cleaning up via :func:`remove_worktree`.

    Args:
        sha: Git commit SHA to materialize.

    Returns:
        Path to the worktree directory.

    Raises:
        SystemExit: When ``git worktree add`` fails (exit code 2).
    """
    tmp = Path(tempfile.mkdtemp(prefix="config-compat-"))
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(tmp), sha],  # noqa: S607
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        # ``git worktree add`` failed (e.g. an unreachable/orphaned SHA), so git
        # never registered the worktree — ``remove_worktree`` can't reclaim it
        # and ``main``'s finally never runs (we exit before returning). Clean up
        # the empty tempdir directly to avoid leaking config-compat-* dirs.
        shutil.rmtree(tmp, ignore_errors=True)
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        sys.stderr.write(
            f"error: failed to create worktree at {sha}: {stderr}\n",
        )
        raise SystemExit(2) from exc
    return tmp


def remove_worktree(path: Path) -> None:
    """Remove a worktree created by :func:`worktree_at_sha`. Best-effort."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],  # noqa: S607
        check=False,
        capture_output=True,
    )


def changed_yaml_files(base_ref: str, paths: list[Path] | None) -> list[Path]:
    """Resolve the set of YAML files to check.

    Args:
        base_ref: Git ref to diff against (e.g. ``origin/main``).
        paths: Explicit file list. When provided, ``base_ref`` is ignored.

    Returns:
        Paths to ``config/*.yaml`` files modified vs ``base_ref`` (or the
        explicit list, filtered to YAML under ``config/``).
    """
    if paths is not None:
        return [
            p
            for p in paths
            if p.suffix in {".yaml", ".yml"}
            and p.is_file()
            and "config" in p.parts
            and not _has_skip_marker(p)
        ]
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", base_ref, "HEAD", "--", "config/"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        Path(line)
        for line in result.stdout.splitlines()
        if line.endswith((".yaml", ".yml"))
        and Path(line).is_file()
        and not _has_skip_marker(Path(line))
    ]


def _validation_env(worktree: Path) -> dict[str, str]:
    """Build the subprocess environment for the schema-validation probe.

    INHERITS the parent environment (do NOT replace it) so the interpreter
    initialises correctly on every platform — a minimal env strips vars some
    platforms require (e.g. Windows ``SYSTEMROOT``/``SystemRoot``), which makes
    the probe fail to import stdlib/site-packages and surface a spurious
    "No module named yaml". Then:

    * STRIP ``MOUSEDROID_*`` so host config overrides (e.g. a developer's
      ``MOUSEDROID_LLM__ENABLED``) never pollute the file-vs-schema check — the
      gate must validate the YAML's *content*, not the host environment.
    * PIN ``PYTHONPATH`` to the deployed SHA's ``src`` so the probe loads the
      *deployed* ``Settings`` schema, overriding any host ``PYTHONPATH``.
    * Force ``MOUSEDROID_MOCK_HARDWARE`` so the schema load never touches real
      hardware.

    Args:
        worktree: Path to the deployed-SHA git worktree.

    Returns:
        Environment mapping for :func:`subprocess.run`.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("MOUSEDROID_")}
    env["PYTHONPATH"] = str((worktree / "src").resolve())
    env["MOUSEDROID_MOCK_HARDWARE"] = "true"
    return env


def validate_yaml_against_schema(
    yaml_path: Path,
    worktree: Path,
) -> str | None:
    """Load ``yaml_path`` against the schema in ``worktree``.

    Args:
        yaml_path: Path to the YAML file (relative to the *current* repo
            root, not the worktree). The file's content is read from
            this side; the schema is loaded from the worktree.
        worktree: Path to the deployed-SHA worktree.

    Returns:
        ``None`` when the YAML loads cleanly; otherwise an error message
        suitable for surfacing in CI.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from mousedroid.config.loader import load_settings\n"
                "from pathlib import Path\n"
                "try:\n"
                "    load_settings(Path(sys.argv[1]))\n"
                "except Exception as exc:\n"
                "    sys.stderr.write(str(exc))\n"
                "    sys.exit(1)\n"
            ),
            str(yaml_path.resolve()),
        ],
        cwd=worktree,
        check=False,
        capture_output=True,
        text=True,
        env=_validation_env(worktree),
    )
    if probe.returncode == 0:
        return None
    return probe.stderr.strip() or "unknown validation error"


def main(argv: list[str] | None = None) -> int:
    """Run the CI gate. Returns process exit code."""
    parser = argparse.ArgumentParser(
        description="Detect config YAML changes incompatible with the deployed image schema.",
    )
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform name (matches deployments/<platform>-image.json)",
    )
    parser.add_argument(
        "--deployments-dir",
        type=Path,
        default=Path("deployments"),
        help="Directory containing <platform>-image.json (default: deployments)",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref to diff against when --changed-files is omitted",
    )
    parser.add_argument(
        "--changed-files",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit list of files to check (skips git diff)",
    )
    args = parser.parse_args(argv)

    deployment = load_deployment(args.deployments_dir, args.platform)
    sys.stdout.write(
        f"deployed image: {deployment.image_tag} @ {deployment.sha[:12]}\n",
    )

    yamls = changed_yaml_files(args.base_ref, args.changed_files)
    if not yamls:
        sys.stdout.write("no config YAML changes detected; nothing to check.\n")
        return 0

    sys.stdout.write(f"checking {len(yamls)} yaml(s) against deployed schema...\n")

    worktree = worktree_at_sha(deployment.sha)
    try:
        failures: list[tuple[Path, str]] = []
        for yaml_path in yamls:
            error = validate_yaml_against_schema(yaml_path, worktree)
            if error is None:
                sys.stdout.write(f"  PASS  {yaml_path}\n")
            else:
                sys.stdout.write(f"  FAIL  {yaml_path}\n")
                failures.append((yaml_path, error))
    finally:
        remove_worktree(worktree)

    if not failures:
        return 0

    sys.stderr.write("\n" + "=" * 72 + "\n")
    sys.stderr.write(
        "Config schema compatibility check FAILED.\n\n"
        "The deployed image's Settings schema rejects one or more changes in "
        "this PR. Merging now would crash-loop the rover when it pulls main "
        "(observed previously with `Extra inputs are not permitted` on "
        "domain_randomization).\n\n"
        "Resolution: rebuild the deployed image after this PR's source "
        "changes land, push the new image, and update "
        f"`{args.deployments_dir}/{args.platform}-image.json` with the new "
        "SHA in this PR or immediately after merge.\n",
    )
    for yaml_path, error in failures:
        sys.stderr.write("\n" + "-" * 72 + "\n")
        sys.stderr.write(f"file: {yaml_path}\n")
        sys.stderr.write(f"error: {error}\n")
    sys.stderr.write("=" * 72 + "\n")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
