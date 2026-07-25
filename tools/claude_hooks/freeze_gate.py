"""PreToolUse hook: capability freeze gate.

Mechanises the planning rule "hardware readiness preempts all in-flight software
streams" at edit time. While the gate feature (``freeze.feature_key``, by default
``F-008``) has not reached ``done`` in the feature catalog, edits to the
configured capability globs are denied. When the feature completes, the gate
self-disables — no code change, no redeploy.

Failure posture is deliberately split:

* **Governance failure fails closed.** A missing, unreadable or malformed
  catalog, or an absent feature key, denies the edit: the gate cannot prove the
  freeze has lifted, and a broken governance input is itself a red flag.
* **Environment failure fails open.** An unexpected internal error (a missing
  dependency in a fresh clone, say) allows the edit with a loud warning rather
  than bricking every write in the session.

An operator can always proceed by setting the override environment variable
named in config; the override is permitted but always logged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO, Any

import yaml
from tools.claude_hooks import hookio
from tools.claude_hooks.config import ConfigError, WorkforceConfig, load_config
from tools.claude_hooks.envflags import env_flag
from tools.claude_hooks.logging_setup import get_logger
from tools.claude_hooks.paths import path_matches_any, resolve_repo_root, to_repo_relative

_logger = get_logger(__name__)

#: Message template shown to the operator when the gate denies an edit.
DENY_TEMPLATE = (
    "Capability freeze in effect: {rel_path} matches frozen path '{pattern}'.\n"
    "Gate feature {feature_key} is '{status}' (required: '{done_status}') in "
    "{features_file}.\n"
    "Rule: hardware readiness preempts all in-flight software streams — land the "
    "hardware gate before capability work.\n"
    "To proceed anyway, set {override_env}=1 (the override is logged)."
)

#: Message shown when the catalog cannot answer the question at all.
UNKNOWN_TEMPLATE = (
    "Capability freeze gate could not verify {feature_key}: {problem}.\n"
    "Denying the edit to {rel_path} (fail-closed: a broken governance input "
    "cannot prove the freeze has lifted).\n"
    "To proceed anyway, set {override_env}=1 (the override is logged)."
)


def _override_active(cfg: WorkforceConfig, env: dict[str, str]) -> bool:
    """Return whether the configured override is switched on.

    Truthiness, not presence: this opens a safety gate, so ``...=0`` and
    ``...=false`` must keep it shut. A presence check would open the freeze on
    exactly the value an operator writes to keep it closed.
    """
    return env_flag(env, cfg.freeze.override_env)


def read_feature_status(features_path: Path, feature_key: str) -> tuple[str | None, str | None]:
    """Return ``(status, problem)`` for ``feature_key`` in the feature catalog.

    Args:
        features_path: Path to the feature catalog YAML.
        feature_key: Feature id to look up, e.g. ``F-008``.

    Returns:
        A two-tuple. On success ``(status, None)``. On failure
        ``(None, problem)`` where ``problem`` is an operator-facing description
        of why the status could not be determined.
    """
    if not features_path.is_file():
        return None, f"catalog {features_path.name} not found"
    try:
        raw = features_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"catalog unreadable ({exc})"
    try:
        parsed: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, f"catalog is not valid YAML ({exc})"

    features: Any = parsed.get("features") if isinstance(parsed, dict) else parsed
    if not isinstance(features, list):
        return None, "catalog has no 'features' list"

    for entry in features:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id", "")).strip() == feature_key:
            status = entry.get("status")
            if not isinstance(status, str) or not status.strip():
                return None, f"{feature_key} has no status field"
            return status.strip(), None
    return None, f"{feature_key} not present in the catalog"


def evaluate(
    payload: dict[str, Any],
    cfg: WorkforceConfig,
    *,
    repo_root: Path,
    env: dict[str, str],
) -> tuple[bool, str]:
    """Decide whether the pending edit is allowed.

    Args:
        payload: Parsed hook payload.
        cfg: Workforce configuration.
        repo_root: Repository root.
        env: Environment mapping.

    Returns:
        ``(allowed, reason)``. ``reason`` is empty when allowed.
    """
    if not cfg.freeze.enabled:
        _logger.debug("freeze_gate_disabled")
        return True, ""

    target = hookio.extract_target_path(payload)
    if target is None:
        _logger.debug("freeze_gate_no_target", tool=hookio.tool_name(payload))
        return True, ""

    rel_path = to_repo_relative(target, repo_root)
    if rel_path is None:
        # Outside the repository: not this repository's capability surface.
        _logger.debug("freeze_gate_outside_repo", target=target)
        return True, ""

    pattern = path_matches_any(rel_path, cfg.freeze.frozen_paths)
    if pattern is None:
        _logger.debug("freeze_gate_path_not_frozen", rel_path=rel_path)
        return True, ""

    features_path = repo_root / cfg.freeze.features_file
    status, problem = read_feature_status(features_path, cfg.freeze.feature_key)

    if problem is not None:
        if _override_active(cfg, env):
            _logger.warning(
                "freeze_gate_override_used",
                rel_path=rel_path,
                pattern=pattern,
                problem=problem,
                override_env=cfg.freeze.override_env,
            )
            return True, ""
        _logger.error(
            "freeze_gate_catalog_unusable",
            rel_path=rel_path,
            features_file=cfg.freeze.features_file,
            problem=problem,
        )
        return False, UNKNOWN_TEMPLATE.format(
            feature_key=cfg.freeze.feature_key,
            problem=problem,
            rel_path=rel_path,
            override_env=cfg.freeze.override_env,
        )

    if status == cfg.freeze.done_status:
        _logger.info(
            "freeze_gate_self_disabled",
            feature_key=cfg.freeze.feature_key,
            status=status,
            rel_path=rel_path,
        )
        return True, ""

    if _override_active(cfg, env):
        _logger.warning(
            "freeze_gate_override_used",
            rel_path=rel_path,
            pattern=pattern,
            feature_key=cfg.freeze.feature_key,
            status=status,
            override_env=cfg.freeze.override_env,
        )
        return True, ""

    _logger.warning(
        "freeze_gate_denied",
        rel_path=rel_path,
        pattern=pattern,
        feature_key=cfg.freeze.feature_key,
        status=status,
    )
    return False, DENY_TEMPLATE.format(
        rel_path=rel_path,
        pattern=pattern,
        feature_key=cfg.freeze.feature_key,
        status=status,
        done_status=cfg.freeze.done_status,
        features_file=cfg.freeze.features_file,
        override_env=cfg.freeze.override_env,
    )


def main(
    argv: list[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Entry point for the PreToolUse freeze gate.

    Args:
        argv: Unused; accepted so the module matches the other hooks' signature.
        stdin: Payload stream. Defaults to :data:`sys.stdin`.
        stdout: Decision stream. Defaults to :data:`sys.stdout`.
        env: Environment mapping. Defaults to :data:`os.environ`.

    Returns:
        Process exit code.
    """
    del argv
    environ = dict(os.environ) if env is None else env
    payload = hookio.read_payload(stdin)

    try:
        repo_root = resolve_repo_root(env=environ)
        cfg = load_config(repo_root=repo_root)
    except ConfigError as exc:
        # Configuration is itself a governance input: refuse to guess.
        _logger.error("freeze_gate_config_invalid", error=str(exc))
        return hookio.emit_deny(
            f"Capability freeze gate could not load its configuration: {exc}",
            stream=stdout,
        )
    except Exception as exc:  # Environment failure must not brick every edit.
        _logger.error("freeze_gate_environment_error", error=str(exc))
        return hookio.emit_allow()

    allowed, reason = evaluate(payload, cfg, repo_root=repo_root, env=environ)
    if allowed:
        return hookio.emit_allow()
    return hookio.emit_deny(reason, stream=stdout)


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
