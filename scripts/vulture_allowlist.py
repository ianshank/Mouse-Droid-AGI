# Vulture whitelist — known-ALIVE symbols the dead-code audit must not flag.
#
# Format (vulture convention): plain attribute references on `_`, one per
# line, optionally grouped with comments. Each entry needs a one-line WHY —
# an unexplained entry is indistinguishable from a stale one.
#
# Curation contract (F-020, WS-8): the audit is findings-only; this file is
# where a human records "verified alive" verdicts so they stop re-surfacing.
# On a Protocol/DI codebase the dominant false-positive classes are:
#   * @runtime_checkable Protocol members (implemented, never called by name
#     inside src/ - the orchestrator calls through the protocol),
#   * pydantic @field_validator/@model_validator methods (invoked by the
#     framework),
#   * factory build_* hooks reached only from tests / CLI entry points.
# Prefer allowlisting the SPECIFIC symbol over raising --min-confidence:
# a global threshold hides real rot alongside the false positives.
#
# The list starts EMPTY on purpose - entries land as findings are triaged
# (rev. B Phase-4 budget note: curation is an ongoing operator activity).
# Example entry shape:
#   _.execute_action  # ArmControllerProtocol member - called via protocol
