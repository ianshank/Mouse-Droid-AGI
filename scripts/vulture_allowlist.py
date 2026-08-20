# ruff: noqa: F821, B018
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
# ---- Protocol members (called via protocol dispatch, not by name) --------

_.execute_action  # ArmControllerProtocol member - orchestrator dispatches
_.execute_primitive  # ArmControllerProtocol member - controller dispatches
_.smooth  # TrajectoryGenerator member - controller uses
_.enforce_limits  # TrajectoryGenerator member - controller uses
_.build  # SACAgent.build - factory dispatches

_.advance  # MockClock - test infrastructure
_.is_running  # JournalProtocol member - harness dispatches
_.constraints  # MissionProtocol member - harness dispatches
_.parent_id  # MissionProtocol member - harness dispatches
_.finished_at_s  # MissionProtocol member - harness dispatches
_.last_error  # MissionProtocol member - harness dispatches
_.timestamp_s  # MissionProtocol member - harness dispatches
_.proposed_action  # MissionProtocol member - harness dispatches
_.executed_action  # MissionProtocol member - harness dispatches
_.active_tasks  # MissionProtocol member - harness dispatches
_.timings  # MissionProtocol member - harness dispatches
_.decided_at_ns  # ApprovalProtocol member - harness dispatches

_.downloaded_at  # CloudProtocol member - cloud dispatch
_.blob_name  # CloudProtocol member - cloud dispatch
_.bucket_name  # CloudProtocol member - cloud dispatch
_.optimize  # EfficiencyProtocol member - optimizer dispatch
_.speed_deg_s  # LD19 scan point attribute - protocol data
_.parse_frame  # LD19 protocol method - driver calls

# ---- Pydantic @field_validator / @model_validator methods ----------------
# These are called by the Pydantic framework, not directly in source.
# Adding every validator name would be fragile; instead the dominant ones
# that vulture flags consistently are listed here.

# ---- Constants consumed via config or indirect reference -----------------

_.DEFAULT_LIDAR_FEATURE_DIM  # Used in config schema defaults
_.DEFAULT_GRADIENT_SCALE  # Used in training config defaults
_.BACKTRACK_SPEED_THRESHOLD  # Used in navigation logic
_.APPROACH_CLEAR_DISTANCE_M  # Used in navigation logic
_.APPROACH_SPEED_THRESHOLD  # Used in navigation logic
_.LIDAR_DEFAULT_BAUD_RATE  # Used in LiDAR config defaults
_.LIDAR_SCAN_TIMEOUT_MULTIPLIER  # Used in LiDAR driver
_.SOFTMAX_EPSILON  # Used in RL policy numerics

# ---- Protocol classes (structural typing — consumed via isinstance) ------

_.EngineTypedWeightUpdatePollerProtocol  # Cloud weight-update consumer
_.CloudLoggingSinkProtocol  # Cloud logging consumer (deferred: no cloud target)
_.CloudMetricsExporterProtocol  # Cloud metrics consumer (deferred: no cloud target)
_.MockClock  # Test infrastructure clock
_.TrajectoryGenerator  # Arm control pipeline
_.CurriculumManager  # Arm training curriculum
