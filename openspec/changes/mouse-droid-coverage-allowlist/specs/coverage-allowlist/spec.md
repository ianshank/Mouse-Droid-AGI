# Enumerated branch-coverage allowlist

## Purpose

Stop unbounded `factory/` and `orchestrator/_` prefixes from exempting
future modules on the changed-line coverage gate.

## Requirements

### Requirement: factory and orchestrator SHALL NOT be directory prefixes

`_ALLOWED_DIR_PREFIXES` SHALL NOT contain `src/mousedroid/factory/` or
`src/mousedroid/orchestrator/_`.

### Requirement: DI split products SHALL be enumerated files

`_ALLOWED_FILES` SHALL list the ADR-017 pure-DI factory modules and the
orchestrator mixin/`_state.py` files. A path absent from that set and
from remaining prefixes SHALL NOT be exempt.

### Requirement: algorithmic factory modules SHALL stay gated

`factory/on_device_learning.py`, `factory/mcp_harness.py`, and
`factory/_replay_batch_helpers.py` SHALL fail the changed-line gate when
their coverage is below the threshold.

### Requirement: new modules SHALL NOT inherit an exemption

`factory/new_builder.py` and `orchestrator/_new_mixin.py` SHALL NOT be
exempt until a reviewed allowlist edit.
