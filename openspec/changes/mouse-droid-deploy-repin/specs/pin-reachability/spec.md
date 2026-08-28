# Spec — pin reachability

## Requirement: a gate-critical SHA MUST be reachable from a remote tag

A SHA is *gate-critical* when some gate resolves it: the deploy pin in
`deployments/jetson-image.json` (worktreed by the config-schema-compat CI
gate) or any `implemented_in` in `features.yaml` (resolved by
`validate.py --strict-git`).

Branch reachability does not satisfy this requirement. A branch is a moving
ref that cleanup tooling is expected to delete; a tag is the durable one.

### Scenario: an unprotected pin is reported and remediable
- **GIVEN** a pinned SHA reachable only from branches
- **WHEN** `scripts/repin_tags.sh` runs in dry-run mode
- **THEN** it names the annotated tag that would protect the SHA
- **AND** it mutates neither the remote nor the local tag list

### Scenario: coverage is reachability, not naming
- **GIVEN** a pinned SHA already reachable from a remote tag under any name
- **WHEN** the script runs
- **THEN** that pin is reported as already covered and no second tag is created

### Scenario: only remote tags count
- **GIVEN** a pinned SHA reachable from a purely local tag
- **WHEN** the script runs
- **THEN** the pin is NOT treated as covered, because the remote would still
  lose the commit

## Requirement: an extracted pin MUST be validated before use as a git argument

Pin values flow from file contents into tag names and git arguments. The
`features.yaml` extraction is structurally constrained by its own pattern; the
`deployments/jetson-image.json` extraction is not — it prints whatever the JSON
value is.

### Scenario: a malformed pin is rejected by format, not by resolvability
- **GIVEN** a deploy pin that is not 40 lowercase hex characters
- **WHEN** the script runs
- **THEN** it exits nonzero naming the format violation
- **AND** the remote is not mutated

### Scenario: a pin naming a branch is rejected despite resolving
- **GIVEN** a deploy pin whose value is `main`
- **AND** `git cat-file -e main^{commit}` succeeds
- **WHEN** the script runs
- **THEN** it is rejected, because tagging it would protect a different commit
  while presenting it as the pinned one

### Scenario: an unresolvable pin fails loudly
- **GIVEN** a well-formed pin that resolves to no commit
- **WHEN** the script runs
- **THEN** it exits nonzero rather than reporting a partial plan as success
