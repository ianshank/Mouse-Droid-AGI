# Runbook — Git Worktrees for Parallel Agent Isolation

> **Scope**: Operator guide and standard operating procedure for using Git worktrees
> to isolate concurrent agent sessions and human development trees.
> **Governance Context**: Implements D-6 convention of F-024 (Claude Workforce Modernization).
> **Related**: `.claude/skills/worktree-flow/SKILL.md`, `.claude/workforce.yaml`,
> `docs/runbooks/claude-workforce-hooks.md`.

---

## 1. Overview & Architectural Motivation

In multi-agent and pair-programming development workflows, running concurrent agents
or subagents within a single working directory creates file collision risks, dirty index states,
and race conditions during automated lint/test execution.

Git worktrees provide clean, independent file system checkouts attached to a shared repository
history (`.git` object database):

* **Parallel Agent Isolation**: Each agent operates in its own directory checkout without mutating
  another agent's or operator's in-progress changes.
* **Zero Overhead Clones**: Shared `.git` storage avoids disk bloat and network fetching.
* **Independent Indexes**: `git add`, `git commit`, and pre-commit hooks execute in isolated workspaces.
* **Deterministic Session Boundary**: Sessions begin with a clean audit and terminate with PR merges.

---

## 2. Naming Conventions & Configuration

Worktree naming follows `.claude/workforce.yaml` (`worktree.prefix`):

```yaml
worktree:
    prefix: mdcw-
```

All agent worktrees MUST be created as sibling directories to the primary repository root:
`../<prefix><change-id>` (e.g. `../mdcw-f024-phase5` or `../mdcw-feat-ci-hardening`).

---

## 3. Worktree Lifecycle Workflow

```text
[Main Repo Root]
       │
       ├─► 1. Audit Active Worktrees (`git worktree list`)
       │
       ├─► 2. Provision Isolated Worktree (`git worktree add ../mdcw-<change-id> -b <change-id>`)
       │
       ├─► 3. Execute Work & Local Tests inside `../mdcw-<change-id>`
       │
       ├─► 4. Commit & Push Feature Branch (`git push -u origin <change-id>`)
       │
       ├─► 5. Open Pull Request & Complete CI Validation
       │
       └─► 6. Post-Merge Teardown (`git worktree remove ../mdcw-<change-id>`)
```

### Step 1: Session Preamble Audit

Before initiating multi-step agent tasks, inspect existing active worktrees:

```bash
git worktree list
```

Ensure no stale, unmerged, or abandoned worktrees linger on disk.

### Step 2: Create Isolated Worktree

Create a new branch and associated worktree outside the primary repo tree:

```bash
git worktree add ../mdcw-<change-id> -b <change-id>
```

For ephemeral detached inspections (e.g., characterization testing against a historical commit):

```bash
git worktree add --detach ../mdcw-inspect-<sha> <sha>
```

### Step 3: Working inside the Worktree

Navigate into the worktree directory:

```bash
cd ../mdcw-<change-id>
```

Verify your active branch and clean working tree:

```bash
git status
```

Run test and validation commands inside the worktree environment. The local pre-commit hooks,
linter, and pytest suites operate transparently against the files in this worktree.

### Step 4: Staging, Committing & Pushing

Stage changes and commit following project commit conventions:

```bash
git add <files>
git commit -m "feat(<scope>): <description>

<rationale>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push -u origin <change-id>
```

### Step 5: Teardown and Cleanup

Once the PR is merged into the default branch, remove the worktree checkout and delete the local tracking branch:

```bash
cd ../Gronk-Droid-Jetson-Nano
git worktree remove ../mdcw-<change-id>
git branch -d <change-id>
```

If a worktree directory was manually deleted from disk without `git worktree remove`, prune the git metadata:

```bash
git worktree prune
```

---

## 4. Integration with Workforce Hooks & Tooling

### Workforce Hooks Resolution

Workforce governance hooks configured in `.claude/settings.json` execute via:

```bash
cd "$CLAUDE_PROJECT_DIR" && python3 -m tools.claude_hooks.<module>
```

In a worktree session, Claude Code sets `$CLAUDE_PROJECT_DIR` to the root of the active worktree.
The `-m tools.claude_hooks.<module>` invocation correctly loads the worktree's local copy of
`tools/claude_hooks/` and `.claude/workforce.yaml`.

### Docker & Container Exclusions

To prevent ephemeral worktrees from polluting Docker build contexts when building Jetson runtime images,
`.dockerignore` explicitly ignores worktree sibling patterns:

```text
# Claude worktrees + agent workspaces
../mdcw-*
../worktrees/
```

### In-Tree Precedents

* `scripts/check_config_compat.py`: Uses `worktree_at_sha` context manager for ephemeral backwards-compatibility validation against pinned base commits.
* `tools/claude_hooks/portability.py`: Ignores `.claude/worktrees/` directory when sweeping workforce assets for path hygiene.

---

## 5. Core Rules & Invariants

1. **Never Share Dirty Trees**: An agent must never execute in a repository with uncommitted human changes. Always spawn a fresh worktree.
2. **One Agent per Worktree**: Never attach multiple concurrent agent sessions to the same worktree directory.
3. **PR-Only Integration**: All work from worktrees must integrate into trunk via reviewed PRs with green CI ladders. Never direct-push to main.
4. **Clean Worktree Teardown**: Always remove worktrees upon PR completion to maintain a tidy filesystem and avoid stale branch references.
