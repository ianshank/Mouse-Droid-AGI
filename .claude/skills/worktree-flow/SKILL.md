---
description: Manage git worktrees for parallel agent isolation per D-6 convention
status: active
---

# Worktree Flow

One worktree per change-id for parallel agent isolation. Agents never share a
worktree with a human's dirty tree; merge back via PR only.

## Create a Worktree

The prefix comes from `.claude/workforce.yaml` (`worktree.prefix`, default `mdcw-`):

```bash
git worktree add ../mdcw-<change-id> -b <change-id>
```

## Session Preamble Audit

At the start of every session, audit active worktrees:

```bash
git worktree list
```

## Rules

1. **One worktree per change-id** — never two agents in the same worktree
2. **Agents never share a human's dirty tree** — always create a fresh worktree
3. **Merge back via PR only** — never direct-push from a worktree to main
4. **Clean up after merge:**
   ```bash
   git worktree remove ../mdcw-<change-id>
   git branch -d <change-id>
   ```

## Existing Precedent

- `scripts/check_config_compat.py::worktree_at_sha` — ephemeral detached worktrees
- `.dockerignore` already has "Claude worktrees + agent workspaces" entry
- Full documentation: `docs/runbooks/worktrees.md`

## Guardrails

- Worktree directories live outside the repo root (`../mdcw-*`)
- Never `git push --force` from a worktree
- The `.dockerignore` entry keeps worktree artifacts out of Docker builds
