---
description: Audit Model Context Protocol (MCP) server configuration, secretless expansion, and worktree state
status: active
---

# MCP & Worktree Audit

Audits the `.mcp.json` project configuration and active worktrees for secretlessness,
schema compliance with `docs/MCP_OPERATOR_GUIDE.md`, and clean execution posture.

## Deterministic Validation

Run the dedicated AQA suite for workforce and MCP assets:

```bash
# Run workforce & MCP configuration regression suite
python -m pytest tests/regression/test_claude_workforce_aqa.py -v
```

## Quick Checks

### 1. Secretless Verification

Verify that `.mcp.json` contains only environment variable expansions (`${VAR}`) and no raw API tokens:

```bash
python -c "
import json, pathlib
data = json.loads(pathlib.Path('.mcp.json').read_text(encoding='utf-8'))
for name, cfg in data.get('mcpServers', {}).items():
    for k, v in cfg.get('env', {}).items():
        if any(tok in k for tok in ('TOKEN', 'SECRET', 'KEY')):
            assert v.startswith('\${') and v.endswith('}'), f'Unexpanded secret in {name}: {k}'
print('All MCP server credentials secretless and valid')
"
```

### 2. Mock Hardware Posture

Confirm the in-tree `mousedroid` server defaults to motion-safe mock hardware:

```bash
python -c "
import json, pathlib
data = json.loads(pathlib.Path('.mcp.json').read_text(encoding='utf-8'))
env = data['mcpServers']['mousedroid']['env']
assert 'MOUSEDROID_MOCK_HARDWARE' in env
print('mousedroid server mock hardware default configured')
"
```

### 3. Worktree State Inspection

Audit active worktrees to ensure parallel isolation per `docs/runbooks/worktrees.md`:

```bash
git worktree list
```

## Guardrails

- Never commit raw personal access tokens or bearer credentials to `.mcp.json`
- Always verify `config/default.yaml` compatibility before changing server arguments
- Ensure all worktrees are cleaned up post-merge with `git worktree remove`
