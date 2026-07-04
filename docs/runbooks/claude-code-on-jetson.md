# Claude Code on Jetson — install + configure runbook

> Extracted from NEXT_STEPS.md (2026-07-03, F-016 truth reconciliation) —
> operator runbook content, not a roadmap item.

Run the Claude Code agent natively on the Jetson Orin Nano so engineers
can drive the rover from a session that has filesystem + git +
`mousedroid` access without round-tripping through a workstation. This
runbook assumes the Jetson is at `ian@mousedroid.local` per
`~/.claude/projects/<this>/memory/reference_jetson_hardware.md`.

### Prerequisites (one-time)

```bash
# SSH into the Jetson
ssh ian@mousedroid.local

# Confirm L4T + arch — Claude Code ships an aarch64 Node binary.
uname -m            # expect: aarch64
cat /etc/nv_tegra_release | head -1   # expect: R36.x (JetPack 6.x)

# Node.js 18+ is required. JetPack 6 ships with Node 12 — upgrade via NodeSource:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version       # expect: v20.x
npm --version        # expect: 10.x
```

### Install Claude Code

```bash
# Global install via npm (no separate binary needed — Node CLI):
sudo npm install -g @anthropic-ai/claude-code

# Verify
claude --version    # expect: 2.x or newer

# First-run auth flow — opens a browser-link prompt. On a headless Jetson,
# copy the URL to your workstation, complete OAuth there, paste the
# returned token back into the Jetson terminal.
claude --setup-token
```

If the Jetson is **fully headless** and you cannot OAuth, supply an
Anthropic API key directly instead:

```bash
# Keep the key ONLY in /etc/mousedroid/docker.env (chmod 600) — the SAME
# key the mousedroid LLM gateway (PR #107) reads, so agent + gateway
# share one managed, revocable copy. NEVER commit it, and don't spread
# copies into shell profiles like ~/.bashrc — secret sprawl makes
# rotation/revocation unreliable. For an interactive session:
set -a; source /etc/mousedroid/docker.env; set +a
# Service-mode reads the same file via systemd EnvironmentFile=.
```

### Configure for the mousedroid repo

```bash
# Claude Code reads CLAUDE.md from the working tree. The repo's
# CLAUDE.md (this file's neighbour) already encodes all the project
# invariants (factory-first DI, no-hardcoded-values, structlog,
# asyncio, mypy --strict, etc.) — no extra config needed.
cd /opt/mousedroid
claude

# Useful Jetson-specific aliases (drop into ~/.bashrc):
alias claude-mousedroid='cd /opt/mousedroid && claude'
alias claude-smoke='cd /opt/mousedroid && claude "run jetson_full_smoke_run.sh and report the SUMMARY.md"'
alias claude-firmware='cd /opt/mousedroid && claude "diagnose the rover ESP32 — see SKILLS.md rover-firmware-diagnosis"'
```

### Recommended Jetson-specific settings

Edit `~/.config/claude-code/settings.json` (or run `claude config`):

```jsonc
{
  // Larger context lets Claude hold the full src/mousedroid/ tree in
  // memory for refactoring sweeps without compaction noise.
  "default_model": "claude-sonnet-4-6",

  // The Jetson's 8 GB unified memory means we should NOT spawn
  // many background subagents in parallel. Cap at 2.
  "max_parallel_subagents": 2,

  // Permission boundary: allow everything UNDER /opt/mousedroid and
  // /etc/mousedroid but DENY writes elsewhere on the Jetson (don't
  // accidentally touch /etc/systemd or ~/.ssh from an agent).
  "permissions": {
    "fileWriteRoots": ["/opt/mousedroid", "/etc/mousedroid", "/tmp"],
    "denyShellCommands": ["sudo rm -rf", "shutdown", "reboot"]
  }
}
```

### Service-mode (optional, for unattended use)

If you want Claude Code running as a background task that can be poked
via SSH (e.g., for the `coderabbit:autofix` skill firing against a
pending PR), drop this systemd unit at
`/etc/systemd/system/claude-code-agent.service`:

```ini
[Unit]
Description=Claude Code agent (operator-driven; not for production decisions)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ian
WorkingDirectory=/opt/mousedroid
EnvironmentFile=/etc/mousedroid/docker.env
ExecStart=/usr/bin/claude --listen 127.0.0.1:9229
Restart=on-failure
RestartSec=10
# Don't expose to the LAN — bind to loopback only. Use SSH port-forwarding
# (ssh -L 9229:127.0.0.1:9229 jetson) from the workstation to attach.

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable claude-code-agent.service
sudo systemctl start claude-code-agent.service
sudo journalctl -u claude-code-agent.service -f
```

### Hardening — secret hygiene + permission boundary

- The same `ANTHROPIC_API_KEY` reaches BOTH Claude Code (via env) AND
  the mousedroid LLM gateway (via `MOUSEDROID_LLM__API_KEY` SecretStr
  override). Keep it in `/etc/mousedroid/docker.env` only — never in
  `~/.bashrc` that gets shared on screen-share.
  - ⚠️ **The currently-deployed key was exposed in a chat transcript and
    MUST be rotated** (current next step #1, top of file). Generate a fresh
    key, replace it in `/etc/mousedroid/docker.env`, restart the container,
    then revoke the old one in the Anthropic console.
- The `denyShellCommands` list above is the minimum; tighten further
  if Claude Code will run with `sudo` privileges (it shouldn't —
  prefer running as user `ian` and only let it ask for sudo
  interactively when needed).
- `/etc/mousedroid/jetson_production.yaml` is bind-mounted into the
  Docker container read-only. Claude Code's `fileWriteRoots` includes
  it for edits, but operators should run `sync_jetson_overlay.sh`
  after edits to refresh the container view.
- The `--listen 127.0.0.1:9229` binding in the systemd unit is
  loopback-only on purpose; opening to `0.0.0.0` would let anyone on
  the WiFi LAN drive the rover via the agent. Use SSH local-forward
  (`ssh -L 9229:127.0.0.1:9229 jetson`) to attach from a workstation.

### Verifying the install

```bash
# Smoke-test the agent against a known-good question:
claude "summarize the mission of this repository in 3 bullets"

# Confirm it can read structured project context:
claude "what does build_llm_gateway return when fallback_backend='none'?"
# expect: a reference to factory.py + the "return primary" branch.
```

### When to NOT use Claude Code on the Jetson

- **In the 30 Hz reactive control loop.** This loop is intentionally
  LLM-free; the safety projector + MCTS + ESP32 driver are the only
  decision-makers there. Claude Code is for operator workflows
  (debugging, smoke runs, doc edits, rebases), NEVER in the rover's
  hot path.
- **During an active mission.** The agent will compete with the
  orchestrator for the Jetson's ~7 GB usable RAM. Park the
  orchestrator (`docker stop mousedroid`) before launching Claude
  Code for any meaningful work.
- **When the rover battery is below `safety.battery_critical_v`.**
  Claude Code's LLM round-trips can take 5-10 s and prolong the
  low-battery condition. Charge first.

---
