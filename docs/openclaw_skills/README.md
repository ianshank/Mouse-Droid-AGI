# OpenClaw Skills for MouseDroidAGI

This directory documents the four publishable OpenClaw skills that ship
with MouseDroidAGI. Each subdirectory contains a `SKILL.md` operators
copy into `~/.openclaw/workspace/skills/` on the **dedicated Mac mini
host** that runs the OpenClaw agent runtime.

## Deployment topology

```
+---------------------------+               +---------------------------+
|  Mac mini (OpenClaw host) |  Tailscale    |   Jetson Orin Nano        |
|  - OpenClaw daemon        | <-----------> |   - MouseDroidAGI         |
|  - SKILL.md packages      |   tailnet     |     (Docker container)    |
|  - SOUL.md persona        |               |   - Telemetry :8080       |
+---------------------------+               |   - MCP :8765             |
                                            +---------------------------+
```

Both hosts join the same Tailscale tailnet. The Jetson exposes:

* **REST mission endpoint** — `https://<jetson>.tail-xxxx.ts.net/api/v1/mission`
* **MCP transport** (SSE or streamable_http) — `https://<jetson>.tail-xxxx.ts.net:8765/sse`

Bearer tokens are required end-to-end (`MOUSEDROID_TELEMETRY_TOKEN` for
REST, `MOUSEDROID_MCP_TOKEN` for MCP). Tailscale ACLs gate the listening
ports; bearer tokens defend in depth.

## OpenClaw-side artefacts (Mac mini)

Place the following in `~/.openclaw/workspace/`:

* `SOUL.md` — MSE-6 personality fragment (curt, dutiful, mildly grumpy;
  refuses missions outside the configured patrol area).
* `openclaw.json` — set `"dmPolicy": "pairing"` to match
  `OpenClawConfig.dm_pairing_required=true`. Unknown senders get a
  pairing code and are ignored until approved on the Mac mini console.
* `skills/<name>/SKILL.md` — one folder per skill (see siblings of this
  README).

## Operator pre-flight check

From the Mac mini, with both hosts joined to the tailnet:

```bash
# 1. REST mission endpoint reachable + bearer enforced
curl -H "Authorization: Bearer $MOUSEDROID_TELEMETRY_TOKEN" \
     -X POST https://jetson.tail-xxxx.ts.net/api/v1/mission \
     -H "Content-Type: application/json" \
     -d '{"nl_command": "stop"}'

# 2. MCP transport reachable + bearer enforced
curl -H "Authorization: Bearer $MOUSEDROID_MCP_TOKEN" \
     -i https://jetson.tail-xxxx.ts.net:8765/sse
```

Expected: HTTP 202 from the REST endpoint with a `trace_id`, and HTTP
200 with `Content-Type: text/event-stream` from the MCP probe.

## Skill catalog

| Skill name | Channel(s) | Actuation? | Description |
|---|---|---|---|
| `mousedroid-navigate` | REST + MCP | **yes** | NL goal → MCTS-planned waypoint route |
| `mousedroid-sensor-report` | REST + MCP | no | LiDAR + IMU + battery snapshot |
| `mousedroid-voice` | REST + MCP | no | Trigger a Piper TTS phrase |
| `mousedroid-world-model` | REST + MCP | no | RSSM latent + recent episodic summary |

## Memory bridge (Phase D)

When `OpenClawConfig.shared_memory_path` is set on the Jetson side, the
orchestrator's POST_TICK hook periodically writes a Markdown snapshot of
the in-memory `EpisodicReplay` to that path. **This is a snapshot, not
durable storage**: the Jetson's episodic memory remains in-process and
restart loses everything since the last export. Treat `MEMORY.md` as a
read-only view, never as authoritative state.
