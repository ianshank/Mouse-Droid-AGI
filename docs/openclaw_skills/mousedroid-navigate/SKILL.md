# mousedroid-navigate

**Channel:** REST + MCP
**Actuation:** **yes** — requires both
`MCPConfig.expose_actuation_tools=true` AND
`OpenClawConfig.require_actuation_ack=true` on the Jetson side.

Translate an NL navigation command into a velocity target via the
mission dispatcher and the existing world model / MCTS planner.

## Sample prompts

* "Patrol the living room"
* "Drive forward slowly until the next obstacle"
* "Hold position"

## REST invocation

```bash
curl -H "Authorization: Bearer $MOUSEDROID_TELEMETRY_TOKEN" \
     -X POST https://jetson.tail-xxxx.ts.net/api/v1/mission \
     -H "Content-Type: application/json" \
     -d '{"nl_command": "patrol the living room", "idempotency_key": "tx-001"}'
```

Successful response (HTTP 202):

```json
{
  "status": "accepted",
  "trace_id": "a1b2c3d4e5f6abcd",
  "command_hash": "1a2b3c4d5e6f",
  "latency_ms": 18.4,
  "goal_vector": {"vx": 0.5, "vy": 0.0, "omega": 0.0}
}
```

## Rate limits

Per `OpenClawConfig.rest_rate_limit_rps` and `rest_rate_limit_burst`
(defaults: 2 rps, burst 4). Over-limit responses return HTTP 429 with a
`retry_after_s` hint.

## Refusal modes

* HTTP 400 `injection_pattern` — command tripped the shared
  prompt-injection filter.
* HTTP 400 `invalid_command` — empty or oversized command.
* HTTP 401 — missing or wrong bearer token.
* HTTP 429 `rate_limited` — token bucket exhausted.
* HTTP 503 `openclaw_disabled` — the Jetson side is not configured for
  OpenClaw (rare; usually means `cfg.openclaw=None` in YAML).

## Safety envelope

The dispatcher returns a `GoalVector` to the orchestrator, which is
still gated by the existing `SafetyMonitorProtocol` per tick. An
emergency-state robot will refuse the goal at the next tick regardless
of how the goal was produced.
