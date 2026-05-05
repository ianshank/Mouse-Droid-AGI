# mousedroid-sensor-report

**Channel:** REST + MCP
**Actuation:** no — read-only.

Return the latest LiDAR, IMU, battery, and health snapshot as a
structured JSON document for OpenClaw consumption. No state mutation;
no actuation.

## Sample prompts

* "What is the robot's battery level?"
* "Report any obstacles in front of me"
* "Give me a full sensor snapshot"

## MCP invocation

OpenClaw automatically lists this skill when the tool whitelist
(`read_distance`, `read_encoders`, `read_battery`, `query_health`) is
present on the Jetson MCP surface.

## REST fallback

OpenClaw skills can also POST through the REST mission endpoint with
the canonical "report sensors" prompt; the orchestrator routes the
intent through the rule-based mission parser and falls back to the
LLM gateway only if needed.

## Rate limits

Same shared bucket as every other skill: per
`OpenClawConfig.rest_rate_limit_rps`/`burst`.
