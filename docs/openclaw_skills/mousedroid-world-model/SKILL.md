# mousedroid-world-model

**Channel:** REST + MCP
**Actuation:** no — read-only.

Dump the RSSM latent state, current pose estimate, and a recent
episodic-replay summary as a structured document for OpenClaw to
reason over.

## Sample prompts

* "Summarise what the droid believes about the room"
* "What was the last surprising experience?"
* "Where does the droid think it is?"

## Schema

```json
{
  "include_belief": true,
  "include_pose": true,
  "episodic_window": 16
}
```

`episodic_window` is bounded to `[0, 512]` to keep MEMORY.md exports
and MCP responses small.

## Memory bridge note

This skill returns an in-process snapshot of the four-tier memory
modules. **There is no LMDB.** When `OpenClawConfig.shared_memory_path`
is configured on the Jetson, the orchestrator's POST_TICK hook also
writes `MEMORY.md` to that path on a configurable cadence
(`OpenClawConfig.export_every_n_ticks`, default 600 = 20 s @ 30 Hz).
Treat that file as a snapshot, not authoritative storage — restarts
lose everything since the last export.
