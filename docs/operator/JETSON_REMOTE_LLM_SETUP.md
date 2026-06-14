# Jetson Remote-LLM Setup (F-006 fix path c)

Operator runbook for routing the Jetson container's `LLMGateway` at a
host-PC Ollama instance over the USB-net bridge — closing F-006 without
needing to wait for a smaller-quant Phi-3 to land on the rover.

## 1. Why this exists

PR #102's live-Jetson probe confirmed that **Phi-3-mini-q4 (~2.4GB) cannot
be GPU-offloaded on the Orin Nano 8GB shared RAM while the orchestrator's
other CUDA contexts (TensorRT, etc.) hold the iGPU heap** — `cudaMalloc OOM`
at `n_gpu_layers=-1`, `=16`, and `=8`. CPU-only baseline takes **269 seconds**
per `translate_mission`, blowing the 500 ms target by 538×. See
`SMOKE_REPORT.md` Addendum D for the full investigation.

`config/jetson_production_remote_llm.yaml` is an opt-in overlay that flips
`cfg.llm.backend` to `"openai_compatible"` (production-tested via PR #99 /
Tier C2.3) so `LLMGateway.translate_mission` becomes an HTTP call to a
host-PC Ollama instance. The USB-net bridge already used today for SSH and
`docker exec` carries the chat-completion JSON at <1 ms RTT, and a modern
consumer GPU (host PC) handles Phi-3-mini-q4 hot inference at 50–200 ms —
well under the new 750 ms `latency_target_ms` (peer-review-tightened from
a too-loose 2000 ms initial draft).

## 2. Host-PC: install + warm up Ollama

On the host PC (typically `192.168.55.100` over the USB-net bridge — verify
with `ip -4 addr show enxXXX` against the bridge interface name):

```bash
# Install per https://ollama.com/download — out of scope for this runbook.
ollama --version    # confirm install (>= 0.1.18 for /v1/* OpenAI compat)

# Pull the model (~2GB download).
ollama pull phi3:mini

# Warm-load it once so the first orchestrator call doesn't pay cold-start cost.
# This is critical — Ollama cold-loads a 2GB GGUF into VRAM in 3–8 s and the
# FIRST /v1/chat/completions then blocks 15–25 s while the model pages in.
# The Jetson-side request_timeout_s defaults to 60 s for exactly this reason,
# but pre-warming converts the visible cold-start latency to zero from the
# Jetson's perspective.
ollama run phi3:mini /dev/null

# Start the server bound to 0.0.0.0 so the Jetson can reach it over the
# USB-net bridge (NOT the default 127.0.0.1).
OLLAMA_HOST=0.0.0.0:11434 ollama serve &

# Confirm it's listening + advertising the model.
curl http://127.0.0.1:11434/v1/models
# Expected: {"object":"list","data":[{"id":"phi3:mini","object":"model",...}]}
```

> **Security note:** `OLLAMA_HOST=0.0.0.0` exposes Ollama on every host
> interface — including ones other than the USB-net bridge. Bind to the
> bridge IP specifically (`OLLAMA_HOST=192.168.55.100:11434`) if your
> host PC is on a network you don't fully control. Set
> `MOUSEDROID_LLM__API_KEY` in `/etc/mousedroid/docker.env` and configure
> Ollama with the corresponding token if the bridge is reachable beyond
> the Jetson. The gateway forwards api_key as `Authorization: Bearer …`
> and never logs the secret (verified by the
> `test_api_key_forwarded_to_cold_ping_as_bearer_but_never_logged`
> regression test).

## 3. Jetson: discover the host-PC IP on the bridge

On the Jetson, find the host PC's IP on the USB-net bridge:

```bash
# Pick the interface name from `ip link show`:
ip -4 addr show usb0  # or enxXXX, varies by kernel
# The host PC is typically the .100 to your .1.

# Sanity-check connectivity before changing any container state:
ping -c 3 192.168.55.100
curl http://192.168.55.100:11434/v1/models
```

If the curl from the Jetson succeeds, the path is clear; if not, the Ollama
process on the host is bound to `127.0.0.1` only (see step 2).

## 4. Jetson: populate `/etc/mousedroid/docker.env`

`docker-compose.jetson.yml` reads the operator-managed env file via the
`env_file` directive landed in PR #101 (F-014 fix), so values set here
reach the container without any host-shell export gymnastics. The
`MOUSEDROID_LLM__*` env vars use the nested-env-strip from PR #101
(commit `d4dab14`) so they cleanly override the YAML overlay.

```bash
# First-time setup: copy the template + sudoedit.
sudo cp config/.env.jetson.example /etc/mousedroid/docker.env
sudoedit /etc/mousedroid/docker.env
```

Uncomment the four lines under the "F-006 follow-up (path c)" section and
populate them with your values:

```bash
MOUSEDROID_LLM__BACKEND=openai_compatible
MOUSEDROID_LLM__BASE_URL=http://192.168.55.100:11434
MOUSEDROID_LLM__MODEL_NAME=phi3:mini
MOUSEDROID_LLM__API_KEY=<rotate-per-host>
```

## 5. Jetson: deploy the remote-LLM overlay

`scripts/sync_jetson_overlay.sh` was generalised in this sprint (PR-2) to
handle multiple overlay pairs via the `MOUSEDROID_EXTRA_OVERLAYS` env var.
Add a one-liner to the systemd unit's `Environment=` block or pass at
invocation:

```bash
# One-shot sync (also runs every time the systemd unit starts).
MOUSEDROID_EXTRA_OVERLAYS="/opt/mousedroid/config/jetson_production_remote_llm.yaml:/etc/mousedroid/jetson_production_remote_llm.yaml" \
sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh

# Audit-mode verify — read-only hash-compare.
MOUSEDROID_EXTRA_OVERLAYS="/opt/mousedroid/config/jetson_production_remote_llm.yaml:/etc/mousedroid/jetson_production_remote_llm.yaml" \
sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh --verify
# Expected: two ``OK overlay_sync_match pair_index=0...`` + ``pair_index=1...`` lines.
```

## 6. Restart the orchestrator with the layered overlay

The orchestrator's `MOUSEDROID_CONFIG` env var only points at the base
`jetson_production.yaml` (single overlay). To layer the remote-LLM overlay
on top, use the **comma-separated** `MOUSEDROID_CONFIGS` list var (or the
`MOUSEDROID_JETSON_CONFIGS` alias) — `resolve_runtime_config_paths()` in
`src/mousedroid/validation/runtime.py` splits these on `,` and applies the
overlays left-to-right. The CSV list vars take precedence over the
single-path `MOUSEDROID_CONFIG` / `MOUSEDROID_JETSON_CONFIG` forms:

```bash
MOUSEDROID_CONFIGS=/etc/mousedroid/jetson_production.yaml,/etc/mousedroid/jetson_production_remote_llm.yaml
```

```bash
sudo systemctl restart mousedroid-docker.service
# or directly:
docker compose -f /opt/mousedroid/docker-compose.jetson.yml up -d --force-recreate mousedroid
```

## 7. Probe the deployed configuration

The F-006 verification probe shipped in this sprint. It's the operator-
facing equivalent of `tools/llm_latency_probe.py` (PR #102) but exercises
the HTTP path:

```bash
docker exec mousedroid python3 /opt/mousedroid/tools/jetson_remote_llm_probe.py \
    --config /etc/mousedroid/jetson_production.yaml \
    --overlay /etc/mousedroid/jetson_production_remote_llm.yaml
```

What you should see in the structured-log output:

- `probe_cfg` — confirms `backend=openai_compatible` + the resolved
  `base_url` / `model_name` (env vs overlay precedence is observable here).
- `tegrastats_before` / `tegrastats_after` — Jetson RAM snapshots. **These
  should NOT move appreciably** — that's the whole point of moving the
  LLM off-Jetson. If they do, something else is loading models locally.
- `remote_llm_models_listed` — the list of models the host PC's Ollama
  advertises. Confirm your configured `model_name` is in this list before
  trusting the rest of the run.
- `llm_start_complete` — gateway came up. Cold-start ms here is the
  HTTPS handshake + first /v1/models round-trip; should be < 100 ms.
- `llm_latency_result` — final verdict. `passed=true` + `elapsed_ms <=
  cfg.llm.latency_target_ms` means F-006 is **closed** on this host.
- `llm_inference_slow` — emitted only if you exceeded the target. Tune
  `MOUSEDROID_LLM__MODEL_NAME=qwen2.5:1.5b` for a smaller / faster
  model, or raise `latency_target_ms` if a slower host is the constraint.

Exit codes (matches PR #102's contract):

- `0` — translate_mission elapsed ≤ `cfg.llm.latency_target_ms`. F-006 done.
- `1` — elapsed > target. The HTTP path works but is too slow on this host.
- `2` — transport / load failure. Check `llm_gateway_load_failed` event
  for the diagnostic hint.
- `3` — config error. Either `cfg.llm.enabled=false` or you tried to
  run this probe with `backend=llama_cpp` (use `llm_latency_probe.py`
  instead for the local path).

## 8. Append the result to SMOKE_REPORT

Once `passed=true` lands, append an Addendum E to `SMOKE_REPORT.md` with
the measured `elapsed_ms`, the chosen `model_name`, and the host-PC
hardware (CPU/GPU model). This closes F-006 in the published smoke report
and gives future operators a reference point for what "good" looks like.

## 9. Backwards compatibility checklist

Things this overlay does NOT change for operators who never deploy it:

- `cfg.llm.backend` default stays `"llama_cpp"` — pre-Tier-C2.3
  deployments are byte-identical.
- `docker-compose.jetson.yml` `env_file: required: false` (PR #101)
  means first-time bringup still works without `/etc/mousedroid/docker.env`.
- `sync_jetson_overlay.sh` with `MOUSEDROID_EXTRA_OVERLAYS` unset is
  functionally equivalent to the pre-F-006 single-pair flow — same files
  synced, same exit codes (regression-tested by
  `test_extra_overlays_unset_preserves_single_pair_behaviour`). The log
  lines now annotate each pair with `pair_index=0`, so byte-for-byte stderr
  output differs from the pre-F-006 script.
- `MOUSEDROID_LLM__*` env vars all default to safe values when unset
  (per `LLMConfig` schema defaults at `src/mousedroid/config/schema.py`).
