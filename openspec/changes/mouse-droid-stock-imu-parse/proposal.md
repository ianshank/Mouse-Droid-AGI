# Proposal — Stock FEEDBACK_BASE_INFO IMU parse (no RSSM slot)

- change_id: mouse-droid-stock-imu-parse
- project: mouse-droid
- status: in progress
- feature_id: F-036
- epic: Rover bring-up
- owner: ianshank
- created: 2026-09-07
- rev: A

## Why

`WaveshareStockCodec.parse_encoders` left vendor keys ``r``/``p``/``y`` on
the floor and documented that as "audit-R4, not a comms concern." WAVE ROVER
is encoder-less, so `EncoderReading.heading_rad` stayed 0.0 on every stock
frame. Closing F-008 later without this parse leaves the chassis with no
heading at all.

Widening `SENSOR_SLOT_MAP` with an `imu` key is an ONNX mask-width change
(packer `valid_mask` is pinned at 5). That is F-040, not this bundle.

## What Changes

- `EncoderReading` gains `roll_rad` / `pitch_rad` / `yaw_rad` / `imu_valid`
  (defaults 0.0 / False).
- Stock codec copies T=1001 `r`/`p`/`y` through `_coerce_float`.
- Sensing consumes yaw via `heading_for_motor()` into the existing 4-float
  motor observation. Legacy codec and default `command_set="legacy"` stay
  IMU-inert.

## Impact

Default YAML and `command_set="legacy"` are byte-identical. Behaviour
changes only on the already-opt-in `waveshare_stock` path.

## Charter

No CHARTER §3 carve-out: (1) no new actuation; (2) no LLM/training in the
hot loop; (3) default `command_set` remains `"legacy"` so existing YAML
loads unchanged. IMU fields default to the pre-feature zeros.

## Spec Deltas

`openspec/changes/mouse-droid-stock-imu-parse/specs/stock-imu-parse/spec.md`

## Tasks

See `openspec/changes/mouse-droid-stock-imu-parse/tasks.md`.

## Validation

`bash scripts/validations/F-036.sh`
