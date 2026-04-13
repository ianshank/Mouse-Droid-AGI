# Hailo-8 HEF Model Compilation Guide

This guide documents the offline workflow for compiling ONNX/PyTorch models
to Hailo Executable Format (HEF) using the Hailo Dataflow Compiler.

HEF compilation runs on an x86 workstation — **not** on the Jetson.

## Prerequisites

- x86 Linux workstation (Ubuntu 20.04+ recommended)
- Hailo Dataflow Compiler (`hailo_dataflow_compiler` pip package)
- Hailo SDK account at [hailo.ai/developer-zone](https://hailo.ai/developer-zone/)
- Python 3.8-3.10 (compiler requirement)

## Step 1: Export YOLO to ONNX

```bash
# From the project repo on your dev machine
pip install ultralytics
yolo export model=models/yolo11_disk_detector.pt format=onnx imgsz=640 opset=11
```

This produces `models/yolo11_disk_detector.onnx`.

## Step 2: Install Hailo Dataflow Compiler

```bash
pip install hailo_dataflow_compiler
```

Or download from the Hailo Developer Zone.

## Step 3: Parse the ONNX Model

```bash
hailo parser onnx models/yolo11_disk_detector.onnx \
    --hw-arch hailo8
```

This produces a Hailo Archive (HAR) file: `yolo11_disk_detector.har`.

## Step 4: Optimize with Calibration Data

INT8 quantization requires a representative calibration dataset (100-1000 images
from the target domain — Tower of Hanoi disks, laundry garments, etc.).

```bash
hailo optimize yolo11_disk_detector.har \
    --hw-arch hailo8 \
    --calib-path calibration_images/ \
    --output-har yolo11_disk_detector_optimized.har
```

## Step 5: Compile to HEF

```bash
hailo compiler yolo11_disk_detector_optimized.har \
    --hw-arch hailo8 \
    --output-dir models/hailo/
```

This produces `models/hailo/yolo11_disk_detector.hef`.

## Step 6: Feature Extractor Compilation

Repeat steps 1-5 for the feature extractor model:

```bash
# Export feature extractor to ONNX (project-specific)
python scripts/export_feature_extractor.py --output models/feature_extractor.onnx

# Parse, optimize, compile
hailo parser onnx models/feature_extractor.onnx --hw-arch hailo8
hailo optimize feature_extractor.har --hw-arch hailo8 --calib-path calibration_images/
hailo compiler feature_extractor_optimized.har --hw-arch hailo8 --output-dir models/hailo/
```

## Step 7: Deploy to Jetson

Copy the compiled HEF files to the Jetson:

```bash
scp models/hailo/*.hef jetson@<jetson-ip>:/opt/mousedroid/models/hailo/
```

Or mount them via the Docker volume in `docker-compose.jetson.yml`.

## Step 8: Enable in Config

Apply the Hailo overlay config:

```yaml
# config/jetson_hailo.yaml
hailo:
  enabled: true
  yolo_hef_path: "/opt/mousedroid/models/hailo/yolo11_disk_detector.hef"
  feature_extractor_hef_path: "/opt/mousedroid/models/hailo/feature_extractor.hef"
```

## Validation

After deployment, verify the Hailo device is detected:

```bash
hailortcli fw-control identify
```

Monitor inference performance:

```bash
hailortcli monitor
```

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `hailort` import fails | Install: `pip install hailort>=4.18` |
| Device not found | Check M.2 seating, verify `/dev/hailo0` exists |
| HEF load fails | Recompile with matching `--hw-arch` for your device |
| PCIe bandwidth issues | Increase `experience.flush_every_n` to reduce NVMe contention |
| Low accuracy after INT8 | Increase calibration dataset size, try per-channel quantization |
