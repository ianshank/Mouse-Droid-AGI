# Product Requirements Document: Pre-built AI Container Integration

## 1. Overview

As an Edge AI Developer,
I want to use pre-compiled container images for memory-intensive dependencies (like `llama-cpp-python`),
So that I can deploy the MouseDroid platform on memory-constrained devices (8GB Jetson Orin Nano) without triggering Out-Of-Memory (OOM) failures during CUDA compilation.

## 2. Background and Context

During the deployment of `feat/jetson-sdcard-64gb`, compiling `llama-cpp-python` from source on the Jetson Orin Nano consistently resulted in OOM errors triggered by the Linux OOM Killer. Even when constraining compilation to a single thread (`MAKEFLAGS="-j1"`) and allocating 16GB of SSD swap, the NVIDIA CUDA Compiler (`cicc` and `ptxas`) exceeded the contiguous memory available. Pure `.whl` downloads were also blocked by DNS/network instability within the constrained edge environment.

## 3. Acceptance Criteria

- **Given** a Jetson Orin Nano with 8GB RAM,
- **When** the developer runs `docker build -t mousedroid:jetson -f Dockerfile.jetson .`,
- **Then** the build must complete successfully without compiling `llama-cpp-python` from source.
- **And** the resulting container must have both `torch` (with CUDA) and `llama_cpp` (with CUDA) successfully installed and importable.
- **And** the deployment must not exceed the device's physical memory during the build phase.

## 4. Out of Scope

- Migrating the entire codebase away from PyTorch (PyTorch remains the core ML framework).
- Setting up cross-compilation toolchains on separate x86_64 build servers (the build must remain standalone on the Jetson).

## 5. Open Questions & Considerations

- Does copying wheels/binaries between two `dustynv` L4T containers cause glibc or Python ABI mismatches? *(Mitigation: ensure both stages use identically versioned tags, e.g., `r36.4.0`)*.

## 6. Success Metrics

- Docker build completes in under 15 minutes (vs. 40+ minutes of failing compilation).
- Zero OOM killer events logged in `dmesg` during the build process.
- 100% pass rate for integration tests verifying GPU backend inference.
