# Advanced Image Sensor Interface v4.0.0 - System Architecture

## Purpose and scope

This document describes the current architecture of the package in
`src/advanced_image_sensor_interface` and the runnable material in `examples/`,
`scripts/`, and `benchmarks/`. It is a current-state description, verified
against source on 2026-10-26, not a target-state design.

The project is a simulation and modeling framework. Protocol drivers generate
and decode frame payloads in Python, power and thermal behavior is modeled, and
no component opens a camera, PHY, or kernel driver. Hardware integration notes
describe how to replace the simulation boundaries; they are not implemented
here.

Diagram assets:

- `assets/system-architecture-v4.0.0.svg` - canonical layered diagram
- `assets/system-architecture-v4.0.0.png` - 2x raster export
- `scripts/generate_architecture_diagram.py` - deterministic SVG generator

## Architecture at a glance

| Layer | Responsibility | Primary modules |
| --- | --- | --- |
| L0 | Sensor configuration, test patterns, declared image formats | `enhanced_sensor.py`, `test_patterns/pattern_generator.py`, `utils/lens_correction.py`, `sensor_interface/calibration/models.py` |
| L1 | Four transport simulations behind one selector | `sensor_interface/protocol/*`, `protocol_selector.py` |
| L2 | Explicit frame decoding, capture pipeline, integrity, MIPI security | `sensor_interface/pipeline.py`, `image_validation.py`, `utils/data_integrity.py`, `protocol/mipi/security.py` |
| L3 | Typed signal, HDR, RAW, and lens processing | `signal_processing.py`, `hdr_processing.py`, `raw_processing.py`, `utils/lens_correction.py` |
| L4 | Stereo depth, calibration solvers, calibration data models | `utils/depth.py`, `calibration/photogrammetry.py`, `calibration/models.py`, `calibration/neural_tuner.py` |
| L5 | Power, performance, buffers, errors, configuration | `advanced_power_management.py`, `performance/*`, `utils/buffer_manager.py`, `error_handling/*`, `config/constants.py` |
| L6 | Outputs, metrics, examples, scripts, verification | `utils/performance_metrics.py`, `examples/`, `scripts/`, `tests/` |

AI/ML scene classification and the adaptive color/noise extensions live in
`examples/ai_ml_enhancements.py` and `examples/custom_extension.py`. They are
demonstrations, not importable package modules. The package itself provides the
scikit-learn calibration tuner, the noise-reduction framework, and the typed
processing pipelines.

## End-to-end data path

1. `EnhancedSensorInterface` synthesizes frames according to a
   `SensorConfiguration` (resolution, bit depth, Bayer format, HDR mode).
2. `ProtocolSelector` scores the registered drivers and activates one. Each
   driver generates unpacked little-endian payload bytes for its advertised
   formats and records measured frame rate, uptime, and throughput.
3. `FrameCodec` decodes the payload for an explicitly declared `ImageFormat`,
   validates length and sample range, optionally verifies a sender-side
   integrity envelope, and returns an owned NumPy array.
4. `FramePipeline` combines capture, decoding, and a typed processing callback.
   It records capture latency and throughput from received bytes rather than
   configured link capacity. Exceptions propagate; failed typed results are
   returned unchanged apart from measured timing.
5. Processors return `SignalProcessingResult`, `HDRProcessingResult`,
   `RAWProcessingResult`, or `LensCorrectionResult`. `FramePipeline` adds
   sequential stereo capture and returns `DepthResult`.
6. Platform services (power, buffers, performance, errors, configuration)
   support every layer. Outputs include processed arrays, depth maps, PLY point
   clouds, PNG visualizations, and metric dictionaries.

### Frame storage contract

`FrameCodec` requires an unsigned integer `ImageFormat`:

| Bit depth | Storage | Bytes per sample |
| --- | --- | --- |
| 8 | `uint8` | 1 |
| 10, 12, 14, 16 | `uint16` | 2 |
| 20 | `uint32` | 4 |

Samples are little-endian and unpacked. Bit-packed and floating-point formats
are rejected by `FrameCodec`; `ImageFormat` itself allows `float32` for
processing paths that do not serialize frames.

## Layer 0 - Sensor input and declared formats

### Enhanced sensor model

`EnhancedSensorInterface` simulates one or more sensors with:

- `SensorResolution` values from VGA through 8K (`7680 x 4320`), plus custom
  resolutions.
- `RAWFormat` values RAW8, RAW10, RAW12, RAW14, RAW16, and RAW20.
- `HDRMode` labels and per-sensor timing, gain, and exposure settings.
- `SensorConfiguration` for sensor count, synchronization, and master sensor.
- `create_8k_sensor_config()` and `create_multi_sensor_config()` presets.

### Test patterns

`PatternGenerator` produces bit-depth-aware arrays for:
`color_bars`, `grayscale_ramp`, `checkerboard`, `slanted_edge`,
`siemens_star`, `zone_plate`, and `noise`.

### Lens profiles

`LensProfile` uses the Brown-Conrady model with radial (`k1`, `k2`, `k3`) and
tangential (`p1`, `p2`) terms. `STANDARD_PROFILES` contains four profiles:
`gopro_wide`, `smartphone_wide`, `dslr_kit`, and `industrial_cs`.
`DistortionType` classifies barrel, pincushion, mustache, and none. Remapping
supports nearest-neighbor and bilinear interpolation.

### Calibration data models

`calibration/models.py` defines `CalibrationResult`, `StereoCalibrationResult`,
`MultiCameraCalibrationResult`, `ColorCalibrationResult`,
`TemporalCalibrationResult`, `CalibrationQualityMetrics`, and
`CalibrationDatabase` (store, get, list, export). `CalibrationPattern` supports
checkerboard and circles grids for object-point generation; other pattern
strings raise `ValueError`.

## Layer 1 - Transport simulations

`ProtocolBase` and `StreamingProtocolBase` define capabilities, status,
connection, streaming, and frame capture. `ProtocolSelector` scores drivers on
bandwidth, distance, power-over-cable, latency, multi-camera support, cost
priority, and reliability priority.

### MIPI CSI-2

- D-PHY v2.5 model with 1-4 lanes and up to 4.5 Gbps per lane (18 Gbps
  aggregate model).
- Virtual channels 0-3, ECC and CRC packet validation.
- Equalization and de-emphasis configuration in `protocol/mipi/v4_1.py`.
- AES-GCM security is an optional application-level layer (see L2).

### CoaXPress

- CXP-1 through CXP-12 speed grades; CXP-12 is 12.5 Gbps per link.
- Register-map simulation, link/control/stream channels, PoCXP status, and
  trigger control.
- Four-link configurations model up to 50 Gbps aggregate.

### GigE Vision

- GVCP/GVSP modeling with packet resend modes.
- 1, 2.5, 5, and 10 GigE speed labels.
- Optional RoCE transport is a NumPy-backed simulation. It models queue pairs,
  memory regions, and completion entries; it does not use hardware RDMA.

### USB3 Vision

- HighSpeed, SuperSpeed, and SuperSpeedPlus link labels (480 Mbps, 5 Gbps,
  10 Gbps).
- `USB3StreamingManager` uses locked FIFO buffer ownership, copy-on-read frame
  payloads, bounded waits, and explicit buffer states.
- Device discovery, hot-plug events, and GenICam-style feature nodes.

Drivers report measured frame rates and traffic in status and statistics.
`disconnect()` propagates stop and transport-close failures instead of hiding
them.

## Layer 2 - Frame path, integrity, and security

### FrameCodec

`FrameCodec` validates an `ImageFormat`, verifies an optional integrity
envelope, enforces exact byte length, reshapes to the declared shape, copies to
an owned array, and re-validates ranges. Invalid envelopes, lengths, or sample
values raise `DataTransferError`.

### FramePipeline

`FramePipeline.capture_frame()` requires a connected and streaming protocol,
captures one payload, decodes it, and records latency plus measured throughput.
`process_frame(operation)` records callback wall time in both the result metrics
and selector history. `process_stereo_pair()` captures left and right frames
sequentially; acquisition is not synchronized or rectified, so calibration and
rectification remain the caller's responsibility.

### Integrity semantics

`utils/data_integrity.py` provides an opt-in application-layer envelope. It is
not transport verification and not authentication; the sender must call
`FrameCodec.encode` with an `IntegrityChecker`.

- CRC-32 uses the standard zlib polynomial (`0xEDB88320`) and fails closed: a
  mismatch is never reported valid.
- Hamming(7,4) returns candidate corrections. Multiple errors can miscorrect, so
  `IntegrityChecker` re-checks the original CRC before accepting a candidate.
- Parity detects errors only.
- `REED_SOLOMON` retains the legacy checksum wire format. It detects changes in
  the redundancy block but cannot repair payload or redundancy bytes.

### MIPI security

`MIPISecurityManager` is available when the optional `[security]` extra installs
`cryptography`:

- AES-128-GCM and AES-256-GCM with 16/32-byte keys, 12-byte nonces, and 16-byte
  tags. CBC modes are rejected.
- Encryption requests raise `ImportError` when the backend is absent.
- Authentication methods: none, pre-shared key, HMAC-SHA256, HMAC-SHA512.
  `CERTIFICATE` is enumerated but not implemented by the verifier.
- Challenges are identity-bound, single-use, expiring, and bounded per config.
- Sessions use separate transmit counters and an authenticated receive replay
  window; bad tags and wrong sessions do not consume replay state.
- Privilege levels: GUEST, USER, OPERATOR, ADMIN.
- Credentials exclude keys and tags from `repr`.

`SecurityManager` in `sensor_interface/security.py` is input and buffer
validation. It is not the cryptographic manager.

## Layer 3 - Typed processing pipelines

All processing entry points return explicit result objects defined in
`types.py`. `ProcessingResult` is the union of `SignalProcessingResult`,
`HDRProcessingResult`, `RAWProcessingResult`, `LensCorrectionResult`, and
`DepthResult`.

### Signal processing

`SignalProcessor` performs noise reduction and color correction. It accepts
8/10/12/14/16/20-bit unsigned integer frames, supports 20-bit `uint32` data, and
rejects float64 and non-finite input with a typed failure.

### HDR processing

`HDRProcessor` implements:

- Tone mapping: Reinhard, Drago, Adaptive, and Gamma.
- Exposure fusion: Mertens and weighted average.
- Single-image and exposure-stack entry points, both returning
  `HDRProcessingResult`.

`MANTIUK` and `DURAND` tone-mapping values and `LAPLACIAN_PYRAMID` /
`GRADIENT_DOMAIN` fusion values are enumerated but fall back to Reinhard and
Mertens respectively with a warning. They are extension points, not implemented
operators.

### RAW processing

`RAWProcessor` implements:

- Demosaicing: Bilinear, Malvar-He-Cutler, and Simple.
- Bayer patterns: RGGB, BGGR, GRBG, and GBRG.
- Black-level correction, white balance, color correction, and gamma.

`AHD` and `VNG` are enumerated but fall back to Simple demosaicing with a
warning. RAW8 through RAW20 are accepted; RAW20 is stored as `uint32`. Failures
return a `RAWProcessingResult` without fabricating a fallback image.

### Lens correction

`LensCorrectionPipeline` applies radial and tangential correction with the
selected interpolation method and returns `LensCorrectionResult`. Failures are
reported in the result rather than as silent no-ops.

## Layer 4 - Depth and calibration

### Stereo depth

`StereoDepthProcessor` supports:

- Block matching with a windowed SAD cost volume.
- Semi-global matching with four cardinal or eight full scanline paths.
- Optional numba kernels with a bit-identical NumPy fallback.
- Optional OpenCV ORB alignment with a NumPy fallback.
- Disparity-to-depth conversion, point-cloud generation, ASCII/binary PLY
  export, and optional trimesh mesh export.

`DepthResult` reports success, error text, disparity/depth maps, valid-pixel
ratio, algorithm name, and point-cloud data. Export failures propagate as failed
results.

### Calibration

- `CalibrationDatabase` stores, retrieves, lists, and exports results.
- `calibrate_camera` implements Zhang's method with per-view homographies,
  closed-form intrinsics, and Levenberg-Marquardt refinement.
- `solve_projection_matrix` implements DLT with RQ decomposition.
- `MultiSensorSynchronizer.calibrate_sensors()` uses OpenCV for pattern
  detection even when `prefer_native_calibration` selects the native solver. The
  solver itself is OpenCV-free; the multi-sensor detection path is not.
- `NeuralCalibrationTuner` uses scikit-learn `MLPRegressor` with Sobel, Harris,
  FFT, and Laplacian features. It does not require TensorFlow or PyTorch.

## Layer 5 - Platform services

### Power management

`AdvancedPowerManager` models seven power states (ACTIVE, IDLE, STANDBY, SLEEP,
DEEP_SLEEP, HIBERNATE, OFF), five modes (PERFORMANCE, BALANCED, POWER_SAVER,
ULTRA_LOW_POWER, CUSTOM), thermal states, DVFS behavior, component control,
budgets, and multi-sensor power allocation. Simulation and hardware backends are
available; the hardware backend requires platform support.

### Performance and caching

- `PerformanceProfiler` and `PerformanceMonitor` measure timing, memory, CPU,
  and throughput.
- `LRUCache`, `MemoryCache`, and `CacheManager` provide size- and TTL-bounded
  caching.
- `PerformanceOptimizer` applies tuning strategies.
- `utils/performance_metrics.py` provides SNR, dynamic-range, and color-accuracy
  calculations.

### Buffers

- `utils/buffer_manager.py` provides pool-backed `BufferManager`,
  `AsyncBufferManager`, and `ManagedBuffer`.
- `sensor_interface/buffer_management.py` models DMA buffers, memory-mapped
  buffers, and zero-copy views.
- USB3 streaming has its own locked FIFO buffer pool with explicit states.

### Errors and configuration

- `error_handling/` provides the `SensorError` hierarchy, circuit breaker,
  retry policies, recovery strategies, and error monitoring.
- `ConfigManager` applies development, testing, and production overrides and
  exposes a thread-safe lazy singleton. Environment variables can override
  timing, security, and processing limits.

## Layer 6 - Outputs, examples, and verification

Outputs include processed arrays, depth maps, PLY point clouds, mesh exports,
PNG visualizations, and metric dictionaries.

### Extension examples

The following are example-only implementations and are not exported by the
installed package:

- `examples/ai_ml_enhancements.py`: `AIEnhancedProcessor`, `SceneClassifier`,
  `NoisePredictor`, `QualityAssessor`.
- `examples/custom_extension.py`: `AINoiseReducer`, `AdaptiveColorCorrector`,
  `HighSpeedMIPIDriver`.

### Verified state (2026-10-26)

- `pytest`: 640 passed, 6 skipped, 646 collected. Skips require numba or trimesh.
- `ruff check .`, `black --check`, and `bandit` are clean.
- mypy reports 0 errors in the configured protocol scope; pyright reports 0.
- Sphinx builds, `CITATION.cff` validates against schema 1.2.0, and
  `uv lock --check` passes.
- Nine Python examples and the notebook cells execute successfully.
- `scripts/simulation.py` passes with default delays and `--fast`.
- `scripts/automated_testing.py` runs 9 unit tests, 1 integration test, and the
  benchmark suite with 0 failures and 0 errors.
- `scripts/data_analysis.py --plot --compare` analyzes two simulation outputs,
  writes 16 plots, and reports 23 comparison metrics.
- `scripts/data_analysis.py` exits 2 when required input files are omitted; this
  is argparse behavior, not a defect.

Hosted GitHub Actions have not run for this branch. All results are local.

## Dependencies and optional extras

Core dependencies: NumPy, SciPy, Matplotlib, Pillow, tqdm, psutil, and
scikit-learn. Optional extras:

- `[security]`: `cryptography` for AES-GCM.
- `[full]`: OpenCV, scikit-image, PyWavelets, trimesh, numba, and additional
  optional integrations. Several entries are platform-gated.
- `[full]` includes CuPy only on non-Darwin platforms.

Optional imports are guarded. Missing numba disables accelerated SGM; missing
trimesh disables mesh PLY export; missing OpenCV disables ORB alignment and
chessboard detection; missing cryptography makes encryption fail closed.

## Known limitations

- All protocols and hardware control paths are simulations.
- RoCE is a NumPy model, not hardware RDMA.
- `CERTIFICATE` authentication is enumerated but not implemented.
- Mantiuk/Durand tone mapping and AHD/VNG demosaicing are enum placeholders that
  fall back to implemented operators.
- `REED_SOLOMON` is checksum detection, not error correction.
- The mypy gate covers protocol drivers and the protocol selector only.
- Six tests skip without numba or trimesh; the security suite skips without the
  `[security]` extra.
- AI/ML scene classifiers and custom extensions are examples, not core modules.
- No hosted CI run exists for the v4.0.0 branch.

## Evidence index

| Claim | Evidence |
| --- | --- |
| Frame storage and pipeline | `sensor_interface/pipeline.py`, `sensor_interface/image_validation.py`, `tests/test_capture_pipeline.py` |
| Typed processing results | `types.py`, `hdr_processing.py`, `raw_processing.py`, `signal_processing.py`, `utils/lens_correction.py` |
| HDR/RAW fallback behavior | `hdr_processing._apply_tone_mapping`, `hdr_processing._apply_exposure_fusion`, `raw_processing._demosaic` |
| Protocol drivers and lifecycle | `sensor_interface/protocol/base.py`, `protocol/mipi/`, `protocol/coaxpress/`, `protocol/gige/`, `protocol/usb3/`, `tests/test_protocol_contracts.py` |
| Integrity semantics | `utils/data_integrity.py`, `tests/test_integrity_contracts.py` |
| AES-GCM and sessions | `protocol/mipi/security.py`, `tests/test_mipi_security.py` |
| Depth and PLY export | `utils/depth.py`, `tests/test_depth_module.py` |
| Native calibration | `calibration/photogrammetry.py`, `tests/test_calibration_solver.py` |
| Multi-sensor calibration | `multi_sensor_sync.py`, `tests/test_calibration_solver.py` |
| Power management | `advanced_power_management.py`, `tests/test_power_management.py` |
| Buffers | `utils/buffer_manager.py`, `sensor_interface/buffer_management.py`, `tests/test_buffer_manager.py` |
| Errors and configuration | `error_handling/`, `config/constants.py` |
| Metrics | `utils/performance_metrics.py`, `tests/test_performance_metrics.py` |
| Example-only AI/ML | `examples/ai_ml_enhancements.py`, `examples/custom_extension.py` |
| Verification counts | `CHANGELOG.md`, `docs/testing_guide.md`, `tests/` |
| Diagram source | `scripts/generate_architecture_diagram.py`, `assets/system-architecture-v4.0.0.svg` |

## Maintaining this document

Regenerate the diagram after changing package modules or layer boundaries:

```bash
python scripts/generate_architecture_diagram.py
uv run --no-project --with cairosvg cairosvg assets/system-architecture-v4.0.0.svg \
  -o assets/system-architecture-v4.0.0.png -s 2
```

Keep claims tied to source files listed in the evidence index. If a module moves,
update both the layer table and the generator's card content.
