# Roadmap

This roadmap outlines the future development tracking for the Advanced Image Sensor Interface.

## Completed Milestones

### v4.0.0 Features (Security, Explicit Frames, Lifecycle Contracts)
- [x] **Real AES-GCM Security**: Optional `[security]` extra (`cryptography`) implements AES-128/256-GCM with 12-byte nonces and 16-byte tags, guarded imports, and fail-closed encryption requests; AES-CBC rejected; identity-bound single-use expiring challenges; separate TX counters and authenticated RX replay windows; secrets excluded from reprs.
- [x] **Explicit Frame Pipeline**: `FrameCodec` and `FramePipeline` connect a selected streaming driver to typed signal/RAW/stereo processing with owned arrays, strict length/dtype/range validation, measured latency and throughput, and an opt-in application-layer CRC/FEC envelope that is never presented as transport verification.
- [x] **Integrity Fail-Closed**: A CRC mismatch is never reported valid; Hamming corrections are re-checked against the original trailer; REED_SOLOMON is documented and implemented as checksum detection, not correction.
- [x] **Protocol Lifecycle Contracts**: Unpacked little-endian 8/10/12/14/16/20-bit frame storage across MIPI, GigE/RoCE, CoaXPress, and USB3; unknown and bit-packed pixel formats raise; measured traffic and frame rates replace configured link capacity; failed disconnect/stop operations stay visible; USB3 buffer ownership uses locked FIFO queues with copy-on-read frames.
- [x] **Processing Format Repair**: 20-bit uint32 support, RAW20 stored as uint32, typed RAW failures without fallback images, explicit rejection of float64 and non-finite frames, and depth PLY export failures propagated as typed results.
- [x] **Dependency and CI Remediation**: Security floors for Pillow 13.0.0, virtualenv 21.14.9, filelock 4.2.1, certifi 2026.10.19, urllib3 2.8.0, tornado 6.5.9, nbconvert 7.17.2, Werkzeug 3.1.9, setuptools 84.1.0, and black 26.10.1; SHA-pinned GitHub Actions; profile-based dependency audits; docs built on pull requests; workflows retargeted to `version-4.0.0`.
- [x] **Dead Code and Wiring**: Removed the unused neural-tuner model factory, unused lens correctors, no-op version self-assignments, and an unused register helper; `FrameCodec`/`FramePipeline` exported from the package surface.
- [x] **Verification**: Full pytest, Ruff, Black, mypy baseline, Bandit triage, dependency audits, wheel matrix, and a documentation build reported in the v4.0.0 changelog.
- [x] **Release Configuration**: tox 4 environments aligned with CI, mypy.ini synced to the pyproject scope, Read the Docs build packages, citation metadata, and repository hygiene updated for v4.0.0.

### v3.3.0 Features (Dependency Maintenance)
- [x] **Dependabot Completion**: All nine open dependabot PRs folded into the release branch with bot authorship preserved.
- [x] **Widened Dependency Ceilings**: opencv-python <6.0.0, pandas <4.0.0, astropy <9.0.0, trimesh <6.0.0, pyzmq <28.0.0; docs requirements updated (psutil <8.0.0, linkify-it-py >=2.2.0, sphinxcontrib-htmlhelp >=2.1.0).
- [x] **CI Tooling**: astral-sh/setup-uv bumped from v9.0.0 to v10.0.1; workflows retargeted to the version-3.3.0 branch.
- [x] **No API or Behavior Changes**: v3.2.0 typed results, depth, and calibration APIs unchanged; 388 tests passing, mypy baseline held at 136.

### v3.2.0 Features (Typed Results, Simulation Delays, 3D/Depth)
- [x] **ProcessingResult Dataclasses**: SignalProcessingResult, HDRProcessingResult, RAWProcessingResult, LensCorrectionResult replace silent None returns with explicit success/error/metrics.
- [x] **Simulation Delay Config**: SimulationDelayConfig wired into MIPI, GigE/RoCE, CoaXPress CXP-12, and USB3 drivers for configurable simulated latencies.
- [x] **3D / Depth Module**: StereoDepthProcessor with Block Matching and full 8-path semi-global matching (4 cardinal + 4 diagonal paths, optionally numba-accelerated with a bit-identical numpy fallback), disparity-to-depth conversion, point cloud generation, dependency-free PLY export, and trimesh-backed mesh PLY export.
- [x] **Native Calibration Solver**: numpy/scipy photogrammetry solver (`calibrate_camera` via Zhang's method, `solve_projection_matrix` via DLT + RQ) calibrates without OpenCV; opt-in for multi-sensor sync via `SyncConfiguration.prefer_native_calibration`.
- [x] **Optional Extras**: PyWavelets and trimesh added to the `[full]` extra with guarded imports (no new required dependencies).
- [x] **Python 3.11+ Requirement**: Security fixes for keras (deserialization, path traversal CVEs) and astropy (RCE CVE); Python 3.10 support removed.
- [x] **Test Coverage**: Added suites for the depth module (incl. 8-path SGM), simulation delays, native calibration solver, and trimesh export; 388 tests passing (395 collected, 7 skipped without optional extras).

### v3.1.0 Features (AI/ML & Implementation Completion) - **RELEASED 2026-08-21**
- [x] **AI/ML Integration Complete**: Neural Calibration Tuner with scikit-learn MLPRegressor fallback.
- [x] **AI/ML Enhancements**: SceneClassifier, NoisePredictor, QualityAssessor with scene-aware processing.
- [x] **Custom Extensions**: AINoiseReducer with scikit-learn denoising, AdaptiveColorCorrector.
- [x] **Complete Multi-Sensor Synchronization**: Feature-based alignment (ORB+RANSAC), Phase Correlation (sub-pixel FFT).
- [x] **MIPI Security Framework Update**: PRE_SHARED_KEY authentication method support in SecurityConfig.
- [x] **All "In a real implementation" TODOs Completed**: Neural tuner, multi-sensor sync, custom extensions, AI/ML enhancements.
- [x] **Documentation Updates**: README accuracy fixes, performance benchmarks verified, CHANGELOG v3.1.0.
- [x] **Scripts & Benchmarks Fixed**: simulation.py, data_analysis.py, noise_analysis.py, benchmarks/__init__.py.
- [x] **All 329 Tests Passing**: Ruff + Black clean, mypy + pyright configured.
- [x] **Documentation**: Converted reST docstrings to Google style in docs/.
- [x] **Documentation**: Added mathematical explanations to complex algorithms in design_specs.md.
- [x] **Documentation**: Updated design_specs.md power states (verified completed).

### v3.0.0 Features (Protocol & Imaging)
- [x] **MIPI D-PHY v2.5**: Support for data rates up to 4.5 Gbps per lane.
- [x] **Data Integrity**: Advanced error correction with CRC-32 and Reed-Solomon FEC.
- [x] **Multi-System Power**: Power management for multi-sensor and multi-ISP arrays.
- [x] **Lens Correction**: Real-time radial and tangential distortion correction.
- [x] **Malvar-He-Cutler Demosaicing**: Gradient-corrected bilinear interpolation for RAW processing.
- [x] **Protocol Driver De-duplication**: Shared base implementations for all 4 protocol drivers.
- [x] **Register I/O Standardization**: Unified `bytes`-based register I/O across all protocols.
- [x] **Silent Failure Documentation**: Explicit warning sections in processing pipeline docstrings.
- [x] **Thread-Safe Configuration**: Double-checked locking for config singleton.

### v3.0.0 Features (Infrastructure)
- [x] Security fixes for CVE-2025-68480 and CVE-2026-0994.
- [x] UV package manager support with dependency-groups.
- [x] Interactive demo notebook and fixed examples.
- [x] Automated benchmark workflow for performance regression testing.
- [x] CITATION.cff, SECURITY.md, and ROADMAP.md documentation.
- [x] Comprehensive test suite: 329 tests passing.
- [x] Code quality: ruff, black, mypy, pyright all passing.

### v2.0.0 Features
- [x] Multi-protocol support (MIPI, CoaXPress, GigE, USB3).
- [x] 8K Resolution support.
- [x] HDR & RAW Processing pipelines.

---

## v4.0+ Planning (Next Release)

### High Priority
- [ ] **Full-Tree Type Checking**: Extend the mypy gate beyond the protocol scope to the remaining modules and burn down the existing baseline.
- [ ] **AI Denoising**: Integration of ML-based noise reduction models (torch/tensorflow backends).
- [ ] **Smart ISP**: AI-driven image signal processor for automatic parameter tuning.
- [ ] **Predictive Power**: AI-based power consumption forecasting.

