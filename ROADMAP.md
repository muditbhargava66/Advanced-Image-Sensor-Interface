# Roadmap

This roadmap outlines the future development tracking for the Advanced Image Sensor Interface.

## Completed Milestones

### v3.2.0 Features (Typed Results, Simulation Delays, 3D/Depth)
- [x] **ProcessingResult Dataclasses**: SignalProcessingResult, HDRProcessingResult, RAWProcessingResult, LensCorrectionResult replace silent None returns with explicit success/error/metrics.
- [x] **Simulation Delay Config**: SimulationDelayConfig wired into MIPI, GigE/RoCE, CoaXPress CXP-12, and USB3 drivers for configurable simulated latencies.
- [x] **3D / Depth Module**: StereoDepthProcessor with Block Matching and full 8-path semi-global matching (4 cardinal + 4 diagonal paths, optionally numba-accelerated with a bit-identical numpy fallback), disparity-to-depth conversion, point cloud generation, dependency-free PLY export, and trimesh-backed mesh PLY export.
- [x] **Native Calibration Solver**: numpy/scipy photogrammetry solver (`calibrate_camera` via Zhang's method, `solve_projection_matrix` via DLT + RQ) calibrates without OpenCV; opt-in for multi-sensor sync via `SyncConfiguration.prefer_native_calibration`.
- [x] **Optional Extras**: PyWavelets and trimesh added to the `[full]` extra with guarded imports (no new required dependencies).
- [x] **Python 3.11+ Requirement**: Security fixes for keras (deserialization, path traversal CVEs) and astropy (RCE CVE); Python 3.10 support removed.
- [x] **Test Coverage**: Added suites for the depth module (incl. 8-path SGM), simulation delays, native calibration solver, and trimesh export; 387 tests passing (394 collected, 7 skipped without optional extras).

### v3.1.0 Features (AI/ML & Implementation Completion)
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

## v3.3+ Planning (Next Release)

### High Priority
- [ ] **AI Denoising**: Integration of ML-based noise reduction models (torch/tensorflow backends).
- [ ] **Smart ISP**: AI-driven image signal processor for automatic parameter tuning.
- [ ] **Predictive Power**: AI-based power consumption forecasting.

