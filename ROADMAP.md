# Roadmap

This roadmap outlines the future development tracking for the Advanced Image Sensor Interface.

## Completed Milestones

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

## v3.1+ Planning (Next Release)

### High Priority
- [ ] **AI Denoising**: Integration of ML-based noise reduction models.
- [ ] **Smart ISP**: AI-driven image signal processor for automatic parameter tuning.
- [ ] **Predictive Power**: AI-based power consumption forecasting.

### Medium Priority
- [ ] **3D / Depth**: Native support for disparity maps and depth calculation.
- [ ] **ProcessingResult Dataclass**: Replace silent failures with explicit result objects.
- [ ] **Simulation Delay Config**: Make protocol driver sleep delays configurable.
- [ ] **Missing Test Coverage**: Add tests for CXP12, RoCE, D-PHY, USB3 streaming, MIPI Security, Lens Correction.

### Documentation
- [ ] Convert remaining reST docstrings to Google style.
- [ ] Add mathematical explanations to remaining complex algorithms.
- [ ] Update design_specs.md power states (already done - verify).

