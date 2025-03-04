# Changelog

All notable changes to the Advanced Image Sensor Interface project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2024-03-04

### Fixed
- Fixed MIPI driver performance optimization test to be deterministic
- Fixed signal processing noise reduction implementation to properly reduce noise
- Fixed power management validation for input configuration parameters
- Fixed handling of zero values in dynamic range calculation
- Fixed voltage stability issues in power management system
- Fixed MIPI driver performance optimization test to be deterministic

### Added
- Comprehensive test suite with 67+ unit tests across all components
- New testing guide documentation with best practices
- Type checking with both MyPy and Pyright
- Clean separation of test fixtures for better test stability
- Improved error handling across all components

### Changed
- Updated dependencies to address security vulnerabilities
- Improved code quality and test reliability
- Enhanced documentation with detailed API references
- Optimized performance testing approach for reliability
- Restructured signal processing pipeline for better noise reduction

## [1.0.0] - 2024-01-15

### Added
- Initial release of Advanced Image Sensor Interface
- MIPI Driver with support for high-speed data transfer
- Signal Processing Pipeline with noise reduction and color correction
- Power Management System with dual-rail support
- Performance Metrics utilities and benchmarking tools
- Comprehensive documentation including API docs, design specs, and performance analysis

[1.0.1]: https://github.com/muditbhargava66/Advanced-Image-Sensor-Interface/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/muditbhargava66/Advanced-Image-Sensor-Interface/releases/tag/v1.0.0