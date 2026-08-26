# Testing Guide

## 1. Introduction

This document provides comprehensive guidance on testing the Advanced Image Sensor Interface project. It covers unit tests, integration tests, and performance benchmarks, along with best practices for maintaining and extending the test suite.

## 2. Testing Framework

The project uses the following testing tools:

- **pytest**: Main testing framework for the full suite (329 collected tests in the current workspace)
- **unittest.mock**: For mocking dependencies during testing
- **pytest-cov**: For measuring test coverage
- **pytest-asyncio**: For async test support
- **numpy**: For data generation and validation in tests
- **ruff**: For code linting and quality checks
- **black**: For code formatting
- **mypy & pyright**: Static-analysis support for maintained source modules

## 3. Test Structure

The tests are organized in the following structure:

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_buffer_manager.py         # Buffer management tests
├── test_config.py                 # Configuration tests
├── test_enhanced_features.py      # HDR, RAW, multi-sensor tests
├── test_image_validation.py       # Image validation tests
├── test_imaging_features.py       # D-PHY, data integrity, lens correction
├── test_integration_pipelines.py  # End-to-end integration tests
├── test_mipi_driver.py            # MIPI driver tests
├── test_mipi_protocol.py          # MIPI protocol tests
├── test_noise_reduction_coverage.py # Noise reduction algorithm tests
├── test_performance_metrics.py    # Performance metrics tests
├── test_power_management.py       # Power management tests
├── test_protocol_extensions.py    # Security, CXP-12, RoCE, USB3 streaming
├── test_protocols.py              # Protocol driver coverage (MIPI, GigE, USB3, CoaXPress)
├── test_security.py               # Security framework tests
└── test_signal_processing.py      # Signal processing tests
```

### Test Categories by Module

- **Protocol Tests**: `test_protocols.py`, `test_protocol_extensions.py`, `test_mipi_driver.py`, `test_mipi_protocol.py`
- **Feature Tests**: `test_enhanced_features.py`, `test_imaging_features.py`
- **Core Tests**: `test_signal_processing.py`, `test_power_management.py`, `test_buffer_manager.py`
- **Validation Tests**: `test_image_validation.py`, `test_security.py`, `test_config.py`
- **AI/ML Tests**: `test_ai_ml_enhancements.py`, `test_neural_tuner.py`, `test_custom_extensions.py`

## 4. Running Tests

### 4.1 Running All Tests

To run all tests:

```bash
pytest
```

### 4.2 Running Specific Test Modules

To run tests for a specific module:

```bash
pytest tests/test_mipi_driver.py
```

### 4.3 Running with Verbose Output

To see detailed test output:

```bash
pytest -v
```

### 4.4 Measuring Test Coverage

To generate a test coverage report:

```bash
pytest --cov=src
```

For a more detailed HTML coverage report:

```bash
pytest --cov=src --cov-report=html
```

## 5. Test Categories

### 5.1 Unit Tests

Unit tests focus on testing individual functions and methods in isolation. Examples include:

- Testing signal processing functions with synthetic data
- Testing power management voltage setting and measurement
- Testing MIPI driver data sending and receiving

### 5.2 Integration Tests

Integration tests verify that multiple components work together correctly. Examples include:

- Testing the end-to-end processing pipeline from data reception to output
- Testing power management's effect on signal processing
- Testing MIPI driver's interaction with signal processing

### 5.3 Performance Tests

Performance tests measure and validate performance characteristics. Examples include:

- Testing MIPI driver data transfer rates
- Testing signal processing pipeline speed
- Testing power management efficiency

## 6. Test Design Principles

### 6.1 Deterministic Tests

All tests should be deterministic, producing the same results on each run. This means:

- Using fixed random seeds for any random operations
- Avoiding dependency on actual timing for performance tests
- Properly mocking external dependencies

Example:
```python
# Use fixed seed for reproducibility
np.random.seed(42)
test_frame = np.random.randint(0, 4096, size=(100, 100))
```

### 6.2 Test Independence

Each test should be independent of other tests:

- Not relying on state changes from previous tests
- Properly cleaning up after tests
- Using fixtures to create fresh test environments

Example:
```python
@pytest.fixture
def signal_processor():
    """Fixture to create a SignalProcessor instance for testing."""
    config = SignalConfig(bit_depth=12, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    return SignalProcessor(config)
```

### 6.3 Proper Mocking

Use mocks to isolate the unit being tested:

- Mock external dependencies
- Mock time-consuming operations
- Mock hardware interactions

Example:
```python
@patch("advanced_image_sensor_interface.sensor_interface.mipi_driver.time.sleep")
def test_transmission_simulation(self, mock_sleep, mipi_driver):
    test_data = b'0' * 1000000  # 1 MB of data
    mipi_driver.send_data(test_data)
    expected_sleep_time = len(test_data) / (mipi_driver.config.data_rate * 1e9 / 8)
    mock_sleep.assert_called_with(pytest.approx(expected_sleep_time, rel=1e-6))
```

## 7. Common Testing Patterns

### 7.1 Testing Configurations

Test with a variety of configuration parameters:

```python
@pytest.mark.parametrize("bit_depth", [8, 10, 12, 14, 16])
def test_different_bit_depths(self, bit_depth):
    config = SignalConfig(bit_depth=bit_depth, noise_reduction_strength=0.1, color_correction_matrix=np.eye(3))
    processor = SignalProcessor(config)
    test_frame = np.random.randint(0, 2**bit_depth, size=(1080, 1920), dtype=np.uint16)
    processed_frame = processor.process_frame(test_frame)
    assert np.max(processed_frame) <= 2**bit_depth - 1
```

### 7.2 Testing Error Handling

Test that functions properly handle error conditions:

```python
def test_error_handling(self, signal_processor):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        signal_processor.process_frame("invalid input")
```

### 7.3 Testing Performance Optimization

Test performance improvements without relying on actual timing:

```python
def test_performance_improvement(self, signal_processor):
    # Swap in the test timing strategy and then set a larger synthetic cost.
    signal_processor.set_timing_strategy_for_test()
    original_time = signal_processor._timing_strategy.get_processing_time()
    signal_processor._timing_strategy.set_processing_time(1.0)
    
    try:
        # Optimize performance
        signal_processor.optimize_performance()
        
        # Verify processing time was reduced
        assert signal_processor._timing_strategy.get_processing_time() < 1.0
    finally:
        # Restore the original processing time to avoid affecting other tests
        signal_processor._timing_strategy.set_processing_time(original_time)
```

## 8. Troubleshooting Common Test Issues

### 8.1 Non-Deterministic Tests

If tests fail intermittently:

- Check for random number generation without fixed seeds
- Check for timing-dependent assertions
- Check for dependencies between tests

### 8.2 Slow Tests

For slow-running tests:

- Use smaller data samples for testing when possible
- Mock time-consuming operations
- Parallelize test execution with pytest-xdist

### 8.3 Mocking Issues

If mocking isn't working as expected:

- Ensure you're mocking the correct path
- Check if you're mocking instance methods or class methods correctly
- Verify that the mocked functions are actually called in the code path being tested

## 9. Extending the Test Suite

When adding new features:

1. Add unit tests for each new function or method
2. Update integration tests to include the new functionality
3. Add performance tests for performance-critical components
4. Run the full test suite to ensure no regressions

## 10. Continuous Integration

The project uses GitHub Actions for continuous integration:

- All tests are run on every push and pull request
- Test coverage reports are generated
- Performance benchmarks are tracked over time

## 11. Documentation

Keep test documentation up to date:

- Each test method should have a clear docstring
- Test modules should describe their purpose
- Test fixtures should be documented
- Complex test setups should include comments

## 12. Current Test Status (v3.1.0)

The Advanced Image Sensor Interface project maintains a comprehensive test suite:

- **329 collected tests** verified locally for the current v3.1.0 workspace
- **Targeted quality gates**: pytest, ruff, black, compileall, mypy, and pyright
- **Coverage reports available on demand** via `pytest --cov=src`
- **Multi-Python version testing** (3.10-3.13)

### Test Distribution
- **Protocol tests**: MIPI, GigE Vision, USB3 Vision, and CoaXPress drivers plus extension coverage
- **Imaging tests**: HDR processing, RAW processing, lens correction, and signal-processing regressions
- **Integration tests**: End-to-end capture and processing flows
- **Core utility tests**: Buffer management, configuration, validation, metrics, power, and security
- **AI/ML tests**: Scene classification, noise prediction, quality assessment, neural calibration tuning, custom extensions

## 13. Conclusion

A comprehensive test suite is critical for maintaining code quality and ensuring the reliability of the Advanced Image Sensor Interface project. By following the guidelines in this document, you can contribute to the robustness of the project through effective testing.
