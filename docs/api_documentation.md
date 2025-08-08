# API Documentation

## 1. Introduction

This document provides comprehensive documentation for the API of the Advanced Image Sensor Interface project (v1.1.0). It covers the MIPI Driver, Signal Processing Pipeline, and Power Management System interfaces.

## 2. MIPI Driver API

### 2.1 Class: MIPIDriver

#### Constructor

```python
MIPIDriver(config: MIPIConfig)
```

- `config`: An instance of `MIPIConfig` containing the driver configuration.
- Raises `ValueError` if any configuration parameter is invalid (e.g., lanes ≤ 0, data_rate ≤ 0, or channel < 0).

#### Methods

##### send_data

```python
send_data(data: bytes) -> bool
```

Sends data over the MIPI interface.

- `data`: Bytes to send.
- Returns: `True` if successful, `False` otherwise.
- Raises `ValueError` if `data` is not of type bytes.

##### receive_data

```python
receive_data(num_bytes: int) -> Optional[bytes]
```

Receives data from the MIPI interface.

- `num_bytes`: Number of bytes to receive.
- Returns: Received data as bytes, or `None` if an error occurred.

##### get_status

```python
get_status() -> Dict[str, Any]
```

Retrieves the current status of the MIPI driver.

- Returns: A dictionary containing status information:
  - `lanes`: Number of MIPI lanes
  - `data_rate`: Data rate in Gbps per lane
  - `channel`: MIPI channel number
  - `error_rate`: Current error rate
  - `throughput`: Current throughput
  - `total_data_sent`: Total bytes sent
  - `total_time`: Total time spent sending data

##### optimize_performance

```python
optimize_performance() -> None
```

Optimizes the driver performance for increased data transfer rates. This increases the data rate by 40% and reduces the error rate by 50%.

### 2.2 Class: MIPIConfig

#### Constructor

```python
MIPIConfig(lanes: int, data_rate: float, channel: int)
```

- `lanes`: Number of data lanes (1-4).
- `data_rate`: Data rate in Gbps per lane.
- `channel`: MIPI channel number.

## 3. Signal Processing API

### 3.1 Class: SignalProcessor

#### Constructor

```python
SignalProcessor(config: SignalConfig)
```

- `config`: An instance of `SignalConfig` containing the processor configuration.

#### Methods

##### process_frame

```python
process_frame(frame: np.ndarray) -> np.ndarray
```

Processes a single frame of image data.

- `frame`: Input frame as a numpy array.
- Returns: Processed frame as a numpy array.
- Raises `ValueError` if `frame` is not a numpy array or has an invalid shape.

##### _apply_noise_reduction

```python
_apply_noise_reduction(frame: np.ndarray) -> np.ndarray
```

Applies noise reduction to the frame using a Gaussian blur approach.

- `frame`: Input frame as a numpy array.
- Returns: Noise-reduced frame as a numpy array.

##### _apply_dynamic_range_expansion

```python
_apply_dynamic_range_expansion(frame: np.ndarray) -> np.ndarray
```

Applies dynamic range expansion to the frame.

- `frame`: Input frame as a numpy array.
- Returns: Frame with expanded dynamic range.

##### _apply_color_correction

```python
_apply_color_correction(frame: np.ndarray) -> np.ndarray
```

Applies color correction to the frame using the color correction matrix.

- `frame`: Input frame as a numpy array.
- Returns: Color-corrected frame as a numpy array.

##### optimize_performance

```python
optimize_performance() -> None
```

Optimizes the signal processing pipeline for increased speed by reducing processing time by 20% and improving noise reduction by 10%.

### 3.2 Class: SignalConfig

#### Constructor

```python
SignalConfig(bit_depth: int, noise_reduction_strength: float, color_correction_matrix: np.ndarray)
```

- `bit_depth`: Bit depth of the image data.
- `noise_reduction_strength`: Strength of noise reduction (0.0 - 1.0).
- `color_correction_matrix`: 3x3 color correction matrix.

## 4. Power Management API

### 4.1 Class: PowerManager

#### Constructor

```python
PowerManager(config: PowerConfig)
```

- `config`: An instance of `PowerConfig` containing the power management configuration.
- Raises `ValueError` if any configuration parameter is invalid (e.g., voltage_main ≤ 0, voltage_io ≤ 0, or current_limit ≤ 0).

#### Methods

##### set_voltage

```python
set_voltage(rail: str, voltage: float) -> bool
```

Sets the voltage for a specific power rail.

- `rail`: Power rail identifier ('main' or 'io').
- `voltage`: Desired voltage in volts.
- Returns: `True` if successful, `False` otherwise.
- Raises `ValueError` if rail is not 'main' or 'io'.
- Raises `Exception` if power consumption exceeds limits.

##### get_power_status

```python
get_power_status() -> Dict[str, Any]
```

Retrieves the current power status.

- Returns: A dictionary containing power status information:
  - `voltage_main`: Main voltage in volts
  - `voltage_io`: I/O voltage in volts
  - `current_main`: Main current in amperes
  - `current_io`: I/O current in amperes
  - `power_consumption`: Total power consumption in watts
  - `temperature`: System temperature in degrees Celsius
  - `noise_level`: Noise level in power delivery

##### optimize_noise_reduction

```python
optimize_noise_reduction() -> None
```

Optimizes power delivery for reduced signal noise. This reduces noise level by 30%.

### 4.2 Class: PowerConfig

#### Constructor

```python
PowerConfig(voltage_main: float, voltage_io: float, current_limit: float)
```

- `voltage_main`: Main voltage in volts.
- `voltage_io`: I/O voltage in volts.
- `current_limit`: Current limit in amperes.

## 5. Utility Functions

### 5.1 Performance Metrics

#### calculate_snr

```python
calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float
```

Calculates the Signal-to-Noise Ratio.

- `signal`: Clean signal or reference image.
- `noise`: Noise component or difference between noisy and clean signal.
- Returns: SNR in decibels.
- Raises `ValueError` if the shapes of signal and noise do not match.

#### calculate_dynamic_range

```python
calculate_dynamic_range(image: np.ndarray) -> float
```

Calculates the dynamic range of an image.

- `image`: Input image.
- Returns: Dynamic range in decibels. Returns 0 if minimum value is 0.

#### calculate_color_accuracy

```python
calculate_color_accuracy(reference_colors: np.ndarray, measured_colors: np.ndarray) -> Tuple[float, np.ndarray]
```

Calculates color accuracy using a simplified Delta E formula.

- `reference_colors`: Array of reference RGB colors.
- `measured_colors`: Array of measured RGB colors.
- Returns: Tuple of mean Delta E value and array of Delta E values for each color.
- Raises `ValueError` if the shapes of reference_colors and measured_colors do not match.

## 6. Example Usage

```python
# Initialize MIPI Driver
mipi_config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
mipi_driver = MIPIDriver(mipi_config)

# Initialize Signal Processor
signal_config = SignalConfig(bit_depth=12, noise_reduction_strength=0.5, color_correction_matrix=np.eye(3))
signal_processor = SignalProcessor(signal_config)

# Initialize Power Manager
power_config = PowerConfig(voltage_main=1.8, voltage_io=3.3, current_limit=1.0)
power_manager = PowerManager(power_config)

# Process a frame
raw_frame = mipi_driver.receive_data(frame_size)
processed_frame = signal_processor.process_frame(raw_frame)

# Optimize performance
mipi_driver.optimize_performance()
signal_processor.optimize_performance()
power_manager.optimize_noise_reduction()

# Get system status
mipi_status = mipi_driver.get_status()
power_status = power_manager.get_power_status()

# Calculate performance metrics
snr = calculate_snr(processed_frame, raw_frame - processed_frame)
dynamic_range = calculate_dynamic_range(processed_frame)
color_accuracy, delta_e_values = calculate_color_accuracy(reference_colors, processed_frame)

print(f"MIPI Status: {mipi_status}")
print(f"Power Status: {power_status}")
print(f"SNR: {snr} dB")
print(f"Dynamic Range: {dynamic_range} dB")
print(f"Color Accuracy (Mean Delta E): {color_accuracy}")
```

## 7. Error Handling

All API functions use Python's built-in exception handling mechanism. Here are the common exceptions you might encounter:

- `ValueError`: Raised when an invalid argument is passed to a function, or for invalid configurations.
- `Exception`: Raised for general errors, such as power limit exceeded.
- `RuntimeError`: Raised when an operation fails due to an unexpected condition.
- `IOError`: Raised when a hardware-related operation fails.

Example of error handling:

```python
try:
    mipi_driver.send_data(frame_data)
except ValueError as e:
    print(f"Invalid data format: {e}")
except Exception as e:
    print(f"Error occurred during data transfer: {e}")
```

## 8. Best Practices

1. **Initialization**: Always initialize the MIPI Driver, Signal Processor, and Power Manager with appropriate configurations before use.

2. **Performance Optimization**: Call the `optimize_performance()` methods after the system has been running for a while to adapt to current conditions.

3. **Error Checking**: Always check the return values of methods like `send_data()` and `set_voltage()` to ensure operations were successful.

4. **Resource Management**: Properly clean up resources when they're no longer needed.

5. **Concurrent Access**: Implement proper synchronization mechanisms when accessing objects from multiple threads.

## 9. Version Compatibility

This API documentation is for version 1.1.0 of the Advanced Image Sensor Interface project. Future versions will maintain backwards compatibility for major version numbers (e.g., 1.x.x).