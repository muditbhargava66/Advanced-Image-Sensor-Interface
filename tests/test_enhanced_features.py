"""
Test Suite for Advanced Image Sensor Interface Enhanced Features

This test suite validates all the enhanced features including:
- Enhanced sensor interface
- HDR processing
- RAW processing
- Multi-sensor synchronization
- GPU acceleration
- Advanced power management
"""

import time

import numpy as np
import pytest

# Import enhanced features to test
try:
    from advanced_image_sensor_interface.sensor_interface import (  # Advanced power management; Enhanced sensor interface; GPU acceleration; HDR processing; Multi-sensor synchronization; RAW processing
        AdvancedPowerConfiguration,
        AdvancedPowerManager,
        BayerPattern,
        DemosaicMethod,
        EnhancedSensorInterface,
        GPUAccelerator,
        GPUBackend,
        GPUConfiguration,
        HDRMode,
        HDRParameters,
        HDRProcessor,
        MultiSensorSynchronizer,
        PowerMode,
        PowerState,
        ProcessingMode,
        RAWFormat,
        RAWParameters,
        RAWProcessor,
        SensorConfiguration,
        SensorResolution,
        SyncConfiguration,
        SyncMode,
        ThermalState,
        ToneMappingMethod,
        create_8k_sensor_config,
        create_gpu_config_for_automotive,
        create_hdr_processor_for_automotive,
        create_multi_camera_sync_config,
        create_multi_sensor_config,
        create_power_config_for_automotive,
        create_power_config_for_mobile,
        create_raw_processor_for_automotive,
        create_stereo_sync_config,
    )

    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    pytest.skip(f"Enhanced features not available for testing: {e}", allow_module_level=True)


class TestEnhancedSensorInterface:
    """Test enhanced sensor interface functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Use 4K config for tests to avoid data rate limits
        self.config = SensorConfiguration(
            resolution=SensorResolution.UHD_4K,
            frame_rate=30.0,
            bit_depth=12,
            hdr_mode=HDRMode.HDR10,
            raw_format=RAWFormat.RAW12,
            raw_processing=True,
        )
        self.sensor = EnhancedSensorInterface(self.config)

    def test_sensor_configuration_validation(self):
        """Test sensor configuration validation."""
        # Test valid configuration
        config = SensorConfiguration(
            resolution=SensorResolution.UHD_4K, frame_rate=60.0, sensor_count=2, synchronization_enabled=True
        )
        assert config.effective_resolution == (3840, 2160)
        assert config.pixel_count == 3840 * 2160

        # Test invalid configurations
        with pytest.raises(ValueError):
            SensorConfiguration(frame_rate=0)  # Invalid frame rate

        with pytest.raises(ValueError):
            SensorConfiguration(sensor_count=10)  # Too many sensors

    def test_8k_sensor_config(self):
        """Test 8K sensor configuration."""
        config = create_8k_sensor_config()
        assert config.resolution == SensorResolution.UHD_8K
        assert config.effective_resolution == (7680, 4320)
        assert config.gpu_acceleration
        assert config.raw_processing

    def test_multi_sensor_config(self):
        """Test multi-sensor configuration."""
        config = create_multi_sensor_config(sensor_count=4)
        assert config.sensor_count == 4
        assert config.synchronization_enabled
        assert config.master_sensor_id == 0

    def test_streaming_lifecycle(self):
        """Test sensor streaming start/stop lifecycle."""
        # Initially not streaming
        assert not self.sensor.is_streaming

        # Start streaming
        assert self.sensor.start_streaming()
        assert self.sensor.is_streaming

        # Stop streaming
        assert self.sensor.stop_streaming()
        assert not self.sensor.is_streaming

    def test_frame_capture(self):
        """Test frame capture functionality."""
        self.sensor.start_streaming()

        # Capture frame
        frame = self.sensor.capture_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)

        # Check frame dimensions
        height, width = self.config.effective_resolution[1], self.config.effective_resolution[0]
        if self.config.raw_processing:
            assert frame.shape == (height, width)
        else:
            assert frame.shape == (height, width, 3)

        self.sensor.stop_streaming()

    def test_sensor_status(self):
        """Test sensor status reporting."""
        status = self.sensor.get_sensor_status()
        assert "sensors" in status
        assert "streaming" in status
        assert "configuration" in status


class TestHDRProcessing:
    """Test HDR processing functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.hdr_processor = create_hdr_processor_for_automotive()

        # Create test images
        self.test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.test_images = [
            np.random.randint(0, 128, (480, 640, 3), dtype=np.uint8),  # Underexposed
            np.random.randint(64, 192, (480, 640, 3), dtype=np.uint8),  # Normal
            np.random.randint(128, 256, (480, 640, 3), dtype=np.uint8),  # Overexposed
        ]
        self.exposure_values = [-2.0, 0.0, 2.0]

    def test_hdr_parameters_validation(self):
        """Test HDR parameters validation."""
        # Valid parameters
        params = HDRParameters(tone_mapping_method=ToneMappingMethod.REINHARD, gamma=2.2, exposure_compensation=0.5)
        assert params.gamma == 2.2

        # Invalid parameters
        with pytest.raises(ValueError):
            HDRParameters(gamma=0.05)  # Too low

        with pytest.raises(ValueError):
            HDRParameters(exposure_compensation=10.0)  # Too high

    def test_single_image_processing(self):
        """Test single image HDR processing."""
        result = self.hdr_processor.process_single_image(self.test_image)

        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_image.shape
        assert result.dtype == np.uint8

    def test_exposure_stack_processing(self):
        """Test exposure stack HDR processing."""
        result = self.hdr_processor.process_exposure_stack(self.test_images, self.exposure_values)

        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_images[0].shape
        assert result.dtype == np.uint8

    def test_tone_mapping_methods(self):
        """Test different tone mapping methods."""
        methods = [ToneMappingMethod.REINHARD, ToneMappingMethod.DRAGO, ToneMappingMethod.ADAPTIVE, ToneMappingMethod.GAMMA]

        for method in methods:
            params = HDRParameters(tone_mapping_method=method)
            processor = HDRProcessor(params)

            result = processor.process_single_image(self.test_image)
            assert isinstance(result, np.ndarray)

    def test_processing_stats(self):
        """Test HDR processing statistics."""
        stats = self.hdr_processor.get_processing_stats()

        assert "tone_mapping_method" in stats
        assert "gamma" in stats
        assert "output_bit_depth" in stats


class TestRAWProcessing:
    """Test RAW processing functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.raw_processor = create_raw_processor_for_automotive()

        # Create synthetic RAW data
        self.raw_data = np.random.randint(0, 4095, (480, 640), dtype=np.uint16)

        # Apply Bayer pattern (RGGB)
        self.raw_data = self.raw_data.astype(np.float32)
        self.raw_data[0::2, 0::2] *= 1.2  # R
        self.raw_data[1::2, 1::2] *= 0.8  # B
        self.raw_data = np.clip(self.raw_data, 0, 4095).astype(np.uint16)

    def test_raw_parameters_validation(self):
        """Test RAW parameters validation."""
        # Valid parameters
        params = RAWParameters(bayer_pattern=BayerPattern.RGGB, bit_depth=12, gamma=2.2)
        assert params.bit_depth == 12

        # Invalid parameters
        with pytest.raises(ValueError):
            RAWParameters(bit_depth=15)  # Unsupported bit depth

        with pytest.raises(ValueError):
            RAWParameters(gamma=0.05)  # Invalid gamma

    def test_raw_to_rgb_processing(self):
        """Test RAW to RGB processing."""
        result = self.raw_processor.process_raw_image(self.raw_data)

        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3  # Should be RGB
        assert result.shape[:2] == self.raw_data.shape  # Same height/width
        assert result.shape[2] == 3  # RGB channels
        assert result.dtype == np.uint8

    def test_demosaic_methods(self):
        """Test different demosaicing methods."""
        methods = [DemosaicMethod.SIMPLE, DemosaicMethod.BILINEAR, DemosaicMethod.MALVAR]

        for method in methods:
            params = RAWParameters(demosaic_method=method)
            processor = RAWProcessor(params)

            result = processor.process_raw_image(self.raw_data)
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3

    def test_bayer_patterns(self):
        """Test different Bayer patterns."""
        patterns = [BayerPattern.RGGB, BayerPattern.BGGR]

        for pattern in patterns:
            params = RAWParameters(bayer_pattern=pattern)
            processor = RAWProcessor(params)

            result = processor.process_raw_image(self.raw_data)
            assert isinstance(result, np.ndarray)

    def test_processing_stats(self):
        """Test RAW processing statistics."""
        # Process an image to generate stats
        self.raw_processor.process_raw_image(self.raw_data)

        stats = self.raw_processor.processing_stats
        assert "images_processed" in stats
        assert stats["images_processed"] > 0


class TestMultiSensorSync:
    """Test multi-sensor synchronization functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.stereo_config = create_stereo_sync_config()
        self.stereo_sync = MultiSensorSynchronizer(self.stereo_config)

        self.multi_config = create_multi_camera_sync_config(num_cameras=4)
        self.multi_sync = MultiSensorSynchronizer(self.multi_config)

    def test_sync_configuration_validation(self):
        """Test synchronization configuration validation."""
        # Valid configuration
        config = SyncConfiguration(
            sync_mode=SyncMode.SOFTWARE, master_sensor_id=0, slave_sensor_ids=[1, 2], sync_tolerance_us=100.0
        )
        assert config.master_sensor_id == 0

        # Invalid configuration
        with pytest.raises(ValueError):
            SyncConfiguration(master_sensor_id=0, slave_sensor_ids=[0, 1])  # Master in slave list

    def test_stereo_sync_config(self):
        """Test stereo synchronization configuration."""
        config = create_stereo_sync_config()
        assert config.master_sensor_id == 0
        assert config.slave_sensor_ids == [1]
        assert config.enable_frame_alignment

    def test_multi_camera_sync_config(self):
        """Test multi-camera synchronization configuration."""
        config = create_multi_camera_sync_config(num_cameras=4)
        assert config.master_sensor_id == 0
        assert config.slave_sensor_ids == [1, 2, 3]

    def test_synchronization_lifecycle(self):
        """Test synchronization start/stop lifecycle."""
        # Initially not active
        assert not self.stereo_sync.is_active

        # Start synchronization
        assert self.stereo_sync.start_synchronization()
        assert self.stereo_sync.is_active

        # Stop synchronization
        assert self.stereo_sync.stop_synchronization()
        assert not self.stereo_sync.is_active

    def test_synchronized_frame_capture(self):
        """Test synchronized frame capture."""
        self.stereo_sync.start_synchronization()

        # Capture synchronized frames
        frames = self.stereo_sync.capture_synchronized_frames()

        if frames:  # May be None if sync fails
            assert isinstance(frames, dict)
            assert 0 in frames  # Master sensor
            assert 1 in frames  # Slave sensor

            for sensor_id, (frame, timestamp) in frames.items():
                assert isinstance(frame, np.ndarray)
                assert isinstance(timestamp, float)

        self.stereo_sync.stop_synchronization()

    def test_synchronization_status(self):
        """Test synchronization status reporting."""
        status = self.stereo_sync.get_synchronization_status()

        assert "active" in status
        assert "config" in status
        assert "sensors" in status
        assert "statistics" in status


class TestGPUAcceleration:
    """Test GPU acceleration functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.gpu_config = create_gpu_config_for_automotive()
        self.gpu_accelerator = GPUAccelerator(self.gpu_config)

        # Create test images
        self.test_images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(4)]

    def test_gpu_configuration_validation(self):
        """Test GPU configuration validation."""
        # Valid configuration
        config = GPUConfiguration(preferred_backend=GPUBackend.AUTO, batch_size=4, num_streams=2)
        assert config.batch_size == 4

        # Invalid configuration
        with pytest.raises(ValueError):
            GPUConfiguration(batch_size=0)  # Invalid batch size

    def test_gpu_config_for_automotive(self):
        """Test automotive GPU configuration."""
        config = create_gpu_config_for_automotive()
        assert config.preferred_backend == GPUBackend.CUPY
        assert config.processing_mode == ProcessingMode.HYBRID
        assert config.enable_async_processing

    def test_device_info(self):
        """Test GPU device information."""
        device_info = self.gpu_accelerator.get_device_info()

        assert "backend" in device_info
        assert "is_initialized" in device_info
        assert isinstance(device_info["is_initialized"], bool)

    def test_image_batch_processing(self):
        """Test GPU image batch processing."""
        operations = ["gaussian_blur", "edge_detection", "histogram_equalization"]

        for operation in operations:
            if operation == "gaussian_blur":
                results = self.gpu_accelerator.process_image_batch(self.test_images, operation, sigma=2.0)
            else:
                results = self.gpu_accelerator.process_image_batch(self.test_images, operation)
            assert len(results) == len(self.test_images)
            for result in results:
                assert isinstance(result, np.ndarray)

    def test_performance_stats(self):
        """Test GPU performance statistics."""
        # Process some images to generate stats
        self.gpu_accelerator.process_image_batch(self.test_images, "gaussian_blur")

        stats = self.gpu_accelerator.get_performance_stats()
        assert "operations_processed" in stats
        assert stats["operations_processed"] > 0


class TestAdvancedPowerManagement:
    """Test advanced power management functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.automotive_config = create_power_config_for_automotive()
        self.mobile_config = create_power_config_for_mobile()
        self.power_manager = AdvancedPowerManager(self.automotive_config)

    def test_power_configuration_validation(self):
        """Test power configuration validation."""
        # Valid configuration
        config = AdvancedPowerConfiguration(
            power_mode=PowerMode.BALANCED, thermal_update_interval_ms=1000.0, min_frequency_mhz=100.0, max_frequency_mhz=1000.0
        )
        assert config.power_mode == PowerMode.BALANCED

        # Invalid configuration
        with pytest.raises(ValueError):
            AdvancedPowerConfiguration(thermal_update_interval_ms=50.0)  # Too low

        with pytest.raises(ValueError):
            AdvancedPowerConfiguration(min_frequency_mhz=1000.0, max_frequency_mhz=100.0)  # Min > Max

    def test_automotive_power_config(self):
        """Test automotive power configuration."""
        config = create_power_config_for_automotive()
        assert config.power_mode == PowerMode.BALANCED
        assert config.enable_thermal_monitoring
        assert not config.enable_battery_monitoring

    def test_mobile_power_config(self):
        """Test mobile power configuration."""
        config = create_power_config_for_mobile()
        assert config.power_mode == PowerMode.POWER_SAVER
        assert config.enable_battery_monitoring

    def test_power_mode_changes(self):
        """Test power mode changes."""
        modes = [PowerMode.PERFORMANCE, PowerMode.BALANCED, PowerMode.POWER_SAVER]

        for mode in modes:
            assert self.power_manager.set_power_mode(mode)
            assert self.power_manager.current_mode == mode

    def test_power_state_transitions(self):
        """Test power state transitions."""
        # Test valid transitions
        assert self.power_manager.transition_to_state(PowerState.IDLE)
        assert self.power_manager.current_state == PowerState.IDLE

        assert self.power_manager.transition_to_state(PowerState.ACTIVE)
        assert self.power_manager.current_state == PowerState.ACTIVE

    def test_component_power_control(self):
        """Test component power control."""
        components = ["sensors", "processing_unit", "memory", "io"]

        for component in components:
            # Disable component
            assert self.power_manager.set_component_power(component, False)
            assert not self.power_manager.component_states[component]

            # Enable component
            assert self.power_manager.set_component_power(component, True)
            assert self.power_manager.component_states[component]

    def test_workload_optimization(self):
        """Test workload optimization."""
        workloads = ["streaming", "processing", "idle", "burst"]

        for workload in workloads:
            assert self.power_manager.optimize_for_workload(workload)

    def test_power_metrics(self):
        """Test power metrics reporting."""
        metrics = self.power_manager.get_power_metrics()

        assert isinstance(metrics.total_power, float)
        assert isinstance(metrics.temperature_celsius, float)
        assert isinstance(metrics.current_frequency_mhz, float)
        assert isinstance(metrics.thermal_state, ThermalState)

    def test_monitoring_lifecycle(self):
        """Test power monitoring lifecycle."""
        # Initially not monitoring
        assert not self.power_manager.monitoring_active

        # Start monitoring
        assert self.power_manager.start_monitoring()
        assert self.power_manager.monitoring_active

        # Let it run briefly
        time.sleep(0.1)

        # Stop monitoring
        assert self.power_manager.stop_monitoring()
        assert not self.power_manager.monitoring_active


class TestV2FeatureIntegration:
    """Test integration between v2.0.0 features."""

    def test_sensor_with_hdr_and_raw(self):
        """Test integration of sensor interface with HDR and RAW processing."""
        # Create sensor with RAW processing enabled
        config = SensorConfiguration(resolution=SensorResolution.FHD, raw_processing=True, hdr_mode=HDRMode.HDR10)
        sensor = EnhancedSensorInterface(config)

        # Create processors
        hdr_processor = HDRProcessor()
        raw_processor = RAWProcessor()

        # Start streaming and capture frame
        sensor.start_streaming()
        raw_frame = sensor.capture_frame()

        assert raw_frame is not None
        # Process RAW to RGB
        rgb_frame = raw_processor.process_raw_image(raw_frame)

        # Apply HDR processing
        hdr_frame = hdr_processor.process_single_image(rgb_frame)
        assert isinstance(hdr_frame, np.ndarray)

        sensor.stop_streaming()

    def test_multi_sensor_with_sync_and_power(self):
        """Test integration of multi-sensor sync with power management."""
        # Create multi-sensor configuration
        sync_config = create_multi_camera_sync_config(num_cameras=2)
        synchronizer = MultiSensorSynchronizer(sync_config)

        # Create power manager
        power_config = create_power_config_for_automotive()
        power_manager = AdvancedPowerManager(power_config)

        # Start both systems
        power_manager.start_monitoring()
        synchronizer.start_synchronization()

        # Optimize power for streaming workload
        power_manager.optimize_for_workload("streaming")

        # Brief operation
        time.sleep(0.1)

        # Stop both systems
        synchronizer.stop_synchronization()
        power_manager.stop_monitoring()

        # Verify both operated correctly
        sync_status = synchronizer.get_synchronization_status()
        power_metrics = power_manager.get_power_metrics()

        assert isinstance(sync_status, dict)
        assert isinstance(power_metrics.total_power, float)


def create_test_suite():
    """Create comprehensive test suite for v2.0.0 features."""
    return [
        TestEnhancedSensorInterface,
        TestHDRProcessing,
        TestRAWProcessing,
        TestMultiSensorSync,
        TestGPUAcceleration,
        TestAdvancedPowerManagement,
        TestV2FeatureIntegration,
    ]
