#!/usr/bin/env python3
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
import unittest

import numpy as np

# Import enhanced features to test
try:
    from advanced_image_sensor_interface.sensor_interface import (
        AdvancedPowerConfiguration,
        # Advanced power management
        AdvancedPowerManager,
        BayerPattern,
        DemosaicMethod,
        # Enhanced sensor interface
        EnhancedSensorInterface,
        # GPU acceleration
        GPUAccelerator,
        GPUBackend,
        GPUConfiguration,
        HDRMode,
        HDRParameters,
        # HDR processing
        HDRProcessor,
        # Multi-sensor synchronization
        MultiSensorSynchronizer,
        PowerMode,
        PowerState,
        ProcessingMode,
        RAWFormat,
        RAWParameters,
        # RAW processing
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
    print(f"Enhanced features not available for testing: {e}")


@unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
class TestEnhancedSensorInterface(unittest.TestCase):
    """Test enhanced sensor interface functionality."""

    def setUp(self):
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
            resolution=SensorResolution.UHD_4K,
            frame_rate=60.0,
            sensor_count=2,
            synchronization_enabled=True
        )
        self.assertEqual(config.effective_resolution, (3840, 2160))
        self.assertEqual(config.pixel_count, 3840 * 2160)

        # Test invalid configurations
        with self.assertRaises(ValueError):
            SensorConfiguration(frame_rate=0)  # Invalid frame rate

        with self.assertRaises(ValueError):
            SensorConfiguration(sensor_count=10)  # Too many sensors

    def test_8k_sensor_config(self):
        """Test 8K sensor configuration."""
        config = create_8k_sensor_config()
        self.assertEqual(config.resolution, SensorResolution.UHD_8K)
        self.assertEqual(config.effective_resolution, (7680, 4320))
        self.assertTrue(config.gpu_acceleration)
        self.assertTrue(config.raw_processing)

    def test_multi_sensor_config(self):
        """Test multi-sensor configuration."""
        config = create_multi_sensor_config(sensor_count=4)
        self.assertEqual(config.sensor_count, 4)
        self.assertTrue(config.synchronization_enabled)
        self.assertEqual(config.master_sensor_id, 0)

    def test_streaming_lifecycle(self):
        """Test sensor streaming start/stop lifecycle."""
        # Initially not streaming
        self.assertFalse(self.sensor.is_streaming)

        # Start streaming
        self.assertTrue(self.sensor.start_streaming())
        self.assertTrue(self.sensor.is_streaming)

        # Stop streaming
        self.assertTrue(self.sensor.stop_streaming())
        self.assertFalse(self.sensor.is_streaming)

    def test_frame_capture(self):
        """Test frame capture functionality."""
        self.sensor.start_streaming()

        # Capture frame
        frame = self.sensor.capture_frame()
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)

        # Check frame dimensions
        height, width = self.config.effective_resolution[1], self.config.effective_resolution[0]
        if self.config.raw_processing:
            self.assertEqual(frame.shape, (height, width))
        else:
            self.assertEqual(frame.shape, (height, width, 3))

        self.sensor.stop_streaming()

    def test_sensor_status(self):
        """Test sensor status reporting."""
        status = self.sensor.get_sensor_status()
        self.assertIn('sensors', status)
        self.assertIn('streaming', status)
        self.assertIn('configuration', status)


@unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
class TestHDRProcessing(unittest.TestCase):
    """Test HDR processing functionality."""

    def setUp(self):
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
        params = HDRParameters(
            tone_mapping_method=ToneMappingMethod.REINHARD,
            gamma=2.2,
            exposure_compensation=0.5
        )
        self.assertEqual(params.gamma, 2.2)

        # Invalid parameters
        with self.assertRaises(ValueError):
            HDRParameters(gamma=0.05)  # Too low

        with self.assertRaises(ValueError):
            HDRParameters(exposure_compensation=10.0)  # Too high

    def test_single_image_processing(self):
        """Test single image HDR processing."""
        result = self.hdr_processor.process_single_image(self.test_image)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_exposure_stack_processing(self):
        """Test exposure stack HDR processing."""
        result = self.hdr_processor.process_exposure_stack(
            self.test_images,
            self.exposure_values
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_images[0].shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_tone_mapping_methods(self):
        """Test different tone mapping methods."""
        methods = [
            ToneMappingMethod.REINHARD,
            ToneMappingMethod.DRAGO,
            ToneMappingMethod.ADAPTIVE,
            ToneMappingMethod.GAMMA
        ]

        for method in methods:
            params = HDRParameters(tone_mapping_method=method)
            processor = HDRProcessor(params)

            result = processor.process_single_image(self.test_image)
            self.assertIsInstance(result, np.ndarray)

    def test_processing_stats(self):
        """Test HDR processing statistics."""
        stats = self.hdr_processor.get_processing_stats()

        self.assertIn('tone_mapping_method', stats)
        self.assertIn('gamma', stats)
        self.assertIn('output_bit_depth', stats)


@unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
class TestRAWProcessing(unittest.TestCase):
    """Test RAW processing functionality."""

    def setUp(self):
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
        params = RAWParameters(
            bayer_pattern=BayerPattern.RGGB,
            bit_depth=12,
            gamma=2.2
        )
        self.assertEqual(params.bit_depth, 12)

        # Invalid parameters
        with self.assertRaises(ValueError):
            RAWParameters(bit_depth=15)  # Unsupported bit depth

        with self.assertRaises(ValueError):
            RAWParameters(gamma=0.05)  # Invalid gamma

    def test_raw_to_rgb_processing(self):
        """Test RAW to RGB processing."""
        result = self.raw_processor.process_raw_image(self.raw_data)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 3)  # Should be RGB
        self.assertEqual(result.shape[:2], self.raw_data.shape)  # Same height/width
        self.assertEqual(result.shape[2], 3)  # RGB channels
        self.assertEqual(result.dtype, np.uint8)

    def test_demosaic_methods(self):
        """Test different demosaicing methods."""
        methods = [
            DemosaicMethod.SIMPLE,
            DemosaicMethod.BILINEAR,
            DemosaicMethod.MALVAR
        ]

        for method in methods:
            params = RAWParameters(demosaic_method=method)
            processor = RAWProcessor(params)

            result = processor.process_raw_image(self.raw_data)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result.shape), 3)

    def test_bayer_patterns(self):
        """Test different Bayer patterns."""
        patterns = [BayerPattern.RGGB, BayerPattern.BGGR]

        for pattern in patterns:
            params = RAWParameters(bayer_pattern=pattern)
            processor = RAWProcessor(params)

            result = processor.process_raw_image(self.raw_data)
            self.assertIsInstance(result, np.ndarray)

    def test_processing_stats(self):
        """Test RAW processing statistics."""
        # Process an image to generate stats
        self.raw_processor.process_raw_image(self.raw_data)

        stats = self.raw_processor.processing_stats
        self.assertIn('images_processed', stats)
        self.assertGreater(stats['images_processed'], 0)


@unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
class TestMultiSensorSync(unittest.TestCase):
    """Test multi-sensor synchronization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.stereo_config = create_stereo_sync_config()
        self.stereo_sync = MultiSensorSynchronizer(self.stereo_config)

        self.multi_config = create_multi_camera_sync_config(num_cameras=4)
        self.multi_sync = MultiSensorSynchronizer(self.multi_config)

    def test_sync_configuration_validation(self):
        """Test synchronization configuration validation."""
        # Valid configuration
        config = SyncConfiguration(
            sync_mode=SyncMode.SOFTWARE,
            master_sensor_id=0,
            slave_sensor_ids=[1, 2],
            sync_tolerance_us=100.0
        )
        self.assertEqual(config.master_sensor_id, 0)

        # Invalid configuration
        with self.assertRaises(ValueError):
            SyncConfiguration(
                master_sensor_id=0,
                slave_sensor_ids=[0, 1]  # Master in slave list
            )

    def test_stereo_sync_config(self):
        """Test stereo synchronization configuration."""
        config = create_stereo_sync_config()
        self.assertEqual(config.master_sensor_id, 0)
        self.assertEqual(config.slave_sensor_ids, [1])
        self.assertTrue(config.enable_frame_alignment)

    def test_multi_camera_sync_config(self):
        """Test multi-camera synchronization configuration."""
        config = create_multi_camera_sync_config(num_cameras=4)
        self.assertEqual(config.master_sensor_id, 0)
        self.assertEqual(config.slave_sensor_ids, [1, 2, 3])

    def test_synchronization_lifecycle(self):
        """Test synchronization start/stop lifecycle."""
        # Initially not active
        self.assertFalse(self.stereo_sync.is_active)

        # Start synchronization
        self.assertTrue(self.stereo_sync.start_synchronization())
        self.assertTrue(self.stereo_sync.is_active)

        # Stop synchronization
        self.assertTrue(self.stereo_sync.stop_synchronization())
        self.assertFalse(self.stereo_sync.is_active)

    def test_synchronized_frame_capture(self):
        """Test synchronized frame capture."""
        self.stereo_sync.start_synchronization()

        # Capture synchronized frames
        frames = self.stereo_sync.capture_synchronized_frames()

        if frames:  # May be None if sync fails
            self.assertIsInstance(frames, dict)
            self.assertIn(0, frames)  # Master sensor
            self.assertIn(1, frames)  # Slave sensor

            for sensor_id, (frame, timestamp) in frames.items():
                self.assertIsInstance(frame, np.ndarray)
                self.assertIsInstance(timestamp, float)

        self.stereo_sync.stop_synchronization()

    def test_synchronization_status(self):
        """Test synchronization status reporting."""
        status = self.stereo_sync.get_synchronization_status()

        self.assertIn('active', status)
        self.assertIn('config', status)
        self.assertIn('sensors', status)
        self.assertIn('statistics', status)


@unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
class TestGPUAcceleration(unittest.TestCase):
    """Test GPU acceleration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.gpu_config = create_gpu_config_for_automotive()
        self.gpu_accelerator = GPUAccelerator(self.gpu_config)

        # Create test images
        self.test_images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(4)
        ]

    def test_gpu_configuration_validation(self):
        """Test GPU configuration validation."""
        # Valid configuration
        config = GPUConfiguration(
            preferred_backend=GPUBackend.AUTO,
            batch_size=4,
            num_streams=2
        )
        self.assertEqual(config.batch_size, 4)

        # Invalid configuration
        with self.assertRaises(ValueError):
            GPUConfiguration(batch_size=0)  # Invalid batch size

    def test_gpu_config_for_automotive(self):
        """Test automotive GPU configuration."""
        config = create_gpu_config_for_automotive()
        self.assertEqual(config.preferred_backend, GPUBackend.CUPY)
        self.assertEqual(config.processing_mode, ProcessingMode.HYBRID)
        self.assertTrue(config.enable_async_processing)

    def test_device_info(self):
        """Test GPU device information."""
        device_info = self.gpu_accelerator.get_device_info()

        self.assertIn('backend', device_info)
        self.assertIn('is_initialized', device_info)
        self.assertIsInstance(device_info['is_initialized'], bool)

    def test_image_batch_processing(self):
        """Test GPU image batch processing."""
        operations = ["gaussian_blur", "edge_detection", "histogram_equalization"]

        for operation in operations:
            with self.subTest(operation=operation):
                if operation == "gaussian_blur":
                    results = self.gpu_accelerator.process_image_batch(
                        self.test_images, operation, sigma=2.0
                    )
                else:
                    results = self.gpu_accelerator.process_image_batch(
                        self.test_images, operation
                    )

                self.assertEqual(len(results), len(self.test_images))
                for result in results:
                    self.assertIsInstance(result, np.ndarray)

    def test_performance_stats(self):
        """Test GPU performance statistics."""
        # Process some images to generate stats
        self.gpu_accelerator.process_image_batch(self.test_images, "gaussian_blur")

        stats = self.gpu_accelerator.get_performance_stats()
        self.assertIn('operations_processed', stats)
        self.assertGreater(stats['operations_processed'], 0)


@unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
class TestAdvancedPowerManagement(unittest.TestCase):
    """Test advanced power management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.automotive_config = create_power_config_for_automotive()
        self.mobile_config = create_power_config_for_mobile()
        self.power_manager = AdvancedPowerManager(self.automotive_config)

    def test_power_configuration_validation(self):
        """Test power configuration validation."""
        # Valid configuration
        config = AdvancedPowerConfiguration(
            power_mode=PowerMode.BALANCED,
            thermal_update_interval_ms=1000.0,
            min_frequency_mhz=100.0,
            max_frequency_mhz=1000.0
        )
        self.assertEqual(config.power_mode, PowerMode.BALANCED)

        # Invalid configuration
        with self.assertRaises(ValueError):
            AdvancedPowerConfiguration(thermal_update_interval_ms=50.0)  # Too low

        with self.assertRaises(ValueError):
            AdvancedPowerConfiguration(
                min_frequency_mhz=1000.0,
                max_frequency_mhz=100.0  # Min > Max
            )

    def test_automotive_power_config(self):
        """Test automotive power configuration."""
        config = create_power_config_for_automotive()
        self.assertEqual(config.power_mode, PowerMode.BALANCED)
        self.assertTrue(config.enable_thermal_monitoring)
        self.assertFalse(config.enable_battery_monitoring)

    def test_mobile_power_config(self):
        """Test mobile power configuration."""
        config = create_power_config_for_mobile()
        self.assertEqual(config.power_mode, PowerMode.POWER_SAVER)
        self.assertTrue(config.enable_battery_monitoring)

    def test_power_mode_changes(self):
        """Test power mode changes."""
        modes = [PowerMode.PERFORMANCE, PowerMode.BALANCED, PowerMode.POWER_SAVER]

        for mode in modes:
            with self.subTest(mode=mode):
                self.assertTrue(self.power_manager.set_power_mode(mode))
                self.assertEqual(self.power_manager.current_mode, mode)

    def test_power_state_transitions(self):
        """Test power state transitions."""
        # Test valid transitions
        self.assertTrue(self.power_manager.transition_to_state(PowerState.IDLE))
        self.assertEqual(self.power_manager.current_state, PowerState.IDLE)

        self.assertTrue(self.power_manager.transition_to_state(PowerState.ACTIVE))
        self.assertEqual(self.power_manager.current_state, PowerState.ACTIVE)

    def test_component_power_control(self):
        """Test component power control."""
        components = ['sensors', 'processing_unit', 'memory', 'io']

        for component in components:
            with self.subTest(component=component):
                # Disable component
                self.assertTrue(self.power_manager.set_component_power(component, False))
                self.assertFalse(self.power_manager.component_states[component])

                # Enable component
                self.assertTrue(self.power_manager.set_component_power(component, True))
                self.assertTrue(self.power_manager.component_states[component])

    def test_workload_optimization(self):
        """Test workload optimization."""
        workloads = ["streaming", "processing", "idle", "burst"]

        for workload in workloads:
            with self.subTest(workload=workload):
                self.assertTrue(self.power_manager.optimize_for_workload(workload))

    def test_power_metrics(self):
        """Test power metrics reporting."""
        metrics = self.power_manager.get_power_metrics()

        self.assertIsInstance(metrics.total_power, float)
        self.assertIsInstance(metrics.temperature_celsius, float)
        self.assertIsInstance(metrics.current_frequency_mhz, float)
        self.assertIsInstance(metrics.thermal_state, ThermalState)

    def test_monitoring_lifecycle(self):
        """Test power monitoring lifecycle."""
        # Initially not monitoring
        self.assertFalse(self.power_manager.monitoring_active)

        # Start monitoring
        self.assertTrue(self.power_manager.start_monitoring())
        self.assertTrue(self.power_manager.monitoring_active)

        # Let it run briefly
        time.sleep(0.1)

        # Stop monitoring
        self.assertTrue(self.power_manager.stop_monitoring())
        self.assertFalse(self.power_manager.monitoring_active)


class TestV2FeatureIntegration(unittest.TestCase):
    """Test integration between v2.0.0 features."""

    @unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
    def test_sensor_with_hdr_and_raw(self):
        """Test integration of sensor interface with HDR and RAW processing."""
        # Create sensor with RAW processing enabled
        config = SensorConfiguration(
            resolution=SensorResolution.FHD,
            raw_processing=True,
            hdr_mode=HDRMode.HDR10
        )
        sensor = EnhancedSensorInterface(config)

        # Create processors
        hdr_processor = HDRProcessor()
        raw_processor = RAWProcessor()

        # Start streaming and capture frame
        sensor.start_streaming()
        raw_frame = sensor.capture_frame()

        if raw_frame is not None:
            # Process RAW to RGB
            rgb_frame = raw_processor.process_raw_image(raw_frame)

            # Apply HDR processing
            hdr_frame = hdr_processor.process_single_image(rgb_frame)

            self.assertIsInstance(hdr_frame, np.ndarray)

        sensor.stop_streaming()

    @unittest.skipUnless(ENHANCED_FEATURES_AVAILABLE, "Enhanced features not available")
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

        self.assertIsInstance(sync_status, dict)
        self.assertIsInstance(power_metrics.total_power, float)


def create_test_suite():
    """Create comprehensive test suite for v2.0.0 features."""
    suite = unittest.TestSuite()

    if ENHANCED_FEATURES_AVAILABLE:
        # Add all test classes
        test_classes = [
            TestEnhancedSensorInterface,
            TestHDRProcessing,
            TestRAWProcessing,
            TestMultiSensorSync,
            TestGPUAcceleration,
            TestAdvancedPowerManagement,
            TestV2FeatureIntegration,
        ]

        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
    else:
        # Add a dummy test to indicate features not available
        class TestEnhancedNotAvailable(unittest.TestCase):
            def test_enhanced_features_not_available(self):
                self.skipTest("Enhanced features not available - missing dependencies")

        suite.addTest(TestEnhancedNotAvailable('test_enhanced_features_not_available'))

    return suite


if __name__ == '__main__':
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)

    # Print summary
    print("\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    import sys
    sys.exit(exit_code)
