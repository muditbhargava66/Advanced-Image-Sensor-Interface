.. Advanced Image Sensor Interface documentation master file, created by
   sphinx-quickstart on Thu Jun 27 18:58:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Advanced Image Sensor Interface Documentation (v3.2.0)
========================================================

Welcome to the Advanced Image Sensor Interface documentation! This is a comprehensive Python framework for camera interface protocols with advanced image processing, multi-sensor synchronization, AI/ML enhancements, neural calibration tuning, custom extensions, and professional-grade calibration capabilities.

Overview
--------

The Advanced Image Sensor Interface provides:

* **Multi-Protocol Support**: MIPI CSI-2 (D-PHY v2.5), CoaXPress (CXP-12), GigE Vision (RoCE), and USB3 Vision protocols
* **Advanced Image Processing**: HDR processing, RAW image pipeline, and GPU acceleration
* **Multi-Sensor Synchronization**: Hardware and software synchronization with ~1ms accuracy (simulation)
* **Professional Calibration**: Comprehensive camera calibration with distortion correction
* **Data Integrity**: CRC-32 validation and Reed-Solomon Forward Error Correction
* **Lens Correction**: Real-time radial and tangential distortion correction (Brown-Conrady model)
* **Enhanced Buffer Management**: Asynchronous buffer operations with intelligent memory pooling
* **Power Management**: Advanced power states (7 states), thermal management, and multi-system coordination
* **Security Framework**: MIPI AES-GCM encryption with key management and PRE_SHARED_KEY authentication
* **AI/ML Enhancements**: Neural Calibration Tuner, SceneClassifier, NoisePredictor, QualityAssessor
* **Custom Extensions**: AINoiseReducer, AdaptiveColorCorrector with scikit-learn backends
* **Comprehensive Testing**: 329 automated tests in the current suite
* **Production-Ready**: 100% linting compliance and robust CI/CD pipeline

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Basic usage:

.. code-block:: python

   from advanced_image_sensor_interface import MIPIDriver, MIPIConfig
   
   # Initialize MIPI driver
   config = MIPIConfig(lanes=4, data_rate=2.5, channel=0)
   driver = MIPIDriver(config)
   
   # Get status
   status = driver.get_status()
   print(f"Driver status: {status}")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   system_architecture.md
   design_specs.md
   api_documentation.md
   testing_guide.md

.. toctree::
   :maxdepth: 2
   :caption: Camera Protocols:

   protocol_comparison_guide.md
   protocol_mipi_csi2.md
   protocol_coaxpress.md
   protocol_gige_vision.md
   protocol_usb3_vision.md
   protocols.md
   protocols_index.md

.. toctree::
   :maxdepth: 2
   :caption: Hardware Integration:

   hardware_integration.md

.. toctree::
   :maxdepth: 2
   :caption: Calibration & Configuration:

   calibration.md

.. toctree::
   :maxdepth: 2
   :caption: Performance & Analysis:

   performance_analysis.md
   system_architecture.md

.. toctree::
   :maxdepth: 2
   :caption: AI/ML Enhancements:

   api_documentation.md

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   api_reference.md
   protocols.md
   protocol_mipi_csi2.md
   protocol_coaxpress.md
   protocol_gige_vision.md
   protocol_usb3_vision.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`