.. Advanced Image Sensor Interface documentation master file, created by
   sphinx-quickstart on Thu Jun 27 18:58:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Advanced Image Sensor Interface Documentation (v2.0.0)
======================================================

Welcome to the Advanced Image Sensor Interface documentation! This is a comprehensive Python framework for camera interface protocols with advanced image processing, multi-sensor synchronization, and professional-grade calibration capabilities.

Overview
--------

The Advanced Image Sensor Interface provides:

* **Multi-Protocol Support**: MIPI CSI-2, CoaXPress, GigE Vision, and USB3 Vision protocols
* **Advanced Image Processing**: HDR processing, RAW image pipeline, and GPU acceleration
* **Multi-Sensor Synchronization**: Hardware and software synchronization with sub-millisecond accuracy
* **Professional Calibration**: Comprehensive camera calibration with distortion correction
* **Enhanced Buffer Management**: Asynchronous buffer operations with intelligent memory pooling
* **Power Management**: Advanced power states and thermal management
* **Comprehensive Testing**: 200+ unit tests with extensive protocol and integration testing
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

   design_specs.md
   api_documentation.md
   api_reference.md
   testing_guide.md

.. toctree::
   :maxdepth: 2
   :caption: Camera Protocols:

   protocols_index.md
   protocol_comparison_guide.md
   protocol_mipi_csi2.md
   protocol_coaxpress.md
   protocol_gige_vision.md
   protocol_usb3_vision.md
   protocols.md

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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
