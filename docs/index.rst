.. Advanced Image Sensor Interface documentation master file, created by
   sphinx-quickstart on Thu Jun 27 18:58:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Advanced Image Sensor Interface Documentation (v1.1.0)
======================================================

Welcome to the Advanced Image Sensor Interface documentation! This is a high-level Python simulation and modeling framework for MIPI CSI-2 image sensor pipelines with comprehensive processing and power modeling capabilities.

Overview
--------

The Advanced Image Sensor Interface provides:

* **MIPI CSI-2 Protocol Simulation**: Complete packet-level simulation with ECC/CRC validation
* **Advanced Signal Processing**: Sophisticated noise reduction and image enhancement algorithms  
* **Power Management Modeling**: Simulates power delivery and noise characteristics
* **Multi-Protocol Support**: MIPI CSI-2, GigE Vision, and CoaXPress protocol models
* **Comprehensive Testing**: 122 unit tests with focused coverage on core functionality
* **AI-Based Calibration**: Neural network parameter tuning and optimization

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
   :caption: Contents:

   design_specs.md
   performance_analysis.md
   api_documentation.md
   testing_guide.md



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
