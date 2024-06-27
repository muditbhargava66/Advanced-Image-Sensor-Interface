"""
Benchmarks Package for Advanced Image Sensor Interface

This package contains benchmark tests for evaluating the performance
and quality of the Advanced Image Sensor Interface project.

Modules:
    speed_tests: Benchmarks for measuring processing speed and throughput.
    noise_analysis: Benchmarks for analyzing noise characteristics and reduction efficacy.
"""

from .speed_tests import run_speed_benchmarks
from .noise_analysis import run_noise_analysis

__all__ = ['run_speed_benchmarks', 'run_noise_analysis']

__version__ = '1.0.0'
__author__ = 'Mudit Bhargava'
__license__ = 'MIT'