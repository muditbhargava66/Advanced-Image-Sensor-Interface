"""
Test Patterns Module

This module provides functionality for generating various test patterns
used in image sensor testing and calibration.

Classes:
    PatternGenerator: Main class for generating test patterns.

Functions:
    get_available_patterns: Returns a list of available test pattern names.
"""

from .._version import __author__, __license__, __version__
from .pattern_generator import PatternGenerator, get_available_patterns

__all__ = ["PatternGenerator", "get_available_patterns"]
