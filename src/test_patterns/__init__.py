"""
Test Patterns Module

This module provides functionality for generating various test patterns
used in image sensor testing and calibration.

Classes:
    PatternGenerator: Main class for generating test patterns.

Functions:
    get_available_patterns: Returns a list of available test pattern names.
"""

from .pattern_generator import PatternGenerator, get_available_patterns

__all__ = ['PatternGenerator', 'get_available_patterns']

__version__ = '1.0.0'
__author__ = 'Mudit Bhargava'
__license__ = 'MIT'