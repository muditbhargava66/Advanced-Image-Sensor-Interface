"""
Pattern Generator for Image Sensor Testing

This module provides a comprehensive set of test pattern generation functions
for image sensor testing and calibration. It includes various standard patterns
such as color bars, grayscale ramps, checkerboards, and more advanced patterns
like slanted edges and Siemens stars.

Classes:
    PatternGenerator: Main class for generating various test patterns.

Functions:
    get_available_patterns: Returns a list of available test pattern names.

Usage:
    generator = PatternGenerator(width=1920, height=1080)
    color_bars = generator.generate_pattern('color_bars')
"""

import logging
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternGenerator:
    """
    A class for generating various test patterns for image sensor testing.

    Attributes
    ----------
        width (int): Width of the generated patterns.
        height (int): Height of the generated patterns.
        bit_depth (int): Bit depth of the generated patterns.

    """

    def __init__(self, width: int = 1920, height: int = 1080, bit_depth: int = 8):
        """
        Initialize the PatternGenerator with specified dimensions and bit depth.

        Args:
        ----
            width (int): Width of the generated patterns. Default is 1920.
            height (int): Height of the generated patterns. Default is 1080.
            bit_depth (int): Bit depth of the generated patterns. Default is 8.

        """
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.max_value = 2**bit_depth - 1
        logger.info(f"PatternGenerator initialized with {width}x{height} resolution and {bit_depth}-bit depth")

    def generate_pattern(self, pattern_name: str, **kwargs: Any) -> np.ndarray:
        """
        Generate a specified test pattern.

        Args:
        ----
            pattern_name (str): Name of the pattern to generate.
            **kwargs: Additional arguments specific to each pattern.

        Returns:
        -------
            np.ndarray: The generated test pattern.

        Raises:
        ------
            ValueError: If the specified pattern name is not recognized.

        """
        pattern_functions = {
            "color_bars": self._generate_color_bars,
            "grayscale_ramp": self._generate_grayscale_ramp,
            "checkerboard": self._generate_checkerboard,
            "slanted_edge": self._generate_slanted_edge,
            "siemens_star": self._generate_siemens_star,
            "zone_plate": self._generate_zone_plate,
            "noise": self._generate_noise,
        }

        if pattern_name not in pattern_functions:
            raise ValueError(f"Unknown pattern name: {pattern_name}")

        logger.info(f"Generating {pattern_name} pattern")
        return pattern_functions[pattern_name](**kwargs)

    def _generate_color_bars(self, num_bars: int = 8) -> np.ndarray:
        """Generate a color bar pattern."""
        pattern = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bar_width = self.width // num_bars
        colors = [
            (255, 255, 255),  # White
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (0, 255, 0),  # Green
            (255, 0, 255),  # Magenta
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            (0, 0, 0),  # Black
        ]
        for i, color in enumerate(colors[:num_bars]):
            pattern[:, i * bar_width : (i + 1) * bar_width] = color
        return pattern

    def _generate_grayscale_ramp(self) -> np.ndarray:
        """Generate a grayscale ramp pattern."""
        return np.linspace(0, self.max_value, self.width, dtype=np.uint8).reshape(1, -1).repeat(self.height, 0)

    def _generate_checkerboard(self, square_size: int = 64) -> np.ndarray:
        """Generate a checkerboard pattern."""
        pattern = np.zeros((self.height, self.width), dtype=np.uint8)
        pattern[::2, ::2] = self.max_value
        pattern[1::2, 1::2] = self.max_value
        return pattern

    def _generate_slanted_edge(self, angle: float = 5.0) -> np.ndarray:
        """Generate a slanted edge pattern."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        xx, yy = np.meshgrid(x, y)
        edge = (xx * np.cos(np.radians(angle)) + yy * np.sin(np.radians(angle))) > self.width / 2
        return edge.astype(np.uint8) * self.max_value

    def _generate_siemens_star(self, num_segments: int = 144) -> np.ndarray:
        """Generate a Siemens star pattern."""
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        angle = np.arctan2(yy, xx)
        pattern = np.sin(angle * num_segments / 2) > 0
        return pattern.astype(np.uint8) * self.max_value

    def _generate_zone_plate(self) -> np.ndarray:
        """Generate a zone plate pattern."""
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        pattern = (1 + np.sin(2 * np.pi * self.width * r**2)) / 2
        return (pattern * self.max_value).astype(np.uint8)

    def _generate_noise(self, mean: float = 128, std_dev: float = 32) -> np.ndarray:
        """Generate a noise pattern."""
        noise = np.random.normal(mean, std_dev, (self.height, self.width))
        return np.clip(noise, 0, self.max_value).astype(np.uint8)


def get_available_patterns() -> list[str]:
    """
    Get a list of available test pattern names.

    Returns
    -------
        List[str]: A list of available pattern names.

    """
    return ["color_bars", "grayscale_ramp", "checkerboard", "slanted_edge", "siemens_star", "zone_plate", "noise"]


# Example usage and testing
if __name__ == "__main__":
    generator = PatternGenerator(width=1920, height=1080, bit_depth=8)

    # Generate and test each pattern
    for pattern_name in get_available_patterns():
        pattern = generator.generate_pattern(pattern_name)
        logger.info(f"Generated {pattern_name} pattern with shape {pattern.shape} and dtype {pattern.dtype}")
        assert pattern.shape in ((1080, 1920), (1080, 1920, 3)), f"Unexpected shape for {pattern_name}"
        assert pattern.dtype == np.uint8, f"Unexpected dtype for {pattern_name}"
        assert np.min(pattern) >= 0 and np.max(pattern) <= 255, f"Values out of range for {pattern_name}"

    logger.info("All patterns generated and validated successfully")

    # Example of generating a specific pattern with custom parameters
    custom_color_bars = generator.generate_pattern("color_bars", num_bars=10)
    logger.info(f"Generated custom color bars pattern with shape {custom_color_bars.shape}")

    # Test error handling
    try:
        generator.generate_pattern("non_existent_pattern")
    except ValueError as e:
        logger.info(f"Successfully caught error: {e!s}")
