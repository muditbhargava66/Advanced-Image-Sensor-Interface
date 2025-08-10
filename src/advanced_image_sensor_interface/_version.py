"""Version information for Advanced Image Sensor Interface."""

__version__ = "2.0.0"
__version_info__ = (2, 0, 0)

# Release information
__title__ = "Advanced Image Sensor Interface"
__description__ = "A comprehensive multi-protocol camera interface framework"
__author__ = "Mudit Bhargava"
__author_email__ = "muditbhargava666@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/muditbhargava66/Advanced-Image-Sensor-Interface"

# Version history
VERSION_HISTORY = {
    "2.0.0": {
        "release_date": "2025-01-10",
        "major_features": [
            "Multi-protocol support (MIPI CSI-2, CoaXPress, GigE Vision, USB3 Vision)",
            "Enhanced sensor interface with 8K resolution support",
            "HDR image processing with multiple tone mapping algorithms",
            "RAW image processing with Bayer demosaicing",
            "Multi-sensor synchronization with <100μs accuracy",
            "GPU acceleration with CUDA/OpenCL support",
            "Advanced power management with thermal monitoring",
            "Professional calibration system",
            "Asynchronous buffer management",
        ],
        "breaking_changes": [
            "API redesign for protocol interface standardization",
            "Configuration schema changes with validation",
            "Updated buffer management API with context managers",
            "Python 3.10+ requirement",
        ],
    },
    "1.1.0": {
        "release_date": "2025-09-08",
        "major_features": [
            "Production-ready CI/CD pipeline",
            "Enhanced documentation and API references",
            "Comprehensive security framework",
            "Advanced image processing algorithms",
            "Performance benchmarking suite",
            "MIPI CSI-2 protocol implementation",
        ],
    },
    "1.0.1": {
        "release_date": "2025-03-04",
        "major_features": [
            "Bug fixes and stability improvements",
            "Comprehensive test suite",
            "Type checking improvements",
            "Enhanced error handling",
        ],
    },
    "1.0.0": {
        "release_date": "2024-01-15",
        "major_features": [
            "Initial release",
            "MIPI Driver implementation",
            "Signal processing pipeline",
            "Power management system",
            "Performance metrics utilities",
        ],
    },
}


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple[int, int, int]:
    """Get the current version as a tuple."""
    return __version_info__


def get_release_info() -> dict[str, str]:
    """Get release information."""
    return {
        "version": __version__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "author_email": __author_email__,
        "license": __license__,
        "url": __url__,
    }
