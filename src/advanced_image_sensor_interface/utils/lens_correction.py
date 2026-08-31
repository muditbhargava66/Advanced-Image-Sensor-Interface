"""
Lens Correction Module

Provides real-time geometric distortion correction for camera lenses.
Supports radial (barrel/pincushion) and tangential (decentering) distortion.

Key Features:
- Radial distortion correction (Brown-Conrady model)
- Tangential distortion correction
- Lens profile management
- GPU-accelerated processing (NumPy vectorized)

Author: Advanced Image Sensor Interface Team
Version: 3.2.0
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..types import LensCorrectionResult, ProcessingMetrics

logger = logging.getLogger(__name__)


class DistortionType(Enum):
    """Types of lens distortion."""

    BARREL = "barrel"  # Negative radial distortion
    PINCUSHION = "pincushion"  # Positive radial distortion
    MUSTACHE = "mustache"  # Complex combination
    NONE = "none"


class InterpolationMethod(Enum):
    """Image interpolation methods."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


@dataclass
class LensProfile:
    """
    Lens calibration profile.

    Contains distortion coefficients and optical center information
    for a specific lens/camera combination.

    Attributes:
        name: Profile identifier
        focal_length_mm: Lens focal length in mm
        sensor_width_mm: Sensor width in mm
        image_width: Image width in pixels
        image_height: Image height in pixels
        k1, k2, k3: Radial distortion coefficients
        p1, p2: Tangential distortion coefficients
        cx, cy: Optical center (principal point)
        fx, fy: Focal lengths in pixels
    """

    name: str = "default"
    focal_length_mm: float = 50.0
    sensor_width_mm: float = 36.0
    image_width: int = 1920
    image_height: int = 1080

    # Radial distortion coefficients (Brown-Conrady model)
    k1: float = 0.0  # Primary radial
    k2: float = 0.0  # Secondary radial
    k3: float = 0.0  # Tertiary radial

    # Tangential distortion coefficients
    p1: float = 0.0  # Decentering
    p2: float = 0.0  # Decentering

    # Optical center (None = image center)
    cx: Optional[float] = None
    cy: Optional[float] = None

    # Focal length in pixels (None = calculated from physical params)
    fx: Optional[float] = None
    fy: Optional[float] = None

    def __post_init__(self) -> None:
        """Initialize derived parameters."""
        # Calculate optical center if not specified
        if self.cx is None:
            self.cx = self.image_width / 2.0
        if self.cy is None:
            self.cy = self.image_height / 2.0

        # Calculate focal length in pixels if not specified
        if self.fx is None:
            self.fx = (self.focal_length_mm * self.image_width) / self.sensor_width_mm
        if self.fy is None:
            self.fy = self.fx  # Assume square pixels

    @classmethod
    def barrel_distortion(cls, strength: float = 0.1, image_size: tuple[int, int] = (1920, 1080)) -> "LensProfile":
        """Create a barrel distortion profile."""
        return cls(
            name="barrel", image_width=image_size[0], image_height=image_size[1], k1=-abs(strength), k2=-abs(strength) * 0.1
        )

    @classmethod
    def pincushion_distortion(cls, strength: float = 0.1, image_size: tuple[int, int] = (1920, 1080)) -> "LensProfile":
        """Create a pincushion distortion profile."""
        return cls(
            name="pincushion", image_width=image_size[0], image_height=image_size[1], k1=abs(strength), k2=abs(strength) * 0.1
        )

    @property
    def distortion_type(self) -> DistortionType:
        """Determine distortion type from coefficients."""
        if abs(self.k1) < 1e-6 and abs(self.k2) < 1e-6:
            return DistortionType.NONE
        elif self.k1 < 0:
            return DistortionType.BARREL
        elif self.k1 > 0:
            return DistortionType.PINCUSHION
        else:
            return DistortionType.MUSTACHE


@dataclass
class CorrectionResult:
    """Result of lens correction operation."""

    image: np.ndarray
    processing_time_ms: float = 0.0
    pixels_corrected: int = 0
    max_displacement: float = 0.0


class RadialDistortionCorrector:
    """
    Radial distortion correction using Brown-Conrady model.

    Corrects barrel and pincushion distortion caused by lens geometry.
    """

    def __init__(self, profile: LensProfile) -> None:
        """
        Initialize radial distortion corrector.

        Args:
            profile: Lens calibration profile
        """
        self.profile = profile
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._maps_valid = False

    def build_correction_maps(self) -> None:
        """Pre-compute distortion correction maps.

        Maps are cached for efficient repeated correction.

        The radial distortion model follows the Brown-Conrady formula::

            r_distorted = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)

        where k1, k2, k3 are the radial distortion coefficients and r is
        the normalised radial distance from the optical centre. Positive
        k values produce barrel distortion (edges bow inward); negative
        values produce pincushion distortion (edges bow outward). The
        mapping is inverted so that the correction maps the distorted
        pixel location back to its undistorted source.
        """
        h, w = self.profile.image_height, self.profile.image_width
        cx, cy = self.profile.cx, self.profile.cy
        fx, fy = self.profile.fx, self.profile.fy
        k1, k2, k3 = self.profile.k1, self.profile.k2, self.profile.k3

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalize coordinates
        x_norm = (x_coords - cx) / fx
        y_norm = (y_coords - cy) / fy

        # Calculate radial distance
        r_squared = x_norm**2 + y_norm**2
        r_fourth = r_squared**2
        r_sixth = r_squared**3

        # Radial distortion factor
        radial_factor = 1 + k1 * r_squared + k2 * r_fourth + k3 * r_sixth

        # Calculate distorted coordinates
        x_distorted = x_norm * radial_factor
        y_distorted = y_norm * radial_factor

        # Convert back to pixel coordinates
        self._map_x = (x_distorted * fx + cx).astype(np.float32)
        self._map_y = (y_distorted * fy + cy).astype(np.float32)
        self._maps_valid = True

        logger.debug(f"Built correction maps: {w}x{h}")

    def correct(self, image: np.ndarray, interpolation: InterpolationMethod = InterpolationMethod.BILINEAR) -> np.ndarray:
        """
        Apply radial distortion correction to an image.

        Args:
            image: Input image (HxWxC or HxW)
            interpolation: Interpolation method

        Returns:
            Corrected image
        """
        if not self._maps_valid:
            # Adjust profile for actual image size if different
            if image.shape[:2] != (self.profile.image_height, self.profile.image_width):
                self.profile.image_height, self.profile.image_width = image.shape[:2]
                self.profile.cx = self.profile.image_width / 2.0
                self.profile.cy = self.profile.image_height / 2.0
            self.build_correction_maps()

        return self._remap_image(image, interpolation)

    def _remap_image(self, image: np.ndarray, interpolation: InterpolationMethod) -> np.ndarray:
        """
        Remap image using pre-computed maps.

        Args:
            image: Input image
            interpolation: Interpolation method

        Returns:
            Remapped image
        """
        if interpolation == InterpolationMethod.NEAREST:
            return self._remap_nearest(image)
        elif interpolation == InterpolationMethod.BILINEAR:
            return self._remap_bilinear(image)
        else:
            return self._remap_bilinear(image)  # Default to bilinear

    def _remap_nearest(self, image: np.ndarray) -> np.ndarray:
        """Remap using nearest neighbor interpolation."""
        h, w = image.shape[:2]
        map_x = np.clip(np.round(self._map_x).astype(np.int32), 0, w - 1)
        map_y = np.clip(np.round(self._map_y).astype(np.int32), 0, h - 1)

        if image.ndim == 3:
            return image[map_y, map_x]
        else:
            return image[map_y, map_x]

    def _remap_bilinear(self, image: np.ndarray) -> np.ndarray:
        """Remap using bilinear interpolation."""
        h, w = image.shape[:2]

        # Clip coordinates to valid range
        x0 = np.clip(np.floor(self._map_x).astype(np.int32), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(np.floor(self._map_y).astype(np.int32), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        # Calculate interpolation weights
        dx = self._map_x - x0
        dy = self._map_y - y0

        # Clamp weights
        dx = np.clip(dx, 0, 1)
        dy = np.clip(dy, 0, 1)

        if image.ndim == 3:
            dx = dx[:, :, np.newaxis]
            dy = dy[:, :, np.newaxis]

        # Bilinear interpolation
        top = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
        bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
        result = top * (1 - dy) + bottom * dy

        return result.astype(image.dtype)


class TangentialDistortionCorrector:
    """
    Tangential distortion correction for lens decentering.

    Corrects distortion caused by lens elements not being perfectly aligned
    (decentering). The tangential distortion model follows the Brown-Conrady
    formulation:

        dx = 2*p1*x*y + p2*(r^2 + 2*x^2)
        dy = p1*(r^2 + 2*y^2) + 2*p2*x*y

    where p1, p2 are the tangential (decentering) distortion coefficients,
    x, y are normalized coordinates relative to the optical center, and
    r^2 = x^2 + y^2. This models the effect of the lens being tilted or
    decentered relative to the image sensor.
    """

    def __init__(self, profile: LensProfile) -> None:
        """
        Initialize tangential distortion corrector.

        Args:
            profile: Lens calibration profile
        """
        self.profile = profile
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._maps_valid = False

    def build_correction_maps(self) -> None:
        """Pre-compute tangential distortion correction maps.

        The tangential distortion (decentering) model computes the displacement
        of a point from its ideal position due to lens misalignment:

            dx = 2*p1*x*y + p2*(r^2 + 2*x^2)
            dy = p1*(r^2 + 2*y^2) + 2*p2*x*y

        This is the standard Brown-Conrady tangential distortion formulation.
        The coefficients p1 and p2 model the decentering of the lens along
        the x and y axes respectively. The correction is inverted so that
        the map transforms from distorted coordinates back to undistorted.
        """
        h, w = self.profile.image_height, self.profile.image_width
        cx, cy = self.profile.cx, self.profile.cy
        fx, fy = self.profile.fx, self.profile.fy
        p1, p2 = self.profile.p1, self.profile.p2

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalize coordinates
        x_norm = (x_coords - cx) / fx
        y_norm = (y_coords - cy) / fy

        # Calculate radial distance
        r_squared = x_norm**2 + y_norm**2

        # Tangential distortion
        dx = 2 * p1 * x_norm * y_norm + p2 * (r_squared + 2 * x_norm**2)
        dy = p1 * (r_squared + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm

        # Calculate distorted coordinates
        x_distorted = x_norm + dx
        y_distorted = y_norm + dy

        # Convert back to pixel coordinates
        self._map_x = (x_distorted * fx + cx).astype(np.float32)
        self._map_y = (y_distorted * fy + cy).astype(np.float32)
        self._maps_valid = True

    def correct(self, image: np.ndarray, interpolation: InterpolationMethod = InterpolationMethod.BILINEAR) -> np.ndarray:
        """Apply tangential distortion correction."""
        if not self._maps_valid:
            if image.shape[:2] != (self.profile.image_height, self.profile.image_width):
                self.profile.image_height, self.profile.image_width = image.shape[:2]
                self.profile.cx = self.profile.image_width / 2.0
                self.profile.cy = self.profile.image_height / 2.0
            self.build_correction_maps()

        # Use RadialDistortionCorrector's remap logic
        radial = RadialDistortionCorrector(self.profile)
        radial._map_x = self._map_x
        radial._map_y = self._map_y
        radial._maps_valid = True
        return radial._remap_image(image, interpolation)


class LensCorrectionPipeline:
    """
    Complete lens correction pipeline.

    Combines radial and tangential distortion correction into a
    unified processing pipeline with caching and optimization.
    """

    def __init__(self, profile: Optional[LensProfile] = None) -> None:
        """
        Initialize lens correction pipeline.

        Args:
            profile: Lens calibration profile (None creates default)
        """
        self.profile = profile or LensProfile()
        self._radial_corrector = RadialDistortionCorrector(self.profile)
        self._tangential_corrector = TangentialDistortionCorrector(self.profile)
        self._combined_map_x: Optional[np.ndarray] = None
        self._combined_map_y: Optional[np.ndarray] = None
        self._maps_valid = False

        # Processing statistics
        self._images_processed = 0
        self._total_processing_time = 0.0

    def set_profile(self, profile: LensProfile) -> None:
        """
        Set new lens profile.

        Args:
            profile: New lens calibration profile
        """
        self.profile = profile
        self._radial_corrector = RadialDistortionCorrector(profile)
        self._tangential_corrector = TangentialDistortionCorrector(profile)
        self._maps_valid = False

    def build_combined_maps(self) -> None:
        """Build combined correction maps for full distortion correction."""
        h, w = self.profile.image_height, self.profile.image_width
        cx, cy = self.profile.cx, self.profile.cy
        fx, fy = self.profile.fx, self.profile.fy

        k1, k2, k3 = self.profile.k1, self.profile.k2, self.profile.k3
        p1, p2 = self.profile.p1, self.profile.p2

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalize coordinates
        x_norm = (x_coords - cx) / fx
        y_norm = (y_coords - cy) / fy

        # Calculate radial distance
        r_squared = x_norm**2 + y_norm**2
        r_fourth = r_squared**2
        r_sixth = r_squared**3

        # Combined distortion model
        # Radial
        radial_factor = 1 + k1 * r_squared + k2 * r_fourth + k3 * r_sixth

        # Tangential
        dx_tangential = 2 * p1 * x_norm * y_norm + p2 * (r_squared + 2 * x_norm**2)
        dy_tangential = p1 * (r_squared + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm

        # Combined distorted coordinates
        x_distorted = x_norm * radial_factor + dx_tangential
        y_distorted = y_norm * radial_factor + dy_tangential

        # Convert back to pixel coordinates
        self._combined_map_x = (x_distorted * fx + cx).astype(np.float32)
        self._combined_map_y = (y_distorted * fy + cy).astype(np.float32)
        self._maps_valid = True

        logger.info(f"Built combined correction maps: {w}x{h}")

    def correct(
        self, image: np.ndarray, interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
    ) -> LensCorrectionResult:
        """
        Apply full lens correction to an image.

        Args:
            image: Input image (HxWxC or HxW)
            interpolation: Interpolation method

        Returns:
            LensCorrectionResult with corrected image and statistics
        """
        start_time = time.perf_counter()

        # Handle size mismatch
        if image.shape[:2] != (self.profile.image_height, self.profile.image_width):
            self.profile.image_height, self.profile.image_width = image.shape[:2]
            self.profile.cx = self.profile.image_width / 2.0
            self.profile.cy = self.profile.image_height / 2.0
            self._maps_valid = False

        # Build maps if needed
        if not self._maps_valid:
            self.build_combined_maps()

        # Apply correction using radial corrector's remap
        corrector = RadialDistortionCorrector(self.profile)
        corrector._map_x = self._combined_map_x
        corrector._map_y = self._combined_map_y
        corrector._maps_valid = True
        corrected = corrector._remap_image(image, interpolation)

        # Calculate statistics
        processing_time = (time.perf_counter() - start_time) * 1000
        self._images_processed += 1
        self._total_processing_time += processing_time

        # Calculate max displacement
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = np.abs(self._combined_map_x - x_coords)
        dy = np.abs(self._combined_map_y - y_coords)
        max_displacement = float(np.sqrt(np.max(dx**2 + dy**2)))

        # Determine distortion type
        radial_correction = abs(self.profile.k1) > 1e-6 or abs(self.profile.k2) > 1e-6 or abs(self.profile.k3) > 1e-6
        tangential_correction = abs(self.profile.p1) > 1e-6 or abs(self.profile.p2) > 1e-6
        distortion_type = self.profile.distortion_type.value

        metrics = ProcessingMetrics(
            processing_time_ms=processing_time,
            algorithm_name="LensCorrectionPipeline",
            parameters={
                "profile_name": self.profile.name,
                "distortion_type": distortion_type,
                "radial_correction": radial_correction,
                "tangential_correction": tangential_correction,
                "k1": self.profile.k1,
                "k2": self.profile.k2,
                "k3": self.profile.k3,
                "p1": self.profile.p1,
                "p2": self.profile.p2,
            },
        )

        return LensCorrectionResult(
            success=True,
            data=corrected,
            warnings=[],
            metrics=metrics,
            pixels_corrected=h * w,
            max_displacement=max_displacement,
            distortion_type=distortion_type,
            radial_correction_applied=radial_correction,
            tangential_correction_applied=tangential_correction,
        )

    def get_statistics(self) -> dict:
        """Get processing statistics."""
        avg_time = self._total_processing_time / self._images_processed if self._images_processed > 0 else 0.0
        return {
            "images_processed": self._images_processed,
            "total_processing_time_ms": self._total_processing_time,
            "average_processing_time_ms": avg_time,
            "profile_name": self.profile.name,
            "distortion_type": self.profile.distortion_type.value,
        }

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._images_processed = 0
        self._total_processing_time = 0.0


# Pre-defined lens profiles for common lenses
STANDARD_PROFILES = {
    "gopro_wide": LensProfile(
        name="gopro_wide", focal_length_mm=3.0, sensor_width_mm=6.17, k1=-0.35, k2=0.12, k3=-0.02, p1=0.0001, p2=0.0002
    ),
    "smartphone_wide": LensProfile(name="smartphone_wide", focal_length_mm=4.5, sensor_width_mm=5.64, k1=-0.15, k2=0.03),
    "dslr_kit": LensProfile(name="dslr_kit", focal_length_mm=18.0, sensor_width_mm=23.6, k1=-0.02, k2=0.005),
    "industrial_cs": LensProfile(
        name="industrial_cs", focal_length_mm=8.0, sensor_width_mm=6.4, k1=-0.08, k2=0.015, p1=0.0005, p2=0.0003  # 1/3" sensor
    ),
}
