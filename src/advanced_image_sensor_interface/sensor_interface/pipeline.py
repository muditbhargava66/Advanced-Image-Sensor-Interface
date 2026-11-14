"""Explicit unpacked frame decoding and synchronous capture-to-processing.

Primary drivers return raw frame bytes by default. The optional CRC/FEC envelope
is application-layer protection: the sender must use FrameCodec.encode before
transmission. Checksumming bytes after reception is not transport verification;
native protocol checksums (including MIPI CRC) remain the driver's responsibility.
"""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np

from ..types import DepthResult, ProcessingResult
from ..utils.data_integrity import IntegrityChecker, IntegrityCheckResult
from ..utils.depth import StereoDepthProcessor
from .image_validation import ImageFormat, ImageValidator
from .protocol.base import DataTransferError, StreamingProtocolBase
from .protocol_selector import ProtocolSelector


class FrameCodec:
    """Encode/decode an explicitly declared integer image format.

    Storage is C-order, little-endian and unpacked: 8 bits use one byte,
    10/12/14/16 bits use two bytes, and 20 bits use four bytes per sample.
    Floating-point and wire-level bit-packed formats are not supported.
    """

    def __init__(self, image_format: ImageFormat, integrity: IntegrityChecker | None = None):
        if image_format.numpy_dtype.kind != "u":
            raise ValueError("FrameCodec requires an unsigned integer image format")
        self.image_format = image_format
        self.integrity = integrity

    def encode(self, image: np.ndarray) -> bytes:
        """Validate samples and serialize; optionally protect at the sender."""
        ImageValidator.validate_image(image, self.image_format)
        payload = image.astype(self.image_format.numpy_dtype.newbyteorder("<"), copy=False).tobytes(order="C")
        return self.integrity.protect(payload) if self.integrity is not None else payload

    def decode(self, payload: bytes) -> np.ndarray:
        """Verify an optional sender envelope and return owned, validated samples.

        Invalid envelopes, lengths, or sample ranges raise DataTransferError.
        Mutable byte buffers are also accepted, but never retained by the array.
        """
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise DataTransferError("Frame payload must be a byte buffer")
        try:
            data = bytes(payload)
            if self.integrity is not None:
                status, data, _ = self.integrity.verify(data)
                if status not in (IntegrityCheckResult.VALID, IntegrityCheckResult.CORRECTABLE):
                    raise DataTransferError(f"Frame integrity verification failed: {status.value}")
            fmt = self.image_format
            expected_bytes = fmt.height * fmt.width * fmt.channels * fmt.numpy_dtype.itemsize
            if len(data) != expected_bytes:
                raise DataTransferError(f"Frame byte length mismatch: expected {expected_bytes}, got {len(data)}")
            image = np.frombuffer(data, dtype=fmt.numpy_dtype.newbyteorder("<")).reshape(fmt.shape)
            # Copy after reshape so the returned array itself owns the storage;
            # convert back to native dtype on big-endian hosts as well.
            image = image.astype(fmt.numpy_dtype, copy=True)
            ImageValidator.validate_image(image, fmt)
            return image
        except (TypeError, ValueError, BufferError) as exc:
            raise DataTransferError(f"Invalid frame payload: {exc}") from exc


class FramePipeline:
    """Capture through the currently selected streaming driver, then process.

    The caller owns connection/streaming lifecycle and explicit format selection.
    Latency covers capture plus decoding; throughput uses actual received bytes
    (including any application envelope), not configured link capacity.
    """

    def __init__(self, selector: ProtocolSelector, codec: FrameCodec):
        self.selector = selector
        self.codec = codec

    def capture_frame(self) -> np.ndarray:
        """Capture and decode one frame, recording measured capture metrics."""
        protocol = self.selector.get_current_protocol()
        protocol_type = self.selector.current_protocol
        if protocol_type is None or not isinstance(protocol, StreamingProtocolBase):
            raise DataTransferError("A streaming protocol must be selected")
        if not protocol.is_connected or not protocol.is_streaming:
            raise DataTransferError("Selected protocol must be connected and streaming")

        start = perf_counter()
        payload = protocol.capture_frame()
        if payload is None:
            raise DataTransferError("Protocol capture returned no frame")
        frame = self.codec.decode(payload)
        elapsed = perf_counter() - start
        self.selector.record_performance_metrics(
            protocol_type,
            {
                "latency_ms": elapsed * 1000.0,
                "throughput_mbps": len(payload) * 8 / elapsed / 1e6 if elapsed > 0 else 0.0,
                "payload_bytes": len(payload),
            },
        )
        return frame

    def process_frame(self, operation: Callable[[np.ndarray], ProcessingResult]) -> ProcessingResult:
        """Pass decoded samples to a typed operation without hiding its failures.

        Record callback wall time both in result.metrics and selector history.
        Exceptions propagate; unsuccessful typed results are returned unchanged
        apart from their measured processing_time_ms.
        """
        protocol_type = self.selector.current_protocol
        frame = self.capture_frame()
        start = perf_counter()
        try:
            result = operation(frame)
        finally:
            elapsed_ms = (perf_counter() - start) * 1000.0
            self.selector.record_performance_metrics(protocol_type, {"processing_time_ms": elapsed_ms})
        result.metrics.processing_time_ms = elapsed_ms
        return result

    def process_stereo_pair(
        self,
        right: "FramePipeline",
        processor: StereoDepthProcessor,
        focal_length_px: float,
        baseline_m: float,
        export_path: str | Path | None = None,
    ) -> DepthResult:
        """Capture sequentially and delegate stereo processing and optional PLY.

        Acquisition is NOT synchronized or automatically rectified. The caller
        owns calibration, rectification, and suitability of the pair for stereo.
        Capture errors return a failed DepthResult without fabricated arrays.
        """
        try:
            left_frame = self.capture_frame()
            right_frame = right.capture_frame()
        except Exception as exc:
            return DepthResult(success=False, error=f"Stereo capture failed: {exc}")
        return processor.process_stereo_pair(left_frame, right_frame, focal_length_px, baseline_m, export_path=export_path)
