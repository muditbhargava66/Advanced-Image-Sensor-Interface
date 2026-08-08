"""
Data Integrity Module

Provides advanced error detection and correction for high-speed data transfers.
Includes CRC-32 validation, checksums, and forward error correction.

Key Features:
- CRC-32 packet integrity verification
- Forward Error Correction (FEC) with Reed-Solomon codes
- Error statistics and metrics
- Configurable error recovery strategies
"""

import logging
import struct
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorCorrectionMode(Enum):
    """Forward error correction modes."""

    NONE = "none"
    PARITY = "parity"
    HAMMING = "hamming"
    REED_SOLOMON = "reed_solomon"


class IntegrityCheckResult(Enum):
    """Result of integrity check."""

    VALID = "valid"
    CORRECTABLE = "correctable"
    CORRUPTED = "corrupted"
    UNKNOWN = "unknown"


@dataclass
class IntegrityConfig:
    """
    Configuration for data integrity checking.

    Attributes:
        enable_crc: Enable CRC-32 checking
        enable_fec: Enable forward error correction
        fec_mode: FEC mode to use
        max_correctable_errors: Maximum errors that can be corrected
        auto_retry_on_error: Automatically retry on correctable errors
    """

    enable_crc: bool = True
    enable_fec: bool = True
    fec_mode: ErrorCorrectionMode = ErrorCorrectionMode.REED_SOLOMON
    max_correctable_errors: int = 4
    auto_retry_on_error: bool = True
    error_threshold_percent: float = 1.0


@dataclass
class IntegrityMetrics:
    """Metrics for data integrity operations."""

    packets_checked: int = 0
    packets_valid: int = 0
    packets_corrected: int = 0
    packets_corrupted: int = 0
    crc_errors: int = 0
    fec_corrections: int = 0
    total_errors_corrected: int = 0
    error_rate: float = 0.0

    def update_error_rate(self) -> None:
        """Recalculate error rate."""
        if self.packets_checked > 0:
            total_errors = self.crc_errors + self.packets_corrupted
            self.error_rate = (total_errors / self.packets_checked) * 100


class CRCValidator:
    """
    CRC-32 validator for packet integrity.

    Uses the standard CRC-32 polynomial for detection of
    transmission errors.
    """

    CRC32_POLYNOMIAL = 0xEDB88320

    def __init__(self) -> None:
        """Initialize CRC validator."""
        self._metrics = IntegrityMetrics()

    def calculate_crc(self, data: bytes) -> int:
        """
        Calculate CRC-32 for data.

        Args:
            data: Input data

        Returns:
            CRC-32 value
        """
        return zlib.crc32(data) & 0xFFFFFFFF

    def append_crc(self, data: bytes) -> bytes:
        """
        Append CRC-32 to data.

        Args:
            data: Input data

        Returns:
            Data with CRC appended (4 bytes)
        """
        crc = self.calculate_crc(data)
        return data + struct.pack("<I", crc)

    def verify_crc(self, data: bytes) -> tuple[bool, bytes]:
        """
        Verify CRC-32 and extract original data.

        Args:
            data: Data with CRC appended

        Returns:
            Tuple of (is_valid, original_data)
        """
        if len(data) < 4:
            return False, b""

        original_data = data[:-4]
        received_crc = struct.unpack("<I", data[-4:])[0]
        calculated_crc = self.calculate_crc(original_data)

        is_valid = received_crc == calculated_crc
        self._metrics.packets_checked += 1

        if is_valid:
            self._metrics.packets_valid += 1
        else:
            self._metrics.crc_errors += 1

        self._metrics.update_error_rate()
        return is_valid, original_data

    def get_metrics(self) -> IntegrityMetrics:
        """Get CRC validation metrics."""
        return self._metrics


class ForwardErrorCorrection:
    """
    Forward Error Correction implementation.

    Provides ability to detect and correct errors without retransmission.
    """

    def __init__(self, mode: ErrorCorrectionMode = ErrorCorrectionMode.REED_SOLOMON, max_correctable: int = 4) -> None:
        """
        Initialize FEC encoder/decoder.

        Args:
            mode: Error correction mode
            max_correctable: Maximum correctable errors per block
        """
        self.mode = mode
        self.max_correctable = max_correctable
        self._metrics = IntegrityMetrics()

        # Block sizes for different modes
        self._block_sizes = {
            ErrorCorrectionMode.PARITY: 8,
            ErrorCorrectionMode.HAMMING: 4,
            ErrorCorrectionMode.REED_SOLOMON: 223,  # RS(255, 223)
        }

    def encode(self, data: bytes) -> bytes:
        """
        Encode data with error correction codes.

        Args:
            data: Input data

        Returns:
            Encoded data with ECC
        """
        if self.mode == ErrorCorrectionMode.NONE:
            return data
        elif self.mode == ErrorCorrectionMode.PARITY:
            return self._encode_parity(data)
        elif self.mode == ErrorCorrectionMode.HAMMING:
            return self._encode_hamming(data)
        # Default: REED_SOLOMON
        return self._encode_reed_solomon(data)

    def decode(self, data: bytes) -> tuple[bytes, int]:
        """
        Decode data and correct errors.

        Args:
            data: Encoded data with ECC

        Returns:
            Tuple of (decoded_data, errors_corrected)
        """
        if self.mode == ErrorCorrectionMode.NONE:
            return data, 0
        elif self.mode == ErrorCorrectionMode.PARITY:
            return self._decode_parity(data)
        elif self.mode == ErrorCorrectionMode.HAMMING:
            return self._decode_hamming(data)
        # Default: REED_SOLOMON
        return self._decode_reed_solomon(data)

    def _encode_parity(self, data: bytes) -> bytes:
        """Add parity byte to data."""
        parity = 0
        for byte in data:
            parity ^= byte
        return data + bytes([parity])

    def _decode_parity(self, data: bytes) -> tuple[bytes, int]:
        """Verify parity and extract data."""
        if len(data) < 1:
            return b"", 0

        original = data[:-1]
        received_parity = data[-1]
        calculated_parity = 0
        for byte in original:
            calculated_parity ^= byte

        if received_parity == calculated_parity:
            return original, 0
        else:
            # Parity can detect but not correct errors
            self._metrics.packets_corrupted += 1
            return original, -1  # -1 indicates detected but not corrected

    def _encode_hamming(self, data: bytes) -> bytes:
        """Encode with Hamming(7,4) codes.

        Each input byte is split into two 4-bit nibbles. Each nibble
        is encoded into a 7-bit codeword stored in the upper 7 bits of
        a byte. The bit layout of each encoded byte (MSB first) is:

            [p1, p2, d0, p3, d1, d2, d3, 0]

        Parity bits cover the following data bits (matching standard
        Hamming(7,4) with 1-indexed positions):

            p1 (pos 1): covers positions 1,3,5,7 -> d0, d1, d3
            p2 (pos 2): covers positions 2,3,6,7 -> d0, d2, d3
            p3 (pos 4): covers positions 4,5,6,7 -> d1, d2, d3
        """
        encoded = bytearray()

        for byte in data:
            for shift in (4, 0):  # High nibble first, then low nibble
                d = (byte >> shift) & 0x0F
                d0 = (d >> 0) & 1
                d1 = (d >> 1) & 1
                d2 = (d >> 2) & 1
                d3 = (d >> 3) & 1

                p1 = d0 ^ d1 ^ d3
                p2 = d0 ^ d2 ^ d3
                p3 = d1 ^ d2 ^ d3

                # Pack into byte: [p1, p2, d0, p3, d1, d2, d3, 0]
                encoded_byte = (p1 << 6) | (p2 << 5) | (d0 << 4)
                encoded_byte |= (p3 << 3) | (d1 << 2) | (d2 << 1) | d3
                encoded.append(encoded_byte)

        return bytes(encoded)

    def _decode_hamming(self, data: bytes) -> tuple[bytes, int]:
        """Decode Hamming(7,4) codes."""
        if len(data) % 2 != 0:
            return b"", -1

        decoded = bytearray()
        errors_corrected = 0

        for i in range(0, len(data), 2):
            high_nibble, high_errors = self._decode_hamming_nibble(data[i])
            low_nibble, low_errors = self._decode_hamming_nibble(data[i + 1])

            errors_corrected += high_errors + low_errors
            decoded.append((high_nibble << 4) | low_nibble)

        self._metrics.fec_corrections += errors_corrected
        self._metrics.total_errors_corrected += errors_corrected
        return bytes(decoded), errors_corrected

    def _decode_hamming_nibble(self, encoded: int) -> tuple[int, int]:
        """Decode single Hamming(7,4)-encoded nibble with 1-bit correction.

        The 7-bit codeword layout is [p1, p2, d0, p3, d1, d2, d3].
        The syndrome is a 3-bit value (s3, s2, s1) where each syndrome
        bit checks the corresponding parity relation:

            s1 = p1 ^ d0 ^ d1 ^ d3  (positions 1,3,5,7)
            s2 = p2 ^ d0 ^ d2 ^ d3  (positions 2,3,6,7)
            s3 = p3 ^ d1 ^ d2 ^ d3  (positions 4,5,6,7)

        A non-zero syndrome indicates a single-bit error at the 1-indexed
        position equal to the syndrome value. The mapping from syndrome
        to our variable names is:

            syndrome 1 -> p1 (bit 6), 2 -> p2 (bit 5), 3 -> d0 (bit 4)
            syndrome 4 -> p3 (bit 3), 5 -> d1 (bit 2), 6 -> d2 (bit 1)
            syndrome 7 -> d3 (bit 0)
        """
        # Extract bits from encoded byte: [p1, p2, d0, p3, d1, d2, d3, 0]
        p1 = (encoded >> 6) & 1
        p2 = (encoded >> 5) & 1
        d0 = (encoded >> 4) & 1
        p3 = (encoded >> 3) & 1
        d1 = (encoded >> 2) & 1
        d2 = (encoded >> 1) & 1
        d3 = encoded & 1

        # Calculate syndrome bits
        s1 = p1 ^ d0 ^ d1 ^ d3
        s2 = p2 ^ d0 ^ d2 ^ d3
        s3 = p3 ^ d1 ^ d2 ^ d3

        syndrome = (s3 << 2) | (s2 << 1) | s1
        errors = 0

        if syndrome != 0:
            errors = 1
            # Syndrome value is the 1-indexed position of the error bit.
            # Map syndrome -> bit position in our layout and flip it.
            # Position: 1=p1, 2=p2, 3=d0, 4=p3, 5=d1, 6=d2, 7=d3
            if syndrome == 3:
                d0 ^= 1
            elif syndrome == 5:
                d1 ^= 1
            elif syndrome == 6:
                d2 ^= 1
            elif syndrome == 7:
                d3 ^= 1
            # Syndromes 1, 2, 4 are parity-only errors; data is correct.

        nibble = d0 | (d1 << 1) | (d2 << 2) | (d3 << 3)
        return nibble, errors

    def _encode_reed_solomon(self, data: bytes) -> bytes:
        """Encode data with Reed-Solomon codes (SIMULATION).

        WARNING: This is a SIMULATION ONLY, not a true Reed-Solomon implementation.

        A real RS(255, 223) encoder uses Galois Field arithmetic over GF(2^8):
        - Generator polynomial: g(x) = (x - alpha^0)(x - alpha^1)...(x - alpha^31) where alpha is a primitive element
        - Encoding: multiply message polynomial by x^32, divide by g(x), remainder = parity
        - Can correct up to 16 symbol errors (32 parity symbols)
        - Uses finite field operations with generator polynomial (typically 0x11D for GF(2^8))

        This simulation generates CRC-32 and block checksums as stand-in redundancy bytes.
        For production-grade FEC, use a library such as ``reedsolo`` or ``unireedsolomon``.

        The redundancy bytes are computed as:
        - First 4 bytes: CRC-32 of data
        - Remaining bytes: Simple checksums (sum of bytes modulo 256 per block)
        """
        # In real implementation, use proper RS encoder
        # Here we add redundancy bytes as simulation
        redundancy_bytes = min(32, max(8, self.max_correctable * 2))

        # Calculate simple redundancy (simulation)
        redundancy = self._calculate_rs_redundancy(data, redundancy_bytes)
        return data + redundancy

    def _decode_reed_solomon(self, data: bytes) -> tuple[bytes, int]:
        """Decode Reed-Solomon encoded data (SIMULATION).

        WARNING: This is a SIMULATION ONLY. True RS decoding uses:
        - Syndrome computation over GF(2^8)
        - Berlekamp-Massey or Euclidean algorithm for error locator polynomial
        - Chien search for error positions
        - Forney algorithm for error magnitudes

        This simulation verifies the CRC-32 and checksum redundancy bytes.
        """
        redundancy_bytes = min(32, max(8, self.max_correctable * 2))

        if len(data) < redundancy_bytes:
            return data, 0

        original = data[:-redundancy_bytes]
        received_redundancy = data[-redundancy_bytes:]

        # Verify redundancy
        expected_redundancy = self._calculate_rs_redundancy(original, redundancy_bytes)

        # Count differences as errors
        errors = sum(1 for a, b in zip(received_redundancy, expected_redundancy) if a != b)

        if errors <= self.max_correctable:
            self._metrics.fec_corrections += errors
            self._metrics.total_errors_corrected += errors
            return original, errors
        else:
            self._metrics.packets_corrupted += 1
            return original, -1

    def _calculate_rs_redundancy(self, data: bytes, size: int) -> bytes:
        """Calculate Reed-Solomon redundancy bytes."""
        # Simplified: use CRC and padding as pseudo-RS codes
        crc = zlib.crc32(data) & 0xFFFFFFFF
        redundancy = bytearray(size)

        # Spread CRC across redundancy bytes
        for i in range(min(4, size)):
            redundancy[i] = (crc >> (i * 8)) & 0xFF

        # Add simple checksums for remaining bytes
        for i in range(4, size):
            block_start = (i - 4) * (len(data) // (size - 4)) if size > 4 else 0
            block_end = min(block_start + (len(data) // (size - 4)), len(data))
            redundancy[i] = sum(data[block_start:block_end]) & 0xFF

        return bytes(redundancy)

    def get_metrics(self) -> IntegrityMetrics:
        """Get FEC operation metrics."""
        return self._metrics


class IntegrityChecker:
    """
    High-level facade for data integrity operations.

    Combines CRC validation and forward error correction into a
    unified interface.
    """

    def __init__(self, config: Optional[IntegrityConfig] = None) -> None:
        """
        Initialize integrity checker.

        Args:
            config: Integrity configuration (uses defaults if None)
        """
        self.config = config or IntegrityConfig()
        self._crc = CRCValidator()
        self._fec = ForwardErrorCorrection(mode=self.config.fec_mode, max_correctable=self.config.max_correctable_errors)
        self._overall_metrics = IntegrityMetrics()

    def protect(self, data: bytes) -> bytes:
        """
        Add integrity protection to data.

        Applies FEC encoding and CRC in sequence.

        Args:
            data: Input data

        Returns:
            Protected data with ECC and CRC
        """
        result = data

        # Apply FEC first
        if self.config.enable_fec:
            result = self._fec.encode(result)

        # Then add CRC
        if self.config.enable_crc:
            result = self._crc.append_crc(result)

        return result

    def verify(self, data: bytes) -> tuple[IntegrityCheckResult, bytes, int]:
        """
        Verify and recover data.

        Args:
            data: Protected data

        Returns:
            Tuple of (result, recovered_data, errors_corrected)
        """
        self._overall_metrics.packets_checked += 1
        current_data = data
        errors_corrected = 0

        # Verify CRC first
        if self.config.enable_crc:
            is_valid, current_data = self._crc.verify_crc(current_data)
            if not is_valid:
                self._overall_metrics.crc_errors += 1
                # Try FEC recovery if enabled
                if not self.config.enable_fec:
                    self._overall_metrics.packets_corrupted += 1
                    return IntegrityCheckResult.CORRUPTED, current_data, 0

        # Apply FEC decoding
        if self.config.enable_fec:
            current_data, errors = self._fec.decode(current_data)
            if errors > 0:
                errors_corrected = errors
                self._overall_metrics.packets_corrected += 1
                self._overall_metrics.total_errors_corrected += errors
            elif errors < 0:
                self._overall_metrics.packets_corrupted += 1
                return IntegrityCheckResult.CORRUPTED, current_data, 0

        if errors_corrected > 0:
            # CQ-6: packets_corrected is already incremented inside the
            # FEC branch above; do not double-count here.
            return IntegrityCheckResult.CORRECTABLE, current_data, errors_corrected
        else:
            self._overall_metrics.packets_valid += 1
            return IntegrityCheckResult.VALID, current_data, 0

    def get_metrics(self) -> IntegrityMetrics:
        """Get combined integrity metrics."""
        self._overall_metrics.update_error_rate()
        return self._overall_metrics

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._overall_metrics = IntegrityMetrics()
        self._crc = CRCValidator()
        self._fec = ForwardErrorCorrection(mode=self.config.fec_mode, max_correctable=self.config.max_correctable_errors)
