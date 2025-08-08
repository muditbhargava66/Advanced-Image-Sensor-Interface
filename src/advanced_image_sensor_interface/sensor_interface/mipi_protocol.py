"""
MIPI CSI-2 Protocol Implementation and Validation

This module provides MIPI CSI-2 packet parsing, formatting, and validation
capabilities for protocol compliance testing and simulation.

Classes:
    MIPIPacket: Base class for MIPI CSI-2 packets
    ShortPacket: MIPI CSI-2 short packet implementation
    LongPacket: MIPI CSI-2 long packet implementation
    MIPIProtocolValidator: Protocol validation and compliance checking

Functions:
    calculate_ecc: Calculate Error Correction Code for packet headers
    calculate_crc: Calculate CRC for long packet data
    validate_packet: Validate packet structure and checksums
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DataType(Enum):
    """MIPI CSI-2 Data Types as defined in the specification."""

    # YUV formats
    YUV420_8BIT = 0x18
    YUV420_10BIT = 0x19
    YUV422_8BIT = 0x1E
    YUV422_10BIT = 0x1F

    # RGB formats
    RGB444 = 0x20
    RGB555 = 0x21
    RGB565 = 0x22
    RGB666 = 0x23
    RGB888 = 0x24

    # RAW formats
    RAW6 = 0x28
    RAW7 = 0x29
    RAW8 = 0x2A
    RAW10 = 0x2B
    RAW12 = 0x2C
    RAW14 = 0x2D
    RAW16 = 0x2E
    RAW20 = 0x2F

    # Short packet types
    FRAME_START = 0x00
    FRAME_END = 0x01
    LINE_START = 0x02
    LINE_END = 0x03
    GENERIC_SHORT_1 = 0x08
    GENERIC_SHORT_2 = 0x09
    GENERIC_SHORT_3 = 0x0A
    GENERIC_SHORT_4 = 0x0B
    GENERIC_SHORT_5 = 0x0C
    GENERIC_SHORT_6 = 0x0D
    GENERIC_SHORT_7 = 0x0E
    GENERIC_SHORT_8 = 0x0F


@dataclass
class MIPIPacket:
    """Base class for MIPI CSI-2 packets."""

    virtual_channel: int  # 0-3
    data_type: DataType

    def __post_init__(self):
        """Validate packet parameters."""
        if not (0 <= self.virtual_channel <= 3):
            raise ValueError("Virtual channel must be 0-3")


@dataclass
class ShortPacket(MIPIPacket):
    """MIPI CSI-2 Short Packet (4 bytes total)."""

    data: int  # 16-bit data field

    def __post_init__(self):
        """Validate short packet parameters."""
        super().__post_init__()
        if not (0 <= self.data <= 0xFFFF):
            raise ValueError("Short packet data must be 16-bit")

    def to_bytes(self) -> bytes:
        """Convert short packet to byte representation."""
        # Packet Header: DI (Data Identifier) + WC (Word Count) + ECC
        di = (self.virtual_channel << 6) | self.data_type.value
        wc_low = self.data & 0xFF
        wc_high = (self.data >> 8) & 0xFF
        ecc = calculate_ecc(di, wc_low, wc_high)

        return bytes([di, wc_low, wc_high, ecc])

    @classmethod
    def from_bytes(cls, data: bytes) -> "ShortPacket":
        """Create short packet from byte representation."""
        if len(data) != 4:
            raise ValueError("Short packet must be 4 bytes")

        di, wc_low, wc_high, ecc = data

        # Validate ECC
        expected_ecc = calculate_ecc(di, wc_low, wc_high)
        if ecc != expected_ecc:
            raise ValueError(f"ECC mismatch: expected {expected_ecc:02X}, got {ecc:02X}")

        virtual_channel = (di >> 6) & 0x3
        data_type = DataType(di & 0x3F)
        packet_data = wc_low | (wc_high << 8)

        return cls(virtual_channel=virtual_channel, data_type=data_type, data=packet_data)


@dataclass
class LongPacket(MIPIPacket):
    """MIPI CSI-2 Long Packet (header + payload + footer)."""

    payload: bytes

    def __post_init__(self):
        """Validate long packet parameters."""
        super().__post_init__()
        if len(self.payload) > 0xFFFF:
            raise ValueError("Long packet payload too large (max 65535 bytes)")

    def to_bytes(self) -> bytes:
        """Convert long packet to byte representation."""
        # Packet Header: DI + WC + ECC
        di = (self.virtual_channel << 6) | self.data_type.value
        wc = len(self.payload)
        wc_low = wc & 0xFF
        wc_high = (wc >> 8) & 0xFF
        ecc = calculate_ecc(di, wc_low, wc_high)

        # Packet Footer: CRC
        crc = calculate_crc(self.payload)
        crc_low = crc & 0xFF
        crc_high = (crc >> 8) & 0xFF

        return bytes([di, wc_low, wc_high, ecc]) + self.payload + bytes([crc_low, crc_high])

    @classmethod
    def from_bytes(cls, data: bytes) -> "LongPacket":
        """Create long packet from byte representation."""
        if len(data) < 6:  # Minimum: 4-byte header + 2-byte CRC
            raise ValueError("Long packet too short")

        # Parse header
        di, wc_low, wc_high, ecc = data[:4]

        # Validate ECC
        expected_ecc = calculate_ecc(di, wc_low, wc_high)
        if ecc != expected_ecc:
            raise ValueError(f"ECC mismatch: expected {expected_ecc:02X}, got {ecc:02X}")

        wc = wc_low | (wc_high << 8)

        if len(data) != 4 + wc + 2:
            raise ValueError(f"Packet length mismatch: expected {4 + wc + 2}, got {len(data)}")

        # Extract payload and CRC
        payload = data[4 : 4 + wc]
        crc_low, crc_high = data[4 + wc : 4 + wc + 2]
        crc = crc_low | (crc_high << 8)

        # Validate CRC
        expected_crc = calculate_crc(payload)
        if crc != expected_crc:
            raise ValueError(f"CRC mismatch: expected {expected_crc:04X}, got {crc:04X}")

        virtual_channel = (di >> 6) & 0x3
        data_type = DataType(di & 0x3F)

        return cls(virtual_channel=virtual_channel, data_type=data_type, payload=payload)


def calculate_ecc(byte1: int, byte2: int, byte3: int) -> int:
    """
    Calculate Error Correction Code for MIPI CSI-2 packet header.

    Args:
        byte1: First byte (Data Identifier)
        byte2: Second byte (Word Count Low)
        byte3: Third byte (Word Count High)

    Returns:
        8-bit ECC value
    """
    # Combine the three bytes into a 24-bit value
    data = (byte3 << 16) | (byte2 << 8) | byte1

    # Calculate parity bits according to MIPI CSI-2 specification
    p0 = 0
    p1 = 0
    p2 = 0
    p3 = 0
    p4 = 0
    p5 = 0

    # P0: XOR of bits 0, 1, 3, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 23
    for bit in [0, 1, 3, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 23]:
        p0 ^= (data >> bit) & 1

    # P1: XOR of bits 0, 2, 3, 5, 6, 9, 10, 12, 13, 16, 17, 20, 21
    for bit in [0, 2, 3, 5, 6, 9, 10, 12, 13, 16, 17, 20, 21]:
        p1 ^= (data >> bit) & 1

    # P2: XOR of bits 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 22, 23
    for bit in [1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 22, 23]:
        p2 ^= (data >> bit) & 1

    # P3: XOR of bits 4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23
    for bit in [4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23]:
        p3 ^= (data >> bit) & 1

    # P4: XOR of bits 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    for bit in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        p4 ^= (data >> bit) & 1

    # P5: XOR of all bits
    p5 = bin(data).count("1") & 1

    return (p5 << 7) | (p4 << 6) | (p3 << 5) | (p2 << 4) | (p1 << 3) | (p0 << 2)


def calculate_crc(data: bytes) -> int:
    """
    Calculate CRC-16 for MIPI CSI-2 long packet payload.

    Args:
        data: Payload bytes

    Returns:
        16-bit CRC value
    """
    # CRC-16-CCITT polynomial: x^16 + x^12 + x^5 + 1 (0x1021)
    polynomial = 0x1021
    crc = 0xFFFF

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
            crc &= 0xFFFF

    return crc


class MIPIProtocolValidator:
    """MIPI CSI-2 protocol validation and compliance checking."""

    def __init__(self):
        """Initialize the protocol validator."""
        self.packet_count = 0
        self.error_count = 0
        self.frame_count = 0
        self.line_count = 0

    def validate_packet(self, packet_data: bytes) -> bool:
        """
        Validate a MIPI CSI-2 packet.

        Args:
            packet_data: Raw packet bytes

        Returns:
            True if packet is valid, False otherwise
        """
        try:
            self.packet_count += 1

            if len(packet_data) == 4:
                # Short packet
                short_packet = ShortPacket.from_bytes(packet_data)
                logger.debug(
                    f"Valid short packet: VC={short_packet.virtual_channel}, " f"DT={short_packet.data_type.name}, Data={short_packet.data:04X}"
                )

                # Track frame/line markers
                if short_packet.data_type == DataType.FRAME_START:
                    self.frame_count += 1
                elif short_packet.data_type == DataType.LINE_START:
                    self.line_count += 1

            else:
                # Long packet
                long_packet = LongPacket.from_bytes(packet_data)
                logger.debug(
                    f"Valid long packet: VC={long_packet.virtual_channel}, "
                    f"DT={long_packet.data_type.name}, Length={len(long_packet.payload)}"
                )

            return True

        except ValueError as e:
            self.error_count += 1
            logger.error(f"Invalid packet: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get validation statistics."""
        return {
            "total_packets": self.packet_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.packet_count, 1),
            "frame_count": self.frame_count,
            "line_count": self.line_count,
        }

    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.packet_count = 0
        self.error_count = 0
        self.frame_count = 0
        self.line_count = 0


# Example usage and testing
if __name__ == "__main__":
    # Test short packet
    short_pkt = ShortPacket(virtual_channel=0, data_type=DataType.FRAME_START, data=0x1234)

    short_bytes = short_pkt.to_bytes()
    print(f"Short packet bytes: {short_bytes.hex()}")

    # Round-trip test
    decoded_short = ShortPacket.from_bytes(short_bytes)
    assert decoded_short.virtual_channel == short_pkt.virtual_channel
    assert decoded_short.data_type == short_pkt.data_type
    assert decoded_short.data == short_pkt.data

    # Test long packet
    payload = b"Hello, MIPI CSI-2 World!" * 10
    long_pkt = LongPacket(virtual_channel=1, data_type=DataType.RAW12, payload=payload)

    long_bytes = long_pkt.to_bytes()
    print(f"Long packet length: {len(long_bytes)} bytes")

    # Round-trip test
    decoded_long = LongPacket.from_bytes(long_bytes)
    assert decoded_long.virtual_channel == long_pkt.virtual_channel
    assert decoded_long.data_type == long_pkt.data_type
    assert decoded_long.payload == long_pkt.payload

    # Test validator
    validator = MIPIProtocolValidator()
    assert validator.validate_packet(short_bytes)
    assert validator.validate_packet(long_bytes)

    # Test invalid packet
    invalid_packet = b"\x00\x01\x02"  # Too short
    assert not validator.validate_packet(invalid_packet)

    stats = validator.get_statistics()
    print(f"Validation statistics: {stats}")

    print("All MIPI CSI-2 protocol tests passed!")
