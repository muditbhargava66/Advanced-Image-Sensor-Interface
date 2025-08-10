"""
Unit Tests for MIPI Protocol

This module contains unit tests for the MIPI CSI-2 protocol implementation
in the Advanced Image Sensor Interface project.

Classes:
    TestMIPIProtocol: Test cases for MIPI CSI-2 packet handling.

Usage:
    Run these tests using pytest:
    $ pytest tests/test_mipi_protocol.py
"""

import pytest
from advanced_image_sensor_interface.sensor_interface.mipi_protocol import (
    DataType,
    LongPacket,
    MIPIProtocolValidator,
    ShortPacket,
    calculate_crc,
    calculate_ecc,
)


class TestMIPIProtocol:
    """Test cases for MIPI CSI-2 protocol implementation."""

    def test_data_types(self):
        """Test MIPI CSI-2 data type enumeration."""
        # Test some key data types
        assert DataType.RAW12.value == 0x2C
        assert DataType.RGB888.value == 0x24
        assert DataType.FRAME_START.value == 0x00
        assert DataType.FRAME_END.value == 0x01

    def test_short_packet_creation(self):
        """Test short packet creation and validation."""
        packet = ShortPacket(virtual_channel=1, data_type=DataType.FRAME_START, data=0x1234)

        assert packet.virtual_channel == 1
        assert packet.data_type == DataType.FRAME_START
        assert packet.data == 0x1234

    def test_short_packet_invalid_parameters(self):
        """Test short packet with invalid parameters."""
        # Invalid virtual channel
        with pytest.raises(ValueError, match="Virtual channel must be 0-3"):
            ShortPacket(virtual_channel=4, data_type=DataType.FRAME_START, data=0x1234)

        # Invalid data range
        with pytest.raises(ValueError, match="Short packet data must be 16-bit"):
            ShortPacket(virtual_channel=0, data_type=DataType.FRAME_START, data=0x10000)

    def test_short_packet_serialization(self):
        """Test short packet serialization and deserialization."""
        original = ShortPacket(virtual_channel=2, data_type=DataType.LINE_START, data=0xABCD)

        # Serialize to bytes
        packet_bytes = original.to_bytes()
        assert len(packet_bytes) == 4

        # Deserialize back
        decoded = ShortPacket.from_bytes(packet_bytes)

        assert decoded.virtual_channel == original.virtual_channel
        assert decoded.data_type == original.data_type
        assert decoded.data == original.data

    def test_long_packet_creation(self):
        """Test long packet creation and validation."""
        payload = b"Hello, MIPI CSI-2 World!"
        packet = LongPacket(virtual_channel=0, data_type=DataType.RAW12, payload=payload)

        assert packet.virtual_channel == 0
        assert packet.data_type == DataType.RAW12
        assert packet.payload == payload

    def test_long_packet_invalid_parameters(self):
        """Test long packet with invalid parameters."""
        # Payload too large
        large_payload = b"x" * 0x10000  # 64KB + 1
        with pytest.raises(ValueError, match="Long packet payload too large"):
            LongPacket(virtual_channel=0, data_type=DataType.RAW12, payload=large_payload)

    def test_long_packet_serialization(self):
        """Test long packet serialization and deserialization."""
        payload = b"Test payload for MIPI CSI-2 long packet validation"
        original = LongPacket(virtual_channel=3, data_type=DataType.RGB888, payload=payload)

        # Serialize to bytes
        packet_bytes = original.to_bytes()
        expected_length = 4 + len(payload) + 2  # Header + payload + CRC
        assert len(packet_bytes) == expected_length

        # Deserialize back
        decoded = LongPacket.from_bytes(packet_bytes)

        assert decoded.virtual_channel == original.virtual_channel
        assert decoded.data_type == original.data_type
        assert decoded.payload == original.payload

    def test_ecc_calculation(self):
        """Test Error Correction Code calculation."""
        # Test with known values
        ecc = calculate_ecc(0x12, 0x34, 0x56)
        assert isinstance(ecc, int)
        assert 0 <= ecc <= 0xFF

        # Test consistency
        ecc1 = calculate_ecc(0x12, 0x34, 0x56)
        ecc2 = calculate_ecc(0x12, 0x34, 0x56)
        assert ecc1 == ecc2

        # Test different inputs give different results
        ecc3 = calculate_ecc(0x12, 0x34, 0x57)
        assert ecc1 != ecc3

    def test_crc_calculation(self):
        """Test CRC-16 calculation."""
        # Test with known data
        data = b"Hello, World!"
        crc = calculate_crc(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

        # Test consistency
        crc1 = calculate_crc(data)
        crc2 = calculate_crc(data)
        assert crc1 == crc2

        # Test different data gives different CRC
        crc3 = calculate_crc(b"Hello, World?")
        assert crc1 != crc3

        # Test empty data
        crc_empty = calculate_crc(b"")
        assert crc_empty == 0xFFFF  # Initial CRC value

    def test_short_packet_ecc_validation(self):
        """Test ECC validation in short packets."""
        # Create valid packet
        packet = ShortPacket(virtual_channel=0, data_type=DataType.FRAME_START, data=0x1234)
        packet_bytes = packet.to_bytes()

        # Corrupt ECC
        corrupted_bytes = bytearray(packet_bytes)
        corrupted_bytes[3] ^= 0x01  # Flip one bit in ECC

        with pytest.raises(ValueError, match="ECC mismatch"):
            ShortPacket.from_bytes(bytes(corrupted_bytes))

    def test_long_packet_crc_validation(self):
        """Test CRC validation in long packets."""
        # Create valid packet
        payload = b"Test payload"
        packet = LongPacket(virtual_channel=0, data_type=DataType.RAW12, payload=payload)
        packet_bytes = packet.to_bytes()

        # Corrupt CRC
        corrupted_bytes = bytearray(packet_bytes)
        corrupted_bytes[-1] ^= 0x01  # Flip one bit in CRC

        with pytest.raises(ValueError, match="CRC mismatch"):
            LongPacket.from_bytes(bytes(corrupted_bytes))

    def test_short_packet_invalid_length(self):
        """Test short packet with invalid length."""
        with pytest.raises(ValueError, match="Short packet must be 4 bytes"):
            ShortPacket.from_bytes(b"\x00\x01\x02")  # Too short

        with pytest.raises(ValueError, match="Short packet must be 4 bytes"):
            ShortPacket.from_bytes(b"\x00\x01\x02\x03\x04")  # Too long

    def test_long_packet_invalid_length(self):
        """Test long packet with invalid length."""
        # Too short (less than minimum 6 bytes)
        with pytest.raises(ValueError, match="Long packet too short"):
            LongPacket.from_bytes(b"\x00\x01\x02\x03\x04")

        # Length mismatch
        # Create a packet header indicating 10 bytes payload but provide different length
        di = 0x2C  # RAW12
        wc_low = 10  # Word count low
        wc_high = 0  # Word count high
        ecc = calculate_ecc(di, wc_low, wc_high)
        header = bytes([di, wc_low, wc_high, ecc])

        # Provide wrong payload length
        wrong_payload = b"short"  # 5 bytes instead of 10
        crc = calculate_crc(wrong_payload)
        wrong_packet = header + wrong_payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])

        with pytest.raises(ValueError, match="Packet length mismatch"):
            LongPacket.from_bytes(wrong_packet)


class TestMIPIProtocolValidator:
    """Test cases for MIPI protocol validator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = MIPIProtocolValidator()
        assert validator.packet_count == 0
        assert validator.error_count == 0
        assert validator.frame_count == 0
        assert validator.line_count == 0

    def test_validate_short_packet(self):
        """Test validation of short packets."""
        validator = MIPIProtocolValidator()

        # Create valid short packet
        packet = ShortPacket(virtual_channel=0, data_type=DataType.FRAME_START, data=0x1234)
        packet_bytes = packet.to_bytes()

        # Validate
        assert validator.validate_packet(packet_bytes) == True

        stats = validator.get_statistics()
        assert stats["total_packets"] == 1
        assert stats["error_count"] == 0
        assert stats["frame_count"] == 1  # FRAME_START increments frame count

    def test_validate_long_packet(self):
        """Test validation of long packets."""
        validator = MIPIProtocolValidator()

        # Create valid long packet
        payload = b"Test payload for validation"
        packet = LongPacket(virtual_channel=1, data_type=DataType.RAW12, payload=payload)
        packet_bytes = packet.to_bytes()

        # Validate
        assert validator.validate_packet(packet_bytes) == True

        stats = validator.get_statistics()
        assert stats["total_packets"] == 1
        assert stats["error_count"] == 0

    def test_validate_invalid_packet(self):
        """Test validation of invalid packets."""
        validator = MIPIProtocolValidator()

        # Invalid packet (too short)
        invalid_packet = b"\x00\x01\x02"

        # Validate
        assert validator.validate_packet(invalid_packet) == False

        stats = validator.get_statistics()
        assert stats["total_packets"] == 1
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 1.0

    def test_validator_statistics(self):
        """Test validator statistics tracking."""
        validator = MIPIProtocolValidator()

        # Create various packets
        frame_start = ShortPacket(0, DataType.FRAME_START, 0x0000)
        line_start = ShortPacket(0, DataType.LINE_START, 0x0001)
        data_packet = LongPacket(0, DataType.RAW12, b"test data")

        # Validate packets
        validator.validate_packet(frame_start.to_bytes())
        validator.validate_packet(line_start.to_bytes())
        validator.validate_packet(data_packet.to_bytes())
        validator.validate_packet(b"invalid")  # Invalid packet

        stats = validator.get_statistics()
        assert stats["total_packets"] == 4
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.25
        assert stats["frame_count"] == 1
        assert stats["line_count"] == 1

    def test_validator_reset(self):
        """Test validator statistics reset."""
        validator = MIPIProtocolValidator()

        # Add some packets
        packet = ShortPacket(0, DataType.FRAME_START, 0x0000)
        validator.validate_packet(packet.to_bytes())
        validator.validate_packet(b"invalid")

        # Check initial stats
        stats = validator.get_statistics()
        assert stats["total_packets"] == 2
        assert stats["error_count"] == 1

        # Reset
        validator.reset_statistics()

        # Check reset stats
        stats = validator.get_statistics()
        assert stats["total_packets"] == 0
        assert stats["error_count"] == 0
        assert stats["frame_count"] == 0
        assert stats["line_count"] == 0

    def test_multiple_frame_line_tracking(self):
        """Test tracking of multiple frames and lines."""
        validator = MIPIProtocolValidator()

        # Simulate frame with multiple lines
        packets = [
            ShortPacket(0, DataType.FRAME_START, 0x0000),
            ShortPacket(0, DataType.LINE_START, 0x0001),
            ShortPacket(0, DataType.LINE_START, 0x0002),
            ShortPacket(0, DataType.LINE_START, 0x0003),
            ShortPacket(0, DataType.FRAME_END, 0x0000),
            ShortPacket(0, DataType.FRAME_START, 0x0001),  # Second frame
            ShortPacket(0, DataType.LINE_START, 0x0001),
        ]

        for packet in packets:
            validator.validate_packet(packet.to_bytes())

        stats = validator.get_statistics()
        assert stats["frame_count"] == 2
        assert stats["line_count"] == 4


if __name__ == "__main__":
    pytest.main([__file__])
