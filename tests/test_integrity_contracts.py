"""Application-envelope regressions: detection must never masquerade as recovery."""

import pytest

from advanced_image_sensor_interface.utils.data_integrity import (
    ErrorCorrectionMode,
    ForwardErrorCorrection,
    IntegrityChecker,
    IntegrityCheckResult,
    IntegrityConfig,
)


def checker_for(mode, *, crc=True, fec=True):
    return IntegrityChecker(IntegrityConfig(enable_crc=crc, enable_fec=fec, fec_mode=mode))


def flip(data, index, mask=1):
    changed = bytearray(data)
    changed[index] ^= mask
    return bytes(changed)


@pytest.mark.parametrize("mode", list(ErrorCorrectionMode))
def test_application_envelope_roundtrip(mode):
    checker = checker_for(mode)
    data = b"\x00\x13\xab\xff"
    assert checker.verify(checker.protect(data)) == (IntegrityCheckResult.VALID, data, 0)
    metrics = checker.get_metrics()
    assert (metrics.packets_checked, metrics.packets_valid, metrics.packets_corrected, metrics.packets_corrupted) == (1, 1, 0, 0)


@pytest.mark.parametrize("fec", [False, True])
@pytest.mark.parametrize("index", [0, -1])
def test_crc_failure_cannot_be_bypassed_by_none_fec(fec, index):
    checker = checker_for(ErrorCorrectionMode.NONE, fec=fec)
    status, _, corrected = checker.verify(flip(checker.protect(b"original"), index))
    assert status is IntegrityCheckResult.CORRUPTED
    assert corrected == 0
    metrics = checker.get_metrics()
    assert (metrics.packets_checked, metrics.packets_corrupted, metrics.crc_errors) == (1, 1, 1)
    assert metrics.packets_corrected == metrics.packets_valid == metrics.total_errors_corrected == 0
    assert metrics.error_rate == 100.0


@pytest.mark.parametrize("mode", list(ErrorCorrectionMode))
@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x00\x00\x00"])
def test_truncated_crc_frame_fails_even_with_fec(mode, payload):
    checker = checker_for(mode)
    assert checker.verify(payload)[0] is IntegrityCheckResult.CORRUPTED
    assert checker.get_metrics().packets_corrected == 0


@pytest.mark.parametrize("mode", list(ErrorCorrectionMode))
def test_original_crc_trailer_is_never_replaced(mode):
    checker = checker_for(mode)
    status, _, corrected = checker.verify(flip(checker.protect(b"payload"), -1))
    assert status is IntegrityCheckResult.CORRUPTED
    assert corrected == 0
    assert checker.get_metrics().packets_corrected == 0


@pytest.mark.parametrize("bit", range(7))
def test_hamming_single_bit_recovery_is_proved_by_original_crc(bit):
    checker = checker_for(ErrorCorrectionMode.HAMMING)
    data = b"\xa5\x13"
    assert checker.verify(flip(checker.protect(data), 0, 1 << bit)) == (IntegrityCheckResult.CORRECTABLE, data, 1)
    metrics = checker.get_metrics()
    assert (metrics.packets_checked, metrics.packets_valid, metrics.packets_corrected, metrics.packets_corrupted) == (1, 0, 1, 0)
    assert metrics.crc_errors == metrics.fec_corrections == metrics.total_errors_corrected == 1
    assert metrics.error_rate == 100.0


@pytest.mark.parametrize("mask", [3, 5, 6, 0x30, 0x60])
def test_crc_rejects_hamming_double_error_miscorrection(mask):
    checker = checker_for(ErrorCorrectionMode.HAMMING)
    status, _, corrected = checker.verify(flip(checker.protect(b"\xa5\x13"), 0, mask))
    assert status is IntegrityCheckResult.CORRUPTED
    assert corrected == 0
    metrics = checker.get_metrics()
    assert metrics.packets_corrected == metrics.fec_corrections == metrics.total_errors_corrected == 0
    assert metrics.packets_corrupted == metrics.crc_errors == 1
    assert metrics.error_rate == 100.0


def test_hamming_correction_plus_corrupted_trailer_is_rejected():
    checker = checker_for(ErrorCorrectionMode.HAMMING)
    corrupted = flip(flip(checker.protect(b"\x13"), 0, 0x10), -1)
    assert checker.verify(corrupted)[0] is IntegrityCheckResult.CORRUPTED
    assert checker.get_metrics().total_errors_corrected == 0


@pytest.mark.parametrize("index", [0, -1])
def test_simulated_rs_redundancy_mismatch_never_claims_correction(index):
    fec = ForwardErrorCorrection(ErrorCorrectionMode.REED_SOLOMON)
    _, corrected = fec.decode(flip(fec.encode(b"payload"), index))
    assert corrected == -1
    assert fec.get_metrics().fec_corrections == fec.get_metrics().total_errors_corrected == 0


@pytest.mark.parametrize("crc", [False, True])
def test_simulated_rs_mismatch_fails_closed_at_facade(crc):
    checker = checker_for(ErrorCorrectionMode.REED_SOLOMON, crc=crc)
    protected = checker.protect(b"payload")
    # Last redundancy byte, not the outer CRC trailer.
    status, _, corrected = checker.verify(flip(protected, -5 if crc else -1))
    assert status is IntegrityCheckResult.CORRUPTED
    assert corrected == 0
    assert checker.get_metrics().packets_corrected == 0


@pytest.mark.parametrize(
    "mode,payload",
    [(ErrorCorrectionMode.PARITY, b""), (ErrorCorrectionMode.HAMMING, b"\x00"), (ErrorCorrectionMode.REED_SOLOMON, b"\x00")],
)
def test_incomplete_fec_codeword_is_rejected_without_crc(mode, payload):
    checker = checker_for(mode, crc=False)
    assert checker.verify(payload)[0] is IntegrityCheckResult.CORRUPTED


@pytest.mark.parametrize("crc", [False, True])
def test_parity_detects_but_cannot_correct(crc):
    checker = checker_for(ErrorCorrectionMode.PARITY, crc=crc)
    status, _, corrected = checker.verify(flip(checker.protect(b"payload"), 0))
    assert status is IntegrityCheckResult.CORRUPTED
    assert corrected == 0
    assert checker.get_metrics().packets_corrected == 0


def test_integrity_metrics_count_each_outcome_once_and_reset():
    checker = checker_for(ErrorCorrectionMode.HAMMING)
    protected = checker.protect(b"\xa5")
    checker.verify(protected)
    checker.verify(flip(protected, 0, 0x10))
    checker.verify(flip(protected, 0, 3))
    metrics = checker.get_metrics()
    assert (metrics.packets_checked, metrics.packets_valid, metrics.packets_corrected, metrics.packets_corrupted) == (3, 1, 1, 1)
    assert metrics.total_errors_corrected == metrics.fec_corrections == 1
    assert metrics.crc_errors == 2
    assert metrics.error_rate == pytest.approx(200 / 3)
    checker.reset_metrics()
    assert checker.get_metrics().packets_checked == checker.get_metrics().total_errors_corrected == 0
