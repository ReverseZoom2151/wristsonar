"""The raw watch transport is exact, bounded, and unambiguous."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from wristsonar.capture.wire import (
    RawPcmWireFrame,
    WireError,
    decode_raw_pcm,
    encode_raw_pcm,
)


def _frame() -> RawPcmWireFrame:
    return RawPcmWireFrame(
        np.asarray([-32768, -7, 0, 11, 32767], dtype="<i2"),
        timestamp_ns=123_456_789,
        sample_rate=48_000,
    )


def test_round_trip_preserves_signed_pcm_and_monotonic_timestamp() -> None:
    decoded = decode_raw_pcm(encode_raw_pcm(_frame()))
    np.testing.assert_array_equal(decoded.samples, _frame().samples)
    assert decoded.timestamp_ns == 123_456_789
    assert decoded.timestamp_s == pytest.approx(0.123456789)
    assert decoded.sample_rate == 48_000


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "truncated"),
        (b"NOPE" + bytes(20), "wrong magic"),
        (b"WSAR" + bytes([2]) + bytes(14), "unsupported"),
    ],
)
def test_rejects_invalid_headers(payload: bytes, message: str) -> None:
    with pytest.raises(WireError, match=message):
        decode_raw_pcm(payload)


def test_rejects_declared_length_that_does_not_match_the_payload() -> None:
    payload = struct.pack("<4sBIQH", b"WSAR", 1, 48_000, 0, 3) + b"\x00\x00"
    with pytest.raises(WireError, match="requires exactly"):
        decode_raw_pcm(payload)


def test_rejects_big_endian_or_empty_samples_at_the_construction_boundary() -> None:
    with pytest.raises(ValueError, match="little-endian"):
        RawPcmWireFrame(np.asarray([1], dtype=">i2"), 0, 48_000)
    with pytest.raises(ValueError, match=r"1\.\.4096"):
        RawPcmWireFrame(np.asarray([], dtype="<i2"), 0, 48_000)
