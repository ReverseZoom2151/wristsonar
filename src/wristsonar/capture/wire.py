"""Versioned binary transport for raw watch PCM callback frames.

The watch must send raw PCM, not precomputed echo profiles: otherwise the host
cannot prove that its live DSP matches the data used for training.  This wire
format is intentionally tiny and length-delimited, so TCP's arbitrary packet
boundaries cannot change an audio callback into a different frame.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from socket import socket

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RawPcmWireFrame",
    "WireError",
    "decode_raw_pcm",
    "encode_raw_pcm",
    "recv_raw_pcm",
]

_MAGIC = b"WSAR"
_VERSION = 1
_HEADER = struct.Struct("<4sBIQH")
_MAX_SAMPLES = 4_096


class WireError(ValueError):
    """A peer sent data that is not one Wristsonar raw PCM frame."""


@dataclass(frozen=True, slots=True)
class RawPcmWireFrame:
    """One timestamped signed-16-bit microphone callback from the watch."""

    samples: NDArray[np.int16]
    timestamp_ns: int
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1 or not 1 <= self.samples.size <= _MAX_SAMPLES:
            raise ValueError(
                f"samples must be a one-dimensional 1..{_MAX_SAMPLES} PCM vector"
            )
        if self.samples.dtype != np.dtype("<i2"):
            raise ValueError("samples must be explicitly little-endian int16 PCM")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns cannot be negative")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def timestamp_s(self) -> float:
        """Monotonic capture time converted at the host boundary."""
        return self.timestamp_ns / 1_000_000_000.0


def encode_raw_pcm(frame: RawPcmWireFrame) -> bytes:
    """Serialize one frame exactly as ``wear/.../RawPcmTransport.kt`` writes it."""
    count = int(frame.samples.size)
    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        frame.sample_rate,
        frame.timestamp_ns,
        count,
    )
    return header + frame.samples.tobytes()


def decode_raw_pcm(payload: bytes) -> RawPcmWireFrame:
    """Decode exactly one full frame, refusing trailing or truncated bytes."""
    if len(payload) < _HEADER.size:
        raise WireError("truncated header")
    magic, version, sample_rate, timestamp_ns, count = _HEADER.unpack_from(payload)
    if magic != _MAGIC:
        raise WireError(f"wrong magic {magic!r}")
    if version != _VERSION:
        raise WireError(f"unsupported raw PCM wire version {version}")
    if not 1 <= count <= _MAX_SAMPLES:
        raise WireError(f"invalid sample count {count}")
    expected = _HEADER.size + count * np.dtype("<i2").itemsize
    if len(payload) != expected:
        raise WireError(
            f"frame is {len(payload)} bytes but header requires exactly {expected}"
        )
    samples = np.frombuffer(payload, dtype="<i2", offset=_HEADER.size).copy()
    try:
        return RawPcmWireFrame(samples, timestamp_ns, sample_rate)
    except ValueError as error:
        raise WireError(str(error)) from error


def recv_raw_pcm(connection: socket) -> RawPcmWireFrame:
    """Read one whole frame from TCP, regardless of its packet boundaries.

    TCP is a byte stream. A single ``recv`` is permitted to return half a
    header or several audio frames, so using it as a message boundary silently
    corrupts audio. This helper reads the fixed header first, then exactly the
    signed-16-bit payload count stated by that header.
    """
    header = _recv_exact(connection, _HEADER.size)
    if header is None:
        raise EOFError("peer closed before a raw PCM header")
    magic, version, _sample_rate, _timestamp_ns, count = _HEADER.unpack(header)
    if magic != _MAGIC:
        raise WireError(f"wrong magic {magic!r}")
    if version != _VERSION:
        raise WireError(f"unsupported raw PCM wire version {version}")
    if not 1 <= count <= _MAX_SAMPLES:
        raise WireError(f"invalid sample count {count}")
    body = _recv_exact(connection, count * np.dtype("<i2").itemsize)
    if body is None:
        raise EOFError("peer closed in the middle of a raw PCM frame")
    return decode_raw_pcm(header + body)


def _recv_exact(connection: socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
