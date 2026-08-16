"""Blocking TCP consumption for a single trusted Wristsonar watch session."""

from __future__ import annotations

from collections.abc import Callable
from socket import socket

from wristsonar.capture.wire import RawPcmWireFrame, recv_raw_pcm

__all__ = ["consume_raw_pcm_connection"]


def consume_raw_pcm_connection(
    connection: socket, on_packet: Callable[[RawPcmWireFrame], None]
) -> int:
    """Consume one TCP stream until its peer closes and return its packet count.

    The callback boundary is intentionally raw PCM.  It lets the CLI combine a
    listener with either live DSP or a diagnostics recorder without granting a
    device-side echo profile any authority over the host's signal contract.
    """
    received = 0
    while True:
        try:
            packet = recv_raw_pcm(connection)
        except EOFError:
            return received
        on_packet(packet)
        received += 1
