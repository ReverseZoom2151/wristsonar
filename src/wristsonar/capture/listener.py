"""Blocking TCP consumption for a single trusted Wristsonar watch session."""

from __future__ import annotations

from collections.abc import Callable
from socket import socket

from wristsonar.capture.wire import (
    RawPcmWireFrame,
    TruncatedFrameError,
    WireError,
    recv_raw_pcm,
)

__all__ = ["WatchSessionError", "consume_raw_pcm_connection"]


class WatchSessionError(RuntimeError):
    """A watch session ended in a way a finished session cannot look like.

    A crashed watch and a completed capture used to be the same return value,
    because a mid-frame ``EOFError`` was caught alongside a clean close.  They
    are opposite outcomes: one is a result, the other is an incident to
    investigate, and only one of them should leave the operator content.
    """

    def __init__(self, message: str, *, packets: int) -> None:
        super().__init__(message)
        self.packets = packets


def consume_raw_pcm_connection(
    connection: socket, on_packet: Callable[[RawPcmWireFrame], None]
) -> int:
    """Consume one TCP stream to a clean close and return its packet count.

    The callback boundary is intentionally raw PCM.  It lets the CLI combine a
    listener with either live DSP or a diagnostics recorder without granting a
    device-side echo profile any authority over the host's signal contract.

    Only a peer that closes between frames is a completed session.  A peer that
    dies mid-frame, sends a frame this protocol does not recognise, or drops
    the connection raises :class:`WatchSessionError` carrying the count so far.
    A malformed frame used to escape as an uncaught ``WireError`` and end the
    process in a traceback.
    """
    received = 0
    while True:
        try:
            packet = recv_raw_pcm(connection)
        except EOFError:
            return received
        except TruncatedFrameError as error:
            raise WatchSessionError(
                f"watch stopped mid-frame after {received} packets: {error}",
                packets=received,
            ) from error
        except WireError as error:
            raise WatchSessionError(
                f"watch sent an unusable frame after {received} packets: {error}",
                packets=received,
            ) from error
        except OSError as error:
            raise WatchSessionError(
                f"watch connection failed after {received} packets: {error}",
                packets=received,
            ) from error
        on_packet(packet)
        received += 1
