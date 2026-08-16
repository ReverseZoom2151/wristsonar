"""Raw TCP sessions remain raw until the host deliberately processes them."""

from __future__ import annotations

from socket import socketpair
from threading import Thread

import numpy as np

from wristsonar.capture.listener import consume_raw_pcm_connection
from wristsonar.capture.wire import RawPcmWireFrame, encode_raw_pcm


def _packet(value: int) -> RawPcmWireFrame:
    return RawPcmWireFrame(
        np.asarray([value, -value], dtype="<i2"), value, 48_000
    )


def test_consumes_packet_boundaries_and_stops_cleanly_at_peer_eof() -> None:
    left, right = socketpair()
    expected = (_packet(1), _packet(2), _packet(3))
    seen: list[RawPcmWireFrame] = []
    try:
        def send() -> None:
            left.sendall(b"".join(encode_raw_pcm(frame) for frame in expected))
            left.close()

        sender = Thread(target=send)
        sender.start()
        assert consume_raw_pcm_connection(right, seen.append) == len(expected)
        sender.join()
    finally:
        left.close()
        right.close()
    assert [item.timestamp_ns for item in seen] == [1, 2, 3]
