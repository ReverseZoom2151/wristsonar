"""Raw TCP sessions remain raw until the host deliberately processes them."""

from __future__ import annotations

from socket import socketpair
from threading import Thread

import numpy as np
import pytest

from wristsonar.capture.listener import WatchSessionError, consume_raw_pcm_connection
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


def test_a_watch_that_dies_mid_frame_does_not_look_like_a_finished_session() -> None:
    """A crashed watch and a completed capture used to be the same return value."""
    left, right = socketpair()
    payload = encode_raw_pcm(_packet(9))
    seen: list[RawPcmWireFrame] = []
    try:

        def send_then_die() -> None:
            left.sendall(encode_raw_pcm(_packet(1)))
            left.sendall(payload[:-3])
            left.close()

        sender = Thread(target=send_then_die)
        sender.start()
        with pytest.raises(WatchSessionError, match="mid-frame after 1 packets"):
            consume_raw_pcm_connection(right, seen.append)
        sender.join()
    finally:
        left.close()
        right.close()
    assert len(seen) == 1


def test_a_malformed_frame_is_reported_rather_than_crashing_the_host() -> None:
    left, right = socketpair()
    try:

        def send_rubbish() -> None:
            left.sendall(b"NOPE" + bytes(20))
            left.close()

        sender = Thread(target=send_rubbish)
        sender.start()
        with pytest.raises(WatchSessionError, match="unusable frame after 0 packets"):
            consume_raw_pcm_connection(right, lambda _: None)
        sender.join()
    finally:
        left.close()
        right.close()
