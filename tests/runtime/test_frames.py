from __future__ import annotations

import json
import socket
import threading
from io import StringIO

import numpy as np
import pytest

from wristsonar.runtime import CaptureFrame, JsonLinesSink, PoseFrame, TcpJsonLinesSink


def test_capture_refuses_an_ambiguous_channel_layout() -> None:
    with pytest.raises(ValueError, match=r"original.differential"):
        CaptureFrame(np.zeros((60, 96), dtype=np.float32), 1.0, 48_000)


def test_jsonl_sink_preserves_the_reference_frame_and_calibration() -> None:
    output = StringIO()
    frame = PoseFrame(
        np.zeros((21, 3), dtype=np.float32), 2.0, 1.9, "baseline@abc", 2.0
    )
    JsonLinesSink(output).emit(frame)
    payload = json.loads(output.getvalue())
    assert payload["frame"] == "wrist-relative"
    assert payload["calibration_minutes"] == 2.0
    assert len(payload["joints"]) == 21


def test_tcp_sink_sends_the_same_jsonl_contract() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    received: list[bytes] = []

    def read_one() -> None:
        connection, _ = server.accept()
        with connection:
            received.append(connection.recv(4096))
        server.close()

    thread = threading.Thread(target=read_one)
    thread.start()
    sink = TcpJsonLinesSink(port=port)
    sink.emit(PoseFrame(np.zeros((21, 3), dtype=np.float32), 2, 2, "m", 0))
    sink.close()
    thread.join(timeout=2)
    assert json.loads(received[0])["type"] == "wristsonar.pose"
