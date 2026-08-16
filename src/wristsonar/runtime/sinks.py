"""Output sinks. JSON Lines is the portable baseline for Blender and XR."""

from __future__ import annotations

import json
import socket
from abc import ABC, abstractmethod
from typing import TextIO

from wristsonar.runtime.frames import PoseFrame

__all__ = ["JsonLinesSink", "Sink", "TcpJsonLinesSink"]


class Sink(ABC):
    @abstractmethod
    def emit(self, frame: PoseFrame) -> None:
        """Consume one pose frame without changing its reference frame."""


class JsonLinesSink(Sink):
    """A dependency-free streaming sink consumed by Blender or a relay."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def emit(self, frame: PoseFrame) -> None:
        self._stream.write(json.dumps(frame.jsonable(), separators=(",", ":")) + "\n")
        self._stream.flush()


class TcpJsonLinesSink(Sink):
    """Local TCP variant used by the Blender script."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None

    def emit(self, frame: PoseFrame) -> None:
        payload = (json.dumps(frame.jsonable(), separators=(",", ":")) + "\n").encode()
        try:
            self._connection().sendall(payload)
        except OSError:
            self.close()
            raise

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _connection(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.create_connection((self._host, self._port))
        return self._socket
