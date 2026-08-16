"""Output sinks. JSON Lines is the portable baseline for Blender and XR."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TextIO

from wristsonar.runtime.frames import PoseFrame

__all__ = ["JsonLinesSink", "Sink"]


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
