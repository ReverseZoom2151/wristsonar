"""Live capture and sink contracts shared by a watch and host process."""

from wristsonar.runtime.frames import CaptureFrame, PoseFrame
from wristsonar.runtime.sinks import JsonLinesSink, Sink

__all__ = ["CaptureFrame", "JsonLinesSink", "PoseFrame", "Sink"]
