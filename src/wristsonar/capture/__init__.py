"""Raw capture validation, alignment, and causal window construction."""

from wristsonar.capture.health import CaptureHealth, DuplexValidator
from wristsonar.capture.processor import (
    CaptureProcessingError,
    EchoWindowAssembler,
    ProcessedFrame,
)
from wristsonar.capture.synchronizer import (
    AlignedPcmFrame,
    PcmSynchronizer,
    SynchronizationError,
)

__all__ = [
    "AlignedPcmFrame",
    "CaptureHealth",
    "CaptureProcessingError",
    "DuplexValidator",
    "EchoWindowAssembler",
    "PcmSynchronizer",
    "ProcessedFrame",
    "SynchronizationError",
]
