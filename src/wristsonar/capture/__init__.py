"""Raw capture validation, alignment, and causal window construction."""

from wristsonar.capture.health import CaptureHealth, DuplexValidator
from wristsonar.capture.listener import consume_raw_pcm_connection
from wristsonar.capture.live import LiveCaptureError, LiveCaptureProcessor
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
from wristsonar.capture.wire import (
    RawPcmWireFrame,
    WireError,
    decode_raw_pcm,
    encode_raw_pcm,
    recv_raw_pcm,
)

__all__ = [
    "AlignedPcmFrame",
    "CaptureHealth",
    "CaptureProcessingError",
    "DuplexValidator",
    "EchoWindowAssembler",
    "LiveCaptureError",
    "LiveCaptureProcessor",
    "PcmSynchronizer",
    "ProcessedFrame",
    "RawPcmWireFrame",
    "SynchronizationError",
    "WireError",
    "consume_raw_pcm_connection",
    "decode_raw_pcm",
    "encode_raw_pcm",
    "recv_raw_pcm",
]
