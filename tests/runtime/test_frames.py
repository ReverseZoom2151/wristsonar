from __future__ import annotations

import json
from io import StringIO

import numpy as np
import pytest

from wristsonar.runtime import CaptureFrame, JsonLinesSink, PoseFrame


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
